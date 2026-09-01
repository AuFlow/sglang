"""Native AR stage for HunyuanImage-3."""

import copy
import os
from typing import Any

import torch
from PIL import Image as PILImage

from sglang.multimodal_gen.configs.sample.hunyuan_image3 import (
    align_hunyuan_image3_resolution,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import load_dict
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.vision import load_image

from .prompts import resolve_system_prompt
from .tokenizer import (
    HunyuanImage3TokenizerWrapper,
    ImageInfo,
    JointImageInfo,
    TokenizerEncodeOutput,
)

logger = init_logger(__name__)

# Max cond images per VAE encode pass (conv activations grow with the batch).
_COND_VAE_ENCODE_CHUNK = 2


def _seed_for_output(seed, output_idx: int):
    if isinstance(seed, list):
        if output_idx < len(seed):
            return int(seed[output_idx])
        return int(seed[0]) + output_idx
    return int(seed) + output_idx


def _cond_image_to_pil(raw_img):
    if isinstance(raw_img, torch.Tensor):
        return PILImage.fromarray(raw_img.cpu().permute(1, 2, 0).numpy())
    try:
        return load_image(raw_img)
    except ValueError as e:
        logger.warning("Failed to load condition image %r: %s", raw_img, e)
        return None


def _vae_downsample_factors(hf_config: Any) -> tuple[int, int]:
    vae_factor = getattr(hf_config, "vae_downsample_factor", [16, 16])
    if isinstance(vae_factor, (list, tuple)):
        vae_h = vae_factor[0]
        vae_w = vae_factor[1] if len(vae_factor) > 1 else vae_factor[0]
    else:
        vae_h = vae_w = int(vae_factor)
    return vae_h, vae_w


def _build_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    image_slices: list[list[slice]],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(0).repeat(batch_size, 1, 1)

    for i in range(batch_size):
        for image_slice in image_slices[i]:
            mask[i, image_slice, image_slice] = True

    return mask.unsqueeze(1)


def _section_shapes(
    sections_row: list | None, fallback_h: int, fallback_w: int
) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for section in sections_row or []:
        stype = section.get("type", "")
        if "image" not in stype:
            continue
        t_h = section.get("token_height", fallback_h)
        t_w = section.get("token_width", fallback_w)
        if isinstance(t_h, list):
            # joint_image: list of [vae, vit] dims
            for h_i, w_i in zip(t_h, t_w):
                shapes.append((int(h_i), int(w_i)))
        else:
            shapes.append((int(t_h), int(t_w)))
    return shapes


def _build_rope_image_info(
    tokenizer_output: TokenizerEncodeOutput,
    batch_size: int,
    token_h: int,
    token_w: int,
    image_info: ImageInfo,
    sections: list | None = None,
) -> list[list[tuple[slice, tuple[int, int]]]]:
    th = image_info.token_height
    tw = image_info.token_width

    per_row_sections = bool(sections) and isinstance(sections[0], list)
    shared_shapes = [] if per_row_sections else _section_shapes(sections, th, tw)

    rope_image_info: list[list[tuple[slice, tuple[int, int]]]] = []
    for b in range(batch_size):
        section_shapes = (
            _section_shapes(sections[b], th, tw) if per_row_sections else shared_shapes
        )
        batch_info: list[tuple[slice, tuple[int, int]]] = []
        shape_idx = 0

        def _collect(slices_row: list[slice], fallback_shape: tuple[int, int]):
            nonlocal shape_idx
            for s in slices_row:
                if shape_idx < len(section_shapes):
                    batch_info.append((s, section_shapes[shape_idx]))
                    shape_idx += 1
                else:
                    batch_info.append((s, fallback_shape))

        # Cond (joint) images precede gen images in the sequence
        _collect(tokenizer_output.cond_vae_image_slices[b], (token_h, token_w))
        _collect(tokenizer_output.cond_vit_image_slices[b], (token_h, token_w))
        _collect(tokenizer_output.gen_image_slices[b], (th, tw))

        rope_image_info.append(batch_info)
    return rope_image_info


class HunyuanImage3AR(PipelineStage):
    """Native AR diffusion-loop stage for HunyuanImage-3."""

    def __init__(
        self,
        ar_model,
        vae,
        tokenizer,
        processor,
        scheduler,
        model_path: str,
        vision_model,
        vision_aligner,
    ):
        super().__init__()
        self.ar_model = ar_model
        self._vae = vae
        self._processor = processor
        self._scheduler = scheduler
        self._model_path = model_path
        self._vision_model = vision_model
        self._vision_aligner = vision_aligner
        self._custom_tokenizer = HunyuanImage3TokenizerWrapper(tokenizer)
        self._gen_config_cache: dict | None = None

    def _generation_config(self) -> dict:
        if self._gen_config_cache is None:
            self._gen_config_cache = load_dict(
                os.path.join(self._model_path, "generation_config.json")
            )
        return self._gen_config_cache

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                memory_intensive=True,
            )
        ]

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    def _rebuild_image_info(self, image_info):
        # Re-create as this module's ImageInfo so isinstance checks in the
        # tokenizer pass (the processor ships its own copy).
        if isinstance(image_info, ImageInfo):
            return image_info
        new_info = ImageInfo.__new__(ImageInfo)
        new_info.__dict__.update(image_info.__dict__)
        return new_info

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor],
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(-1, hidden_size).contiguous()
        attention_mask = attention_mask.contiguous()
        cos, sin = custom_pos_emb
        cos = cos.contiguous()
        sin = sin.contiguous()

        # Broadcast from rank 0 for deterministic TP collectives
        if model_parallel_is_initialized():
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                hidden_states = tp_group.broadcast(hidden_states, src=0)
                attention_mask = tp_group.broadcast(attention_mask, src=0)
                cos = tp_group.broadcast(cos, src=0)
                sin = tp_group.broadcast(sin, src=0)

        output = self.ar_model.forward_block(
            hidden_states,
            attention_mask,
            (cos, sin),
            timestep=timestep,
        )
        # batch_size may differ after TP broadcast
        actual_batch = attention_mask.shape[0]
        actual_seq_len = output.shape[0] // actual_batch
        return output.view(actual_batch, actual_seq_len, hidden_size)

    def _instantiate_vae_tokens_first_step(
        self,
        hidden_states: torch.Tensor,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, n_embd = hidden_states.shape
        t_emb = self.ar_model.time_embed(timesteps)
        image_seq, token_h, token_w = self.ar_model.patch_embed(images, t_emb)
        image_scatter_index = (
            torch.arange(seqlen, device=hidden_states.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        image_scatter_index = image_scatter_index.masked_select(image_mask.bool()).reshape(bsz, -1)
        hidden_states.scatter_(
            dim=1,
            index=image_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=image_seq,
        )
        return hidden_states

    def _instantiate_timestep_tokens(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        timestep_index: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, n_embd = hidden_states.shape
        timestep_emb = self.ar_model.timestep_emb(timesteps).reshape(bsz, -1, n_embd)

        if timestep_index.dtype == torch.bool:
            index = (
                torch.arange(seqlen, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
            ts_scatter_index = index.masked_select(timestep_index).reshape(bsz, -1)
        else:
            ts_scatter_index = timestep_index.long()

        num_positions = ts_scatter_index.shape[1]
        timestep_emb = timestep_emb.expand(-1, num_positions, -1)
        hidden_states.scatter_(
            dim=1,
            index=ts_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
            src=timestep_emb,
        )
        return hidden_states

    def _extract_diffusion_pred(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        image_mask: torch.Tensor,
        token_h: int,
        token_w: int,
    ) -> torch.Tensor:
        n_embd = hidden_states.size(-1)
        t_emb = self.ar_model.time_embed_2(timesteps)

        image_output = hidden_states.masked_select(
            image_mask.unsqueeze(-1).bool()
        ).reshape(-1, token_h * token_w, n_embd)

        return self.ar_model.final_layer(image_output, t_emb, token_h, token_w)

    @staticmethod
    def _resize_and_crop_center(image, target_width: int, target_height: int):
        tw, th = target_width, target_height
        w, h = image.size
        tr = th / tw
        r = h / w
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))
        resized = image.resize((resize_width, resize_height), PILImage.Resampling.LANCZOS)
        crop_left = int(round((resize_width - tw) / 2.0))
        crop_top = int(round((resize_height - th) / 2.0))
        return resized.crop((crop_left, crop_top, crop_left + tw, crop_top + th))

    def _preprocess_cond_image(self, pil_image):
        processor = self._processor
        pil_image = pil_image.convert("RGB")
        orig_width, orig_height = pil_image.size

        vae_h_factor, vae_w_factor = _vae_downsample_factors(self.ar_model.hf_config)

        base_size, ratio_idx = processor.vae_reso_group.get_base_size_and_ratio_index(
            orig_width, orig_height
        )
        base_size = int(base_size)
        ratio_idx = int(ratio_idx)
        reso = processor.vae_reso_group[ratio_idx]
        target_width = int(reso.width)
        target_height = int(reso.height)

        vae_input = self._resize_and_crop_center(pil_image, target_width, target_height)
        vae_tensor = processor.pil_image_to_tensor(vae_input)

        vae_info = ImageInfo(
            image_type="vae",
            image_width=target_width,
            image_height=target_height,
            token_width=target_width // vae_w_factor,
            token_height=target_height // vae_h_factor,
            base_size=base_size,
            ratio_index=ratio_idx,
        )

        vit_processor = processor.vit_processor
        vit_inputs = vit_processor(pil_image, return_tensors="pt")
        vit_tensor = vit_inputs["pixel_values"].squeeze(0)
        spatial_shapes = vit_inputs["spatial_shapes"].squeeze(0)
        pixel_attention_mask = vit_inputs["pixel_attention_mask"].squeeze(0)
        vit_token_h = int(spatial_shapes[0].item())
        vit_token_w = int(spatial_shapes[1].item())
        vit_patch_size = getattr(vit_processor, "patch_size", 1)
        if isinstance(vit_patch_size, (tuple, list)):
            vit_patch_size = int(vit_patch_size[0])

        vit_info = ImageInfo(
            image_type="siglip2",
            image_width=vit_token_w * vit_patch_size,
            image_height=vit_token_h * vit_patch_size,
            token_width=vit_token_w,
            token_height=vit_token_h,
            image_token_length=int(vit_tensor.shape[0]),
        )

        joint_info = JointImageInfo(
            vae_image_info=vae_info,
            vision_image_info=vit_info,
            vision_encoder_kwargs={
                "spatial_shapes": spatial_shapes,
                "pixel_attention_mask": pixel_attention_mask,
            },
        )
        vae_info.image_tensor = vae_tensor
        vit_info.image_tensor = vit_tensor
        return joint_info

    def _vae_encode_cond_image(self, vae_tensor, device):
        vae = self._vae
        vae_tensor = self._cond_vae_tensor_5d(vae_tensor).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            latents = vae.encode(vae_tensor).latent_dist.sample()
            if vae.shift_factor:
                latents.sub_(vae.shift_factor)
            if vae.scaling_factor:
                latents.mul_(vae.scaling_factor)

        latents = latents.squeeze(2)
        t = torch.zeros((latents.shape[0],))
        return t, latents.squeeze(0)

    def _preprocess_cond_images(self, per_request_raw_conds: list[list]) -> list[list]:
        per_request_joint_infos: list[list] = []
        for raw_conds in per_request_raw_conds:
            joints = []
            for raw_img in raw_conds or []:
                pil_img = _cond_image_to_pil(raw_img)
                if pil_img is None:
                    logger.warning(
                        "Skipping unsupported condition image (%s)",
                        type(raw_img).__name__,
                    )
                    continue
                joints.append(self._preprocess_cond_image(pil_img))
            per_request_joint_infos.append(joints)
        return per_request_joint_infos

    def _encode_conditions(self, per_request_joint_infos, device):
        flat_infos = [
            info for joints in per_request_joint_infos for info in joints
        ]
        request_bounds = []
        offset = 0
        for joints in per_request_joint_infos:
            request_bounds.append((offset, offset + len(joints)))
            offset += len(joints)

        if not flat_infos:
            empty_embeds: list[list] = [[] for _ in per_request_joint_infos]
            return (
                empty_embeds,
                [None] * len(per_request_joint_infos),
                [[] for _ in per_request_joint_infos],
            )

        try:
            batched = self._encode_conditions_batched(flat_infos, device)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(
                e
            ).lower():
                raise
            logger.warning("Batched cond encoding OOM; falling back to sequential")
            torch.get_device_module().empty_cache()
            batched = None
        if batched is None:
            batched = self._encode_conditions_sequential(flat_infos, device)
        vae_embeds, t_values, vit_embeds = batched

        per_request_vae_embeds: list[list] = []
        per_request_t: list = []
        per_request_vit_embeds: list[list] = []
        for start, end in request_bounds:
            per_request_vae_embeds.append(vae_embeds[start:end])
            t_slice = t_values[start:end]
            per_request_t.append(
                torch.cat(t_slice, dim=0) if t_slice else None
            )
            per_request_vit_embeds.append(vit_embeds[start:end])
        return per_request_vae_embeds, per_request_t, per_request_vit_embeds

    @staticmethod
    def _cond_vae_tensor_5d(image_tensor) -> torch.Tensor:
        tensor = image_tensor
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)
        return tensor

    def _encode_conditions_sequential(self, flat_infos, device):
        vae_embeds: list = []
        t_values: list = []
        vit_embeds: list = []
        # Autocast matches the diffusion loop so cached embeddings stay
        # bit-identical with in-loop computation.
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=torch.bfloat16,
            enabled=True,
        ):
            for info in flat_infos:
                t_i, latents_i = self._vae_encode_cond_image(
                    info.vae_image_info.image_tensor, device
                )
                # Keep t 1D: time_embed output feeds the ResBlock scale/shift split.
                t_i_emb = self.ar_model.time_embed(t_i.to(device))
                if latents_i.dim() == 3:
                    latents_i = latents_i.unsqueeze(0)
                image_i_seq, _, _ = self.ar_model.patch_embed(
                    latents_i.to(device), t_i_emb
                )
                vae_embeds.append(image_i_seq)
                t_values.append(t_i)

                vit_kwargs = self._cond_vit_kwargs(info)
                packed = self._pack_cond_vit_patches(info)
                if packed is None:
                    raise ValueError(
                        "Cannot pack cond ViT patches for spatial_shapes "
                        f"{vit_kwargs['spatial_shapes']}"
                    )
                pixels_packed, length = packed
                image_embed = self._vision_model(
                    pixels_packed.unsqueeze(0).to(device),
                    spatial_shapes=vit_kwargs["spatial_shapes"],
                    cu_seqlens=torch.tensor(
                        [0, length], dtype=torch.int32, device=device
                    ),
                    max_seqlen=length,
                )
                image_embed = self._vision_aligner(image_embed)
                # Re-pad to the processor's padded length (scatter contract).
                expected = info.vision_image_info.image_tensor.shape[0]
                emb = image_embed[0]
                if expected > length:
                    emb = torch.cat(
                        [emb, emb.new_zeros(expected - length, emb.shape[-1])],
                        dim=0,
                    )
                vit_embeds.append(emb)
        return vae_embeds, t_values, vit_embeds

    @staticmethod
    def _cond_vit_kwargs(info) -> dict:
        vit_kwargs = {
            "spatial_shapes": info.vision_encoder_kwargs["spatial_shapes"],
            "attention_mask": info.vision_encoder_kwargs[
                "pixel_attention_mask"
            ],
        }
        if vit_kwargs["spatial_shapes"].ndim == 1:
            vit_kwargs["spatial_shapes"] = vit_kwargs[
                "spatial_shapes"
            ].unsqueeze(0)
        if vit_kwargs["attention_mask"].ndim == 1:
            vit_kwargs["attention_mask"] = vit_kwargs[
                "attention_mask"
            ].unsqueeze(0)
        return vit_kwargs

    @staticmethod
    def _pack_cond_vit_patches(info) -> tuple[torch.Tensor, int] | None:
        vit_kwargs = HunyuanImage3AR._cond_vit_kwargs(info)
        row = vit_kwargs["spatial_shapes"][0]
        token_h = int(row[0].item())
        token_w = int(row[1].item())
        length = token_h * token_w
        pixels = info.vision_image_info.image_tensor
        if length <= 0 or length > pixels.shape[0]:
            return None
        return pixels[:length], length

    def _encode_conditions_batched(self, flat_infos, device):
        # Returns None when the images cannot be stacked (shape mismatch).
        vae_tensors = [
            self._cond_vae_tensor_5d(info.vae_image_info.image_tensor)
            for info in flat_infos
        ]
        if len({tuple(t.shape) for t in vae_tensors}) > 1:
            return None

        vit_tensors = [info.vision_image_info.image_tensor for info in flat_infos]
        feat_dim = vit_tensors[0].shape[-1]
        if any(t.ndim != 2 or t.shape[-1] != feat_dim for t in vit_tensors):
            return None

        # SRT Siglip2 takes packed variable-length input via cu_seqlens;
        # the packed output is re-padded afterwards (scatter contract).
        packed_pixels: list = []
        spatial_shape_rows: list = []
        lengths: list = []
        for info in flat_infos:
            vit_kwargs = self._cond_vit_kwargs(info)
            packed = self._pack_cond_vit_patches(info)
            if packed is None:
                return None
            pixels, length = packed
            packed_pixels.append(pixels)
            spatial_shape_rows.append(vit_kwargs["spatial_shapes"][0])
            lengths.append(length)

        # Autocast matches the diffusion loop so cached embeddings stay
        # bit-identical with in-loop computation.
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=torch.bfloat16,
            enabled=True,
        ):
            vae_embeds: list = []
            t_values: list = []
            for chunk_start in range(0, len(vae_tensors), _COND_VAE_ENCODE_CHUNK):
                chunk = vae_tensors[chunk_start : chunk_start + _COND_VAE_ENCODE_CHUNK]
                batched_vae = torch.cat(
                    [t.to(dtype=vae_tensors[0].dtype) for t in chunk], dim=0
                )
                t_chunk, latents_chunk = self._vae_encode_cond_image(
                    batched_vae, device
                )
                # Keep t 1D: time_embed output feeds the ResBlock scale/shift split.
                t_emb = self.ar_model.time_embed(t_chunk.to(device))
                if latents_chunk.dim() == 3:
                    latents_chunk = latents_chunk.unsqueeze(0)
                image_seq, _, _ = self.ar_model.patch_embed(
                    latents_chunk.to(device), t_emb
                )
                vae_embeds.extend(
                    image_seq[i] for i in range(image_seq.shape[0])
                )
                t_values.extend(
                    t_chunk[i : i + 1] for i in range(len(chunk))
                )

            pixel_values = torch.cat(
                [
                    p.to(device=device, dtype=vit_tensors[0].dtype)
                    for p in packed_pixels
                ],
                dim=0,
            ).unsqueeze(0)
            # spatial_shapes must stay on CPU (asserted by the SRT embeddings).
            spatial_shapes = torch.stack(spatial_shape_rows, dim=0)
            cu_seqlens = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                dtype=torch.int32, device=device,
            )
            image_embed = self._vision_model(
                pixel_values,
                spatial_shapes=spatial_shapes,
                cu_seqlens=cu_seqlens,
                max_seqlen=max(lengths),
            )
            image_embed = self._vision_aligner(image_embed)
            dim = image_embed.shape[-1]
            vit_embeds: list = []
            offset = 0
            for i, length in enumerate(lengths):
                emb = image_embed[0, offset : offset + length]
                expected = vit_tensors[i].shape[0]
                if expected > length:
                    emb = torch.cat(
                        [emb, emb.new_zeros(expected - length, dim)], dim=0
                    )
                vit_embeds.append(emb)
                offset += length
        return vae_embeds, t_values, vit_embeds

    def _scatter_cond_vae_tokens_batched(
        self, hidden_states, per_request_vae_embeds,
        cond_vae_slices_rows, n_req, do_cfg,
    ):
        n_embd = hidden_states.shape[-1]
        for r, embeds in enumerate(per_request_vae_embeds):
            if not embeds:
                continue
            target_rows = [r] + ([n_req + r] if do_cfg else [])
            for row in target_rows:
                slices_row = cond_vae_slices_rows[row]
                for i, s in enumerate(slices_row):
                    positions = torch.arange(
                        s.start, s.stop, device=hidden_states.device
                    )
                    hidden_states[row, positions] = embeds[i].reshape(-1, n_embd)
        return hidden_states

    def _scatter_cond_vit_tokens_batched(
        self, hidden_states, per_request_vit_embeds, cond_vit_slices_rows,
    ):
        for r, embeds in enumerate(per_request_vit_embeds):
            for i, embed in enumerate(embeds):
                s = cond_vit_slices_rows[r][i]
                positions = torch.arange(
                    s.start, s.stop, device=hidden_states.device
                )
                hidden_states[r, positions] = embed[: s.stop - s.start].to(
                    hidden_states.dtype
                )
        return hidden_states

    @staticmethod
    def _normalize_bot_task(bot_task: str) -> str:
        if bot_task in ("none", "vanilla"):
            return "image"
        if bot_task == "think_recaption":
            return "think"
        return bot_task

    def _resolve_generation_params(self, reqs: list[Req], raw_conds_rows: list):
        head = reqs[0]
        width, height = self._effective_resolution(head, raw_conds_rows[0])
        image_info = self._processor.build_gen_image_info(f"{height}x{width}")
        height = image_info.image_height
        width = image_info.image_width
        token_h = image_info.token_height
        token_w = image_info.token_width
        image_info = self._rebuild_image_info(image_info)
        for req in reqs:
            req.width, req.height = width, height

        guidance_scale = head.guidance_scale
        num_inference_steps = head.num_inference_steps
        return (
            width, height, token_h, token_w, image_info,
            guidance_scale, num_inference_steps,
        )

    def _build_tokenizer_kwargs(
        self, reqs: list[Req], image_info, tokenizer_bot_task: str, cfg_factor: int
    ) -> dict[str, Any]:
        gen_config = self._generation_config()
        tokenizer_kwargs: dict[str, Any] = dict(
            batch_prompt=[req.prompt for req in reqs],
            mode="gen_image",
            bot_task=tokenizer_bot_task,
            sequence_template=gen_config.get("sequence_template", "pretrain"),
            drop_think=gen_config.get("drop_think", False),
            cfg_factor=cfg_factor,
            image_base_size=self._processor.vae_reso_group.base_size,
        )
        resolved_prompt = resolve_system_prompt(
            reqs[0].system_prompt, bot_task=tokenizer_bot_task
        )
        if resolved_prompt is not None:
            tokenizer_kwargs["batch_system_prompt"] = (
                [resolved_prompt.strip()] * len(reqs)
            )
        cot_texts = [req.cot_text for req in reqs]
        if any(cot is not None for cot in cot_texts):
            tokenizer_kwargs["batch_cot_text"] = cot_texts
        tokenizer_kwargs["batch_gen_image_info"] = [image_info] * len(reqs)
        return tokenizer_kwargs

    @staticmethod
    def _tok_field(
        tokenizer_output: TokenizerEncodeOutput, name: str, device: torch.device
    ):
        value = getattr(tokenizer_output, name)
        return value.to(device) if torch.is_tensor(value) else value

    def _parse_tokenizer_output(
        self, tokenizer_output: TokenizerEncodeOutput, device: torch.device
    ):
        input_ids = tokenizer_output.tokens.to(device)
        return dict(
            input_ids=input_ids,
            actual_batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            image_mask=tokenizer_output.gen_image_mask.to(device),
            timestep_index=self._tok_field(
                tokenizer_output, "gen_timestep_scatter_index", device
            ),
            cond_timestep_scatter_index=self._tok_field(
                tokenizer_output, "cond_timestep_scatter_index", device
            ),
            cond_vae_slices_rows=tokenizer_output.cond_vae_image_slices,
            cond_vit_slices_rows=tokenizer_output.cond_vit_image_slices,
        )

    def _build_attention_and_rope(
        self, tokenizer_output, tokenizer_sections, actual_batch_size: int,
        seq_len: int, token_h: int, token_w: int, image_info, device,
    ):
        gen_slices = tokenizer_output.gen_image_slices
        joint_slices = tokenizer_output.joint_image_slices
        image_slices = [
            joint_slices[i] + gen_slices[i] for i in range(actual_batch_size)
        ]
        attention_mask = _build_causal_attention_mask(
            actual_batch_size, seq_len, image_slices, device
        )

        rope_image_info = _build_rope_image_info(
            tokenizer_output, actual_batch_size, token_h, token_w, image_info,
            sections=tokenizer_sections,
        )
        cos, sin = self.ar_model.cached_rope(
            seq_len, device, rope_image_info=rope_image_info
        )
        return attention_mask, cos, sin

    def _latent_dims(self, height: int, width: int) -> tuple[int, int, int]:
        hf_config = self.ar_model.hf_config
        if hasattr(hf_config, "vae") and isinstance(hf_config.vae, dict):
            latent_channels = hf_config.vae["latent_channels"]
        else:
            latent_channels = getattr(hf_config, "latent_channels", 32)
        vae_h, vae_w = _vae_downsample_factors(hf_config)
        return latent_channels, height // vae_h, width // vae_w

    def _prepare_noise(
        self, reqs: list[Req], latent_channels: int, latent_h: int,
        latent_w: int, device: torch.device,
    ) -> torch.Tensor:
        # One generator per request keeps each request bit-identical with its
        # single-request run.
        noise_rows = []
        for req in reqs:
            generator = torch.Generator(device=device)
            if req.seed is not None:
                generator.manual_seed(req.seed)
            noise_rows.append(
                torch.randn(
                    1, latent_channels, latent_h, latent_w,
                    generator=generator, device=device, dtype=torch.bfloat16,
                )
            )
        return torch.cat(noise_rows, dim=0)

    @staticmethod
    def _collect_raw_cond_images(req: Req):
        raw_cond_images = req.condition_image
        if raw_cond_images is None and req.image_path is not None:
            image_path = req.image_path
            raw_cond_images = image_path if isinstance(image_path, list) else [image_path]
        if raw_cond_images is not None and not isinstance(raw_cond_images, (list, tuple)):
            raw_cond_images = [raw_cond_images]
        return raw_cond_images

    @staticmethod
    def _effective_resolution(req: Req, raw_cond_images) -> tuple[int, int]:
        # TI2I inherits the reference size unless the user set width/height.
        user_explicit_fields = getattr(
            req.sampling_params, "_explicit_fields", set()
        )
        first_cond_pil = (
            _cond_image_to_pil(raw_cond_images[0]) if raw_cond_images else None
        )
        if (
            first_cond_pil is not None
            and "width" not in user_explicit_fields
            and "height" not in user_explicit_fields
        ):
            return align_hunyuan_image3_resolution(*first_cond_pil.size)
        return align_hunyuan_image3_resolution(req.width, req.height)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        clones = self._expand_multi_output(batch)
        outputs = self._forward_batched(clones)
        if len(outputs) == 1:
            return outputs[0]
        batch.latents = torch.cat([out.latents for out in outputs], dim=0)
        batch.width, batch.height = outputs[0].width, outputs[0].height
        return batch

    def run_grouped_requests(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> list[Req]:
        flat: list[Req] = []
        for batch in batches:
            flat.extend(self._expand_multi_output(batch))

        if len(flat) <= 1:
            return [self(req, server_args) for req in flat]

        results: list[Req | None] = [None] * len(flat)
        batchable: list[tuple[int, Req]] = []
        for index, req in enumerate(flat):
            if self._is_batchable(req):
                batchable.append((index, req))
            else:
                results[index] = self(req, server_args)

        buckets: dict[Any, list[tuple[int, Req]]] = {}
        for index, req in batchable:
            buckets.setdefault(self._batch_bucket_key(req), []).append((index, req))

        for group in buckets.values():
            if len(group) == 1:
                index, req = group[0]
                results[index] = self(req, server_args)
                continue
            outputs = self._forward_batched([req for _, req in group])
            for (index, _), output in zip(group, outputs):
                results[index] = output

        return [result for result in results if result is not None]

    @staticmethod
    def _expand_multi_output(req: Req) -> list[Req]:
        num_outputs = int(req.num_outputs_per_prompt)
        if num_outputs == 1:
            return [req]
        clones: list[Req] = []
        for output_idx in range(num_outputs):
            clone = copy.copy(req)
            clone.sampling_params = copy.copy(req.sampling_params)
            clone.extra = dict(req.extra)
            clone.metrics = copy.deepcopy(req.metrics)
            clone.num_outputs_per_prompt = 1
            clone.seed = _seed_for_output(req.seed, output_idx)
            clone.seeds = None
            clone.generator = None
            if req.request_id is not None:
                clone.request_id = f"{req.request_id}:{output_idx}"
                if clone.metrics is not None:
                    clone.metrics.request_id = clone.request_id
            clones.append(clone)
        return clones

    @staticmethod
    def _is_batchable(req: Req) -> bool:
        return not req.is_warmup

    def _batch_bucket_key(self, req: Req):
        raw_cond_images = self._collect_raw_cond_images(req)
        n_cond = len(raw_cond_images) if raw_cond_images else 0
        width, height = self._effective_resolution(req, raw_cond_images)
        return (
            width,
            height,
            req.num_inference_steps,
            req.guidance_scale,
            self._normalize_bot_task(req.bot_task),
            req.system_prompt,
            n_cond,
        )

    @torch.no_grad()
    def _forward_batched(self, reqs: list[Req]) -> list[Req]:
        tokenizer = self._custom_tokenizer
        n_req = len(reqs)
        head = reqs[0]

        per_request_raw_conds = [self._collect_raw_cond_images(req) for req in reqs]
        has_cond = any(bool(conds) for conds in per_request_raw_conds)
        (
            width, height, token_h, token_w, image_info,
            guidance_scale, num_inference_steps,
        ) = self._resolve_generation_params(reqs, per_request_raw_conds)
        do_cfg = guidance_scale > 1.0
        cfg_factor = 2 if do_cfg else 1

        # cpu_offload may leave weights on CPU
        device = get_local_torch_device()
        model_device = self.ar_model.model.embed_tokens.weight.device
        if model_device.type == "cpu":
            self.log_info("Moving AR model from CPU to %s", device)
            self.ar_model.to(device)
        else:
            device = model_device

        tokenizer_bot_task = self._normalize_bot_task(head.bot_task)
        tokenizer_kwargs = self._build_tokenizer_kwargs(
            reqs, image_info, tokenizer_bot_task, cfg_factor
        )

        per_request_joint_infos: list[list] = []
        if has_cond:
            per_request_joint_infos = self._preprocess_cond_images(
                per_request_raw_conds
            )
            tokenizer_kwargs["batch_cond_image_info"] = per_request_joint_infos

        tokenizer_output_dict = tokenizer.apply_chat_template(**tokenizer_kwargs)
        tokenizer_output = tokenizer_output_dict["output"]
        tokenizer_sections = tokenizer_output_dict["sections"]

        tok = self._parse_tokenizer_output(tokenizer_output, device)
        input_ids = tok["input_ids"]
        actual_batch_size = tok["actual_batch_size"]
        image_mask = tok["image_mask"]
        timestep_index = tok["timestep_index"]

        attention_mask, cos, sin = self._build_attention_and_rope(
            tokenizer_output, tokenizer_sections, actual_batch_size,
            tok["seq_len"], token_h, token_w, image_info, device,
        )

        scheduler = self._scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        latent_channels, latent_h, latent_w = self._latent_dims(height, width)
        latents = self._prepare_noise(
            reqs, latent_channels, latent_h, latent_w, device
        )

        per_request_vae_embeds: list[list] = []
        per_request_t: list = []
        per_request_vit_embeds: list[list] = []
        has_cond_encoded = False
        if has_cond and any(per_request_joint_infos):
            self._vision_model.to(device)
            self._vision_model.eval()
            self._vision_aligner.to(device)
            self._vision_aligner.eval()
            (
                per_request_vae_embeds, per_request_t, per_request_vit_embeds,
            ) = self._encode_conditions(per_request_joint_infos, device)
            has_cond_encoded = True

        cond_vae_slices_rows = tok["cond_vae_slices_rows"]
        cond_vit_slices_rows = tok["cond_vit_slices_rows"]
        cond_timestep_scatter_index = tok["cond_timestep_scatter_index"]

        for step_idx, t in enumerate(self.progress_bar(timesteps, batch=head)):
            latent_model_input = scheduler.scale_model_input(latents, t)
            if do_cfg:
                latent_model_input = torch.cat([latent_model_input] * cfg_factor, dim=0)
            latent_bs = latent_model_input.shape[0]
            t_expand = t.repeat(latent_bs).to(device)

            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=torch.bfloat16,
                enabled=True,
            ), set_forward_context(
                current_timestep=step_idx,
                attn_metadata=None,
                forward_batch=head,
            ):
                # Re-embed the full input_ids every step; shortening produces garbage.
                hidden_states = self.ar_model.model.get_input_embeddings(input_ids)
                hidden_states = self._instantiate_vae_tokens_first_step(
                    hidden_states, latent_model_input, t_expand, image_mask,
                )
                if timestep_index is not None:
                    hidden_states = self._instantiate_timestep_tokens(
                        hidden_states, t_expand, timestep_index,
                    )

                if has_cond_encoded:
                    hidden_states = self._scatter_cond_vae_tokens_batched(
                        hidden_states, per_request_vae_embeds,
                        cond_vae_slices_rows, n_req, do_cfg,
                    )
                    hidden_states = self._scatter_cond_vit_tokens_batched(
                        hidden_states, per_request_vit_embeds,
                        cond_vit_slices_rows,
                    )
                    if cond_timestep_scatter_index is not None:
                        all_cond_t = torch.cat(per_request_t, dim=0).repeat(cfg_factor)
                        hidden_states = self._instantiate_timestep_tokens(
                            hidden_states,
                            all_cond_t.to(hidden_states.device),
                            cond_timestep_scatter_index,
                        )

                # CFG: run cond/uncond halves separately (halves peak attention memory).
                if do_cfg:
                    half = hidden_states.shape[0] // 2
                    out_cond = self._backbone_forward(
                        hidden_states[:half],
                        attention_mask[:half], (cos[:half], sin[:half]),
                        timestep=t_expand[:half],
                    )
                    out_uncond = self._backbone_forward(
                        hidden_states[half:],
                        attention_mask[half:], (cos[half:], sin[half:]),
                        timestep=t_expand[half:],
                    )
                    backbone_out = torch.cat([out_cond, out_uncond], dim=0)
                else:
                    backbone_out = self._backbone_forward(
                        hidden_states, attention_mask, (cos, sin),
                        timestep=t_expand,
                    )

                pred = self._extract_diffusion_pred(
                    backbone_out, t_expand, image_mask, token_h, token_w,
                )

            pred = pred.float()

            if do_cfg:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            latent_dtype = latents.dtype
            latents = scheduler.step(pred, t, latents, return_dict=False)[0].to(dtype=latent_dtype)

        # [B, C, H, W] -> [B, C, 1, H, W]
        latents = latents.to(torch.bfloat16)
        for i, req in enumerate(reqs):
            req.latents = latents[i : i + 1].unsqueeze(2)

        self.log_info(
            "AR stage: %d req(s) -> latents %s (%dx%d)",
            n_req,
            tuple(latents.shape),
            height,
            width,
        )
        return reqs
