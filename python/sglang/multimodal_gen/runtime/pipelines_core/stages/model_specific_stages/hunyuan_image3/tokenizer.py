"""Tokenizer wrapper for HunyuanImage-3: multimodal ``apply_chat_template``
over a base HF tokenizer (self-contained mirror of vllm-omni's wrapper).
"""

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ImageInfo:
    """Stores image metadata for the tokenizer (mirrors vllm-omni's class)."""

    def __init__(
        self,
        image_type: str = None,
        image_width: int = None,
        image_height: int = None,
        token_width: int = None,
        token_height: int = None,
        image_token_length: int = None,
        base_size: int = None,
        ratio_index: int = None,
        **kwargs,
    ):
        self.image_type = image_type
        self.image_width = image_width
        self.image_height = image_height
        self.token_width = token_width
        self.token_height = token_height
        self.image_token_length = (
            image_token_length
            if image_token_length is not None
            else (
                token_width * token_height
                if token_width is not None and token_height is not None
                else None
            )
        )
        self.base_size = base_size
        self.ratio_index = ratio_index

        self.add_timestep_token = kwargs.get("add_timestep_token", True)
        self.add_guidance_token = kwargs.get("add_guidance_token", False)
        self.use_front_boi_token = kwargs.get("use_front_boi_token", True)
        self.add_image_shape_token = kwargs.get("add_image_shape_token", True)

    @property
    def meta_info(self):
        if self.image_type in ["vae", "gen_image", "joint_image"]:
            return dict(
                token_length=self.image_token_length,
                add_timestep_token=self.add_timestep_token,
                add_guidance_token=self.add_guidance_token,
                use_front_boi_token=self.use_front_boi_token,
                add_image_shape_token=self.add_image_shape_token,
                base_size=self.base_size,
                ratio_idx=self.ratio_index,
                token_height=self.token_height,
                token_width=self.token_width,
                image_height=self.image_height,
                image_width=self.image_width,
            )
        raise ValueError(f"Unknown image type '{self.image_type}'")


class JointImageInfo:
    """Dual VAE + ViT metadata for a joint image; ``token_length`` is
    ``[vae_len, vit_len]``.
    """

    def __init__(self, vae_image_info: ImageInfo, vision_image_info: ImageInfo,
                 vision_encoder_kwargs: dict = None):
        self.vae_image_info = vae_image_info
        self.vision_image_info = vision_image_info
        self.vision_encoder_kwargs = vision_encoder_kwargs or {}

        self.image_type = "joint_image"
        self.image_token_length = (
            vae_image_info.image_token_length + vision_image_info.image_token_length
        )
        self.add_timestep_token = vae_image_info.add_timestep_token
        self.use_front_boi_token = vae_image_info.use_front_boi_token
        self.add_image_shape_token = vae_image_info.add_image_shape_token

    @property
    def meta_info(self):
        return dict(
            token_length=[
                self.vae_image_info.image_token_length,
                self.vision_image_info.image_token_length,
            ],
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            add_image_shape_token=self.add_image_shape_token,
            base_size=self.vae_image_info.base_size,
            ratio_idx=self.vae_image_info.ratio_index,
            token_height=[
                self.vae_image_info.token_height,
                self.vision_image_info.token_height,
            ],
            token_width=[
                self.vae_image_info.token_width,
                self.vision_image_info.token_width,
            ],
            image_height=[
                self.vae_image_info.image_height,
                self.vision_image_info.image_height,
            ],
            image_width=[
                self.vae_image_info.image_width,
                self.vision_image_info.image_width,
            ],
        )


@dataclass
class TokenizerEncodeOutput:
    tokens: torch.Tensor = None
    gen_image_slices: list = None
    gen_image_mask: torch.Tensor = None
    gen_timestep_scatter_index: torch.Tensor = None
    cond_timestep_scatter_index: torch.Tensor = None
    cond_vae_image_mask: torch.Tensor = None
    cond_vit_image_mask: torch.Tensor = None
    cond_vae_image_slices: list = None
    cond_vit_image_slices: list = None
    joint_image_slices: list = None


class _Conversation:
    roles: list = ["User", "Assistant"]
    sep: str = "\n\n"


class HunyuanImage3TokenizerWrapper:
    """Wraps a base HF tokenizer with multimodal ``apply_chat_template``
    (mirrors vllm-omni's ``TokenizerWrapper``).
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.bos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.boi_token_id = self.tokenizer.convert_tokens_to_ids("<boi>")
        self.eoi_token_id = self.tokenizer.convert_tokens_to_ids("<eoi>")
        self.img_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.cfg_token_id = self.tokenizer.convert_tokens_to_ids("<cfg>")
        self.end_recaption_token_id = self.tokenizer.convert_tokens_to_ids("</recaption>")
        self.end_think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.joint_img_sep_token_id = self.tokenizer.convert_tokens_to_ids("<joint_img_sep>")
        self.special_token_map = self.tokenizer.added_tokens_encoder

    @staticmethod
    def _pad(tensors, pad_val=0):
        max_len = max(t.shape[0] for t in tensors)
        out = []
        for t in tensors:
            if t.shape[0] < max_len:
                t = F.pad(t, (0, max_len - t.shape[0]), value=pad_val)
            out.append(t)
        return out

    def _get_cot_sections(self, cot_text, uncond_kwargs):
        """Parse <think>/</think> or <recaption>/</recaption> blocks."""
        if not cot_text:
            return []
        if "<think>" in cot_text and "</think>" in cot_text:
            before = cot_text.split("<think>")[0]
            think = cot_text.split("<think>")[1].split("</think>")[0]
            after = cot_text.split("</think>")[1]
            return (
                self._get_cot_sections(before, uncond_kwargs)
                + [
                    dict(type="text", text="<think>"),
                    dict(type="text", text=think, **uncond_kwargs),
                    dict(type="text", text="</think>"),
                ]
                + self._get_cot_sections(after, uncond_kwargs)
            )
        if "<recaption>" in cot_text and "</recaption>" in cot_text:
            before = cot_text.split("<recaption>")[0]
            recaption = cot_text.split("<recaption>")[1].split("</recaption>")[0]
            after = cot_text.split("</recaption>")[1]
            return (
                self._get_cot_sections(before, uncond_kwargs)
                + [
                    dict(type="text", text="<recaption>"),
                    dict(type="text", text=recaption, **uncond_kwargs),
                    dict(type="text", text="</recaption>"),
                ]
                + self._get_cot_sections(after, uncond_kwargs)
            )
        return [dict(type="text", text=cot_text, **uncond_kwargs)]

    def encode_text(self, *texts, uncond_enabled=None, uncond_p=None):
        """Encode text(s); with ``uncond_p == 1.0`` (the CFG unconditional
        pass) eligible texts are replaced by ``<cfg>`` tokens."""
        if uncond_enabled is None:
            uncond_enabled = [True] * len(texts)
        elif isinstance(uncond_enabled, bool):
            uncond_enabled = [uncond_enabled] * len(texts)
        assert len(uncond_enabled) == len(texts)

        do_uncond_drop = uncond_p == 1.0
        text_tokens = []
        for text, uncond_flag in zip(texts, uncond_enabled):
            text_token = self.tokenizer.encode(text, add_special_tokens=False)
            if uncond_flag and do_uncond_drop:
                text_token = [self.cfg_token_id] * len(text_token)
            text_tokens.extend(text_token)
        return text_tokens

    def encode_sequence(self, template, token_source):
        """Assemble token sequence from *template* (e.g. ``text-text-gen_image-text``)."""
        keys = template.split("-")
        index_indicator = {k: 0 for k in token_source}
        for v in token_source.values():
            assert isinstance(v, (list, tuple))

        assert set(keys) == set(token_source.keys())
        _key_counts = {k: 0 for k in keys}
        for k in keys:
            _key_counts[k] += 1
        for k, c in _key_counts.items():
            assert len(token_source[k]) == c

        token_seq = [self.bos_token_id]
        token_count = 1
        extra = defaultdict(list)

        for key in keys:
            source = token_source[key][index_indicator[key]]

            if key == "text":
                token_seq.extend(source)
                extra["<text>_start"].append(token_count)
                if "<cfg>_start" not in extra and len(source) > 0 and source[0] == self.cfg_token_id:
                    extra["<cfg>_start"].append(token_count)
                token_count += len(source)
                extra["<text>_end"].append(token_count - 1)
                if len(source) > 0 and source[-1] == self.end_think_token_id:
                    extra["<think>_end"].append(token_count - 1)
                if len(source) > 0 and source[-1] == self.end_recaption_token_id:
                    extra["<recaption>_end"].append(token_count - 1)

            elif key == "gen_image":
                if source["front_boi"]:
                    token_seq.append(self.boi_token_id)
                    extra["boi"].append(token_count)
                    token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq, token_count, extra,
                    add_timestep_token=source["timestep"],
                    add_guidance_token=source["guidance"],
                    add_image_shape_token=source["image_shape"],
                    base_size=source["base_size"],
                    ratio_idx=source["ratio_idx"],
                    image_type=key,
                )
                if not source["front_boi"]:
                    token_seq.append(self.boi_token_id)
                    extra["boi"].append(token_count)
                    token_count += 1
                token_seq.extend([self.img_token_id] * source["length"] + [self.eoi_token_id])
                extra["<img>_start"].append(token_count)
                extra["<all_img>_start"].append(token_count)
                token_count += source["length"]
                extra["<img>_end"].append(token_count - 1)
                extra["<all_img>_end"].append(token_count - 1)
                extra["eoi"].append(token_count)
                token_count += 1

            elif key == "joint_image":
                # Dual layout: VAE tokens + <joint_img_sep> + ViT tokens
                assert isinstance(source["length"], list) and len(source["length"]) == 2, (
                    "joint_image length should be a list of two integers [vae_len, vit_len]"
                )
                vae_len, vit_len = source["length"]
                if source["front_boi"]:
                    token_seq.append(self.boi_token_id)
                    extra["boi"].append(token_count)
                    token_count += 1
                # No guidance token for joint_image
                token_count = self._add_image_meta_info_token(
                    token_seq, token_count, extra,
                    add_timestep_token=source["timestep"],
                    add_guidance_token=False,
                    add_image_shape_token=source["image_shape"],
                    base_size=source["base_size"],
                    ratio_idx=source["ratio_idx"],
                    image_type=key,
                )
                if not source["front_boi"]:
                    token_seq.append(self.boi_token_id)
                    extra["boi"].append(token_count)
                    token_count += 1
                token_seq.extend([self.img_token_id] * vae_len)
                extra["<vae_img>_start"].append(token_count)
                extra["<joint_img>_start"].append(token_count)
                extra["<all_img>_start"].append(token_count)
                token_count += vae_len
                extra["<vae_img>_end"].append(token_count - 1)
                extra["<all_img>_end"].append(token_count - 1)
                token_seq.append(self.joint_img_sep_token_id)
                extra["joint_img_sep"].append(token_count)
                token_count += 1
                token_seq.extend([self.img_token_id] * vit_len)
                extra["<vit_img>_start"].append(token_count)
                extra["<all_img>_start"].append(token_count)
                token_count += vit_len
                extra["<vit_img>_end"].append(token_count - 1)
                extra["<joint_img>_end"].append(token_count - 1)
                extra["<all_img>_end"].append(token_count - 1)
                token_seq.append(self.eoi_token_id)
                extra["eoi"].append(token_count)
                token_count += 1

            else:
                raise ValueError(f"Unsupported key: {key}")
            index_indicator[key] += 1

        return token_seq, extra

    def _add_image_meta_info_token(
        self, token_seq, token_count, extra_token_pos,
        add_timestep_token=False, add_image_shape_token=False,
        add_guidance_token=False, base_size=None, ratio_idx=None,
        image_type=None,
    ):
        if add_image_shape_token:
            token_seq.extend([
                self.special_token_map[f"<img_size_{base_size}>"],
                self.special_token_map[f"<img_ratio_{ratio_idx}>"],
            ])
            token_count += 2
        if add_timestep_token:
            token_seq.extend([self.special_token_map["<timestep>"]])
            extra_token_pos["timestep"].append(token_count)
            if image_type == "gen_image":
                extra_token_pos["gen_timestep"].append(token_count)
            elif image_type == "joint_image":
                extra_token_pos["cond_timestep"].append(token_count)
            token_count += 1
        if add_guidance_token:
            token_seq.extend([self.special_token_map["<guidance>"]])
            extra_token_pos["guidance"].append(token_count)
            token_count += 1
        return token_count

    def encode_general(self, sections):
        """Encode a list of section dicts into a ``TokenizerEncodeOutput``."""
        template = "-".join(s["type"] for s in sections)

        token_source = defaultdict(list)
        for section in sections:
            if section["type"] == "text":
                text = self.encode_text(
                    section["text"],
                    uncond_enabled=section.get("uncond_enabled"),
                    uncond_p=section.get("uncond_p"),
                )
                token_source["text"].append(text)
            elif section["type"] == "gen_image":
                token_source["gen_image"].append(dict(
                    length=section["token_length"],
                    timestep=section.get("add_timestep_token", False),
                    guidance=section.get("add_guidance_token", False),
                    front_boi=section.get("use_front_boi_token", False),
                    image_shape=section.get("add_image_shape_token", False),
                    base_size=section.get("base_size"),
                    ratio_idx=section.get("ratio_idx"),
                ))
            elif section["type"] == "joint_image":
                token_source["joint_image"].append(dict(
                    length=section["token_length"],
                    timestep=section.get("add_timestep_token", False),
                    guidance=section.get("add_guidance_token", False),
                    front_boi=section.get("use_front_boi_token", False),
                    image_shape=section.get("add_image_shape_token", False),
                    base_size=section.get("base_size"),
                    ratio_idx=section.get("ratio_idx"),
                ))
            else:
                raise ValueError(f"Invalid section type: {section['type']}")

        full_token_seq, extra = self.encode_sequence(
            template=template, token_source=dict(token_source),
        )
        full_tensor = torch.tensor(full_token_seq, dtype=torch.long)

        gen_ts_idx = torch.tensor(extra["gen_timestep"], dtype=torch.long) if "gen_timestep" in extra else None
        cond_ts_idx = torch.tensor(extra["cond_timestep"], dtype=torch.long) if "cond_timestep" in extra else None

        gen_image_slices = []
        gen_image_mask = None
        if "<img>_start" in extra and "<img>_end" in extra:
            gen_image_slices = [slice(s, e + 1) for s, e in zip(extra["<img>_start"], extra["<img>_end"])]
            gen_image_mask = torch.zeros_like(full_tensor, dtype=torch.bool)
            for sl in gen_image_slices:
                gen_image_mask[sl] = True

        joint_image_slices = []
        cond_vae_image_mask = None
        cond_vit_image_mask = None
        cond_vae_image_slices = []
        cond_vit_image_slices = []
        if "<vae_img>_start" in extra and "<vae_img>_end" in extra:
            cond_vae_image_slices = [slice(s, e + 1) for s, e in zip(extra["<vae_img>_start"], extra["<vae_img>_end"])]
            cond_vae_image_mask = torch.zeros_like(full_tensor, dtype=torch.bool)
            for sl in cond_vae_image_slices:
                cond_vae_image_mask[sl] = True
        if "<vit_img>_start" in extra and "<vit_img>_end" in extra:
            cond_vit_image_slices = [slice(s, e + 1) for s, e in zip(extra["<vit_img>_start"], extra["<vit_img>_end"])]
            cond_vit_image_mask = torch.zeros_like(full_tensor, dtype=torch.bool)
            for sl in cond_vit_image_slices:
                cond_vit_image_mask[sl] = True
        if "<joint_img>_start" in extra and "<joint_img>_end" in extra:
            joint_image_slices = [slice(s, e + 1) for s, e in zip(extra["<joint_img>_start"], extra["<joint_img>_end"])]

        return TokenizerEncodeOutput(
            tokens=full_tensor,
            gen_image_slices=gen_image_slices,
            gen_image_mask=gen_image_mask,
            gen_timestep_scatter_index=gen_ts_idx,
            cond_timestep_scatter_index=cond_ts_idx,
            cond_vae_image_mask=cond_vae_image_mask,
            cond_vit_image_mask=cond_vit_image_mask,
            cond_vae_image_slices=cond_vae_image_slices,
            cond_vit_image_slices=cond_vit_image_slices,
            joint_image_slices=joint_image_slices,
        )

    def apply_chat_template(
        self,
        batch_prompt,
        batch_gen_image_info,
        batch_cond_image_info=None,
        batch_system_prompt=None,
        batch_cot_text=None,
        sequence_template="pretrain",
        cfg_factor=1,
    ):
        """Main entry point — image-generation mode of vllm-omni's
        ``TokenizerWrapper.apply_chat_template``."""
        batch_size = len(batch_prompt)
        if not isinstance(batch_system_prompt, list):
            batch_system_prompt = [batch_system_prompt] * batch_size
        if not isinstance(batch_gen_image_info, list):
            batch_gen_image_info = [batch_gen_image_info] * batch_size
        batch_cot_text = batch_cot_text or [None] * batch_size
        batch_cond_image_info = batch_cond_image_info or [[] for _ in range(batch_size)]

        batch_message_list = []
        for prompt, sys_p, cot, img_info, cond_imgs in zip(
            batch_prompt, batch_system_prompt, batch_cot_text,
            batch_gen_image_info, batch_cond_image_info,
        ):
            ml = []
            if sys_p:
                ml.append(dict(role="system", type="text", content=sys_p))
            if len(cond_imgs) > 0:
                ml.extend([
                    dict(role="user", type="joint_image", content=c)
                    for c in cond_imgs
                ])
            ml.append(dict(role="user", type="text", content=prompt))
            if cot is not None:
                ml.append(dict(role="assistant", type="text", content=cot))
            ml.append(dict(role="assistant", type="gen_image", content=img_info))
            batch_message_list.append(ml)

        output, sections = self._apply_general_template(
            message_list=batch_message_list,
            sequence_template=sequence_template,
            cfg_factor=cfg_factor,
            batchify=True,
        )
        return dict(output=output, sections=sections)

    def _apply_general_template(
        self, message_list, sequence_template="instruct",
        uncond_p=0.0, cfg_factor=1, batchify=False,
    ):
        if batchify:
            return self._batch_gen_infer(
                infer_fn=self._apply_general_template,
                infer_fn_kwargs_list=[
                    dict(message_list=ml_i, sequence_template=sequence_template)
                    for ml_i in message_list
                ],
                do_classifier_free_guidance=cfg_factor > 1,
            )

        conv = _Conversation()
        uncond_kwargs = dict(uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)

        if sequence_template == "instruct":
            answer_prefix, answer_suffix = "<answer>", "</answer>"
        else:
            answer_prefix, answer_suffix = "", ""

        if sequence_template == "pretrain":
            system_suffix = user_prefix = user_suffix = bot_prefix = bot_suffix = ""
        else:
            system_suffix = conv.sep
            user_prefix = f"{conv.roles[0]}: "
            user_suffix = conv.sep
            bot_prefix = f"{conv.roles[1]}: "
            bot_suffix = conv.sep

        sections: list[dict] = []
        cur_idx = 0
        while cur_idx < len(message_list):
            for role, pfx, sfx, apfx, asfx in [
                ("system", "", system_suffix, "", ""),
                ("user", user_prefix, user_suffix, "", ""),
                ("assistant", bot_prefix, bot_suffix, answer_prefix, answer_suffix),
            ]:
                sub, cur_idx = self._process_successive(
                    message_list, cur_idx, role, pfx, sfx, apfx, asfx,
                    uncond_kwargs=uncond_kwargs,
                )
                sections.extend(sub)

        output = self.encode_general(sections=sections)
        return output, sections

    def _process_successive(self, message_list, cur_idx, role,
                            prefix, suffix, answer_prefix="", answer_suffix="",
                            uncond_kwargs=None):
        if uncond_kwargs is None:
            uncond_kwargs = {}
        sub_sections: list[dict] = []
        while cur_idx < len(message_list) and message_list[cur_idx]["role"] == role:
            msg = message_list[cur_idx]
            if msg["type"] == "text":
                text = msg["content"]
                if role == "system":
                    sub_sections.append(dict(type="text", text=text))
                elif role == "assistant":
                    if ("<think>" in text and "</think>" in text) or (
                        "<recaption>" in text and "</recaption>" in text
                    ):
                        sub_sections.extend(self._get_cot_sections(text, uncond_kwargs))
                    else:
                        sub_sections.append(dict(type="text", text=text, **uncond_kwargs))
                else:
                    # User text: no answer tags, but uncond_kwargs apply for CFG
                    sub_sections.append(
                        dict(type="text", text=f"{answer_prefix}{text}{answer_suffix}", **uncond_kwargs)
                    )
            elif msg["type"] in ("gen_image", "joint_image"):
                info = msg["content"]
                expected_cls = JointImageInfo if msg["type"] == "joint_image" else ImageInfo
                assert isinstance(info, expected_cls), (
                    f"Expected {expected_cls.__name__}, got {type(info).__name__}"
                )
                if role == "assistant" and msg["type"] == "gen_image":
                    sub_sections.append(dict(type="text", text=answer_prefix))
                sub_sections.append(dict(type=msg["type"], **info.meta_info))
                if role == "assistant" and msg["type"] == "gen_image":
                    sub_sections.append(dict(type="text", text=answer_suffix))
            else:
                raise ValueError(f"Unknown message type: {msg['type']}")
            cur_idx += 1

        if sub_sections:
            sub_sections.insert(0, dict(type="text", text=prefix))
            sub_sections.append(dict(type="text", text=suffix))
        return sub_sections, cur_idx

    def _batch_gen_infer(self, infer_fn, infer_fn_kwargs_list,
                         do_classifier_free_guidance=False):
        """Run infer_fn per row; with CFG also run the unconditional pass
        (``uncond_p=1.0``) and merge cond rows followed by uncond rows."""
        cond_outputs, cond_sections = [], []
        uncond_outputs, uncond_sections = [], []
        for kw in infer_fn_kwargs_list:
            cond_kw = {**kw, "uncond_p": 0.0} if do_classifier_free_guidance else kw
            output, sections = infer_fn(**cond_kw)
            cond_outputs.append(output)
            cond_sections.append(sections)
            if do_classifier_free_guidance:
                output, sections = infer_fn(**{**kw, "uncond_p": 1.0})
                uncond_outputs.append(output)
                uncond_sections.append(sections)
        return (
            self._merge_encode_outputs(cond_outputs + uncond_outputs),
            cond_sections + uncond_sections,
        )

    def _merge_encode_outputs(self, outputs):
        merged = {}
        for key in TokenizerEncodeOutput.__dataclass_fields__:
            vals = [getattr(o, key) for o in outputs]
            if isinstance(vals[0], torch.Tensor):
                if "mask" in key:
                    pad_val = 0.0
                elif key == "tokens":
                    pad_val = self.special_token_map.get("<pad>", self.pad_token_id)
                else:
                    pad_val = False
                merged[key] = torch.stack(self._pad(vals, pad_val=pad_val), dim=0)
            elif isinstance(vals[0], list):
                merged[key] = vals
            elif vals[0] is None:
                merged[key] = None
            else:
                merged[key] = vals
        return TokenizerEncodeOutput(**merged)
