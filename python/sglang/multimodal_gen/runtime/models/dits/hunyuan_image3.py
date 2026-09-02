"""HunyuanImage-3 AR backbone + diffusion I/O for multimodal_gen.

Ported from the official HunyuanImage-3 repository
(`modeling_hunyuan_image_3.py`).
"""

import re
import types
from typing import Iterable, Optional, Tuple

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.resnet import ResnetBlock2D
from einops import rearrange
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.hunyuan import (
    HunYuanMLP,
    HunYuanSparseMoeBlock,
    _get_cla_factor,
    _is_moe,
)
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig

from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import HunyuanImage3DitConfig

from .hunyuan_image3_utils import (
    CachedRoPE,
    HunYuanRotary2DEmbedder,
    image_attention,
)

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Checkpoint weight names of the non-AR parts (VAE, ViT), skipped during
# backbone weight loading.
UNEXPECTED_KEYWORDS = [
    "vae",
    "vision_aligner",
    "vision_model",
]

# Official diffusion-I/O checkpoint key fragments mapped onto the diffusers
# building-block attribute names they run on (see UNetDown/UNetUp below).
_DIFFUSION_IO_KEY_MAP = [
    ("in_layers.0.", "norm1."),
    ("in_layers.2.", "conv1."),
    ("emb_layers.1.", "time_emb_proj."),
    ("out_layers.0.", "norm2."),
    ("out_layers.3.", "conv2."),
    ("skip_connection.", "conv_shortcut."),
    ("mlp.0.", "mlp.linear_1."),
    ("mlp.2.", "mlp.linear_2."),
]


_RES_BLOCK_KWARGS = dict(
    eps=1e-5,
    non_linearity="swish",
    time_embedding_norm="scale_shift",
    kernel="sde_vp",
)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations (GELU MLP over the
    official sinusoidal projection: cos/sin order, no freq downscale shift)."""

    def __init__(self, hidden_size, frequency_embedding_size=256, out_size=None):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.mlp = TimestepEmbedding(
            in_channels=frequency_embedding_size,
            time_embed_dim=hidden_size,
            act_fn="gelu",
            out_dim=out_size,
        )

    def forward(self, t):
        t_freq = self.time_proj(t).to(self.mlp.linear_1.weight.dtype)
        return self.mlp(t_freq)


class UNetDown(nn.Module):
    """Patch embed: converts noise latents (B, C, H, W) into sequence embeddings."""

    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels,
                 out_channels, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size

        self.model = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        ])
        if self.patch_size == 1:
            self.model.append(ResnetBlock2D(
                in_channels=hidden_channels,
                out_channels=out_channels,
                dropout=dropout,
                temb_channels=emb_channels,
                **_RES_BLOCK_KWARGS,
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResnetBlock2D(
                    in_channels=hidden_channels,
                    out_channels=(hidden_channels if (i + 1) * 2 != self.patch_size else out_channels),
                    dropout=dropout,
                    temb_channels=emb_channels,
                    down=True,
                    **_RES_BLOCK_KWARGS,
                ))

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, ResnetBlock2D):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, token_h, token_w


class UNetUp(nn.Module):
    """Final layer: converts backbone output sequence into noise predictions."""

    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels,
                 out_channels, dropout=0.0, out_norm=False):
        super().__init__()
        self.patch_size = patch_size
        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(ResnetBlock2D(
                in_channels=in_channels,
                out_channels=hidden_channels,
                dropout=dropout,
                temb_channels=emb_channels,
                **_RES_BLOCK_KWARGS,
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResnetBlock2D(
                    in_channels=in_channels if i == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    temb_channels=emb_channels,
                    up=True,
                    **_RES_BLOCK_KWARGS,
                ))

        if out_norm:
            self.model.append(nn.Sequential(
                nn.GroupNorm(32, hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            ))
        else:
            self.model.append(nn.Conv2d(
                hidden_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x, t, token_h, token_w):
        x = rearrange(x, "b (h w) c -> b c h w", h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResnetBlock2D):
                x = module(x, t)
            else:
                x = module(x)
        return x


def _make_rope(config: PretrainedConfig, head_dim: int, rope_theta, rope_scaling, max_position):
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        rope_scaling["rope_type"] = "default"
    return get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=rope_theta,
        rope_scaling=rope_scaling,
        is_neox_style=True,
    )


class HunYuanAttention(nn.Module):
    """Self-attention; CLA followers attend to the master's K/V via ``kv_states``."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.hidden_size = hidden_size
        self.is_cross_attention = is_cross_attention
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        if is_cross_attention:
            self.q_proj = ColumnParallelLinear(
                hidden_size, hidden_size, bias=bias, quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = _make_rope(
            config, self.head_dim, rope_theta, rope_scaling, max_position_embeddings
        )
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=True,
        )

        self.image_rope2d_emb = HunYuanRotary2DEmbedder(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        if self.use_qk_norm:
            self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
            self.query_layernorm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)

    def _apply_rope(
        self,
        positions,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        custom_pos_emb,
    ):
        """Apply the 2D image RoPE (diffusion path) or the 1D text RoPE."""
        if custom_pos_emb is not None:
            return self.image_rope2d_emb(q, k, hidden_states, custom_pos_emb)
        return self.rotary_emb(positions, q, k)

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        kv_states=None,
        attention_mask=None,
        custom_pos_emb=None,
    ):
        q_len, hidden_size = hidden_states.size()
        hidden_states = hidden_states.reshape(-1, hidden_size)

        if self.is_cross_attention:
            # CLA follower: attend to the master layer's K/V.
            ori_k, v = kv_states
            k = ori_k
            q, _ = self.q_proj(hidden_states)
            q, _ = self._apply_rope(
                positions, q, torch.empty_like(k), hidden_states, custom_pos_emb,
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self._apply_rope(
                positions, q, k, hidden_states, custom_pos_emb
            )
            ori_k = k

        if self.use_qk_norm:
            # Master layers use NPU fused RMSNorm; followers use generic RMSNorm.
            if self.is_cross_attention:
                q = self.query_layernorm(
                    q.view(-1, self.num_heads, self.head_dim).contiguous()
                )
                k = self.key_layernorm(
                    k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
                )
            else:
                import torch_npu

                q = torch_npu.npu_rms_norm(
                    q.view(-1, self.num_heads, self.head_dim).contiguous(),
                    gamma=self.query_layernorm.weight.float(),
                    epsilon=self.rms_norm_eps,
                )[0]
                k = torch_npu.npu_rms_norm(
                    k.view(-1, self.num_kv_heads, self.head_dim).contiguous(),
                    gamma=self.key_layernorm.weight.float(),
                    epsilon=self.rms_norm_eps,
                )[0]

        if attention_mask is not None:
            attn_output = image_attention(q, k, v, attention_mask)
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            attn_output = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))

        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        output = output.reshape(q_len, -1)
        return output, (ori_k, v)


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(
        self, config: PretrainedConfig, layer_id: int,
        quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        # intermediate_size may be a scalar or a per-layer list in the config
        intermediate_size = getattr(config, "intermediate_size", 0)
        if isinstance(intermediate_size, list):
            intermediate_size = intermediate_size[layer_id]
        self.intermediate_size = intermediate_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling = dict(rope_scaling)
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)

        cla_factor = _get_cla_factor(config)
        attention_type = "cross" if layer_id % cla_factor != 0 else "self"
        attn_kwargs = dict(
            config=config, hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config, bias=attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn = HunYuanAttention(**attn_kwargs, is_cross_attention=attention_type == "cross")

        if _is_moe(config):
            self.mlp = HunYuanSparseMoeBlock(
                config=config, layer_id=layer_id, quant_config=quant_config,
            )
        else:
            self.mlp = HunYuanMLP(
                hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
                hidden_act=config.hidden_act, quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False), prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions, hidden_states, forward_batch, residual,
        kv_states=None, attention_mask=None, custom_pos_emb=None,
    ):
        if attention_mask is not None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            
            hidden_states, ori_kv_states = self.self_attn(
                positions=positions, hidden_states=hidden_states,
                forward_batch=forward_batch, kv_states=kv_states,
                attention_mask=attention_mask,
                custom_pos_emb=custom_pos_emb,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states, ori_kv_states = self.self_attn(
                positions=positions, hidden_states=hidden_states,
                forward_batch=forward_batch, kv_states=kv_states,
            )
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, ori_kv_states


class HunyuanImage3Model(nn.Module):
    def __init__(
        self, config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList([
            HunyuanImage3DecoderLayer(
                config=config, layer_id=i, quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        cla_factor = _get_cla_factor(self.config)
        prev_kv_states = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual, kv_states = layer(
                positions, hidden_states, forward_batch, residual, prev_kv_states,
            )
            if getattr(self.config, "use_cla", False) and i % cla_factor == 0:
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_block(self, hidden_states, attention_mask, custom_pos_emb):
        residual = None
        cla_factor = _get_cla_factor(self.config)
        prev_kv_states = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual, kv_states = layer(
                None, hidden_states, None, residual,
                prev_kv_states, attention_mask, custom_pos_emb,
            )
            if getattr(self.config, "use_cla", False) and i % cla_factor == 0:
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

        return hidden_states.contiguous()

    def _split_qkv_weight(self, qkv):
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        num_key_value_groups = num_attention_heads // num_kv_heads
        hidden_size = self.config.hidden_size
        attention_head_dim = hidden_size // num_attention_heads

        qkv = qkv.reshape(num_kv_heads, num_key_value_groups + 2, attention_head_dim, hidden_size)
        q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
        return torch.concat((q, k, v))


class HunyuanImage3ForCausalMM(CachableDiT):
    """Top-level HunyuanImage-3 model for diffusion pipeline."""

    def __init__(
        self, config: HunyuanImage3DitConfig, prefix: str = "", **kwargs,
    ):
        super().__init__(config=config, **kwargs)

        arch_config = self.config

        self.model = HunyuanImage3Model(
            arch_config, prefix=f"{prefix}.model",
        )

        # ---- Diffusion I/O modules ----
        patch_size = getattr(arch_config, "patch_size", 1)
        patch_embed_hidden_dim = getattr(arch_config, "patch_embed_hidden_dim", 1024)
        img_proj_type = getattr(arch_config, "img_proj_type", "unet")
        # latent_channels may be top-level or nested under arch_config.vae
        if isinstance(getattr(arch_config, "vae", None), dict):
            latent_channels = arch_config.vae["latent_channels"]
        else:
            latent_channels = arch_config.latent_channels

        if img_proj_type == "unet":
            self.timestep_emb = TimestepEmbedder(hidden_size=arch_config.hidden_size)
            self.patch_embed = UNetDown(
                patch_size=patch_size,
                emb_channels=arch_config.hidden_size,
                in_channels=latent_channels,
                hidden_channels=patch_embed_hidden_dim,
                out_channels=arch_config.hidden_size,
            )
            self.time_embed = TimestepEmbedder(hidden_size=arch_config.hidden_size)
            self.final_layer = UNetUp(
                patch_size=patch_size,
                emb_channels=arch_config.hidden_size,
                in_channels=arch_config.hidden_size,
                hidden_channels=patch_embed_hidden_dim,
                out_channels=latent_channels,
                out_norm=True,
            )
            self.time_embed_2 = TimestepEmbedder(hidden_size=arch_config.hidden_size)
        else:
            raise ValueError(f"Unknown img_proj_type: {img_proj_type}")

        head_dim = arch_config.hidden_size // arch_config.num_attention_heads
        self.cached_rope = CachedRoPE(
            rope_theta=arch_config.rope_theta,
            head_dim=head_dim,
            rope_type=getattr(arch_config, "rope_type", "2d"),
        )

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, **kwargs):
        """DiT-style forward for denoising stage."""
        return hidden_states

    def forward_block(self, hidden_states, attention_mask, custom_pos_emb, timestep=None):
        # TeaCache gate: skip layers on similar steps, reuse the cached
        # residual; one decision covers the packed CFG batch.
        if timestep is not None and self.should_skip_forward_for_cached_states(
            timestep=timestep
        ):
            return self.retrieve_cached_states(hidden_states).contiguous()

        output = self.model.forward_block(hidden_states, attention_mask, custom_pos_emb)

        if timestep is not None:
            self.maybe_cache_states(output, hidden_states)
        return output

    # TeaCache — see runtime/cache/teacache.py

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        ctx = self._get_teacache_context()
        # Gate consumed by forward_block / maybe_cache_states.
        self.enable_teacache = ctx is not None
        if ctx is None:
            return False
        # Cond/uncond rows share one packed forward (same timestep), so skip
        # boundaries are step-unit based — no CFG doubling.
        start_skipping, end_skipping = ctx.teacache_params.get_skip_boundaries(
            ctx.num_inference_steps, do_cfg=False
        )
        is_boundary_step = (
            ctx.current_timestep < start_skipping
            or ctx.current_timestep >= end_skipping
        )
        # Timestep-conditioned input for the L1 similarity decision.
        modulated_inp = self.time_embed(kwargs["timestep"])
        should_calc = self._compute_teacache_decision(
            modulated_inp=modulated_inp,
            is_boundary_step=is_boundary_step,
            coefficients=ctx.coefficients,
            teacache_thresh=ctx.teacache_thresh,
        )
        return not should_calc

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache the backbone residual for the packed [tokens, hidden] tensor."""
        if not self.enable_teacache:
            return
        self.previous_residual = hidden_states - original_hidden_states

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.previous_residual

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        split_params_mapping = [
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
            (
                ".qkv_proj", ".qkv_proj",
                num_attention_heads + num_kv_heads * 2,
                [("q", num_attention_heads), ("k", num_kv_heads), ("v", num_kv_heads)],
                self.model._split_qkv_weight,
            ),
        ]

        cla_factor = _get_cla_factor(self.config)

        # Expert mapping for FusedMoE loading (matching vllm-omni); remaps
        # to fused gate_and_up_proj checkpoint keys.
        expert_weights_remapping = {
            "gate_proj": ("gate_and_up_proj", 1, 2),
            "up_proj": ("gate_and_up_proj", 0, 2),
        }
        expert_params_mapping = []
        if _is_moe(self.config):
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set = set()
        _ckpt_dtype_logged = False

        for name, loaded_weight in weights:
            if not _ckpt_dtype_logged:
                logger.info(
                    "  checkpoint weight dtype: %s (param dtype: %s)",
                    loaded_weight.dtype,
                    next(iter(params_dict.values())).dtype if params_dict else "?",
                )
                _ckpt_dtype_logged = True
            if any(keyword in name for keyword in UNEXPECTED_KEYWORDS):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            # Diffusion I/O runs on diffusers blocks; remap the official
            # checkpoint names onto their attribute names.
            if name.startswith(
                ("patch_embed.", "final_layer.", "time_embed", "timestep_emb")
            ):
                for old, new in _DIFFUSION_IO_KEY_MAP:
                    if old in name:
                        name = name.replace(old, new)
            if "gate_proj_bias" in name:
                name = name.replace("gate_proj_bias", "gate_proj.bias")
            if "up_proj_bias" in name:
                name = name.replace("up_proj_bias", "up_proj.bias")
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            if name.endswith("wte.weight"):
                name = name.replace("wte.weight", "embed_tokens.weight")
            if name.endswith("ln_f.weight"):
                name = name.replace("ln_f.weight", "norm.weight")
            if "mlp.gate.wg." in name:
                name = name.replace("wg.", "")

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                if weight_name == ".q_proj" and cla_factor > 1:
                    match = re.search(r"layers\.(\d+)", name)
                    if match and int(match.group(1)) % cla_factor != 0:
                        continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            for param_name, weight_name, den, split_param, func in split_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den
                param = params_dict[name]
                weight_loader = param.weight_loader
                chunk = func(loaded_weight) if func is not None else loaded_weight
                offset = 0
                for shard_id, num in split_param:
                    new_offset = offset + num * units
                    weight_loader(param, chunk[offset:new_offset], shard_id)
                    offset = new_offset
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            # Expert weights: FusedMoE.make_expert_params_mapping +
            # expert_weights_remapping handle the fused gate_and_up_proj format.
            is_expert_weight = False
            is_found = False
            found_num = 0
            if _is_moe(self.config) and "mlp.experts" in name:
                if not getattr(self, "_expert_ckpt_logged", False):
                    logger.info("  expert ckpt key sample: %s", name)
                    self._expert_ckpt_logged = True
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    offset = 0
                    den = 1
                    for (
                        mapped_weight_substr,
                        origin_weight_info,
                    ) in expert_weights_remapping.items():
                        if mapped_weight_substr in weight_name:
                            origin_weight_name, offset, den = origin_weight_info
                            weight_name = weight_name.replace(
                                mapped_weight_substr, origin_weight_name
                            )
                            break
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    found_num += 1
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader

                    if den > 1:
                        assert loaded_weight.shape[0] % den == 0
                        units = loaded_weight.shape[0] // den
                        loaded_weight_shard = loaded_weight[
                            offset * units : offset * units + units
                        ]
                    else:
                        loaded_weight_shard = loaded_weight

                    weight_loader(
                        param,
                        loaded_weight_shard,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(name_mapped)
                    is_found = True
                    if found_num == den:
                        break
            if is_found:
                continue
            if is_expert_weight:
                # Recognised as expert weight but not mapped locally
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Log missing weights; filter out expected missing patterns
        all_param_names = set(params_dict.keys())
        missing = all_param_names - loaded_params
        if missing:
            significant_missing = [
                n for n in missing
                if not any(k in n for k in ["rotary_emb"])
            ]
            if significant_missing:
                logger.warning(
                    "Weight loading: %d/%d params loaded, %d MISSING:",
                    len(loaded_params), len(all_param_names), len(significant_missing),
                )
                for n in sorted(significant_missing)[:30]:
                    logger.warning("  MISSING: %s", n)
                if len(significant_missing) > 30:
                    logger.warning("  ... and %d more", len(significant_missing) - 30)
            else:
                logger.info(
                    "Weight loading: %d/%d params loaded (all accounted for)",
                    len(loaded_params), len(all_param_names),
                )
        else:
            logger.info(
                "Weight loading: %d/%d params loaded (complete)",
                len(loaded_params), len(all_param_names),
            )

        return loaded_params


class LightProjector(nn.Module):
    """ViT embedding → transformer dim projection."""

    def __init__(self, config):
        config = types.SimpleNamespace(**config)
        super().__init__()

        if config.projector_type == "linear":
            self.layers = nn.Linear(config.input_dim, config.n_embed)
        elif config.projector_type == "mlp_gelu":
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, config.depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            self.layers = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

    def forward(self, x):
        return self.layers(x)


EntryClass = [HunyuanImage3ForCausalMM]
