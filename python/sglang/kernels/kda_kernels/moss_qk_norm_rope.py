# SPDX-License-Identifier: Apache-2.0
"""Paired MOSS Q/K normalization and RoPE, preserving BF16 rounding."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.norm.rmsnorm_cast_before_weight_bitexact import (
    _norm_rope_kernel,
    _valid_rmsnorm_cast_before_weight_rope,
)
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _paired_norm_rope_kernel(
    Q,
    K,
    QWeight,
    KWeight,
    Cos,
    Sin,
    QOut,
    KOut,
    M: tl.constexpr,
    HEADS: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
):
    # Reuse the single-input implementation so its reduction tree and each
    # intermediate BF16 rounding stay identical. The second grid dimension
    # combines two launches without increasing each CTA's register footprint.
    if tl.program_id(1) == 0:
        _norm_rope_kernel(Q, QWeight, Cos, Sin, QOut, M, HEADS, Q_STRIDE)
    else:
        _norm_rope_kernel(K, KWeight, Cos, Sin, KOut, M, HEADS, K_STRIDE)


def _fake_paired_norm_rope(q, k, q_weight, k_weight, cos, sin):
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )


@register_custom_op(
    op_name="moss_qk_norm_rope_bitexact",
    mutates_args=[],
    fake_impl=_fake_paired_norm_rope,
)
def moss_qk_norm_rope_bitexact(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize/rotate [B, S, H, 128] Q/K in one launch.

    Accept contiguous projections and strided views of a packed QKV projection.
    Outputs are contiguous, inputs are never mutated, and Q/K may have different
    token strides. Only CUDA BF16 and identical Q/K shapes are supported.
    """
    if not (
        q.shape == k.shape
        and _valid_rmsnorm_cast_before_weight_rope(q, q_weight, cos, sin)
        and _valid_rmsnorm_cast_before_weight_rope(k, k_weight, cos, sin)
    ):
        raise ValueError("Unsupported paired MOSS Q/K RMSNorm + RoPE inputs.")
    with torch.cuda.device(q.device):
        q_out, k_out = _fake_paired_norm_rope(q, k, q_weight, k_weight, cos, sin)
        rows = q.numel() // 128
        _paired_norm_rope_kernel[(triton.cdiv(rows, 16), 2)](
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            q_out,
            k_out,
            rows,
            q.shape[2],
            q.stride(1),
            k.stride(1),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, k_out
