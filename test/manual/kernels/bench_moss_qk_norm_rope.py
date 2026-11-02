# SPDX-License-Identifier: Apache-2.0
"""Compare separate/paired launches at MOSS-Image's 960x1728 token length.

Run: python test/manual/kernels/bench_moss_qk_norm_rope.py
Reports eager and CUDA-graph kernel time; this is not an end-to-end benchmark.
"""

import torch

from sglang.kernels.jit.benchmark.marker import do_bench
from sglang.kernels.ops.attention.moss_qk_norm_rope import (
    moss_qk_norm_rope_bitexact,
)
from sglang.kernels.ops.diffusion import rmsnorm_cast_before_weight_rope_bitexact


def main():
    print(torch.cuda.get_device_name())
    for seq in (6480, 6608, 6992):
        for packed in (False, True):
            qkv = torch.randn(1, seq, 3, 24, 128, device="cuda", dtype=torch.bfloat16)
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            if not packed:
                qkv = torch.stack((q, k))
            qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            kw = torch.randn_like(qw)
            phase = torch.randn(seq, 64, device="cuda")
            cos, sin = (
                phase.cos().repeat(1, 2).bfloat16(),
                phase.sin().repeat(1, 2).bfloat16(),
            )

            def separate(storage, qw, kw, cos, sin, packed=packed):
                q, k = (
                    (storage[:, :, 0], storage[:, :, 1])
                    if packed
                    else (storage[0], storage[1])
                )
                return (
                    rmsnorm_cast_before_weight_rope_bitexact(q, qw, cos, sin),
                    rmsnorm_cast_before_weight_rope_bitexact(k, kw, cos, sin),
                )

            def paired(storage, qw, kw, cos, sin, packed=packed):
                q, k = (
                    (storage[:, :, 0], storage[:, :, 1])
                    if packed
                    else (storage[0], storage[1])
                )
                return moss_qk_norm_rope_bitexact(q, k, qw, kw, cos, sin)

            inputs = (qkv, qw, kw, cos, sin)
            for got, want in zip(paired(*inputs), separate(*inputs)):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            # marker flushes L2 and rotates input storage in graph mode. Pass
            # the backing storage so cloning preserves packed Q/K strides.
            for mode, use_graph in (("eager", False), ("graph", True)):
                old_s, new_s = (
                    do_bench(
                        fn,
                        input_args=inputs,
                        use_cuda_graph=use_graph,
                        metrics=(0.5,),
                        disable_log_bandwidth=True,
                    ).times[0]
                    for fn in (separate, paired)
                )
                print(
                    f"seq={seq} packed={packed} mode={mode} cold_l2=True: "
                    f"separate={old_s * 1e6:.2f}us paired={new_s * 1e6:.2f}us "
                    f"speedup={old_s / new_s:.3f}x"
                )


if __name__ == "__main__":
    main()
