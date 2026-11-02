# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.kernels.ops.attention.moss_qk_norm_rope import (
    moss_qk_norm_rope_bitexact,
)
from sglang.kernels.ops.diffusion import rmsnorm_cast_before_weight_rope_bitexact
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMOSSQKNormRoPE(CustomTestCase):
    def test_matches_separate_launches(self):
        # Independent and packed projections, tail tiles, a multi-sample batch,
        # and the image-token length of the reported 960x1728 workload.
        torch.manual_seed(42)
        for batch, seq, packed in ((1, 17, False), (2, 19, True), (1, 6480, True)):
            with self.subTest(batch=batch, seq=seq, packed=packed):
                shape = (batch, seq, 24, 128)
                if packed:
                    storage = torch.randn(
                        batch, seq, 3, 24, 128, device="cuda", dtype=torch.bfloat16
                    )
                    q, k = storage[:, :, 0], storage[:, :, 1]
                else:
                    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                    k = torch.randn_like(q)
                qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
                kw = torch.randn_like(qw)
                phase = torch.randn(batch * seq, 64, device="cuda")
                cos = phase.cos().repeat(1, 2).bfloat16()
                sin = phase.sin().repeat(1, 2).bfloat16()
                before = (q.clone(), k.clone())
                expected = (
                    rmsnorm_cast_before_weight_rope_bitexact(q, qw, cos, sin),
                    rmsnorm_cast_before_weight_rope_bitexact(k, kw, cos, sin),
                )
                actual = moss_qk_norm_rope_bitexact(q, k, qw, kw, cos, sin)
                for got, want in zip(actual, expected):
                    self.assertTrue(torch.equal(got, want))
                    self.assertTrue(got.is_contiguous())
                self.assertTrue(torch.equal(q, before[0]))
                self.assertTrue(torch.equal(k, before[1]))

                # Mixed token strides must also use each input's own layout.
                mixed = moss_qk_norm_rope_bitexact(q, k.contiguous(), qw, kw, cos, sin)
                for got, want in zip(mixed, expected):
                    self.assertTrue(torch.equal(got, want))

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = moss_qk_norm_rope_bitexact(q, k, qw, kw, cos, sin)
                graph.replay()
                for got, want in zip(captured, expected):
                    self.assertTrue(torch.equal(got, want))

                with self.assertRaises(ValueError):
                    moss_qk_norm_rope_bitexact(q, k[:, :-1], qw, kw, cos, sin)
                with self.assertRaises(ValueError):
                    moss_qk_norm_rope_bitexact(q.float(), k, qw, kw, cos, sin)


if __name__ == "__main__":
    unittest.main()
