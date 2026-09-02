from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import (
    HunyuanImage3DitConfig,
)
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig, T5Config
from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import (
    HunyuanImage3VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


@dataclass
class HunyuanImage3PipelineConfig(SpatialImagePipelineConfig):
    """HunyuanImage-3 pipeline config."""

    vae_precision: str = "fp32"

    should_use_guidance: bool = True
    task_type: ModelTaskType = ModelTaskType.TI2I

    vae_tiling: bool = False
    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=HunyuanImage3DitConfig)

    vae_config: VAEConfig = field(default_factory=HunyuanImage3VAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(T5ArchConfig(num_heads=6)),)
    )

    enable_autocast: bool = False

    def __post_init__(self):
        self.vae_scale_factor = self.vae_config.get_vae_scale_factor()

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # HunyuanImage-3.0 is an ~80B-total / 13B-active MoE whose per-rank
        # weights (~61 GiB at TP=2) fill a 61 GiB accelerator with no headroom
        # for activations, so it cannot run fully resident. It streams via
        # layerwise DiT offload -- HunyuanImage3ForCausalMM is a
        # LayerwiseOffloadableModuleMixin over "model.layers" -- the same
        # mechanism MiniMaxH3 uses for its 61.73 GiB DiT, and it needs no FSDP.
        #   * dit_layerwise_offload_modes=("auto","memory") makes the DiT
        #     eligible for automatic layerwise offload in those perf modes.
        #   * keep_resident_min_available_gb=120 is deliberate: the image-gen
        #     keep-resident default (45 GiB) would, at ~60 GiB free, strip the
        #     DiT back out of layerwise offload (and flip dit_cpu_offload off),
        #     leaving it resident -> OOM. A 120 GiB bar skips that on 61 GiB
        #     cards, while genuinely large cards still keep the DiT resident.
        # CFG is handled internally by the AR stage's cond/uncond batching, so
        # framework CFG-parallel is disabled (it would split ranks that all
        # need the weights).
        return ModelDeploymentConfig(
            keep_resident_min_available_gb=120,
            dit_layerwise_offload_modes=("auto", "memory"),
            auto_enable_cfg_parallel=False,
        )

    def supports_dynamic_batching(self):
        # The AR stage batches compatible requests in one diffusion loop
        # (run_grouped_requests), falling back to single-request execution.
        return True

    def supports_native_grouped_requests(self):
        return True

    def supports_batching_image_conditioning(self):
        # TI2I requests carry per-request conditioning (per-row masks/scatter/
        # RoPE); requests are bucketed by resolution and condition-image count.
        return True

    def supports_sequential_dit_inference(self):
        return True

    def supports_sequential_multi_output_inference(self):
        return current_platform.is_npu()
