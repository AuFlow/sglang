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
        # HunyuanImage-3.0 is an ~80B-total / 13B-active MoE. Its per-rank
        # weights (~76 GiB at TP=2) exceed a 61 GiB accelerator, so it must not
        # inherit the generic image-gen keep-resident default (45 GiB): that
        # keeps the DiT resident, builds the whole backbone on-device, and OOMs
        # during MoE weight construction. Require a very high free-memory bar
        # before the DiT stays resident, so normal cards keep DiT offload
        # enabled and run it via FSDP (--use-fsdp-inference) CPU-offloaded
        # per-layer sharding instead. CFG is handled internally by the AR
        # stage's cond/uncond batching, so framework CFG-parallel is disabled
        # (it would split ranks that all need the sharded weights).
        return ModelDeploymentConfig(
            keep_resident_min_available_gb=120,
            keep_resident_components=("dit", "vae"),
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
