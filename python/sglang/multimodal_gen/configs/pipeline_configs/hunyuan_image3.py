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
        # weights (~61 GiB at TP=2) fill a 61 GiB accelerator with no headroom,
        # so it cannot run fully resident and must stream. On NPU the only
        # streaming path is FSDP inference: automatic layerwise DiT offload is
        # CUDA-only (auto_tune.maybe_adjust_auto_default_layerwise_offload
        # returns early when not is_cuda()), and DiT component-offload moves the
        # whole ~61 GiB module to the device at once -> OOM. So do NOT pin a
        # high keep_resident threshold here: that keeps dit_cpu_offload=True ->
        # COMPONENT_OFFLOAD. Leave it unset so auto-tune keeps the DiT RESIDENT,
        # which is what should_use_fsdp_for_component requires; then run with
        # --use-fsdp-inference true on cards that cannot hold the backbone.
        # CFG is handled internally by the AR stage's cond/uncond batching, so
        # framework CFG-parallel is disabled.
        return ModelDeploymentConfig(
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
