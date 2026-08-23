from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT = 16

VALID_BOT_TASKS = {"auto", "image", "think", "recaption", "think_recaption", "img_ratio", "none"}

SYSTEM_PROMPT_PRESETS = {
    "none", "en_unified", "en_vanilla", "en_recaption",
    "en_think_recaption", "dynamic", "auto",
}


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    """Sampling parameters for HunyuanImage-3."""

    negative_prompt: str = ""
    num_frames: int = 1
    guidance_scale: float = 2.5
    num_inference_steps: int = 50

    # Tokenizer bot_task: controls the bot response prefix. Default "image"
    # adds no bot prefix for gen_image mode.
    bot_task: str = "image"

    # Preset name (see SYSTEM_PROMPT_PRESETS) or raw custom text.
    system_prompt: str | None = "en_unified"

    # Pre-generated CoT text from AR stage (think/recaption output)
    cot_text: str | None = None

    # Supported resolutions (height, width) - must be divisible by 16
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1024, 1024),  # 1:1
            (768, 1024),  # 3:4 portrait
            (1024, 768),  # 4:3 landscape
            (720, 1280),  # 9:16 portrait
            (1280, 720),  # 16:9 landscape
        ]
    )

    def _adjust(self, server_args):
        requested_width = self.width
        requested_height = self.height
        if self.width is not None and self.height is not None:
            self.width, self.height = align_hunyuan_image3_resolution(
                self.width, self.height
            )
            if (self.width, self.height) != (
                requested_width,
                requested_height,
            ):
                logger.warning(
                    "HunyuanImage-3 requires dimensions divisible by %s; adjusted "
                    "requested resolution from %sx%s to %sx%s",
                    HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
                    requested_width,
                    requested_height,
                    self.width,
                    self.height,
                )
        if self.bot_task not in VALID_BOT_TASKS:
            logger.warning(
                f"Invalid bot_task '{self.bot_task}'. Must be one of {VALID_BOT_TASKS}. "
                f"Defaulting to 'image'."
            )
            self.bot_task = "image"
        super()._adjust(server_args)


def align_hunyuan_image3_dimension(value: int) -> int:
    """Round a HunyuanImage-3 dimension up to a supported multiple."""
    return max(
        HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
        (value + HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT - 1)
        // HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT
        * HUNYUAN_IMAGE3_RESOLUTION_ALIGNMENT,
    )


def align_hunyuan_image3_resolution(width: int, height: int) -> tuple[int, int]:
    """Align both width and height to supported multiples."""
    return align_hunyuan_image3_dimension(width), align_hunyuan_image3_dimension(height)
