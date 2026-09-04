"""Resolution helpers shared by HunyuanImage-3 pipeline stages."""

import math

RESTORE_SIZE_EXTRA_KEY = "hunyuan_image3_restore_size"


def resolve_hunyuan_image3_output_resolution(
    width: int,
    height: int,
    explicit_fields: set[str],
    reference_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Return the raw target size used for native aspect-ratio bucketing.

    Explicit dimensions take precedence. Image editing without an explicit
    size inherits the unmodified reference size instead of the generic
    1280x720 pipeline default.
    """
    if reference_size is not None and not {"width", "height"} & explicit_fields:
        width, height = reference_size
    return int(width), int(height)


def calculate_hunyuan_image3_restored_size(
    target_width: int,
    target_height: int,
    target_area: int,
) -> tuple[int, int]:
    """Scale a target aspect ratio to ``target_area`` without cropping."""
    if target_width <= 0 or target_height <= 0:
        raise ValueError("HunyuanImage-3 target dimensions must be positive")
    if target_area <= 0:
        raise ValueError("HunyuanImage-3 target area must be positive")

    scale = math.sqrt(target_area / (target_width * target_height))
    restored_width = max(1, round(target_width * scale))
    restored_height = max(1, round(target_height * scale))
    return restored_width, restored_height
