"""HunyuanImage-3 decoding with target-aspect-ratio restoration."""

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from .resolution import RESTORE_SIZE_EXTRA_KEY


def resize_hunyuan_image3_decoded_frames(
    frames: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Resize decoded frames to ``(width, height)`` with antialiased bicubic."""
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError("HunyuanImage-3 restored dimensions must be positive")
    if frames.ndim not in (4, 5):
        raise ValueError(
            "HunyuanImage-3 decoded frames must be BCHW or BCFHW, "
            f"but got shape {tuple(frames.shape)}"
        )

    input_height, input_width = frames.shape[-2:]
    if input_width * output_height == input_height * output_width:
        return frames

    original_dtype = frames.dtype
    if frames.ndim == 5:
        batch_size, channels, num_frames, _, _ = frames.shape
        flat_frames = (
            frames.permute(0, 2, 1, 3, 4)
            .reshape(batch_size * num_frames, channels, input_height, input_width)
            .float()
        )
    else:
        batch_size, channels, _, _ = frames.shape
        num_frames = None
        flat_frames = frames.float()

    resized = F.interpolate(
        flat_frames,
        size=(output_height, output_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(original_dtype)

    if num_frames is None:
        return resized
    return (
        resized.reshape(batch_size, num_frames, channels, output_height, output_width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


class HunyuanImage3DecodingStage(DecodingStage):
    """Decode native-bucket latents and restore the requested aspect ratio."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        output_batch = super().forward(batch, server_args)
        restore_size = batch.extra.get(RESTORE_SIZE_EXTRA_KEY)
        if restore_size is None or not isinstance(output_batch.output, torch.Tensor):
            return output_batch

        output_batch.output = resize_hunyuan_image3_decoded_frames(
            output_batch.output,
            restore_size,
        )
        if output_batch.trajectory_decoded is not None:
            output_batch.trajectory_decoded = [
                resize_hunyuan_image3_decoded_frames(frames, restore_size)
                for frames in output_batch.trajectory_decoded
            ]
        batch.width = output_batch.output.shape[-1]
        batch.height = output_batch.output.shape[-2]
        return output_batch
