"""Resolution-contract tests for HunyuanImage-3 image generation and editing."""

from types import SimpleNamespace

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan_image3 import (
    HunyuanImage3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.ar_stage import (
    HunyuanImage3AR,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.decoding import (
    HunyuanImage3DecodingStage,
    resize_hunyuan_image3_decoded_frames,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.resolution import (
    RESTORE_SIZE_EXTRA_KEY,
    calculate_hunyuan_image3_restored_size,
    resolve_hunyuan_image3_output_resolution,
)


def test_edit_without_explicit_size_follows_reference_aspect_ratio():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1280,
        height=720,
        explicit_fields=set(),
        reference_size=(1000, 333),
    )

    assert (width, height) == (1000, 333)


def test_explicit_size_is_not_pre_aligned_before_processor():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1025,
        height=577,
        explicit_fields={"width", "height"},
    )

    assert (width, height) == (1025, 577)


def test_explicit_size_takes_precedence_over_reference_image():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1025,
        height=577,
        explicit_fields={"width", "height"},
        reference_size=(1000, 333),
    )

    assert (width, height) == (1025, 577)


def test_one_explicit_dimension_prevents_reference_size_override():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=640,
        height=720,
        explicit_fields={"width"},
        reference_size=(333, 1000),
    )

    assert (width, height) == (640, 720)


def test_text_to_image_without_reference_keeps_raw_request_size():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1025,
        height=577,
        explicit_fields=set(),
    )

    assert (width, height) == (1025, 577)


def test_restored_size_preserves_explicit_target_aspect_ratio():
    restored_width, restored_height = calculate_hunyuan_image3_restored_size(
        target_width=1000,
        target_height=700,
        target_area=1024**2,
    )

    assert (restored_width, restored_height) == (1224, 857)
    native_ratio_error = abs(1216 / 832 - 1000 / 700)
    restored_ratio_error = abs(restored_width / restored_height - 1000 / 700)
    assert restored_ratio_error < native_ratio_error


def test_decoded_frames_are_resized_to_restored_aspect_ratio():
    frames = torch.zeros(1, 3, 1, 32, 64)

    restored = resize_hunyuan_image3_decoded_frames(frames, (76, 54))

    assert restored.shape == (1, 3, 1, 54, 76)


def test_decoded_frames_with_matching_ratio_are_not_resampled():
    frames = torch.zeros(1, 3, 1, 36, 64)

    restored = resize_hunyuan_image3_decoded_frames(frames, (1280, 720))

    assert restored is frames


def test_decoded_image_batch_is_resized_to_restored_aspect_ratio():
    images = torch.zeros(2, 3, 32, 64)

    restored = resize_hunyuan_image3_decoded_frames(images, (76, 54))

    assert restored.shape == (2, 3, 54, 76)


def test_decoding_stage_restores_outputs_and_trajectories(monkeypatch):
    frames = torch.zeros(2, 3, 1, 32, 64)
    trajectory = torch.zeros(2, 3, 1, 32, 64)

    def fake_decode(_stage, _batch, _server_args):
        return OutputBatch(output=frames, trajectory_decoded=[trajectory])

    monkeypatch.setattr(DecodingStage, "forward", fake_decode)
    stage = object.__new__(HunyuanImage3DecodingStage)
    batch = Req(sampling_params=SamplingParams(prompt="test", width=1000, height=700))
    batch.extra[RESTORE_SIZE_EXTRA_KEY] = (76, 54)

    output = stage.forward(batch, SimpleNamespace())

    assert output.output.shape == (2, 3, 1, 54, 76)
    assert output.trajectory_decoded[0].shape == (2, 3, 1, 54, 76)
    assert (batch.width, batch.height) == (76, 54)


def test_multi_output_request_keeps_restored_size_metadata(monkeypatch):
    restore_size = (1224, 857)
    outputs = [
        SimpleNamespace(
            latents=torch.zeros(1, 1, 1, 1, 1),
            width=1216,
            height=832,
            extra={RESTORE_SIZE_EXTRA_KEY: restore_size},
        )
        for _ in range(2)
    ]
    stage = object.__new__(HunyuanImage3AR)
    monkeypatch.setattr(stage, "_expand_multi_output", lambda _batch: [object()] * 2)
    monkeypatch.setattr(stage, "_forward_batched", lambda _batches: outputs)
    batch = Req(
        sampling_params=SamplingParams(
            prompt="test",
            width=1000,
            height=700,
            num_outputs_per_prompt=2,
        )
    )

    output = stage.forward(batch, SimpleNamespace())

    assert output is batch
    assert output.latents.shape[0] == 2
    assert output.extra[RESTORE_SIZE_EXTRA_KEY] == restore_size


def test_pipeline_config_delegates_condition_image_sizing_to_native_stage():
    config = HunyuanImage3PipelineConfig()
    reference = Image.new("RGB", (1000, 333))

    assert config.calculate_condition_image_size(reference, 1280, 720) is None
    assert config.prepare_calculated_size(reference) is None


def test_input_validation_does_not_apply_generic_32_pixel_edit_resize():
    reference = Image.new("RGB", (1000, 333))
    batch = Req(
        sampling_params=SamplingParams(prompt="edit", width=1008, height=336),
        condition_image=reference,
    )
    batch.extra["explicit_fields"] = ["width", "height"]
    server_args = SimpleNamespace(pipeline_config=HunyuanImage3PipelineConfig())

    InputValidationStage().preprocess_condition_image(
        batch,
        server_args,
        condition_image_width=reference.width,
        condition_image_height=reference.height,
    )

    assert batch.condition_image[0].size == (1000, 333)
    assert (batch.width, batch.height) == (1008, 336)
