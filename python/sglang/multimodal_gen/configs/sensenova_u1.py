# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the native SenseNova-U1 integration."""

import json
import os

from PIL import Image

SENSENOVA_U1_REQUEST_EXTRA_KEY = "sensenova_u1"

SENSENOVA_U1_MODEL_IDS = {
    "sensenova/sensenova-u1.5-8b-mot",
}
SENSENOVA_U1_ADAPTER_ONLY_MODEL_IDS = {
    "sensenova/sensenova-u1.5-8b-mot-loras",
}

SENSENOVA_U1_CFG_NORM_CHOICES = (
    "none",
    "global",
    "channel",
    "cfg_zero_star",
)
SENSENOVA_U1_RESOLUTION_ALIGNMENT = 32

DEFAULT_CFG_NORM = "none"
DEFAULT_TIMESTEP_SHIFT = 3.0
DEFAULT_ENABLE_TIMESTEP_SHIFT = True
DEFAULT_CFG_INTERVAL = (0.0, 1.0)
DEFAULT_IMG_CFG_SCALE = 1.0
DEFAULT_T_EPS = 0.02
DEFAULT_THINK_MODE = False


def derive_cache_branch_count(
    *, is_edit: bool, cfg_scale: float, img_cfg_scale: float
) -> int:
    """Return the stable denoising branch count used by SenseNova generation.

    This deliberately mirrors the generation loops' exact comparisons.  Cache-DiT
    must not use a tolerance here: a cache profile which differs from the model's
    real conditional schedule can advance residual state on the wrong branch.
    """
    if not is_edit:
        return 2 if cfg_scale > 1 else 1

    if cfg_scale == 1 and img_cfg_scale == 1:
        return 1
    if img_cfg_scale == 1 or cfg_scale == img_cfg_scale:
        return 2
    return 3


def _flatten_rgba_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    return background


def is_sensenova_u1_model(model_path: str) -> bool:
    """Identify SenseNova-U1 Hub IDs and local base checkpoints."""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path) as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(config, dict):
            return False
        architectures = config.get("architectures", [])
        return (
            config.get("model_type") == "neo_chat"
            and isinstance(architectures, list)
            and "NEOChatModel" in architectures
        )

    return model_path.rstrip("/").lower() in SENSENOVA_U1_MODEL_IDS


def is_sensenova_u1_adapter_only_model(model_path: str) -> bool:
    return model_path.rstrip("/").lower() in SENSENOVA_U1_ADAPTER_ONLY_MODEL_IDS
