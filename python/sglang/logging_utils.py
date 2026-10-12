# SPDX-License-Identifier: Apache-2.0
"""Stream routing shared by SRT and diffusion logging."""

import logging
import sys


class MaxLevelFilter(logging.Filter):
    """Keep records below ERROR off stderr-based error collectors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.ERROR


def configure_root_logger(
    level: int, formatter: logging.Formatter, *, force: bool = True
) -> None:
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(MaxLevelFilter())

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level, handlers=[stdout_handler, stderr_handler], force=force
    )


def configure_uvicorn_streams(config: dict) -> None:
    """Split Uvicorn runtime logs while preserving its access handler."""
    config.setdefault("filters", {})["sglang_below_error"] = {
        "()": MaxLevelFilter,
    }
    default_handler = config["handlers"]["default"]
    default_handler["stream"] = "ext://sys.stdout"
    filters = default_handler.setdefault("filters", [])
    if "sglang_below_error" not in filters:
        filters.append("sglang_below_error")
    config["handlers"]["sglang_error"] = {
        "class": "logging.StreamHandler",
        "formatter": default_handler["formatter"],
        "level": "ERROR",
        "stream": "ext://sys.stderr",
    }
    handlers = config["loggers"]["uvicorn"]["handlers"]
    if "sglang_error" not in handlers:
        handlers.append("sglang_error")
