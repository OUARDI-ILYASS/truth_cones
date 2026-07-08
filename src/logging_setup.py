"""Centralized logging configuration.

Use ``get_logger(__name__)`` in every module instead of ``print``.
Long-running sweeps emit progress through the logger so that runs can be
piped to log files without losing structure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def configure_root_logger(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """Configure the root logger. Idempotent.

    Args:
        level: log level (e.g. logging.INFO).
        log_file: if set, also write to this file.
        fmt: log message format.
        datefmt: date format.
    """
    global _root_configured
    if _root_configured:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Clear any pre-existing handlers (notebooks, prior runs, etc.)
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)

    # Quiet noisy third-party loggers
    for name in ["transformers", "huggingface_hub", "accelerate", "urllib3"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Configures the root logger on first call."""
    if not _root_configured:
        configure_root_logger()
    return logging.getLogger(name)
