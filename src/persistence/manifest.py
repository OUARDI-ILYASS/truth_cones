"""Hybrid persistence: JSON manifests (audit trail) + .pt files (bit-perfect tensors).

Each pipeline stage writes both:
  - JSON manifest: human-readable metadata, hyperparameters, summary
    metrics (Tab. master, Tab. hyperparams, Tab. cone_asr, etc.)
  - .pt weights: bit-perfect cone basis V or direction r for downstream
    loading (DIM → TDO warm-start, TDO → TCO incremental init, TCO →
    evaluation)

Same timestamp links a manifest to its weights file. The JSON is the
audit trail; the .pt is the handoff.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from src.logging_setup import get_logger

logger = get_logger(__name__)


def _json_default(obj: Any) -> Any:
    """Serialization fallback for non-JSON-native types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_manifest(
    manifest: Dict[str, Any],
    output_dir: Path,
    filename: str,
) -> Path:
    """Write a manifest JSON file with auto-created parent dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    logger.info("Manifest written: %s", path)
    return path


def write_weights(
    weights: Dict[str, torch.Tensor],
    output_dir: Path,
    filename: str,
) -> Path:
    """Write a torch state-dict-like file.

    Typically contains {"best_vectors": (k, d)} for TCO or
    {"best_vectors": (1, d)} for TDO.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(weights, path)
    logger.info("Weights written: %s", path)
    return path


def make_run_filenames(
    model_name: str,
    *,
    suffix: str = "",
    cone_dim: int | None = None,
) -> Dict[str, str]:
    """Generate timestamped, slugged filenames for a pipeline run.

    Returns dict with keys "timestamp", "manifest", "weights".
    cone_dim embeds k in the filename for easy identification.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model_name.replace("/", "_")
    parts = [slug]
    if cone_dim is not None:
        parts.append(f"d{cone_dim}")
    if suffix:
        parts.append(suffix)
    parts.append(timestamp)
    base = "_".join(parts)
    return {
        "timestamp": timestamp,
        "manifest":  f"{base}_manifest.json",
        "weights":   f"{base}_weights.pt",
    }