"""Experiment 4 — DIM vs Cone Alignment CLI driver.

Pure post-hoc analysis: reads ``.pt`` artifacts from Experiments 2 and 3
and computes cosine similarities. No model loading, no GPU required.

Usage:
    python scripts/run_alignment.py --config configs/alignment.yaml
    python scripts/run_alignment.py --config configs/alignment.yaml --model qwen-2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AlignmentConfig, load_config
from src.evaluation.alignment import compute_cosine_similarities, summarize_alignment
from src.logging_setup import configure_root_logger, get_logger
from src.persistence.results_summary import build_alignment_summary, write_results_summary

logger = get_logger(__name__)


def _load_dim_vector(tdo_summary_path: Path, model_name: str) -> torch.Tensor | None:
    """Load the DIM vector for a model from Experiment 2 weights file."""
    if not tdo_summary_path.exists():
        return None
    with open(tdo_summary_path) as f:
        summary = json.load(f)
    entry = summary["models"].get(model_name)
    if entry is None:
        return None
    weights_path = tdo_summary_path.parent / entry["weights_file"]
    if not weights_path.exists():
        return None
    return torch.load(weights_path, map_location="cpu")["dim_direction"].float()


def _load_cone_bases(
    tco_dir: Path,
    model_name: str,
    cone_dims: list[int],
) -> dict[int, torch.Tensor]:
    """Load best_vectors for each cone_dim of a model from Experiment 3."""
    summary_path = tco_dir / "results_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        summary = json.load(f)

    entry = summary["models"].get(model_name)
    if entry is None:
        return {}

    bases: dict[int, torch.Tensor] = {}
    for d in cone_dims:
        cone_entry = entry.get("cones", {}).get(str(d))
        if cone_entry is None:
            continue
        weights_path = tco_dir / cone_entry["weights_file"]
        if not weights_path.exists():
            continue
        weights = torch.load(weights_path, map_location="cpu")
        bases[d] = weights[cone_entry["basis_key"]].float()
    return bases


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4 — DIM vs Cone Alignment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    configure_root_logger(log_file=args.log_file)
    cfg = load_config(args.config, AlignmentConfig)
    assert isinstance(cfg, AlignmentConfig)

    tdo_summary_path = cfg.tdo_dir / "results_summary.json"
    tco_summary_path = cfg.tco_dir / "results_summary.json"

    if not tco_summary_path.exists():
        logger.error("Missing %s. Run Experiment 3 first.", tco_summary_path)
        sys.exit(1)

    with open(tco_summary_path) as f:
        tco_summary = json.load(f)

    per_model: dict[str, dict] = {}

    for model_name in tco_summary["models"]:
        logger.info("Processing %s...", model_name)
        dim_vector = _load_dim_vector(tdo_summary_path, model_name)
        if dim_vector is None:
            logger.warning("Skipping %s: no DIM vector found.", model_name)
            continue

        cone_bases = _load_cone_bases(cfg.tco_dir, model_name, cfg.cone_dims)
        if not cone_bases:
            logger.warning("Skipping %s: no cone bases found.", model_name)
            continue

        cosines = compute_cosine_similarities(dim_vector, cone_bases, cfg.cone_dims)
        cosines_serializable = {
            str(d): {f"v{i+1}": float(c) for i, c in enumerate(cos_list)}
            for d, cos_list in cosines.items()
        }
        summary_stats = summarize_alignment(cosines)
        per_model[model_name] = {
            "cosine_similarities":  cosines_serializable,
            **summary_stats,
        }
        logger.info(
            "  v1=%.2e  max(v2..)=%.2e  %s",
            summary_stats["v1_dim_alignment"],
            summary_stats["max_other_alignment"],
            summary_stats["interpretation"],
        )

    summary = build_alignment_summary(per_model)
    write_results_summary(summary, cfg.output_dir / "results_summary.json")


if __name__ == "__main__":
    main()
