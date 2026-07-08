"""results_summary.json builders.

Each experiment writes one of these. The analysis notebooks
(``analysis_paper_figures.ipynb``, ``analysis_cross_experiment_table.ipynb``)
read **only** these files. This decouples computation from presentation —
rerunning training does not invalidate figures, and updating analysis code
does not require re-running the experiments.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.logging_setup import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat()


def write_results_summary(payload: Dict[str, Any], path: Path) -> None:
    """Write a results_summary.json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("results_summary written: %s", path)


# ── Experiment 1 ──────────────────────────────────────────────────────────────


def build_patching_handoff(
    selections_dict: Dict[str, Dict[str, Any]],
    *,
    schema_version: str = "1.0",
) -> Dict[str, Any]:
    """Build the layer_selection.json payload for Experiment 1.

    Use ``upsert_layer_selection`` from ``handoff.py`` instead when running
    model-by-model — this builder is for batch writes.
    """
    return {
        "schema_version": schema_version,
        "produced_by":    "experiment_1",
        "timestamp":      _now_iso(),
        "selections":     selections_dict,
    }


# ── Experiment 2: TDO ─────────────────────────────────────────────────────────


def build_tdo_summary(
    per_model_results: Dict[str, Dict[str, Any]],
    *,
    schema_version: str = "1.0",
) -> Dict[str, Any]:
    """Schema:

    ``models[model_name]``:
        - ``add_layer``, ``alpha``
        - ``weights_file`` (relative to output_dir)
        - ``eval``: ``{dim, cone_1d}`` × ``{asr_ablation, asr_addition}``
        - ``training``: ``{lowest_loss, did_early_stop, total_steps}``
    """
    return {
        "schema_version": schema_version,
        "produced_by":    "experiment_2",
        "timestamp":      _now_iso(),
        "models":         per_model_results,
    }


# ── Experiment 3: TCO ─────────────────────────────────────────────────────────


def build_tco_summary(
    per_model_results: Dict[str, Dict[str, Any]],
    cone_dims: List[int],
    *,
    schema_version: str = "1.0",
) -> Dict[str, Any]:
    """Schema:

    ``models[model_name]``:
        - ``add_layer``, ``alpha``
        - ``cone_dims_evaluated``: list[int]
        - ``optimal_dim``: int
        - ``cones[d]``: weights_file, mc_asr_ablation {mean,std,...},
                        lowest_loss, did_early_stop
    """
    # Pre-compute optimal_dim for each model (downstream experiments rely on it)
    for model_name, m in per_model_results.items():
        if "cones" in m and m["cones"]:
            best = max(
                m["cones"],
                key=lambda d: m["cones"][d].get("mc_asr_ablation", {}).get("mean", -1),
            )
            m["optimal_dim"] = int(best)
        m.setdefault("cone_dims_evaluated", cone_dims)

    return {
        "schema_version": schema_version,
        "produced_by":    "experiment_3",
        "timestamp":      _now_iso(),
        "cone_dims":      cone_dims,
        "models":         per_model_results,
    }


# ── Experiment 4: alignment ───────────────────────────────────────────────────


def build_alignment_summary(
    per_model_results: Dict[str, Dict[str, Any]],
    *,
    schema_version: str = "1.0",
) -> Dict[str, Any]:
    """Schema: ``models[model_name].cosine_similarities[d] = {v1, v2, ...}``."""
    return {
        "schema_version": schema_version,
        "produced_by":    "experiment_4",
        "timestamp":      _now_iso(),
        "models":         per_model_results,
    }


# ── Experiment 5: retention ───────────────────────────────────────────────────


def build_retention_summary(
    per_model_results: Dict[str, Dict[str, Any]],
    *,
    n_retain_prompts: int,
    kl_threshold: float,
    schema_version: str = "1.0",
) -> Dict[str, Any]:
    """Schema:

    ``models[model_name]``:
        - ``cone_dims[d]``: ``{mc_kl_mean, mc_kl_std, basis_kl, surgical}``
        - ``all_surgical``: bool
        - ``worst_dim``: int  ``worst_kl``: float
    """
    # Pre-compute summary stats per model
    for model_name, m in per_model_results.items():
        cone_dims = m.get("cone_dims", {})
        if not cone_dims:
            continue
        all_surg = all(c.get("surgical", False) for c in cone_dims.values())
        worst_d  = max(cone_dims, key=lambda d: cone_dims[d].get("mc_kl_mean", 0.0))
        m["all_surgical"] = all_surg
        m["worst_dim"]    = int(worst_d)
        m["worst_kl"]     = cone_dims[worst_d]["mc_kl_mean"]

    return {
        "schema_version":    schema_version,
        "produced_by":       "experiment_5",
        "timestamp":         _now_iso(),
        "n_retain_prompts":  n_retain_prompts,
        "kl_threshold":      kl_threshold,
        "models":            per_model_results,
    }
