"""Normalized Patching Effect (NPE) and l* selection logic (§3.2).

NPE(l, i) = (M_patched − M_corrupt) / (M_clean − M_corrupt)   (Eq. 3)

Measures how much restoring a single (layer, position) activation from
the clean cache recovers clean-run logit difference. Originally from
Wang et al. (2023); the paper applies it following Zhang et al. (2024),
who prefer logit difference M(p) over probability because it is linear
in the residual stream and decomposes cleanly under patching.

Layer selection (§3.2): sweep NPE over all layers and positions,
aggregate across 50 contrastive pairs × 3 datasets (cities, animals,
elements), then select l* as the most-downstream layer in NPE group (b)
with mean NPE > 0.1.

The three NPE groups (Marks & Tegmark 2024):
  (a) Entity token (STR-perturbed) — encodes the factual association
  (b) End-of-statement punctuation, mid layers — propositional judgment
  (c) Colon token ":", mid-to-late layers — prediction readout
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def normalized_patching_effect(
    patched: float,
    clean: float,
    corrupted: float,
    eps: float = 1e-8,
) -> float:
    """NPE(l, i) = (M_patched − M_corrupt) / (M_clean − M_corrupt).

    Eq. 3 (§3.2). Lies in [0, 1] for typical experiments: 1.0 = full
    restoration of clean-run logit difference, 0.0 = no recovery.
    """
    denom = (clean - corrupted) + eps
    return float((patched - corrupted) / denom)


def select_layer_from_npz(
    npz_paths: list[Path],
    *,
    model_name: str,
    n_layers_total: int | None = None,
    end_stm_token_only: bool = True,
    group_b_start_frac: float = 0.20,
    group_b_end_frac: float = 0.90,
    npe_threshold: float = 0.1,
) -> dict:
    """Aggregate NPE across datasets and select l* from saved .npz files.

    Loads per-dataset NPE matrices (from activation patching sweeps),
    averages across datasets, then delegates to select_target_layer.

    Args:
        npz_paths:           saved NPE .npz files (one per dataset).
        model_name:          model slug for filename parsing.
        n_layers_total:      L. Inferred from NPE shape if None.
        end_stm_token_only:  if True, use only position −5 (end-of-
                             statement punctuation) for candidate scoring.
        group_b_start_frac:  lower bound of group (b) as fraction of L.
        group_b_end_frac:    upper bound of group (b) as fraction of L.
        npe_threshold:       minimum NPE at position −5 to qualify.

    Returns:
        (selected_layer, selection_method, evidence_dict).
    """

    npz_paths = [Path(p) for p in npz_paths]
    if not npz_paths:
        raise ValueError("npz_paths is empty.")

    per_dataset_avg: dict[str, np.ndarray] = {}
    token_labels_ref = None

    for path in npz_paths:
        if not path.exists():
            logger.warning("Missing npz: %s", path)
            continue
        data = np.load(path, allow_pickle=True)
        npe = data["npe"]
        labels = data["token_labels"].tolist()
        m = re.match(rf"{re.escape(model_name)}_(.+?)_heatmap_\d+_\d+_weights$", path.stem)
        key = m.group(1) if m else path.stem
        key = path.stem.replace("_heatmap", "")
        per_dataset_avg[key] = npe
        if token_labels_ref is None:
            token_labels_ref = labels
        logger.info("Loaded %s: shape=%s, n_pairs=%s", path.name, npe.shape,
                    int(data["n_pairs"]) if "n_pairs" in data.files else "?")

    if not per_dataset_avg:
        raise ValueError("No npz files loaded successfully.")

    # Align token dimensions across datasets (they may differ in
    # prefix length), keeping only the shared suffix.
    global_min = min(a.shape[1] for a in per_dataset_avg.values())
    per_dataset_avg = {k: a[:, -global_min:] for k, a in per_dataset_avg.items()}
    # Cross-dataset average (§3.2: "aggregating across models and datasets").
    cross_dataset_avg = np.stack(list(per_dataset_avg.values()), axis=0).mean(axis=0)

    if n_layers_total is None:
        n_layers_total = cross_dataset_avg.shape[0]

    selected, method, evidence = select_target_layer(
        cross_dataset_avg,
        end_stm_token_only=end_stm_token_only,
        group_b_start_frac=group_b_start_frac,
        group_b_end_frac=group_b_end_frac,
        npe_threshold=npe_threshold,
    )

    evidence.update({
        "datasets_aggregated": list(per_dataset_avg.keys()),
        "per_dataset_peak_layer": {
            k: int(np.argmax(m[:, -5])) for k, m in per_dataset_avg.items()
        },
        "source_npz_files": [str(p) for p in npz_paths],
    })

    return selected, method, evidence


def select_target_layer(
    npe_matrix: np.ndarray,            # (L, n_tokens)
    *,
    end_stm_token_only: bool = True,
    group_b_start_frac: float = 0.20,
    group_b_end_frac: float = 0.90,
    npe_threshold: float = 0.1,
) -> Tuple[int, str, dict]:
    """Select l* as the most-downstream layer in NPE group (b) (§3.2).

    Group (b) per Marks & Tegmark (2024): end-of-statement punctuation
    in mid-to-late layers (propositional judgment formation).

    The candidate window [start_frac × L, end_frac × L) restricts
    search to mid-to-late layers. Within that window, layers whose NPE
    at position −5 exceeds npe_threshold are candidates. l* is the
    most-downstream (deepest) qualifying layer.

    Falls back to global-argmax if no candidate qualifies.

    Args:
        npe_matrix:          (L, n_tokens) cross-dataset averaged NPE.
        end_stm_token_only:  if True, score by position −5 only.
        group_b_start_frac:  lower bound of candidate window (fraction of L).
        group_b_end_frac:    upper bound.
        npe_threshold:       minimum NPE to qualify as group (b).

    Returns:
        (selected_layer, selection_method, evidence_dict).
    """
    n_layers, n_tokens = npe_matrix.shape

    # Position −5 = end-of-statement punctuation (§3.2 footnote).
    end_stm_tok_npe = npe_matrix[:, -5] if end_stm_token_only else npe_matrix.mean(axis=1)
    layer_mean_npe = npe_matrix.mean(axis=1)

    start = int(group_b_start_frac * n_layers)
    end = int(group_b_end_frac * n_layers)

    candidate_layers: List[int] = [
        layer
        for layer in range(start, end)
        if end_stm_tok_npe[layer] >= npe_threshold
    ]

    if candidate_layers:
        selected = max(candidate_layers)        # most downstream in group (b)
        method = "most_downstream_group_b"
    else:
        selected = int(layer_mean_npe.argmax())
        method = "global_max_mean_npe"

    evidence = {
        "n_layers":              n_layers,
        "group_b_range":         [start, end],
        "npe_threshold":         npe_threshold,
        "n_candidate_layers":    len(candidate_layers),
        "selected_layer_npe":    float(end_stm_tok_npe[selected]),
        "selected_layer_mean_npe": float(layer_mean_npe[selected]),
    }

    return selected, method, evidence