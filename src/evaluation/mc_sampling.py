"""Monte Carlo ASR evaluation of a learned cone (§3.6, §4.2 / RQ2).

After training a k-D cone, sample directions uniformly from the
non-negative orthant of S^{k-1} (the cone interior) and compute ASR
for each. Reported in Tab. cone_asr.

  MC ASR mean  → "does the entire cone mediate truth?"
  MC ASR std   → "are there dead zones in the interior?"

Paper parameters (§3.6): 32 MC directions per cone. MC std = 0 or
near-zero at k ≤ 2 for all models; grows at k=4 for Qwen-2.5-1.5B
and Llama-3.1-8B where over-parameterization leaves interior pockets
unconstrained (§4.2).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from src.data.tdo_dataset import TDODataset
from src.evaluation.asr import evaluate_asr
from src.interventions.truth_cone import TruthCone
from src.logging_setup import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def mc_evaluate_cone(
    model: Any,
    cone: TruthCone,
    eval_dataset: TDODataset,
    intervention_type: str = "ablation",
    *,
    add_layer: int = -1,
    alpha: float = 0.0,
    n_mc: int = 256,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Sample n_mc directions from Cone(V) and compute ASR for each.

    Directions are sampled via cone.sample_directions (Gaussian folding
    to positive orthant, §3.5 Algorithm 2 Sample function).

    Args:
        model:             frozen nnsight LanguageModel.
        cone:              trained TruthCone with orthonormal basis V.
        eval_dataset:      TDODataset (ID 20% split or OOD elements).
        intervention_type: 'ablation' (ASR_abl) or 'addition' (ASR_add).
        add_layer:         l* (required for addition).
        alpha:             ||θ_DIM||_2 (required for addition).
        n_mc:              number of MC directions to sample.
        batch_size:        evaluation batch size.

    Returns:
        Dict with mean/std/median/min/max ASR and full sample list.
    """
    mc_dirs = cone.sample_directions(n_mc)  # (n_mc, d)
    asr_samples: list[float] = []

    log_every = max(1, n_mc // 8)
    for s_idx in range(n_mc):
        report = evaluate_asr(
            model=model,
            cone=cone,
            dataset=eval_dataset,
            intervention_type=intervention_type,
            direction=mc_dirs[s_idx],
            alpha=alpha,
            add_layer=add_layer,
            batch_size=batch_size,
        )
        asr_samples.append(report["asr"])

        if (s_idx + 1) % log_every == 0:
            logger.info(
                "    MC %3d/%d  running mean ASR=%.3f",
                s_idx + 1, n_mc, float(np.mean(asr_samples)),
            )

    arr = np.array(asr_samples)
    return {
        "intervention_type":   intervention_type,
        "n_mc":                n_mc,
        "asr_mean":            float(arr.mean()),
        "asr_std":             float(arr.std()),
        "asr_median":          float(np.median(arr)),
        "asr_min":             float(arr.min()),
        "asr_max":             float(arr.max()),
        "asr_samples":         asr_samples,
    }