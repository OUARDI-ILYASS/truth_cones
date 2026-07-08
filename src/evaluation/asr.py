"""Answer Switching Rate (ASR) — §3.1, §3.6.

The primary causal metric. ASR measures the fraction of prompts whose
argmax token flips under intervention (Definition 2.1):

  ASR_abl = |{p_true : argmax flips to "False" after ablation}| / |{p_true}|
  ASR_add = |{p_false : argmax flips to "True" after addition}| / |{p_false}|

ASR_abl tests causal necessity (axiom 2-i): ablating r from true
statements should flip them to False.
ASR_add tests causal sufficiency (axiom 1): adding α·r to false
statements should flip them to True.

For cone evaluation (§4.2), MC ASR averages over 32 directions sampled
uniformly from Cone(V). MC std = 0 means every direction works equally
well — no dead zones in the cone interior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.tdo_dataset import TDODataset, tdo_collate
from src.interventions.truth_cone import TruthCone
from src.logging_setup import get_logger
from src.utils.alpha import get_statement_end_position

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_asr(
    model: Any,
    cone: TruthCone,
    dataset: TDODataset,
    intervention_type: str,            # 'ablation' | 'addition'
    direction: torch.Tensor,
    *,
    alpha: Optional[float] = None,
    add_layer: Optional[int] = None,
    batch_size: int = 4,
) -> Dict[str, float]:
    """Compute ASR for a single direction r.

    For ablation: runs f_ablate(r)(p_true) and counts flips to "False".
    For addition: runs f_add(r, l*, α)(p_false) and counts flips to "True".

    Args:
        model:             frozen nnsight LanguageModel.
        cone:              TruthCone module (dispatches ablate/add).
        dataset:           TDODataset. Ablation uses p_true; addition uses p_false.
        intervention_type: 'ablation' (necessity) or 'addition' (sufficiency).
        direction:         (d,) unit-norm truth direction or cone sample.
        alpha:             ||θ_DIM||_2, required for addition (§3.3).
        add_layer:         l*, required for addition (§3.2).
        batch_size:        evaluation batch size.

    Returns:
        {'asr', 'mean_logit_diff', 'sample_size', 'intervention_type'}.
    """
    if intervention_type not in {"ablation", "addition"}:
        raise ValueError(f"intervention_type must be 'ablation' or 'addition', got {intervention_type!r}")
    if intervention_type == "addition" and (alpha is None or add_layer is None):
        raise ValueError("Addition requires alpha and add_layer.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tdo_collate)

    successes = 0
    total = 0
    logit_diffs: list[float] = []

    for batch in loader:
        # Ablation: intervene on p_true, expect flip to t_false.
        # Addition: intervene on p_false, expect flip to t_true.
        if intervention_type == "ablation":
            prompts    = batch["p_true"]
            t_target   = batch["t_false"]   # expected after ablation
            t_baseline = batch["t_true"]    # original prediction
        else:
            prompts    = batch["p_false"]
            t_target   = batch["t_true"]    # expected after addition
            t_baseline = batch["t_false"]   # original prediction

        with model.trace() as tracer:
            with tracer.invoke(prompts):
                if intervention_type == "ablation":
                    # f_ablate(r): Eq. 1, applied ∀ l, i
                    cone.ablate(direction)
                else:
                    # f_add(r, l*, α): Eq. 2, at layer l*
                    cone.add(direction, alpha=alpha, layer_idx=add_layer)
                logits = model.lm_head.output[:, -1, :].save()

        # FLAG: ASR criterion differs from paper.
        #
        # Paper (§3.1): "fraction of factual prompts whose ARGMAX TOKEN
        # flips under intervention." This means:
        #   success = (argmax of intervened logits) == t_target
        # i.e. the target token must be THE top prediction.
        #
        # Code checks: logit(t_target) > logit(t_baseline)
        # i.e. the target token must beat the baseline token, but need
        # not be the global argmax.
        #
        # These differ when a third token (neither "True" nor "False")
        # has the highest logit after intervention. The paper's argmax
        # criterion would count that as a failure; the code's pairwise
        # comparison would count it as a success if t_target > t_baseline.
        #
        # In practice, the restricted vocabulary (only "True"/"False"
        # considered) likely makes this moot, but the code should match
        # the paper's stated definition for correctness.
        resolved_logits = logits.value if hasattr(logits, "value") else logits
        for i in range(len(prompts)):
            l_target   = resolved_logits[i, t_target[i]]
            l_baseline = resolved_logits[i, t_baseline[i]]
            successes += int(l_target > l_baseline)
            total += 1
            logit_diffs.append((l_target - l_baseline).item())

    return {
        "intervention_type": intervention_type,
        "asr":               successes / total if total else 0.0,
        "mean_logit_diff":   float(np.mean(logit_diffs)) if logit_diffs else 0.0,
        "sample_size":       total,
    }


@torch.no_grad()
def evaluate_dim_baseline(
    model: Any,
    dim_vector: torch.Tensor,
    dataset: TDODataset,
    add_layer: int,
    alpha: float,
    batch_size: int = 4,
) -> Dict[str, Dict[str, float]]:
    """Evaluate ASR for the DIM baseline (§3.3, §4.1 / RQ1).

    Wraps θ_DIM in a non-trainable k=1 TruthCone and evaluates both
    ASR_abl and ASR_add. Results appear in Tab. tdo_vs_dim.

    Args:
        model:      frozen nnsight LanguageModel.
        dim_vector: (d,) θ_DIM = μ_+ − μ_− from §3.3.
        dataset:    TDODataset for evaluation (ID 20% split or OOD elements).
        add_layer:  l* from activation patching (§3.2).
        alpha:      ||θ_DIM||_2 (§3.3).
        batch_size: evaluation batch size.

    Returns:
        {'ablation': {asr, ...}, 'addition': {asr, ...}}.
    """
    # Wrap DIM in a non-trainable k=1 cone for intervention dispatch.
    cone = TruthCone(
        module=model.model,
        hidden_size=model.config.hidden_size,
        n_vectors=1,
        init_vectors=dim_vector.unsqueeze(0),
        trainable=False,
    )
    cone.to(model.device)

    direction = cone.fn_vectors[0].data
    return {
        "ablation": evaluate_asr(
            model, cone, dataset, "ablation", direction, alpha=alpha, add_layer=add_layer,
            batch_size=batch_size,
        ),
        "addition": evaluate_asr(
            model, cone, dataset, "addition", direction,
            alpha=alpha, add_layer=add_layer, batch_size=batch_size,
        ),
    }