"""KL retention — L_ret in training (§3.4) and surgicality metric (§3.6, §4.4 / RQ4).

Training role (L_ret):
    KL(f_ablate(r)(p_retain) || f(p_retain))
    Penalizes directions that damage general capabilities on filtered
    Alpaca instructions. Weight λ_ret = 1.0 (Tab. hyperparams).

Evaluation role (§3.6, Tab. KL):
    For each (model, k) pair, sample 32 MC directions from Cone(V),
    compute mean KL over last 30 token positions of 100 held-out Alpaca
    instructions filtered of factual-recall keywords. A cone is
    "surgical" if mean MC KL < 0.1 (Arditi et al. 2024), ~96,000
    token-level comparisons per cone.

Key finding (§4.4): three models fail surgicality at k=1 (KL > 0.1);
all six recover at k=2 (KL < 0.05). The co-failure/co-recovery with
ASR is the clearest evidence for multi-dimensional truth structure.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.interventions.truth_cone import TruthCone
from src.logging_setup import get_logger

logger = get_logger(__name__)


def kl_div_fn(
    logits_intervened: torch.Tensor,
    logits_clean: torch.Tensor,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL(p_intervened || p_clean) in float64 for numerical stability.

    Computes KL divergence between softmax distributions derived from
    the intervened and clean logits. Used in both L_ret during training
    (§3.4, Algorithm 1 ComputeLoss line 5) and the evaluation protocol
    (§3.6).

    Args:
        logits_intervened: logits from f_ablate(r)(p_retain).
        logits_clean:      logits from f(p_retain) (no intervention).
        reduction:         passed to F.kl_div.

    Returns:
        Scalar KL divergence.
    """
    li = logits_intervened.to(torch.float64)
    lc = logits_clean.to(torch.float64)
    return F.kl_div(
        F.log_softmax(li, dim=-1),
        F.softmax(lc, dim=-1),
        reduction=reduction,
    )


@torch.no_grad()
def _kl_for_direction(
    model: Any,
    cone: TruthCone,
    direction: torch.Tensor,
    retain_prompts: List[str],
    batch_size: int,
    logit_window: int,
) -> float:
    """Mean KL divergence over all retain prompts for one direction.

    For each batch: run f(p_retain) clean, then run f_ablate(r)(p_retain),
    compute KL over last logit_window token positions.
    """
    kl_values: list[float] = []

    for i in range(0, len(retain_prompts), batch_size):
        batch = retain_prompts[i : i + batch_size]
        # Clean forward pass: f(p_retain)
        with model.trace() as tracer:
            with tracer.invoke(batch):
                clean_logits = model.lm_head.output[:, -logit_window:, :].save()

        # Ablated forward pass: f_ablate(r)(p_retain)
        with model.trace() as tracer:
            with tracer.invoke(batch):
                cone.ablate(direction)
                ablated_logits = model.lm_head.output[:, -logit_window:, :].save()

        # KL over flattened (batch × window) token positions
        cl  = clean_logits.float().reshape(-1, clean_logits.shape[-1])
        abl = ablated_logits.float().reshape(-1, ablated_logits.shape[-1])
        kl_values.append(kl_div_fn(abl, cl).item())

    return float(np.mean(kl_values))


@torch.no_grad()
def evaluate_retention_kl(
    model: Any,
    cone_basis: torch.Tensor,           # (k, d)
    retain_prompts: List[str],
    n_mc: int = 64,
    batch_size: int = 4,
    logit_window: int = 30,
    kl_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Full KL retention evaluation for a cone (§3.6, §4.4 / RQ4).

    Computes:
      - Per-basis-vector KL: one value per v_i (boundary check)
      - MC KL: n_mc directions sampled from the cone (interior check)
      - Surgicality flag: mc_kl_mean < kl_threshold

    Paper parameters (§3.6):
      - 100 filtered Alpaca prompts (passed as retain_prompts)
      - 32 MC directions per cone
      - 30 token positions per prompt
      - threshold = 0.1 (Arditi et al. 2024)

    Args:
        model:          frozen nnsight LanguageModel.
        cone_basis:     (k, d) learned orthonormal basis V.
        retain_prompts: filtered Alpaca instructions (paper: 100).
        n_mc:           MC directions to sample (paper: 32).
        batch_size:     evaluation batch size.
        logit_window:   last N token positions for KL (paper: 30).
        kl_threshold:   surgicality threshold (paper: 0.1).

    Returns:
        Dict with basis_kl, mc_kl_mean, mc_kl_std, surgical flag.
    """
    cone_dim, hidden_size = cone_basis.shape
    cone = TruthCone(
        module=model.model,
        hidden_size=hidden_size,
        n_vectors=cone_dim,
        init_vectors=cone_basis,
        trainable=False,
    )
    cone.to(model.device)

    # FLAG: n_mc default is 64. Paper says 32 MC directions (§3.6:
    # "sample 32 directions from the cone interior"). Callers should
    # pass n_mc=32 to match the paper. Default should be 32, not 64.

    # ── Per-basis-vector KL (boundary of cone) ────────────────────────────────
    basis_kl: list[float] = []
    for v_idx in range(cone_dim):
        d = cone.fn_vectors[v_idx].data.clone()
        d = d / (d.norm() + 1e-8)
        kl = _kl_for_direction(model, cone, d, retain_prompts, batch_size, logit_window)
        basis_kl.append(kl)
        logger.info("    v%d KL = %.4f", v_idx + 1, kl)

    # ── MC KL across cone interior ────────────────────────────────────────────
    # FLAG: sample_directions samples from positive orthant of S^{k-1}
    # (Cone(V)), matching the paper's evaluation protocol. But the
    # module docstring of TruthCone previously said "cone's unit
    # hypersphere" which was corrected — verify consistency.
    mc_dirs = cone.sample_directions(n_mc)
    mc_kl: list[float] = []
    for s_idx in range(n_mc):
        kl = _kl_for_direction(
            model, cone, mc_dirs[s_idx], retain_prompts, batch_size, logit_window,
        )
        mc_kl.append(kl)

    mc_arr = np.array(mc_kl)
    mc_kl_mean = float(mc_arr.mean())
    # Surgical if mean MC KL < 0.1 (Arditi et al. 2024, Definition 2.1 axiom 2-ii).
    surgical = mc_kl_mean < kl_threshold

    logger.info(
        "  MC KL mean=%.4f std=%.4f  %s",
        mc_kl_mean, float(mc_arr.std()),
        "surgical ✓" if surgical else "exceeds threshold ✗",
    )

    return {
        "basis_kl":      basis_kl,
        "basis_kl_mean": float(np.mean(basis_kl)),
        "mc_kl_mean":    mc_kl_mean,
        "mc_kl_std":     float(mc_arr.std()),
        "mc_kl_samples": mc_kl,
        "n_mc":          n_mc,
        "surgical":      surgical,
        "kl_threshold":  kl_threshold,
    }