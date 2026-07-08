"""Truth Direction Optimization (TDO) — Algorithm 1, §3.4.

Learns a single truth direction r ∈ S^{d-1} by gradient descent on
the composite loss (§3.4):

    L(r) = λ_abl · L_abl(r) + λ_add · L_add(r) + λ_ret · L_ret(r)

where:
    L_abl = CE(f_ablate(r)(p_true),  t_false)   — necessity
    L_add = CE(f_add(r,l*,α)(p_false), t_true)  — sufficiency
    L_ret = KL(f_ablate(r)(p_retain) || f(p_retain)) — surgicality

Initialized from θ_DIM / ||θ_DIM|| (warm-start, Algorithm 1 line 1).
After each gradient step, r is renormalized to S^{d-1} (line 6).

Adapted from Wollschläger et al. (2025) Algorithm 1 (RDO) to the
truth domain: refusal targets become True/False targets, harmful/safe
prompt pairs become true/false statement pairs.

For k > 1, use truth_cone_optimization (Algorithm 2, §3.5).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import LossWeights, TrainingHyperparams
from src.data.tdo_dataset import TDODataset, tdo_collate
from src.evaluation.kl import kl_div_fn
from src.interventions.projection import projection_einops
from src.interventions.truth_cone import TruthCone
from src.logging_setup import get_logger

logger = get_logger(__name__)


def truth_direction_optimization(
    model: Any,
    train_dataset: TDODataset,
    add_layer: int,
    alpha: float,
    training: TrainingHyperparams,
    loss_weights: LossWeights,
    init_vector: Optional[torch.Tensor] = None,
    retain_window: int = 30,
) -> Dict[str, Any]:
    """Learn a single truth direction r via Algorithm 1.

    Args:
        model:         frozen nnsight LanguageModel (bfloat16).
        train_dataset: triples (p_true, p_false, p_retain) per §3.4.
        add_layer:     l* from activation patching (§3.2).
        alpha:         ||θ_DIM||_2, calibrated steering magnitude (§3.3).
        training:      lr=10^-2, effective_batch=16, etc. (Tab. hyperparams).
        loss_weights:  (λ_abl, λ_add, λ_ret) = (1.0, 0.2, 1.0) from
                       Wollschläger Tab. 3, held fixed across models.
        init_vector:   (d,) DIM warm-start. Algorithm 1 line 1:
                       r ← θ_DIM / ||θ_DIM||.
        retain_window: last N token positions for L_ret (paper: 30).

    Returns:
        dict with metadata, results (best_vector, lowest_loss), and
        per-step history (loss components, vector trajectory).
    """
    # TruthCone with k=1 degenerates to a single direction (§3.4).
    cone = TruthCone(
        module=model.model,
        hidden_size=model.config.hidden_size,
        n_vectors=1,
        init_vectors=init_vector.unsqueeze(0) if init_vector is not None else None,
    )
    cone.to(model.device)

    # AdamW, lr = 10^-2 (Tab. hyperparams). Paper: weight_decay = 0.
    optimizer = torch.optim.AdamW(
        cone.parameters(),
        lr=training.lr,
        betas=(0.9, 0.98),
        weight_decay=training.weight_decay,
        amsgrad=True,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=tdo_collate,
    )

    # Physical batch 1, effective batch 16 (gradient accumulation).
    accumulation_steps = training.effective_batch_size // training.batch_size

    history = {
        "vectors":    [],
        "total_loss": [],
        "abl_loss":   [],
        "add_loss":   [],
        "ret_loss":   [],
        "lr_changes": [],
    }

    lowest_loss = float("inf")
    best_vector: Optional[torch.Tensor] = None
    patience_counter = 0
    lr_reduce_counter = 0
    early_stopped = False

    batch_abl = batch_add = batch_ret = 0.0

    logger.info("TDO start  add_layer=%d  alpha=%.4f  lr=%.4f", add_layer, alpha, training.lr)

    for epoch in range(training.epochs):
        logger.info("Epoch %d/%d", epoch + 1, training.epochs)
        for step, batch in enumerate(tqdm(loader, leave=False, desc="TDO")):
            p_true   = batch["p_true"]
            p_false  = batch["p_false"]
            p_retain = batch["p_retain"]
            t_true   = batch["t_true"]
            t_false  = batch["t_false"]

            direction = cone.fn_vectors[0]  # single basis vector r

            # ── L_add = CE(f_add(r, l*, α)(p_false), t_true) — sufficiency ───
            with model.trace() as tracer:
                with tracer.invoke(p_false):
                    cone.add(direction, alpha, add_layer)
                    logits = model.lm_head.output[:, -1, :]
                    target = torch.tensor(t_true, dtype=torch.long, device=model.device)
                    loss_add = F.cross_entropy(logits, target)
                    batch_add += loss_add.detach().item()
                    (loss_weights.addition * loss_add).backward(retain_graph=True)
                    del loss_add
                    torch.cuda.empty_cache()

            # ── L_abl = CE(f_ablate(r)(p_true), t_false) — necessity ─────────
            with model.trace() as tracer:
                with tracer.invoke(p_true):
                    cone.ablate(direction)
                    logits = model.lm_head.output[:, -1, :]
                    target = torch.tensor(t_false, dtype=torch.long, device=model.device)
                    loss_abl = F.cross_entropy(logits, target)
                    batch_abl += loss_abl.detach().item()
                    (loss_weights.ablation * loss_abl).backward(retain_graph=True)
                    del loss_abl
                    torch.cuda.empty_cache()

            # ── L_ret = KL(f_ablate(r)(p_retain) || f(p_retain)) — surgicality
            # FLAG: Clean retain logits recomputed every step. TCO caches them
            # once per batch (Win 1). Same optimization applies here — the
            # clean forward pass f(p_retain) is independent of r.
            with model.trace() as tracer:
                with tracer.invoke(p_retain):
                    clean_logits = model.lm_head.output[:, -retain_window:].save()

            with model.trace() as tracer:
                with tracer.invoke(p_retain):
                    cone.ablate(direction)
                    intervened_logits = model.lm_head.output[:, -retain_window:]
                    loss_ret = kl_div_fn(intervened_logits, clean_logits).mean()
                    batch_ret += loss_ret.detach().item()
                    (loss_weights.retention * loss_ret).backward(retain_graph=True)

            # ── Gradient accumulation (Algorithm 1, lines 5–6) ────────────────
            if (step + 1) % accumulation_steps == 0:
                # Spherical gradient projection (§8.5):
                # g_⊥ = g − (rᵀg) r — project onto tangent plane of S^{d-1}.
                for v in cone.fn_vectors:
                    if v.grad is not None:
                        v.grad.sub_(
                            projection_einops(v.grad.unsqueeze(0), v.data).squeeze(0)
                        )

                torch.nn.utils.clip_grad_norm_(cone.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Algorithm 1 line 6: r ← r / ||r||
                cone.normalize()

                # ── Logging ───────────────────────────────────────────────────
                batch_abl /= accumulation_steps
                batch_add /= accumulation_steps
                batch_ret /= accumulation_steps
                total_loss = batch_abl + batch_add + batch_ret

                history["total_loss"].append(total_loss)
                history["abl_loss"].append(batch_abl)
                history["add_loss"].append(batch_add)
                history["ret_loss"].append(batch_ret)
                history["vectors"].append(cone.get_basis_matrix().detach().cpu().clone())

                # ── Early stopping + LR schedule (not in Algorithm 1) ─────────
                # Wollschläger Tab. 3: "divide by 1/10 up to 2 times."
                if total_loss < lowest_loss:
                    lowest_loss = total_loss
                    best_vector = cone.get_basis_matrix().detach().cpu().clone()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= training.patience:
                    if lr_reduce_counter >= training.n_lr_reduce:
                        early_stopped = True
                        break
                    optimizer.param_groups[0]["lr"] /= 10.0
                    history["lr_changes"].append(step)
                    lr_reduce_counter += 1
                    patience_counter = 0
                    logger.info(
                        "  LR reduced to %.6f at step %d",
                        optimizer.param_groups[0]["lr"], step,
                    )

                batch_abl = batch_add = batch_ret = 0.0

        if early_stopped:
            break

    if best_vector is None:
        best_vector = cone.get_basis_matrix().detach().cpu()

    return {
        "metadata": {
            "add_layer": add_layer,
            "alpha": alpha,
            "cone_dim": 1,
            "training": training.model_dump(),
            "loss_weights": loss_weights.model_dump(),
            "retain_window": retain_window,
        },
        "results": {
            "best_vectors": best_vector,        # (1, d)
            "lowest_loss": lowest_loss,
            "did_early_stop": early_stopped,
        },
        "history": history,
    }