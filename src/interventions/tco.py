"""Truth Cone Optimization (TCO) — generalization of TDO to k dimensions.

Extends TDO by sampling ``n_sample`` directions uniformly from the cone's
unit hypersphere at every accumulation step and computing the composite
loss on each. This forces the optimizer to constrain the *entire cone
volume*, not just its boundary basis vectors — preventing the degenerate
solution where only the basis vectors satisfy the loss while the interior
contains "dead zones".

For ``cone_dim=1`` this is mathematically equivalent to TDO.
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


def truth_cone_optimization(
    model: Any,
    train_dataset: TDODataset,
    cone_dim: int,
    add_layer: int,
    alpha: float,
    training: TrainingHyperparams,
    loss_weights: LossWeights,
    init_vectors: Optional[torch.Tensor] = None,
    retain_window: int = 30,
    optimize_basis: bool = True,
) -> Dict[str, Any]:
    """Learn a k-D orthonormal truth cone via gradient descent + MC sampling.

    The training step computes the composite loss on TWO sources of directions:
      1. ``n_sample`` MC-sampled directions from the cone's unit hypersphere
         (interior coverage — scaled by 1/n_sample).
      2. The ``cone_dim`` explicit basis vectors ``cone.fn_vectors``
         (boundary anchoring — scaled by 1/cone_dim).

    Both contributions backpropagate into the same basis parameters, mirroring
    Wollschläger et al. (2025) Algorithm 2 / RDO ``optimize_basis=True``.

    Args:
        model:           nnsight LanguageModel.
        train_dataset:   TDODataset of triplets.
        cone_dim:        target subspace dimensionality (k).
        add_layer:       layer index for L_add injection.
        alpha:           injection magnitude.
        training:        hyperparameters; ``training.n_sample`` controls the
                         number of MC directions per step.
        loss_weights:    λ_abl, λ_add, λ_ret.
        init_vectors:    optional ``(k, D)`` warm-start.
        retain_window:   trailing-token window for L_ret KL.
        optimize_basis:  if True, additionally backprop the composite loss
                         through each explicit basis vector (recommended).

    Returns:
        dict with ``metadata``, ``results`` (best_vectors + final loss), and
        ``history`` (per-step loss components and basis trajectory).
    """
    cone = TruthCone(
        module=model.model,
        hidden_size=model.config.hidden_size,
        n_vectors=cone_dim,
        init_vectors=init_vectors,
    )
    cone.to(model.device)

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

    accumulation_steps = training.effective_batch_size // training.batch_size
    n_sample = training.n_sample

    history = {
        "vectors":         [],
        "total_loss":      [],
        "abl_loss":        [],
        "add_loss":        [],
        "ret_loss":        [],
        "sample_abl_loss": [],
        "sample_add_loss": [],
        "sample_ret_loss": [],
        "basis_abl_loss":  [],
        "basis_add_loss":  [],
        "basis_ret_loss":  [],
        "lr_changes":      [],
    }

    lowest_loss = float("inf")
    best_basis: Optional[torch.Tensor] = None
    patience_counter = 0
    lr_reduce_counter = 0
    early_stopped = False

    # Separate trackers for MC-sample and basis-vector contributions
    batch_sample_abl = batch_sample_add = batch_sample_ret = 0.0
    batch_basis_abl  = batch_basis_add  = batch_basis_ret  = 0.0

    logger.info(
        "TCO start  cone_dim=%d  add_layer=%d  alpha=%.4f  n_sample=%d  optimize_basis=%s",
        cone_dim, add_layer, alpha, n_sample, optimize_basis,
    )

    for epoch in range(training.epochs):
        logger.info("Epoch %d/%d", epoch + 1, training.epochs)
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"TCO d={cone_dim}")):
            p_true   = batch["p_true"]
            p_false  = batch["p_false"]
            p_retain = batch["p_retain"]
            t_true   = batch["t_true"]
            t_false  = batch["t_false"]

            # ──────────────────────────────────────────────────────────────────
            # (1) MC sampling: n_sample directions from current cone interior
            # ──────────────────────────────────────────────────────────────────
            sample_directions = cone.sample_directions(n_sample)  # (n_sample, D)

            for s_idx in range(n_sample):
                direction = sample_directions[s_idx]  # (D,)

                # L_add
                with model.trace() as tracer:
                    with tracer.invoke(p_false):
                        cone.add(direction, alpha, add_layer)
                        logits = model.lm_head.output[:, -1, :]
                        target = torch.tensor(t_true, dtype=torch.long, device=model.device)
                        loss_add = F.cross_entropy(logits, target) / cone_dim
                        batch_sample_add += loss_add.detach().item() / n_sample
                        (loss_weights.addition * loss_add / n_sample).backward(retain_graph=True)

                # L_abl
                with model.trace() as tracer:
                    with tracer.invoke(p_true):
                        cone.ablate(direction)
                        logits = model.lm_head.output[:, -1, :]
                        target = torch.tensor(t_false, dtype=torch.long, device=model.device)
                        loss_abl = F.cross_entropy(logits, target) / cone_dim
                        batch_sample_abl += loss_abl.detach().item() / n_sample
                        (loss_weights.ablation * loss_abl / n_sample).backward(retain_graph=True)

                # L_ret
                with model.trace() as tracer:
                    with tracer.invoke(p_retain):
                        clean_logits = model.lm_head.output[:, -retain_window:].save()

                with model.trace() as tracer:
                    with tracer.invoke(p_retain):
                        cone.ablate(direction)
                        intervened_logits = model.lm_head.output[:, -retain_window:]
                        loss_ret = kl_div_fn(intervened_logits, clean_logits).mean()
                        batch_sample_ret += loss_ret.detach().item() / n_sample
                        (loss_weights.retention * loss_ret / n_sample).backward(retain_graph=True)

            # ──────────────────────────────────────────────────────────────────
            # (2) Explicit basis-vector optimization (rdo.py optimize_basis=True)
            # ──────────────────────────────────────────────────────────────────
            if optimize_basis:
                for fn_vector in cone.fn_vectors:

                    # L_add on basis vector
                    with model.trace() as tracer:
                        with tracer.invoke(p_false):
                            cone.add(fn_vector, alpha, add_layer)
                            logits = model.lm_head.output[:, -1, :]
                            target = torch.tensor(t_true, dtype=torch.long, device=model.device)
                            basis_loss_add = F.cross_entropy(logits, target) / cone_dim
                            batch_basis_add += basis_loss_add.detach().item()
                            (loss_weights.addition * basis_loss_add).backward(retain_graph=True)

                    # L_abl on basis vector
                    with model.trace() as tracer:
                        with tracer.invoke(p_true):
                            cone.ablate(fn_vector)
                            logits = model.lm_head.output[:, -1, :]
                            target = torch.tensor(t_false, dtype=torch.long, device=model.device)
                            basis_loss_abl = F.cross_entropy(logits, target) / cone_dim
                            batch_basis_abl += basis_loss_abl.detach().item()
                            (loss_weights.ablation * basis_loss_abl).backward(retain_graph=True)

                    # L_ret on basis vector
                    with model.trace() as tracer:
                        with tracer.invoke(p_retain):
                            basis_clean_logits = model.lm_head.output[:, -retain_window:].save()

                    with model.trace() as tracer:
                        with tracer.invoke(p_retain):
                            cone.ablate(fn_vector)
                            basis_intervened_logits = model.lm_head.output[:, -retain_window:]
                            basis_loss_ret = (
                                kl_div_fn(basis_intervened_logits, basis_clean_logits).mean() / cone_dim
                            )
                            batch_basis_ret += basis_loss_ret.detach().item()
                            (loss_weights.retention * basis_loss_ret).backward(retain_graph=True)

            # ──────────────────────────────────────────────────────────────────
            # Accumulation step
            # ──────────────────────────────────────────────────────────────────
            if (step + 1) % accumulation_steps == 0:
                # Spherical gradient projection per basis vector
                for v in cone.fn_vectors:
                    if v.grad is not None:
                        v.grad.sub_(
                            projection_einops(v.grad.unsqueeze(0), v.data).squeeze(0)
                        )

                torch.nn.utils.clip_grad_norm_(cone.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Re-orthogonalize the basis after the gradient step
                cone.orthogonalize()

                # Average per-bucket losses over the accumulation window
                batch_sample_abl /= accumulation_steps
                batch_sample_add /= accumulation_steps
                batch_sample_ret /= accumulation_steps
                batch_basis_abl  /= accumulation_steps
                batch_basis_add  /= accumulation_steps
                batch_basis_ret  /= accumulation_steps

                # Fused per-component losses (sample + basis) drive early stopping
                batch_abl = batch_sample_abl + batch_basis_abl
                batch_add = batch_sample_add + batch_basis_add
                batch_ret = batch_sample_ret + batch_basis_ret
                total_loss = batch_abl + batch_add + batch_ret

                history["total_loss"].append(total_loss)
                history["abl_loss"].append(batch_abl)
                history["add_loss"].append(batch_add)
                history["ret_loss"].append(batch_ret)
                history["sample_abl_loss"].append(batch_sample_abl)
                history["sample_add_loss"].append(batch_sample_add)
                history["sample_ret_loss"].append(batch_sample_ret)
                history["basis_abl_loss"].append(batch_basis_abl)
                history["basis_add_loss"].append(batch_basis_add)
                history["basis_ret_loss"].append(batch_basis_ret)
                history["vectors"].append(cone.get_basis_matrix().detach().cpu().clone())

                if total_loss < lowest_loss:
                    lowest_loss = total_loss
                    best_basis = cone.get_basis_matrix().detach().cpu().clone()
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

                # Reset both buckets for next accumulation window
                batch_sample_abl = batch_sample_add = batch_sample_ret = 0.0
                batch_basis_abl  = batch_basis_add  = batch_basis_ret  = 0.0

        if early_stopped:
            break

    if best_basis is None:
        best_basis = cone.get_basis_matrix().detach().cpu()

    return {
        "metadata": {
            "add_layer":      add_layer,
            "alpha":          alpha,
            "cone_dim":       cone_dim,
            "n_sample":       n_sample,
            "optimize_basis": optimize_basis,
            "training":       training.model_dump(),
            "loss_weights":   loss_weights.model_dump(),
            "retain_window":  retain_window,
        },
        "results": {
            "best_vectors":   best_basis,        # shape (cone_dim, D)
            "lowest_loss":    lowest_loss,
            "did_early_stop": early_stopped,
        },
        "history": history,
    }