"""Truth Cone Optimization — Algorithm 2 (§3.5), optimized variant.

Learns an orthonormal cone basis V = [v_1, ..., v_k] by minimizing a
composite loss on both MC-sampled interior directions and basis vectors:

    L = L_sample + L_basis                         (Algorithm 2, lines 8–10)
    L_sample = E_{u ~ Sample(V)} [ComputeLoss(u)]  (MC interior, prevents basis collapse)
    L_basis  = (1/k) Σ ComputeLoss(v_i)            (boundary, stabilizes ASR lower bound)

Each ComputeLoss evaluates the three-term objective from Algorithm 1:
    λ_abl · CE(f_ablate(r)(p_true), t_false)        (necessity)
  + λ_add · CE(f_add(r,l*,α)(p_false), t_true)      (sufficiency)
  + λ_ret · KL(f_ablate(r)(p_retain) || f(p_retain)) (surgicality)

After each gradient step, V is re-orthonormalized via modified
Gram–Schmidt (Algorithm 2, line 12).

Speedups vs naive implementation:
  1. Clean retain logits cached once per batch (was: recomputed per direction)
  2. k=1 skipped at driver level (equivalent to TDO — applied in run_tco.py)
  3. MC samples + basis vectors fused into one direction loop
  4. Three separate traces per direction but each does a single backward
     (was: 3× backward with retain_graph on a shared graph)
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
    """Learn a k-dimensional orthonormal truth cone (Algorithm 2).

    Args:
        model:          frozen language model loaded via nnsight.
        train_dataset:  triples (p_true, p_false, p_retain).
        cone_dim:       cone dimension k. Paper tests k ∈ {1,2,3,4}.
        add_layer:      target layer l* from activation patching (§3.2).
        alpha:          steering magnitude ||θ_DIM||_2 (§3.3).
        training:       lr, epochs, batch_size, effective_batch_size, etc.
        loss_weights:   (λ_abl, λ_add, λ_ret) = (1.0, 0.2, 1.0) per Tab. hyperparams.
        init_vectors:   (k, d) warm-start. For k=1: DIM direction.
                        For k>1: previous (k-1)-cone augmented with v_rand (§3.5).
        retain_window:  last N token positions for KL retention (paper: 30).
        optimize_basis: whether to include L_basis. Paper always does (Algorithm 2).
    """
    cone = TruthCone(
        module=model.model,
        hidden_size=model.config.hidden_size,
        n_vectors=cone_dim,
        init_vectors=init_vectors,
    )
    cone.to(model.device)

    # AdamW with lr = 10^-2 (§8.4, Tab. hyperparams).
    # Paper (Wollschläger Tab. 3) specifies weight_decay = 0.
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

    # Paper: physical batch 1, effective batch 16 (gradient accumulation).
    accumulation_steps = training.effective_batch_size // training.batch_size
    # Paper: 16 MC directions per accumulation step (§3.5).
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

    batch_sample_abl = batch_sample_add = batch_sample_ret = 0.0
    batch_basis_abl  = batch_basis_add  = batch_basis_ret  = 0.0

    sample_factor = 1.0 / (cone_dim * n_sample)
    basis_factor  = 1.0 / cone_dim

    logger.info(
        "TCO (optimized) cone_dim=%d add_layer=%d alpha=%.4f n_sample=%d optimize_basis=%s",
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

            target_true  = torch.tensor(t_true,  dtype=torch.long, device=model.device)
            target_false = torch.tensor(t_false, dtype=torch.long, device=model.device)

            # ── Win 1: cache clean retain logits once per batch ───────────────
            # f(p_retain) in L_ret = KL(f_ablate(r)(p_retain) || f(p_retain)).
            # Independent of direction r, so computed once and reused across
            # all n_sample + k directions below.
            with torch.no_grad():
                with model.trace() as tracer:
                    with tracer.invoke(p_retain):
                        clean_logits = model.lm_head.output[:, -retain_window:].save()

            # ── Win 3: fuse MC samples + basis into one direction loop ────────
            # Algorithm 2 lines 8–9: L = L_sample + L_basis.
            # Instead of two separate loops, concatenate all directions and
            # tag each as sample or basis for correct weighting.
            sample_dirs = cone.sample_directions(n_sample)              # (n_sample, d)
            if optimize_basis:
                basis_dirs = torch.stack(list(cone.fn_vectors), dim=0)  # (k, d)
                all_dirs   = torch.cat([sample_dirs, basis_dirs], dim=0)
                n_total    = n_sample + cone_dim
            else:
                all_dirs   = sample_dirs
                n_total    = n_sample

            for d_idx in range(n_total):
                direction = all_dirs[d_idx]
                is_basis  = d_idx >= n_sample
                factor    = basis_factor if is_basis else sample_factor

                # L_add = CE(f_add(r, l*, α)(p_false), t_true)  — sufficiency
                with model.trace() as tracer:
                    with tracer.invoke(p_false):
                        cone.add(direction, alpha, add_layer)
                        logits_add = model.lm_head.output[:, -1, :]
                        loss_add = F.cross_entropy(logits_add, target_true)
                        la_log = loss_add.detach().item().save()
                        (loss_weights.addition * loss_add * factor).backward(retain_graph=True)

                # L_abl = CE(f_ablate(r)(p_true), t_false)  — necessity
                with model.trace() as tracer:
                    with tracer.invoke(p_true):
                        cone.ablate(direction)
                        logits_abl = model.lm_head.output[:, -1, :]
                        loss_abl = F.cross_entropy(logits_abl, target_false)
                        lb_log = loss_abl.detach().item().save()
                        (loss_weights.ablation * loss_abl * factor).backward(retain_graph=True)

                # L_ret = KL(f_ablate(r)(p_retain) || f(p_retain))  — surgicality
                with model.trace() as tracer:
                    with tracer.invoke(p_retain):
                        cone.ablate(direction)
                        logits_ret = model.lm_head.output[:, -retain_window:]
                        loss_ret = kl_div_fn(logits_ret, clean_logits).mean()
                        lr_log = loss_ret.detach().item().save()
                        (loss_weights.retention * loss_ret * factor).backward(retain_graph=True)

                if is_basis:
                    batch_basis_add += la_log * factor
                    batch_basis_abl += lb_log * factor
                    batch_basis_ret += lr_log * factor
                else:
                    batch_sample_add += la_log * factor
                    batch_sample_abl += lb_log * factor
                    batch_sample_ret += lr_log * factor

            # ── Gradient accumulation step (Algorithm 2, lines 11–12) ─────────
            if (step + 1) % accumulation_steps == 0:
                # Spherical gradient projection (§8.5):
                # g_⊥ = g − (v^T g) v — project gradient onto tangent plane
                # of unit sphere at v, so the step only rotates v.
                for v in cone.fn_vectors:
                    if v.grad is not None:
                        v.grad.sub_(
                            projection_einops(v.grad.unsqueeze(0), v.data).squeeze(0)
                        )

                torch.nn.utils.clip_grad_norm_(cone.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Re-orthonormalize V via modified Gram–Schmidt (Alg. 2 line 12).
                cone.orthogonalize()

                # ── Logging ───────────────────────────────────────────────────
                batch_sample_abl /= accumulation_steps
                batch_sample_add /= accumulation_steps
                batch_sample_ret /= accumulation_steps
                batch_basis_abl  /= accumulation_steps
                batch_basis_add  /= accumulation_steps
                batch_basis_ret  /= accumulation_steps

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

                # ── Early stopping + LR schedule (not in Algorithm 2) ─────────
                # Implementation-level engineering: reduce LR by 10× on
                # plateau, up to n_lr_reduce times. Wollschläger Tab. 3:
                # "divide by 1/10 up to 2 times, every 5 batches if plateaued."
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
            "best_vectors":   best_basis,
            "lowest_loss":    lowest_loss,
            "did_early_stop": early_stopped,
        },
        "history": history,
    }