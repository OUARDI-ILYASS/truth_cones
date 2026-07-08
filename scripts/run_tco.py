"""Experiment 3 — Truth Cone Optimization (k-D) CLI driver.

Usage:
    python scripts/run_tco.py --config configs/tco.yaml
    python scripts/run_tco.py --config configs/tco.yaml --model qwen-2.5-1.5b-instruct


For each model and each cone_dim ∈ {1..5}:
  1. Load alpha and target layer from Experiment 1 + Experiment 2 outputs
  2. Build TDODataset
  3. Warm-start: d=1 from DIM, d>1 from previous (d-1) basis + new random vector
  4. Run TCO with MC sampling during training
  5. Evaluate with 256 MC samples on the held-out set
  6. Save manifest + .pt + update results_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import subprocess

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TCOConfig, load_config
from src.data.factual import load_factual_datasets, prepare_factual_split
from src.data.prompts import apply_chat_template_fn, apply_few_shot_fn, apply_retain_template
from src.data.retain import load_retain_dataset, split_retain
from src.data.tdo_dataset import TDODataset
from src.evaluation.mc_sampling import mc_evaluate_cone
from src.interventions.tco_optimized import truth_cone_optimization
from src.interventions.truth_cone import TruthCone
from src.logging_setup import configure_root_logger, get_logger
from src.models.loader import load_model
from src.persistence.handoff import load_layer_selection
from src.persistence.manifest import make_run_filenames, write_manifest, write_weights
from src.persistence.results_summary import build_tco_summary, write_results_summary

logger = get_logger(__name__)


def _load_tdo_artifacts(tdo_summary_path: Path, model_name: str, output_dir: Path):
    """Read alpha and DIM vector for a model from Experiment 2 outputs.

    Args:
        tdo_summary_path: path to exp2_tdo/results_summary.json.
        model_name:       which model to look up.
        output_dir:       exp2_tdo/ directory (where weights .pt files live).

    Returns:
        ``(alpha: float, dim_vector: torch.Tensor)``.
    """
    if not tdo_summary_path.exists():
        raise FileNotFoundError(
            f"{tdo_summary_path} not found. Run Experiment 2 first."
        )
    with open(tdo_summary_path) as f:
        summary = json.load(f)

    if model_name not in summary["models"]:
        raise KeyError(f"{model_name} not in TDO summary.")
    entry = summary["models"][model_name]

    alpha = float(entry["alpha"])
    weights_path = output_dir / entry["weights_file"]
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found.")
    weights = torch.load(weights_path, map_location="cpu")
    dim_vector = weights["dim_direction"].float()
    return alpha, dim_vector


def run_for_model(model_spec, cfg: TCOConfig, layer_selections) -> dict:
    """Run TCO sweep for a single model. Returns per-model results dict."""
    if model_spec.name not in layer_selections:
        raise KeyError(f"{model_spec.name} not in layer_selection.json")
    target_layer = layer_selections[model_spec.name].add_layer

    alpha, dim_vector = _load_tdo_artifacts(
        cfg.tdo_summary_path, model_spec.name,
        output_dir=cfg.tdo_summary_path.parent,
    )

    handle = load_model(
        model_spec.name, quantization=model_spec.quantization,
        target_layer=target_layer,
    )
    model = handle.model

    # ── Data ──────────────────────────────────────────────────────────────────
    factual_df = load_factual_datasets(cfg.factual_datasets)
    factual_train, factual_val = prepare_factual_split(
        factual_df, train_ratio=cfg.train_ratio,
    )
    retain_df = load_retain_dataset(cfg.retain_dataset)
    retain_train, retain_val = split_retain(retain_df, train_ratio=cfg.train_ratio)

    train_dataset = TDODataset(
        factual_df=factual_train,
        retain_df=retain_train,
        tokenizer=model.tokenizer,
        apply_few_shot_fn=apply_few_shot_fn,
        apply_retain_template_fn=apply_retain_template,
    )
    val_dataset = TDODataset(
        factual_df=factual_val,
        retain_df=retain_val,
        tokenizer=model.tokenizer,
        apply_few_shot_fn=apply_few_shot_fn,
        apply_retain_template_fn=apply_retain_template,
    )

    ood_df = load_factual_datasets(cfg.ood_test)

    ood_dataset = TDODataset(
        factual_df=ood_df,
        retain_df=retain_val,
        tokenizer=model.tokenizer,
        apply_few_shot_fn=apply_few_shot_fn,
        apply_retain_template_fn=apply_retain_template,
    )

    # in run_tco.py, after load_model, before the cone_dim loop
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke("warmup"):
                _ = model.lm_head.output.save()

    cones_results: dict[str, dict] = {}
    prev_basis: torch.Tensor | None = dim_vector.unsqueeze(0) 

    for cone_dim in cfg.cone_dims:
        logger.info("\n=== %s  d=%d ===", model_spec.name, cone_dim)

        # Warm-start: d=1 from DIM, d>1 from prev (d-1) + new random vector
        if cone_dim == 1:
            init_vectors = dim_vector.unsqueeze(0)            # (1, D)
        else:
            new_vec = torch.randn(1, handle.hidden_size)
            init_vectors = torch.cat([prev_basis, new_vec], dim=0)

        

        tco_out = truth_cone_optimization(
            model=model,
            train_dataset=train_dataset,
            cone_dim=cone_dim,
            add_layer=target_layer,
            alpha=alpha,
            training=cfg.training,
            loss_weights=cfg.loss_weights,
            init_vectors=init_vectors,
        )

        best_basis = tco_out["results"]["best_vectors"]   # (cone_dim, D)
        prev_basis = best_basis

        # ── MC evaluation: 256 directions ────────────────────────────────────
        eval_cone = TruthCone(
            module=model.model,
            hidden_size=handle.hidden_size,
            n_vectors=cone_dim,
            init_vectors=best_basis,
            trainable=False,
        )
        eval_cone.to(model.device)

        mc_eval = mc_evaluate_cone(
            model=model,
            cone=eval_cone,
            eval_dataset=val_dataset,
            intervention_type="ablation",
            add_layer=target_layer,
            alpha=alpha,
            n_mc=cfg.mc_eval_samples,
            batch_size=cfg.training.batch_size,
        )
        logger.info(
            "  d=%d  MC ASR mean=%.3f std=%.3f",
            cone_dim, mc_eval["asr_mean"], mc_eval["asr_std"],
        )

        mc_eval_ood = mc_evaluate_cone(
            model=model,
            cone=eval_cone,
            eval_dataset=ood_dataset,
            intervention_type="ablation",
            add_layer=target_layer,
            alpha=alpha,
            n_mc=cfg.mc_eval_samples,
            batch_size=cfg.training.batch_size,
        )
        logger.info(
            "  d=%d  MC ASR ood mean=%.3f std=%.3f",
            cone_dim, mc_eval_ood["asr_mean"], mc_eval_ood["asr_std"],
        )

        # ── Save ──────────────────────────────────────────────────────────────
        names = make_run_filenames(model_spec.name, cone_dim=cone_dim)
        weights_path = write_weights(
            weights={
                "best_vectors":  best_basis,
                "dim_direction": dim_vector,
            },
            output_dir=cfg.output_dir,
            filename=names["weights"],
        )

        write_manifest(
            manifest={
                "model_name":      model_spec.name,
                "cone_dim":        cone_dim,
                "add_layer":       target_layer,
                "alpha":           alpha,
                "weights_file":    names["weights"],
                "metadata":        tco_out["metadata"],
                "results": {
                    "lowest_loss":    tco_out["results"]["lowest_loss"],
                    "did_early_stop": tco_out["results"]["did_early_stop"],
                },
                "history_metrics": {
                    "abl_loss":   tco_out["history"]["abl_loss"],
                    "add_loss":   tco_out["history"]["add_loss"],
                    "ret_loss":   tco_out["history"]["ret_loss"],
                    "lr_changes": tco_out["history"]["lr_changes"],
                },
                "mc_eval": mc_eval,
                "mc_eval_ood": mc_eval_ood,
            },
            output_dir=cfg.output_dir,
            filename=names["manifest"],
        )

        cones_results[str(cone_dim)] = {
            "weights_file":      names["weights"],
            "basis_key":         "best_vectors",
            "mc_asr_ablation":   {
                "mean":   mc_eval["asr_mean"],
                "std":    mc_eval["asr_std"],
                "median": mc_eval["asr_median"],
                "min":    mc_eval["asr_min"],
                "max":    mc_eval["asr_max"],
            },
            "mc_asr_ablation_ood":   {
                "mean":   mc_eval_ood["asr_mean"],
                "std":    mc_eval_ood["asr_std"],
                "median": mc_eval_ood["asr_median"],
                "min":    mc_eval_ood["asr_min"],
                "max":    mc_eval_ood["asr_max"],
            },
            "lowest_loss":    tco_out["results"]["lowest_loss"],
            "did_early_stop": tco_out["results"]["did_early_stop"],
        }

    return {
        "add_layer":           target_layer,
        "alpha":               alpha,
        "cone_dims_evaluated": cfg.cone_dims,
        "cones":               cones_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 — TCO")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    configure_root_logger(log_file=args.log_file)
    cfg = load_config(args.config, TCOConfig)
    assert isinstance(cfg, TCOConfig)

    layer_selections = load_layer_selection(cfg.layer_selection_path)

    models = (
        [m for m in cfg.models if m.name == args.model]
        if args.model is not None else cfg.models
    )
    if not models:
        logger.error("No models matched.")
        sys.exit(1)

    per_model_results: dict[str, dict] = {}

    for model_spec in models:
        logger.info("\n%s\n=== TCO: %s ===\n%s", "=" * 60, model_spec.name, "=" * 60)
        try:
            per_model_results[model_spec.name] = run_for_model(
                model_spec, cfg, layer_selections,
            )
        except Exception as exc:                       # noqa: BLE001
            logger.exception("TCO failed for %s: %s", model_spec.name, exc)
        finally:
            # Expand ~ to the full home directory path and clear the hub cache
            # cache_path = sys.path.expanduser("~/.cache/huggingface/hub/*")
            subprocess.run(f"rm -rf ~/.cache/huggingface/hub/*", shell=True, check=False)

    summary = build_tco_summary(per_model_results, cfg.cone_dims)
    write_results_summary(summary, cfg.output_dir /  f"{args.model}results_summary.json")


if __name__ == "__main__":
    main()