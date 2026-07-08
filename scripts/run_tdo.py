"""Experiment 2 — Truth Direction Optimization (1-D) CLI driver.

Usage:
    python scripts/run_tdo.py --config configs/tdo.yaml
    python scripts/run_tdo.py --config configs/tdo.yaml --model qwen-2.5-7b-instruct


For each model:
  1. Load target layer from layer_selection.json
  2. Build TDODataset (factual triplets + Alpaca retain)
  3. Extract DIM direction at the target layer
  4. Run TDO with DIM warm-start
  5. Evaluate DIM and 1-D cone ASR (ablation + addition)
  6. Save manifest + .pt + update results_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TDOConfig, load_config
from src.data.factual import load_factual_datasets, prepare_factual_split
from src.data.prompts import apply_chat_template_fn, apply_few_shot_fn, apply_retain_template
from src.data.retain import load_retain_dataset, split_retain
from src.data.tdo_dataset import TDODataset
from src.evaluation.asr import evaluate_asr, evaluate_dim_baseline
from src.interventions.tdo import truth_direction_optimization
from src.interventions.truth_cone import TruthCone
from src.logging_setup import configure_root_logger, get_logger
from src.models.loader import load_model
from src.persistence.handoff import load_layer_selection
from src.persistence.manifest import make_run_filenames, write_manifest, write_weights
from src.persistence.results_summary import build_tdo_summary, write_results_summary
from src.utils.alpha import calibrate_alpha, dim_direction_from_activations

logger = get_logger(__name__)


def run_for_model(model_spec, cfg: TDOConfig, layer_selections) -> dict:
    """Run TDO for a single model. Returns the per-model results dict."""
    if model_spec.name not in layer_selections:
        raise KeyError(
            f"{model_spec.name} not in layer_selection.json. "
            "Run Experiment 1 for this model first."
        )
    target_layer = layer_selections[model_spec.name].add_layer

    handle = load_model(
        model_spec.name,
        quantization=model_spec.quantization,
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

    # ── DIM extraction + alpha calibration ───────────────────────────────────
    train_prompts = apply_few_shot_fn(
        factual_train["statement"].tolist(),
        factual_train["examples"].tolist()
    )
    train_labels = factual_train["label"].tolist()
    dim_raw, dim_unit = dim_direction_from_activations(
        model=model,
        prompts=train_prompts,
        labels=train_labels,
        target_layer=target_layer,
        batch_size=cfg.training.batch_size,
        effective_batch_size=cfg.training.effective_batch_size
    )
    alpha = calibrate_alpha(dim_raw)

    # ── DIM baseline evaluation ──────────────────────────────────────────────
    logger.info("Evaluating DIM baseline...")
    dim_eval = evaluate_dim_baseline(
        model=model,
        dim_vector=dim_raw,
        dataset=val_dataset,
        add_layer=target_layer,
        alpha=alpha,
        batch_size=cfg.training.batch_size,
    )

    dim_ood_eval = evaluate_dim_baseline(
        model=model,
        dim_vector=dim_raw,
        dataset=ood_dataset,
        add_layer=target_layer,
        alpha=alpha,
        batch_size=cfg.training.batch_size,
    )


    logger.info("DIM ASR  abl=%.3f  add=%.3f",
                dim_eval["ablation"]["asr"], dim_eval["addition"]["asr"])

    logger.info("DIM ASR ood abl=%.3f  add=%.3f",
                dim_ood_eval["ablation"]["asr"], dim_ood_eval["addition"]["asr"])

    # ── TDO training ──────────────────────────────────────────────────────────
    tdo_out = truth_direction_optimization(
        model=model,
        train_dataset=train_dataset,
        add_layer=target_layer,
        alpha=alpha,
        training=cfg.training,
        loss_weights=cfg.loss_weights,
        init_vector=dim_unit,
    )
    best_vectors = tdo_out["results"]["best_vectors"]   # (1, D)
    cone_direction = best_vectors[0]

    # ── 1-D cone evaluation ──────────────────────────────────────────────────
    logger.info("Evaluating 1-D cone...")
    eval_cone = TruthCone(
        module=model.model,
        hidden_size=handle.hidden_size,
        n_vectors=1,
        init_vectors=best_vectors,
        trainable=False,
    )
    eval_cone.to(model.device)

    cone_abl = evaluate_asr(
        model, eval_cone, val_dataset, "ablation",
        direction=eval_cone.fn_vectors[0].data, alpha=alpha, add_layer=target_layer,
        batch_size=cfg.training.batch_size,
    )
    cone_add = evaluate_asr(
        model, eval_cone, val_dataset, "addition",
        direction=eval_cone.fn_vectors[0].data,
        alpha=alpha, add_layer=target_layer,
        batch_size=cfg.training.batch_size,
    )

    cone_abl_ood = evaluate_asr(
        model, eval_cone, ood_dataset, "ablation",
        direction=eval_cone.fn_vectors[0].data, alpha=alpha, add_layer=target_layer,
        batch_size=cfg.training.batch_size,
    )
    cone_add_ood = evaluate_asr(
        model, eval_cone, ood_dataset, "addition",
        direction=eval_cone.fn_vectors[0].data,
        alpha=alpha, add_layer=target_layer,
        batch_size=cfg.training.batch_size,
    )

    logger.info("1-D Cone ASR  abl=%.3f  add=%.3f",
                cone_abl["asr"], cone_add["asr"])

    logger.info("1-D Cone ASR ood dataset abl=%.3f  add=%.3f",
                cone_abl_ood["asr"], cone_add_ood["asr"])

    # ── Persist ───────────────────────────────────────────────────────────────
    names = make_run_filenames(model_spec.name)
    weights_path = write_weights(
        weights={
            "best_vectors":   best_vectors,
            "dim_direction":  dim_unit.cpu(),
            "dim_norm":       torch.tensor(alpha),
        },
        output_dir=cfg.output_dir,
        filename=names["weights"],
    )

    write_manifest(
        manifest={
            "model_name":      model_spec.name,
            "add_layer":       target_layer,
            "alpha":           alpha,
            "weights_file":    names["weights"],
            "metadata":        tdo_out["metadata"],
            "results": {
                "lowest_loss":     tdo_out["results"]["lowest_loss"],
                "did_early_stop":  tdo_out["results"]["did_early_stop"],
            },
            "history_metrics": {
                "abl_loss":   tdo_out["history"]["abl_loss"],
                "add_loss":   tdo_out["history"]["add_loss"],
                "ret_loss":   tdo_out["history"]["ret_loss"],
                "lr_changes": tdo_out["history"]["lr_changes"],
            },
            "eval": {
                "dim":     {"asr_ablation": dim_eval["ablation"]["asr"],
                            "asr_addition": dim_eval["addition"]["asr"]},
                "dim_ood":     {"asr_ablation": dim_ood_eval["ablation"]["asr"],
                            "asr_addition": dim_ood_eval["addition"]["asr"]},
                "cone_1d": {"asr_ablation": cone_abl["asr"],
                            "asr_addition": cone_add["asr"]},
                "cone_1d_ood": {"asr_ablation": cone_abl_ood["asr"],
                            "asr_addition": cone_add_ood["asr"]},
            },
        },
        output_dir=cfg.output_dir,
        filename=names["manifest"],
    )

    return {
        "add_layer":      target_layer,
        "alpha":          alpha,
        "weights_file":   names["weights"],
        "dim_vector_key": "dim_direction",
        "cone_vector_key": "best_vectors",
        "eval": {
            "dim":     {"asr_ablation": dim_eval["ablation"]["asr"],
                        "asr_addition": dim_eval["addition"]["asr"]},
            "dim_ood":     {"asr_ablation": dim_ood_eval["ablation"]["asr"],
                        "asr_addition": dim_ood_eval["addition"]["asr"]},
            "cone_1d": {"asr_ablation": cone_abl["asr"],
                        "asr_addition": cone_add["asr"]},
            "cone_1d_ood": {"asr_ablation": cone_abl_ood["asr"],
                        "asr_addition": cone_add_ood["asr"]},
        },
        "training": {
            "lowest_loss":    tdo_out["results"]["lowest_loss"],
            "did_early_stop": tdo_out["results"]["did_early_stop"],
            "total_steps":    len(tdo_out["history"]["total_loss"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 — TDO")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (overrides config). Optional.")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    configure_root_logger(log_file=args.log_file)
    cfg = load_config(args.config, TDOConfig)
    assert isinstance(cfg, TDOConfig)

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
        logger.info("\n=== TDO: %s ===", model_spec.name)
        try:
            per_model_results[model_spec.name] = run_for_model(
                model_spec, cfg, layer_selections,
            )
        except Exception as exc:                       # noqa: BLE001
            logger.exception("TDO failed for %s: %s", model_spec.name, exc)
        finally:
            # Expand ~ to the full home directory path and clear the hub cache
            # cache_path = sys.path.expanduser("~/.cache/huggingface/hub/*")
            subprocess.run(f"rm -rf ~/.cache/huggingface/hub/*", shell=True, check=False)

    summary = build_tdo_summary(per_model_results)
    write_results_summary(summary, cfg.output_dir / "results_summary.json")

if __name__ == "__main__":
    main()
