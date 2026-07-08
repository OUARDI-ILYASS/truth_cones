"""Experiment 5 — Retention via KL Divergence CLI driver.


Usage:
    python scripts/run_retention.py --config configs/retention.yaml
    python scripts/run_retention.py --config configs/retention.yaml --model qwen-2.5-1.5b-instruct


For each model and each cone_dim:
  1. Load saved cone basis from Experiment 3
  2. Sample 200 non-factual Alpaca prompts (filtered)
  3. For each basis vector and 64 MC samples, measure KL divergence
     between clean and ablated distributions
  4. Apply surgicality threshold (KL < 0.1)
  5. Save manifest + update results_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RetentionConfig, load_config
import subprocess
from src.data.prompts import apply_retain_template
from src.data.retain import filter_retain_dataset, load_retain_dataset
from src.evaluation.kl import evaluate_retention_kl
from src.logging_setup import configure_root_logger, get_logger
from src.models.loader import load_model
from src.persistence.manifest import make_run_filenames, write_manifest
from src.persistence.results_summary import build_retention_summary, write_results_summary

logger = get_logger(__name__)


def _load_cone_bases(
    tco_dir: Path,
    model_name: str,
    cone_dims: list[int],
) -> dict[int, torch.Tensor]:
    """Reuse the same loader as Experiment 4."""
    summary_path = tco_dir / "results_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        summary = json.load(f)
    entry = summary["models"].get(model_name)
    if entry is None:
        return {}
    bases: dict[int, torch.Tensor] = {}
    for d in cone_dims:
        cone_entry = entry.get("cones", {}).get(str(d))
        if cone_entry is None:
            continue
        weights_path = tco_dir / cone_entry["weights_file"]
        if not weights_path.exists():
            continue
        weights = torch.load(weights_path, map_location="cpu")
        bases[d] = weights[cone_entry["basis_key"]].float()
    return bases


def run_for_model(
    model_spec,
    cfg: RetentionConfig,
    retain_prompts_raw: list[str],
) -> dict:
    """KL evaluation for a single model across all cone dimensions."""
    handle = load_model(model_spec.name, quantization=model_spec.quantization)
    model = handle.model

    # Format retain prompts through the model's chat template
    retain_prompts = apply_retain_template(model.tokenizer, retain_prompts_raw)
    logger.info("Retain prompts formatted: %d", len(retain_prompts))

    # Load cone bases from Experiment 3
    cone_bases = _load_cone_bases(cfg.tco_dir, model_spec.name, cfg.cone_dims)
    if not cone_bases:
        logger.warning("No cone bases for %s. Skipping.", model_spec.name)
        return {}

         # in run_tco.py, after load_model, before the cone_dim loop
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke("warmup"):
                _ = model.lm_head.output.save()

    
    cone_results: dict[str, dict] = {}
    for cone_dim, basis in cone_bases.items():
        logger.info("=== %s  d=%d ===", model_spec.name, cone_dim)
        result = evaluate_retention_kl(
            model=model,
            cone_basis=basis,
            retain_prompts=retain_prompts,
            n_mc=cfg.n_mc_kl,
            batch_size=8,
            logit_window=cfg.logit_window,
            kl_threshold=cfg.kl_threshold,
        )
        cone_results[str(cone_dim)] = {
            "mc_kl_mean":     result["mc_kl_mean"],
            "mc_kl_std":      result["mc_kl_std"],
            "basis_kl":       result["basis_kl"],
            "basis_kl_mean":  result["basis_kl_mean"],
            "surgical":       result["surgical"],
            "n_mc":           result["n_mc"],
        }

    # ── Manifest ──────────────────────────────────────────────────────────────
    names = make_run_filenames(model_spec.name, suffix="retention")
    write_manifest(
        manifest={
            "model_name":        model_spec.name,
            "n_retain_prompts":  len(retain_prompts),
            "kl_threshold":      cfg.kl_threshold,
            "n_mc_kl":           cfg.n_mc_kl,
            "logit_window":      cfg.logit_window,
            "cone_dims":         cone_results,
        },
        output_dir=cfg.output_dir,
        filename=names["manifest"],
    )

    return {"cone_dims": cone_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 5 — Retention via KL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    configure_root_logger(log_file=args.log_file)
    cfg = load_config(args.config, RetentionConfig)
    assert isinstance(cfg, RetentionConfig)

    # ── Build the fixed retain prompt set (model-independent) ─────────────────
    retain_df = load_retain_dataset(cfg.retain_dataset)
    retain_df = filter_retain_dataset(retain_df, cfg.factual_filter_keywords)
    retain_eval_df = retain_df.sample(
        n=min(cfg.n_retain_prompts, len(retain_df)),
        random_state=42,
    ).reset_index(drop=True)
    retain_prompts_raw = retain_eval_df["instruction"].tolist()
    logger.info("Retain eval set size: %d", len(retain_prompts_raw))

    models = (
        [m for m in cfg.models if m.name == args.model]
        if args.model is not None else cfg.models
    )
    if not models:
        logger.error("No models matched.")
        sys.exit(1)

    per_model: dict[str, dict] = {}
    for model_spec in models:
        logger.info("\n%s\n=== Retention: %s ===\n%s", "=" * 60, model_spec.name, "=" * 60)
        try:
            res = run_for_model(model_spec, cfg, retain_prompts_raw)
            if res:
                per_model[model_spec.name] = res
        except Exception as exc:                       # noqa: BLE001
            logger.exception("Retention failed for %s: %s", model_spec.name, exc)
        finally:
            # Expand ~ to the full home directory path and clear the hub cache
            # cache_path = sys.path.expanduser("~/.cache/huggingface/hub/*")
            subprocess.run(f"rm -rf ~/.cache/huggingface/hub/*", shell=True, check=False)

    summary = build_retention_summary(
        per_model,
        n_retain_prompts=len(retain_prompts_raw),
        kl_threshold=cfg.kl_threshold,
    )
    write_results_summary(summary, cfg.output_dir / "results_summary.json")


if __name__ == "__main__":
    main()
