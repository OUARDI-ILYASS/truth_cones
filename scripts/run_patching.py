"""Experiment 1 — Activation patching CLI driver.

Usage:
    python scripts/run_patching.py --config configs/patching.yaml
    python scripts/run_patching.py --config configs/patching.yaml --model qwen-2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so `python scripts/...` works without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PatchingConfig, load_config
from src.data.prompts import generate_prompt
from src.interventions.patching import ActivationPatching
from src.logging_setup import configure_root_logger, get_logger
from src.models.loader import load_model
from src.persistence.handoff import LayerSelection, upsert_layer_selection
from src.persistence.manifest import make_run_filenames, write_manifest
from src.utils.npe import select_target_layer, select_layer_from_npz
from src.utils.visualization import plot_npe_heatmap, save_static_heatmap, save_heatmap_png

logger = get_logger(__name__)


def _load_contrastive_pairs(
    csv_path: Path,
    n_prompts: int,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Sample ``n_prompts`` contrastive (clean, corrupted) STR pairs from a CSV.

    The CSV must contain columns ``clean_prompt, corrupted_prompt``.
    Only pairs with identical token length survive; this is enforced
    downstream by ``ActivationPatching.run``.
    """
    df = pd.read_csv(csv_path)

    # df_sampled = df.sample(n=3, random_state=seed)
    df_sampled = df.sample(n=n_prompts, random_state=seed)

    return list(zip(df_sampled['clean_prompt'], df_sampled['corrupted_prompt']))


def run_for_model(model_spec, datasets, cfg: PatchingConfig) -> None:
    """Run the activation patching sweep for a single model across all datasets."""
    handle = load_model(model_spec.name, quantization=model_spec.quantization)
    sweeper = ActivationPatching(handle.model)

    true_token_id  = handle.model.tokenizer("TRUE").input_ids[-1]
    false_token_id = handle.model.tokenizer("FALSE").input_ids[-1]

    aggregated_npe: dict[str, np.ndarray] = {}

    for ds in datasets:
        logger.info("=== Dataset: %s (model: %s) ===", ds.name, model_spec.name)
        examples = [tuple(ex) for ex in ds.examples] if ds.examples else None
        pairs = _load_contrastive_pairs(ds.csv_path, cfg.n_prompts_per_dataset)

        if not pairs:
            logger.warning("No usable contrastive pairs for %s. Skipping.", ds.name)
            continue

        per_pair_npe: list[np.ndarray] = []
        sample_pair_for_labels: tuple[str, str] | None = None

        for clean_stmt, corrupt_stmt in pairs:
            logger.info("=== clean stm: %s | corrupted stm: %s ===", clean_stmt, corrupt_stmt)
            clean_prompt = generate_prompt(clean_stmt, examples)
            corr_prompt  = generate_prompt(corrupt_stmt, examples)

            try:
                clean_toks = handle.model.tokenizer(clean_prompt).input_ids
                corr_toks  = handle.model.tokenizer(corr_prompt).input_ids
                if len(clean_toks) != len(corr_toks):
                    logger.info("=== length mismatch — skip this pair ===")
                    continue  # length mismatch — skip this pair

                # Compute sweep_start_idx as first token of last example
                newline_id = handle.model.tokenizer("\n").input_ids[-1]
                sweep_start = (
                    len(clean_toks) - 1
                    - clean_toks[::-1].index(newline_id)
                    + 1
                )

                result = sweeper.run(
                    clean_prompt=clean_prompt,
                    corrupted_prompt=corr_prompt,
                    true_token_id=true_token_id,
                    false_token_id=false_token_id,
                    sweep_start_idx=sweep_start,
                )
                per_pair_npe.append(result.npe_matrix)
                if sample_pair_for_labels is None:
                    sample_pair_for_labels = (corr_prompt, sweep_start, result.token_labels)
            except Exception as exc:                  # noqa: BLE001
                logger.warning("Skipping pair due to error: %s", exc)
                continue

        if not per_pair_npe:
            logger.warning("No successful pairs for %s/%s.", model_spec.name, ds.name)
            continue

        min_pos = min(a.shape[1] for a in per_pair_npe)
        per_pair_npe = [a[:, -min_pos:] for a in per_pair_npe]
        avg_npe = np.stack(per_pair_npe, axis=0).mean(axis=0)
        aggregated_npe[ds.name] = avg_npe

        # Save raw matrix
        names = make_run_filenames(model_spec.name, suffix=f"{ds.name}_heatmap")
        npz_path = cfg.output_dir / f"{Path(names['weights']).stem}.npz"
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        token_labels = sample_pair_for_labels[2][-min_pos:]

        np.savez(
            npz_path,
            npe=avg_npe,
            token_labels=np.array(token_labels, dtype=object),
            n_pairs=len(per_pair_npe),
        )
        logger.info("NPE matrix saved: %s", npz_path)

        # Render heatmap
        fig = plot_npe_heatmap(
            avg_npe,
            token_labels=token_labels,
            title=f"Activation Patching — {model_spec.name} / {ds.name}",
        )
        cfg.figures_dir.mkdir(parents=True, exist_ok=True)
        html_path = cfg.figures_dir / f"{model_spec.name}_{ds.name}_heatmap.html"
        png_path  = cfg.figures_dir / f"{model_spec.name}_{ds.name}_heatmap.png"
        fig.write_html(str(html_path))
        try:
            # save_static_heatmap(fig, png_path)
            save_heatmap_png(avg_npe, png_path, token_labels=token_labels, 
                 title=f"Activation Patching — {model_spec.name} / {ds.name}")
        except Exception as exc:                      # noqa: BLE001
            logger.warning("Could not write PNG (kaleido?): %s", exc)

    if not aggregated_npe:
        logger.error("No NPE matrices produced for %s.", model_spec.name)
        return

    # Average across datasets for layer selection
    global_min = min(a.shape[1] for a in aggregated_npe.values())
    aggregated_npe = {k: a[:, -global_min:] for k, a in aggregated_npe.items()}
    cross_dataset_avg = np.stack(list(aggregated_npe.values()), axis=0).mean(axis=0)
    selected_layer, method, evidence = select_target_layer(
        cross_dataset_avg,
        end_stm_token_only=True,
        group_b_start_frac=cfg.group_b_start_frac,
        group_b_end_frac=cfg.group_b_end_frac,
        npe_threshold=cfg.npe_threshold,
    )
    evidence.update({
        "datasets_aggregated":     list(aggregated_npe.keys()),
        f"per_dataset_peak_layer": {
            ds_name: int(np.argmax(m[:, -5]))
            for ds_name, m in aggregated_npe.items()
        },
    })

    selection = LayerSelection(
        add_layer=selected_layer,
        token_position=-5,
        selection_method=method,
        n_layers_total=handle.n_layers,
        normalized_depth=round(selected_layer / handle.n_layers, 3),
        evidence=evidence,
    )

    selection_path = cfg.output_dir / "layer_selection.json"
    upsert_layer_selection(selection_path, model_spec.name, selection)

    # Per-model manifest
    names = make_run_filenames(model_spec.name, suffix="patching")
    write_manifest(
        manifest={
            "model":             model_spec.name,
            "n_layers":          handle.n_layers,
            "datasets":          list(aggregated_npe.keys()),
            "selected_layer":    selected_layer,
            "selection_method":  method,
            "evidence":          evidence,
        },
        output_dir=cfg.output_dir,
        filename=names["manifest"],
    )

    logger.info(
        "Done %s: selected layer=%d (depth=%.2f, method=%s)",
        model_spec.name, selected_layer, selected_layer / handle.n_layers, method,
    )


def reselect_for_model(
    model_name: str,
    output_dir: Path,
    cfg: PatchingConfig,
    **kwargs,
) -> dict:
    paths = sorted(Path(output_dir).glob(f"{model_name}_*_heatmap_*_weights.npz"))
    if not paths:
        raise FileNotFoundError(f"No heatmap npz files for {model_name} in {output_dir}")
    selected_layer, method, evidence = select_layer_from_npz(paths, model_name=model_name, end_stm_token_only=True,
        group_b_start_frac=cfg.group_b_start_frac,
        group_b_end_frac=cfg.group_b_end_frac,
        npe_threshold=cfg.npe_threshold,)

    selection = LayerSelection(
        add_layer=selected_layer,
        token_position=-5,
        selection_method=method,
        n_layers_total=evidence["n_layers"],
        normalized_depth=round(selected_layer / evidence["n_layers"], 3),
        evidence=evidence,
    )

    selection_path = cfg.output_dir / "layer_selection.json"
    upsert_layer_selection(selection_path, model_name, selection)

    # Per-model manifest
    names = make_run_filenames(model_name, suffix="patching")
    write_manifest(
        manifest={
            "model":             model_name,
            "n_layers":          evidence["n_layers"],
            "datasets":          evidence["datasets_aggregated"],
            "selected_layer":    selected_layer,
            "selection_method":  method,
            "evidence":          evidence,
        },
        output_dir=cfg.output_dir,
        filename=names["manifest"],
    )

    logger.info(
        "Done %s: selected layer=%d (depth=%.2f, method=%s)",
        model_name, selected_layer, selected_layer / evidence["n_layers"], method,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 — Activation Patching")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (overrides config). Optional.")
    parser.add_argument("--log-file", type=Path, default=None)

    parser.add_argument("--from-file", type=str, default=None)
    args = parser.parse_args()

    configure_root_logger(log_file=args.log_file)
    cfg = load_config(args.config, PatchingConfig)
    assert isinstance(cfg, PatchingConfig)

    if args.from_file:
        out = reselect_for_model(args.from_file, Path("experimental_outputs/exp1_patching"), cfg)
        return

    models = (
        [m for m in cfg.models if m.name == args.model]
        if args.model is not None
        else cfg.models
    )
    if not models:
        logger.error("No models matched. Available: %s", [m.name for m in cfg.models])
        sys.exit(1)

    for model_spec in models:
        logger.info("\n%s\n=== %s ===\n%s", "=" * 60, model_spec.name, "=" * 60)
        run_for_model(model_spec, cfg.datasets, cfg)


if __name__ == "__main__":
    main()
