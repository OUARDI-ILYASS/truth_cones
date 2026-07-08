"""Merge 1D cone data from Exp2 into Exp3 results_summary.json."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("merge_1d_cones")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge 1D cone weights + metadata from Exp2 into Exp3."
    )
    parser.add_argument(
        "--exp2-dir",
        type=Path,
        required=True,
        help="Source dir with Exp2 results_summary.json and 1D weights.",
    )
    parser.add_argument(
        "--exp3-dir",
        type=Path,
        required=True,
        help="Target Exp3 dir with results_summary.json to patch.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        logger.error("Missing JSON: %s", path)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote updated JSON: %s", path)


def copy_weights(src_dir: Path, dst_dir: Path, weights_file: str) -> bool:
    src = src_dir / weights_file
    dst = dst_dir / weights_file
    if not src.is_file():
        logger.warning("Weights file missing in Exp2: %s", src)
        return False
    if dst.exists():
        logger.info("Weights already present in Exp3, skipping copy: %s", dst.name)
        return True
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Copied weights: %s -> %s", src.name, dst_dir)
    return True


def merge(exp2_dir: Path, exp3_dir: Path) -> None:
    exp2_summary_path = exp2_dir / "results_summary.json"
    exp3_summary_path = exp3_dir / "results_summary.json"

    exp2 = load_json(exp2_summary_path)
    exp3 = load_json(exp3_summary_path)

    exp2_models = exp2.get("models", {})
    exp3_models = exp3.setdefault("models", {})

    if not exp2_models:
        logger.error("No models in Exp2 summary. Abort.")
        sys.exit(1)

    updated, skipped = 0, 0

    for model_name, exp2_entry in exp2_models.items():
        weights_file = exp2_entry.get("weights_file")
        basis_key = exp2_entry.get("cone_vector_key")

        if not weights_file or not basis_key:
            logger.warning(
                "Model %s missing weights_file or cone_vector_key. Skip.",
                model_name,
            )
            skipped += 1
            continue

        if model_name not in exp3_models:
            logger.warning("Model %s absent from Exp3 summary. Skip.", model_name)
            skipped += 1
            continue

        if not copy_weights(exp2_dir, exp3_dir, weights_file):
            skipped += 1
            continue

        exp3_entry = exp3_models[model_name]
        cones = exp3_entry.setdefault("cones", {})

        if "1" in cones:
            logger.info("Model %s already has cones['1']. Overwriting.", model_name)

        cones["1"] = {
            "weights_file": weights_file,
            "basis_key": basis_key,
        }

        dims_evaluated = exp3_entry.setdefault("cone_dims_evaluated", [])
        if 1 not in dims_evaluated:
            dims_evaluated.insert(0, 1)
            dims_evaluated.sort()

        logger.info("Patched Exp3 metadata: %s cones['1']", model_name)
        updated += 1

    top_dims = exp3.setdefault("cone_dims", [])
    if 1 not in top_dims:
        top_dims.insert(0, 1)
        top_dims.sort()

    save_json(exp3, exp3_summary_path)
    logger.info("Done. Updated: %d, Skipped: %d", updated, skipped)


def main() -> None:
    args = parse_args()
    if not args.exp2_dir.is_dir():
        logger.error("Exp2 dir not found: %s", args.exp2_dir)
        sys.exit(1)
    if not args.exp3_dir.is_dir():
        logger.error("Exp3 dir not found: %s", args.exp3_dir)
        sys.exit(1)
    merge(args.exp2_dir, args.exp3_dir)


if __name__ == "__main__":
    main()