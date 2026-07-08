"""Retain (Alpaca) dataset loading and filtering.

The retain dataset is used as the regularization signal for ``L_ret``:
ablating the truth direction on these prompts must not deviate from the
clean output distribution.

For Experiment 5 (retention) we additionally filter out instructions that
could invoke factual recall, so KL divergence measures purely *collateral*
damage rather than the intended truth-direction effect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logging_setup import get_logger

logger = get_logger(__name__)


def load_retain_dataset(path: Path) -> pd.DataFrame:
    """Load Alpaca-style data with an ``instruction`` column."""
    if not path.exists():
        raise FileNotFoundError(f"Retain dataset not found: {path}")

    if path.suffix == ".json":
        with open(path) as f:
            df = pd.DataFrame(json.load(f))
    elif path.suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported retain dataset format: {path.suffix}")

    if "instruction" not in df.columns:
        raise ValueError(f"{path} must have an 'instruction' column.")

    logger.info("Loaded retain dataset: %d instructions", len(df))
    return df


def filter_retain_dataset(
    df: pd.DataFrame,
    factual_keywords: List[str],
) -> pd.DataFrame:
    """Drop instructions containing any factual-recall keyword.

    Used by Experiment 5 so that the KL divergence measurement isolates
    *collateral* damage from the intended truth-direction effect.
    """
    lowered = [kw.lower() for kw in factual_keywords]
    mask = df["instruction"].str.lower().apply(
        lambda inst: not any(kw in inst for kw in lowered)
    )
    filtered = df[mask].copy()
    logger.info(
        "Retain filter: kept %d / %d (%.1f%%)",
        len(filtered), len(df), 100.0 * len(filtered) / max(len(df), 1),
    )
    return filtered


def split_retain(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standard train/eval split for retain instructions."""
    tr, va = train_test_split(df, train_size=train_ratio, random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)
