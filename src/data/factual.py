"""Factual dataset loaders.

Three sources: cities, animals, elements. Each CSV has columns
``statement, label`` where label ∈ {0, 1}.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DatasetSpec
from src.logging_setup import get_logger

logger = get_logger(__name__)


def load_factual_datasets(specs: List[DatasetSpec]) -> pd.DataFrame:
    """Load and concatenate factual CSVs.

    Each loaded DataFrame is tagged with a ``source`` column equal to the
    dataset name so that downstream code can group by origin if needed.
    """
    frames: list[pd.DataFrame] = []
    for spec in specs:
        if not spec.csv_path.exists():
            raise FileNotFoundError(f"Factual dataset not found: {spec.csv_path}")
        df = pd.read_csv(spec.csv_path)
        if not {"statement", "label"}.issubset(df.columns):
            raise ValueError(
                f"{spec.csv_path} must have columns ['statement', 'label']."
            )
        df["source"] = spec.name
        df["examples"] = [spec.examples] * len(df)
        frames.append(df)
        logger.info("Loaded %s: %d rows", spec.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined factual dataset: %d rows total", len(combined))
    return combined


def prepare_factual_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-source train/eval split.

    The split is performed *per source* before concatenation, ensuring every
    source dataset is proportionally represented in both train and eval.
    Statement-level overlap is impossible because the split is on row indices.
    """
    train_frames, val_frames = [], []
    for source, sub in df.groupby("source"):
        tr, va = train_test_split(sub, train_size=train_ratio, random_state=seed)
        train_frames.append(tr)
        val_frames.append(va)
        logger.info("  %-12s train=%d  val=%d", source, len(tr), len(va))

    train_df = pd.concat(train_frames).reset_index(drop=True)
    val_df = pd.concat(val_frames).reset_index(drop=True)
    logger.info("Total split: train=%d  val=%d", len(train_df), len(val_df))
    return train_df, val_df
