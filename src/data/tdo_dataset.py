"""TDODataset — dataset of triplets used by TDO and TCO.

Each item is ``(p_true, p_false, p_retain, t_true, t_false)``:
  - p_true:   formatted prompt with a TRUE statement     (ablation target)
  - p_false:  formatted prompt with a FALSE statement    (addition target)
  - p_retain: general Alpaca instruction                  (KL retain target)
  - t_true / t_false: token IDs for the binary forced-choice labels.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import pandas as pd
from torch.utils.data import Dataset

from src.logging_setup import get_logger

logger = get_logger(__name__)


class TDODataset(Dataset):
    """Triplet dataset for Truth Direction / Cone Optimization."""

    def __init__(
        self,
        factual_df: pd.DataFrame,
        retain_df: pd.DataFrame,
        tokenizer,
        apply_few_shot_fn: Callable[[List[str], List[List[Tuple[str, str]]]], List[str]],
        apply_retain_template_fn: Callable[[List[str], List[str]], List[str]],
    ) -> None:
        
        true_df = factual_df[factual_df["label"] == 1]
        false_df = factual_df[factual_df["label"] == 0]

        true_statements = apply_few_shot_fn(true_df["statement"].tolist(), true_df["examples"].tolist())
        false_statements = apply_few_shot_fn(false_df["statement"].tolist(), false_df["examples"].tolist())
        retain_instructions = retain_df["instruction"].tolist()

        random.shuffle(true_statements)
        random.shuffle(false_statements)
        random.shuffle(retain_instructions)

        # Balance the three pools: take min size to guarantee equal triplets
        n = min(len(true_statements), len(false_statements), len(retain_instructions))
        raw_true = true_statements[:n]
        raw_false = false_statements[:n]
        raw_retain = retain_instructions[:n]

        self.p_true: List[str] = raw_true
        self.p_false: List[str] = raw_false
        self.p_retain: List[str] = apply_retain_template_fn(tokenizer, raw_retain)

        self.t_true_id: int = tokenizer.encode("True", add_special_tokens=False)[-1]
        self.t_false_id: int = tokenizer.encode("False", add_special_tokens=False)[-1]

        logger.info(
            "TDODataset built: %d triplets  (t_true=%d, t_false=%d)",
            n, self.t_true_id, self.t_false_id,
        )

    def __len__(self) -> int:
        return len(self.p_true)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "p_true":   self.p_true[idx],
            "p_false":  self.p_false[idx],
            "p_retain": self.p_retain[idx],
            "t_true":   self.t_true_id,
            "t_false":  self.t_false_id,
        }


def tdo_collate(batch: List[Dict]) -> Dict:
    """Custom collate that keeps prompts as lists of strings.

    nnsight handles tokenization per-sample inside ``model.trace``, so we
    don't pre-tokenize. This avoids padding issues with variable-length
    prompts and keeps the collate function trivial.
    """
    return {
        "p_true":   [item["p_true"]   for item in batch],
        "p_false":  [item["p_false"]  for item in batch],
        "p_retain": [item["p_retain"] for item in batch],
        "t_true":   [item["t_true"]   for item in batch],
        "t_false":  [item["t_false"]  for item in batch],
    }
