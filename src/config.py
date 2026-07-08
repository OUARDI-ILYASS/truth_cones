"""Pydantic configuration schemas.

Each pipeline stage loads a YAML file → validates it through these
models. This eliminates magic numbers in scripts and gives every
hyperparameter a documented, type-checked home.

Hyperparameter source: Wollschläger et al. (2025) Tab. 3, held
fixed across all six models (§4.1, Tab. hyperparams).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Shared building blocks ────────────────────────────────────────────────────


class ModelSpec(BaseModel):
    """A single model entry: HuggingFace ID + key in config.ini."""

    name: str = Field(..., description="Key in config.ini, e.g. 'qwen-2.5-7b-instruct'.")
    quantization: bool = False


class DatasetSpec(BaseModel):
    """A factual dataset spec (cities / animals / elements, §4.1)."""

    name: str
    csv_path: Path
    examples: List[List[str]] = Field(
        default_factory=list,
        description="Few-shot examples as [statement, label] pairs.",
    )


class LossWeights(BaseModel):
    """Composite loss weights (λ_abl, λ_add, λ_ret) from Wollschläger Tab. 3.

    Held fixed across all models and all k (§4.1, Tab. hyperparams).
    """

    ablation: float = 1.0    # λ_abl
    addition: float = 0.2    # λ_add
    retention: float = 1.0   # λ_ret


class TrainingHyperparams(BaseModel):
    """Optimization hyperparameters shared by TDO (§3.4) and TCO (§3.5).

    Defaults match Wollschläger Tab. 3 / Tab. hyperparams.
    """

    lr: float = 1e-2                  # §8.4: AdamW lr
    batch_size: int = 1               # physical batch
    effective_batch_size: int = 16    # gradient accumulation
    epochs: int = 1
    patience: int = 5                 # early stopping patience
    n_lr_reduce: int = 2              # max LR reductions (÷10 each)
    n_sample: int = 16                # MC directions per step (§3.5)
    weight_decay: float = 0.0         # Wollschläger Tab. 3: 0


# ── Pipeline-stage configs ────────────────────────────────────────────────────


class PatchingConfig(BaseModel):
    """Activation patching (§3.2) — layer localization."""

    models: List[ModelSpec]
    datasets: List[DatasetSpec]
    n_prompts_per_dataset: int = 50   # §3.2: "50 contrastive pairs per dataset"

    npe_threshold: float = 0.1        # §3.2: "mean NPE > 0.1"
    group_b_start_frac: float = 0.20
    group_b_end_frac: float = 0.90

    output_dir: Path = Path("experimental_outputs/exp1_patching")
    figures_dir: Path = Path("figures/exp1_patching")


class TDOConfig(BaseModel):
    """Truth Direction Optimization — Algorithm 1, §3.4."""

    models: List[ModelSpec]
    factual_datasets: List[DatasetSpec]
    ood_test: List[DatasetSpec]
    retain_dataset: Path
    train_ratio: float = 0.8          # 80/20 split (§4.1)
    cone_dim: int = 1                 # k = 1 for TDO
    training: TrainingHyperparams = Field(default_factory=TrainingHyperparams)
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    layer_selection_path: Path = Path("experimental_outputs/exp1_patching/layer_selection.json")
    output_dir: Path = Path("experimental_outputs/exp2_tdo")
    figures_dir: Path = Path("figures/exp2_tdo")


class TCOConfig(BaseModel):
    """Truth Cone Optimization — Algorithm 2, §3.5."""

    models: List[ModelSpec]
    factual_datasets: List[DatasetSpec]
    ood_test: List[DatasetSpec]
    retain_dataset: Path
    train_ratio: float = 0.8          # 80/20 split (§4.1)

    cone_dims: List[int] = Field(default_factory=lambda: [1, 2, 3, 4])  # §4.2, Tab. cone_asr

    training: TrainingHyperparams = Field(default_factory=TrainingHyperparams)
    loss_weights: LossWeights = Field(default_factory=LossWeights)

    mc_eval_samples: int = 32         # §3.6: "32 directions sampled from each cone"

    layer_selection_path: Path = Path("experimental_outputs/exp1_patching/layer_selection.json")
    tdo_summary_path: Path = Path("experimental_outputs/exp2_tdo/results_summary.json")
    output_dir: Path = Path("experimental_outputs/exp3_tco")
    figures_dir: Path = Path("figures/exp3_tco")


class AlignmentConfig(BaseModel):
    """DIM–cone geometric alignment — §3.6, §4.3 / RQ3."""

    cone_dims: List[int] = Field(default_factory=lambda: [1, 2, 3, 4])  # §4.3, Tab. orthogonality

    tdo_dir: Path = Path("experimental_outputs/exp2_tdo")
    tco_dir: Path = Path("experimental_outputs/exp3_tco")
    output_dir: Path = Path("experimental_outputs/exp4_alignment")
    figures_dir: Path = Path("figures/exp4_alignment")


class RetentionConfig(BaseModel):
    """KL retention evaluation — §3.6, §4.4 / RQ4."""

    models: List[ModelSpec]
    retain_dataset: Path

    n_retain_prompts: int = 100        # §3.6: "100 held-out Alpaca instructions"

    n_mc_kl: int = 32                 # §3.6: "sample 32 directions from the cone interior"

    logit_window: int = 30            # paper: last 30 token positions
    kl_threshold: float = 0.1        # Arditi et al. (2024)

    cone_dims: List[int] = Field(default_factory=lambda: [1, 2, 3, 4])  # §4.4, Tab. KL

    factual_filter_keywords: List[str] = Field(
        default_factory=lambda: [
            "capital", "country", "element", "animal", "species", "history",
            "year", "born", "died", "invented", "discovered", "located",
            "true or false", "fact", "trivia",
        ]
    )
    tco_dir: Path = Path("experimental_outputs/exp3_tco")
    output_dir: Path = Path("experimental_outputs/exp5_retention")
    figures_dir: Path = Path("figures/exp5_retention")

    @field_validator("kl_threshold")
    @classmethod
    def _kl_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("kl_threshold must be positive")
        return v


# ── Loader ────────────────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict:
    """Load a YAML config into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: Path, schema: type[BaseModel]) -> BaseModel:
    """Load + validate a YAML config against a Pydantic schema."""
    raw = load_yaml(path)
    return schema(**raw)