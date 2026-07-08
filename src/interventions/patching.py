"""Denoising activation patching for layer localization (§3.2).

Sweeps over (layer, token position) pairs, restoring one clean-run
activation at a time into the corrupted run, and measures the
Normalized Patching Effect (NPE) on the logit difference
M(p) = logit(p)_{t_true} − logit(p)_{t_false}  (§3.2, Eq. 3).

The output NPE matrix is used to identify three groups of causally
implicated hidden states (following Marks & Tegmark 2024):
  (a) entity token — encodes the factual association
  (b) end-of-statement punctuation in mid layers — propositional judgment
  (c) colon token in mid-to-late layers — prediction readout
l* is selected as the deepest layer in group (b) with mean NPE > 0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from tqdm import trange

from src.logging_setup import get_logger
from src.utils.npe import normalized_patching_effect

logger = get_logger(__name__)


@dataclass
class PatchingResult:
    """Output of a single (model, dataset, prompt-pair) patching sweep."""

    npe_matrix: np.ndarray              # (L, n_tokens) NPE values per (layer, position)
    token_labels: List[str]             # decoded token + absolute index for the x-axis
    clean_diff: float                   # M(p_clean): logit difference on clean prompt
    corrupt_diff: float                 # M(p_corrupt): logit difference on corrupted prompt
    sweep_start_idx: int                # absolute token index where sweep begins
    seq_len: int                        # total prompt length in tokens


class ActivationPatching:
    """Single denoising activation patching sweep (§3.2).

    Implements the three-run protocol from Meng et al. (2023) /
    Heimersheim & Janiak (2024), adapted for True/False logit
    difference as in Marks & Tegmark (2024):

      1. Clean run on p_clean  → cache all residual-stream activations
      2. Corrupted run on p_corrupt (STR-perturbed) → baseline logits
      3. For each (l, i): run p_corrupt but restore x_i^(l) from clean
         cache → compute NPE(l, i)

    Usage:
        sweeper = ActivationPatching(model)
        result  = sweeper.run(clean_prompt, corrupted_prompt,
                              true_token_id, false_token_id, sweep_start_idx)
    """

    def __init__(self, model) -> None:
        self.model = model
        self.layers = model.model.layers
        self.n_layers: int = model.config.num_hidden_layers

    @torch.no_grad()
    def _cache_run(self, prompt: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass, caching per-layer residual streams and final logits.

        Returns:
            activations: list of L tensors, each (1, seq_len, d).
                         These are the residual-stream states x_i^(l).
            logits:      (1, seq_len, vocab_size) final-layer logits.
        """
        activations: list[torch.Tensor] = []
        with self.model.trace(prompt):
            for layer in self.layers:
                # layer.output is (hidden_states, ...) — [0] is the
                # residual stream x_i^(l) ∈ ℝ^d at this layer.
                activations.append(layer.output[0].save())
            logits = self.model.output.logits.save()
        return activations, logits

    def run(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        true_token_id: int,
        false_token_id: int,
        sweep_start_idx: int,
    ) -> PatchingResult:
        """Run denoising patching sweep over the residual stream.

        For each (layer l, position i), restores the clean activation
        x_i^(l)(p_clean) into the corrupted forward pass and measures
        NPE(l, i) = (M_patched − M_corrupt) / (M_clean − M_corrupt).

        Args:
            clean_prompt:      p_clean — true statement the model answers correctly.
            corrupted_prompt:  p_corrupt — STR-perturbed prompt that flips the
                               ground-truth label while preserving syntax and
                               token count (§3.2).
            true_token_id:     vocab id for "True"  (t_true).
            false_token_id:    vocab id for "False" (t_false).
            sweep_start_idx:   first token position to patch. Positions before
                               this (e.g. few-shot prefix) are skipped.

        Returns:
            PatchingResult with the (L × n_tokens) NPE matrix.
        """
        clean_toks = self.model.tokenizer(clean_prompt).input_ids
        corr_toks = self.model.tokenizer(corrupted_prompt).input_ids
        if len(clean_toks) != len(corr_toks):
            raise ValueError(
                f"Token length mismatch: clean={len(clean_toks)} corrupted={len(corr_toks)}. "
                "STR requires equal-length prompts."
            )
        seq_len = len(clean_toks)

        # ── Step 1: cache clean & corrupted forward passes ────────────────────
        logger.info("Caching clean run...")
        clean_acts, clean_logits = self._cache_run(clean_prompt)
        logger.info("Caching corrupted run...")
        _, corrupt_logits = self._cache_run(corrupted_prompt)

        # Logit difference M(p) = logit(t_true) − logit(t_false) at last position
        clean_diff = (
            clean_logits[0, -1, true_token_id] - clean_logits[0, -1, false_token_id]
        ).item()
        corrupt_diff = (
            corrupt_logits[0, -1, true_token_id] - corrupt_logits[0, -1, false_token_id]
        ).item()
        logger.info("Baselines  M_clean=%.4f  M_corrupt=%.4f  gap=%.4f",
                    clean_diff, corrupt_diff, clean_diff - corrupt_diff)

        # ── Step 2: sweep (layer × position) ──────────────────────────────────
        n_sweep_tokens = seq_len - sweep_start_idx
        npe_matrix = np.zeros((self.n_layers, n_sweep_tokens), dtype=np.float32)

        # FLAG: n_tokens computed but unused — n_sweep_tokens duplicates it.
        # Remove one.

        for layer_idx in trange(self.n_layers, desc="Sweeping layers"):
            # Batch all token positions for this layer into one forward pass:
            # one copy of p_corrupt per position, each with a different
            # single-position restoration from the clean cache.
            batched_input = [corrupted_prompt] * n_sweep_tokens

            with torch.no_grad():
                with self.model.trace(batched_input):
                    layer_output = self.model.model.layers[layer_idx].output[0]
                    # layer_output: (n_sweep_tokens, seq_len, d)

                    # Restore clean activation at exactly one position per
                    # batch element — the corrupted-with-restoration run.
                    for sweep_idx, token_idx in enumerate(range(sweep_start_idx, seq_len)):
                        clean_act = clean_acts[layer_idx][0, token_idx, :]  # (d,)
                        layer_output[sweep_idx, token_idx, :] = clean_act

                    # M_patched for each position
                    logits = self.model.output.logits  # (n_sweep_tokens, seq_len, vocab)
                    raw_diffs = (
                        logits[:, -1, true_token_id] - logits[:, -1, false_token_id]
                    ).save()  # (n_sweep_tokens,)

            # NPE(l, i) = (M_patched − M_corrupt) / (M_clean − M_corrupt)
            for sweep_idx in range(n_sweep_tokens):
                npe_matrix[layer_idx, sweep_idx] = normalized_patching_effect(
                    patched=raw_diffs[sweep_idx].item(),
                    clean=clean_diff,
                    corrupted=corrupt_diff,
                )

        # ── Step 3: build token labels ────────────────────────────────────────
        token_labels = [
            f"{self.model.tokenizer.decode(corr_toks[idx])} ({idx})"
            for idx in range(sweep_start_idx, seq_len)
        ]

        return PatchingResult(
            npe_matrix=npe_matrix,
            token_labels=token_labels,
            clean_diff=clean_diff,
            corrupt_diff=corrupt_diff,
            sweep_start_idx=sweep_start_idx,
            seq_len=seq_len,
        )