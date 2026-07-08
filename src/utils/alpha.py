"""DIM direction extraction and alpha calibration (§3.3).

θ_DIM = μ_+ − μ_−   where μ_+ and μ_− are mean activations at (l*, −5)
over the true and false training subsets. Following Marks & Tegmark
(2024) and Arditi et al. (2024).

θ_DIM is normalized to a unit vector for ablation and used at its
natural norm α := ||θ_DIM||_2 for addition, calibrating the steering
magnitude to the model's representational scale at l*. Per-model α
ranges from 4.07 (Llama-3.1-8B) to 106.56 (Gemma-2-9B) (Tab. master).

Also provides get_statement_end_position, which locates the "."
token — the end-of-statement punctuation at position i = −5 (§3.2
footnote).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
from tqdm import tqdm

from src.logging_setup import get_logger

logger = get_logger(__name__)


def get_statement_end_position(tokenizer, formatted_prompt: str) -> int:
    """Find the token index of the last "." in the prompt.

    This locates the end-of-statement punctuation — position i = −5
    in the paper's prompt structure (§3.2 footnote). Handles BPE
    fusion (e.g. "Italy." as a single token).
    """
    ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    for i in range(len(ids) - 1, -1, -1):
        if "." in tokenizer.decode([ids[i]]):
            return i
    raise ValueError(f"No '.' token found in prompt: {formatted_prompt!r}")


@torch.no_grad()
def _extract_activations_at_layer(
    model: Any,
    prompts: List[str],
    target_layer: int,
    *,
    batch_size: int = 1,
    effective_batch_size: int = 16,
) -> torch.Tensor:
    """Cache the statement-end activation x_i^(l*) for each prompt.

    Extracts the activation at (l*, i) where i is the end-of-statement
    punctuation position per prompt (found by get_statement_end_position).

    Returns:
        (N, d) tensor of cached activations.
    """
    chunks: list[torch.Tensor] = []
    n = len(prompts)
    pbar = tqdm(range(0, n, effective_batch_size), desc=f"DIM @L{target_layer}")

    for i in pbar:
        end = min(i + effective_batch_size, n)
        eff_batch = []

        for j in range(i, end, batch_size):
            sub_end = min(j + batch_size, end)
            batch_prompts = prompts[j:sub_end]

            # Compute per-sample statement-end positions BEFORE trace.
            # get_statement_end_position operates on strings, not
            # nnsight proxies — must run outside trace context.
            positions = [
                get_statement_end_position(model.tokenizer, fp)
                for fp in batch_prompts
            ]

            with torch.no_grad():
                with model.trace(batch_prompts):
                    layer_out = model.model.layers[target_layer].output[0]
                    # Extract x_i^(l*) at the statement-end position
                    act_slice = torch.stack(
                        [layer_out[k, positions[k], :] for k in range(len(batch_prompts))],
                        dim=0,
                    ).save()

            resolved = act_slice.value if hasattr(act_slice, "value") else act_slice
            eff_batch.append(resolved.detach().cpu())

        chunks.append(torch.cat(eff_batch, dim=0))

    return torch.cat(chunks, dim=0)


@torch.no_grad()
def dim_direction_from_activations(
    model: Any,
    prompts: List[str],
    labels: List[int],
    target_layer: int,
    *,
    batch_size: int = 1,
    effective_batch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute θ_DIM = μ_+ − μ_− at (l*, statement-end position).

    Following Marks & Tegmark (2024) and Arditi et al. (2024) (§3.3).

    Args:
        model:         frozen nnsight LanguageModel.
        prompts:       formatted factual prompts.
        labels:        1 = true, 0 = false (same length as prompts).
        target_layer:  l* from activation patching (§3.2).
        batch_size, effective_batch_size: chunking for memory.

    Returns:
        (θ_DIM_scaled, θ_DIM_unit) — both shape (d,).
        θ_DIM_scaled has norm α = ||θ_DIM||_2 for addition calibration.
        θ_DIM_unit is the unit direction for ablation and TDO warm-start.
    """
    if len(prompts) != len(labels):
        raise ValueError("prompts and labels must have equal length")

    logger.info("prompts: %s", prompts[1])
    acts = _extract_activations_at_layer(
        model, prompts, target_layer,
        batch_size=batch_size, effective_batch_size=effective_batch_size,
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)

    # θ_DIM = μ_+ − μ_− (§3.3)
    mu_pos = acts[label_tensor == 1].mean(dim=0)
    mu_neg = acts[label_tensor == 0].mean(dim=0)

    dim = mu_pos - mu_neg
    dim_unit = dim / (dim.norm() + 1e-8)

    # FLAG: Projection-rescaling step not in the paper.
    #
    # Paper (§3.3) says:
    #   θ_DIM = μ_+ − μ_−   (unnormalized, used as-is for α)
    #   θ_DIM / ||θ_DIM||    (unit, used for ablation + TDO init)
    #   α = ||θ_DIM||_2
    #
    # Code computes:
    #   diff_projection = (μ_+ − μ_−) · dim_unit   (scalar projection)
    #   dim_scaled = diff_projection * dim_unit      (re-scaled unit vector)
    #
    # Since dim_unit = dim / ||dim||, the projection (μ_+ − μ_−) · dim_unit
    # = ||dim||. So dim_scaled = ||dim|| * dim_unit = dim. The operation
    # is a no-op — it reconstructs the original dim vector. The extra
    # computation is harmless but confusing. Simplify to just return dim.
    diff_projection = (mu_pos - mu_neg) @ dim_unit
    dim_scaled = diff_projection * dim_unit

    logger.info("DIM extracted ‖DIM‖=%.4f ‖DIM_scaled‖=%.4f layer=%d N=%d",
                dim.norm().item(), dim_scaled.norm().item(), target_layer, len(prompts))
    return dim_scaled, dim_unit


def calibrate_alpha(dim_vector: torch.Tensor) -> float:
    """α = ||θ_DIM||_2 (§3.3, following Wollschläger et al. 2025).

    Grounds the addition magnitude in the model's representational
    scale at l*, removing one degree of freedom.
    """
    return float(dim_vector.norm().item())