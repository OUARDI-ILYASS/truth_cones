"""Tensor projection utility.

The single primitive that all directional interventions depend on:
project an activation tensor onto a unit-norm direction.
"""
"""Orthogonal projection onto a unit direction.

Implements proj_r̂(x) = (r̂ᵀ x) r̂  from §3.1 Eq. 1. This is the inner
primitive of directional ablation:

    x̃ = x − proj_r̂(x)        (ablation: remove truth component)

and is also used by modified Gram–Schmidt (Algorithm 2, line 12) to
keep the cone basis orthonormal.
"""

from __future__ import annotations

import torch


# FLAG: Name says "einops" but uses torch.einsum, not the einops library.
# einops (https://github.com/arogozhnikov/einops) is a separate package
# with a different API (rearrange/reduce/repeat). This function uses
# Einstein summation via torch.einsum. Consider renaming to
# projection_einsum or just projection.
def projection_einops(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project activation onto a unit direction: proj_r̂(x) = (r̂ᵀ x) r̂.

    This is the rank-1 projection r̂ r̂ᵀ x (§3.1). Ablation subtracts
    this from x to erase the truth component; Gram–Schmidt subtracts it
    to enforce orthogonality between basis vectors.

    Assumes direction is unit-norm. If not, the result is scaled by
    ||direction||² — callers (TruthCone.orthogonalize) must ensure
    normalization.

    Args:
        activation: (batch, ..., d) — residual-stream activations.
                    Intermediate axes cover token positions or other
                    broadcast dimensions.
        direction:  (d,) — unit-norm direction r̂ ∈ S^{d-1}.

    Returns:
        Same shape as activation. The component of each activation
        vector that lies along direction.

    Implementation — torch.einsum notation
    ───────────────────────────────────────
    einsum implements Einstein summation convention: repeated indices
    are summed over, free indices are kept.

    Step 1: dot = einsum("b...d, d -> b...", activation, direction)
      Subscripts:  b = batch,  ... = any intermediate axes,  d = hidden dim.
      "b...d, d -> b..."  means: for each (b, ...) position, contract
      (sum) over d.  This computes the scalar dot product r̂ᵀ x per
      position, yielding shape (batch, ...).

      Equivalent loop (slow):
        for each b, each intermediate index *:
          dot[b, *] = Σ_d  activation[b, *, d] * direction[d]

    Step 2: out = einsum("b..., d -> b...d", dot, direction)
      "b..., d -> b...d"  means: broadcast dot over a new d axis and
      multiply element-wise with direction. No index is repeated on the
      same input, so no contraction — this is an outer product per
      position. Result shape: (batch, ..., d).

      Equivalent: out[b, *, d] = dot[b, *] * direction[d]

    Together: out = (r̂ᵀ x) r̂, the orthogonal projection.
    """
    # Step 1: scalar projection  (r̂ᵀ x)  per batch element
    dot = torch.einsum("b...d,d->b...", activation, direction)
    # Step 2: scale direction by dot product  →  (r̂ᵀ x) r̂
    return torch.einsum("b...,d->b...d", dot, direction)