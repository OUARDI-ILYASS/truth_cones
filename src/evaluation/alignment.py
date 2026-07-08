"""DIM–cone geometric alignment — §3.6 / §4.3 (RQ3).

Post-hoc analysis on saved tensors (no model loading needed). For each
model and cone dimension k ∈ {1, 2, 3, 4}, computes |cos(θ_DIM, v_i)|
for every basis vector v_i.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from src.logging_setup import get_logger

logger = get_logger(__name__)

ALIGNMENT_MEANINGFUL = 0.1


def compute_cosine_similarities(
    dim_vector: torch.Tensor,                 # (d,) θ_DIM, unit-norm
    cone_bases: Dict[int, torch.Tensor],      # {k: (k, d)} learned cone bases
    cone_dims: List[int],                     # e.g. [1, 2, 3, 4]
) -> Dict[int, List[float]]:
    """Compute |cos(θ_DIM, v_i)| for each cone dimension and basis vector.

    Both DIM and basis vectors are renormalized defensively before the dot
    product to guard against numerical drift in saved tensors.

    Returns:
        {k: [cos(θ_DIM, v_1), ..., cos(θ_DIM, v_k)]} for each k.
    """
    if dim_vector is None:
        return {d: [float("nan")] * d for d in cone_dims}

    dim_unit = dim_vector / (dim_vector.norm() + 1e-8)
    out: Dict[int, List[float]] = {}

    for d in cone_dims:
        if d not in cone_bases:
            out[d] = [float("nan")] * d
            continue
        basis = cone_bases[d]                                 # (k, d)
        basis_unit = basis / (basis.norm(dim=-1, keepdim=True) + 1e-8)
        # cos(θ_DIM, v_i) = v̂_i · θ̂_DIM
        out[d] = (basis_unit @ dim_unit).tolist()

    return out


def summarize_alignment(cosines: Dict[int, List[float]]) -> Dict[str, float | str]:
    """Reduce raw cosines to summary stats for results_summary.json.

    Uses the largest available k as the reference dimension (paper uses
    k=4 as reference in Tab. orthogonality). Reports:
      - v1 alignment:  |cos(θ_DIM, v_1)| at reference k
      - max other:     max |cos(θ_DIM, v_i)| for i > 1 (should be < 0.1)
      - interpretation flag

    """
    if not cosines:
        return {
            "v1_dim_alignment": float("nan"),
            "max_other_alignment": float("nan"),
            "interpretation": "no_data",
        }

    ref_dim = max(cosines.keys())
    cos_list = cosines[ref_dim]
    if not cos_list:
        return {
            "v1_dim_alignment": float("nan"),
            "max_other_alignment": float("nan"),
            "interpretation": "no_data",
        }

    v1_sim = abs(cos_list[0])
    rest = [abs(c) for c in cos_list[1:]] if len(cos_list) > 1 else []
    max_rest = max(rest) if rest else float("nan")

    if v1_sim >= ALIGNMENT_MEANINGFUL and (not rest or max_rest < ALIGNMENT_MEANINGFUL):
        interpretation = "only_v1_aligns"
    elif v1_sim < ALIGNMENT_MEANINGFUL:
        interpretation = "no_dim_alignment"
    else:
        interpretation = "multiple_axes_align"

    return {
        "reference_cone_dim":   ref_dim,
        "v1_dim_alignment":     float(v1_sim),
        "max_other_alignment":  float(max_rest) if rest else float("nan"),
        "interpretation":       interpretation,
    }