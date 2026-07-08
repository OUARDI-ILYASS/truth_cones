"""TruthCone — learnable k-dimensional polyhedral cone in the residual stream.

Implements the orthonormal basis V = [v_1, ..., v_k] from §3.5 (TCO).
The cone is Cone(V) = {Σ λ_i v_i : λ_i ≥ 0} \ {0}, i.e. one orthant of
span(V), not the full subspace. For k=1 this recovers TDO (§3.4): a single
truth direction warm-started from DIM.

Owns a ParameterList of basis vectors and exposes:
  - Training:   sample_directions (MC interior sampling), orthogonalize
  - Inference:  ablate, add (via Strategy pattern)

Used by both TDO training (k=1) and TCO training/evaluation (k≥1).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.interventions.ablation import AblationIntervention
from src.interventions.addition import AdditionIntervention
from src.interventions.projection import projection_einops


class TruthCone(nn.Module):
    """Learnable k-dimensional orthonormal cone basis in the residual stream.

    Maintains k unit vectors in ℝ^d, kept orthonormal via modified
    Gram–Schmidt after every gradient step (Algorithm 2, line 12).

    The polyhedral cone Cone(V) = {Σ λ_i v_i : λ_i ≥ 0} \ {0} is the
    set of non-negative combinations of the basis. Every direction
    r ∈ Cone(V) must satisfy the two causal axioms (Definition 2.1):
    monotonic scaling and surgical factuality ablation.

    Args:
        module:        the model body (e.g. model.model) — used to resolve
                       device/dtype and to enumerate layers during ablation.
        hidden_size:   residual-stream dimension d.
        n_vectors:     cone dimension k (1 for TDO, ≥2 for TCO).
        init_vectors:  optional (k, d) initialization (DIM warm-start for v_1,
                       augmented random vectors for v_2..v_k per §3.5).
        trainable:     whether basis vectors have requires_grad=True.
                       False for post-training evaluation.
    """

    def __init__(
        self,
        module: Any,
        hidden_size: int,
        n_vectors: int,
        init_vectors: Optional[torch.Tensor] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if n_vectors < 1:
            raise ValueError(f"n_vectors must be ≥ 1, got {n_vectors}")

        self.module = module
        self.hidden_size = hidden_size
        self.n_vectors = n_vectors

        # Basis V = [v_1, ..., v_k] as a ParameterList (§3.5).
        self.fn_vectors = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(hidden_size, dtype=torch.float32),
                    requires_grad=trainable,
                )
                for _ in range(n_vectors)
            ]
        )

        if init_vectors is not None:
            for i, v in enumerate(init_vectors):
                if i < n_vectors:
                    normed = v / (v.norm() + 1e-6)
                    self.fn_vectors[i].data = normed.detach().clone().float()

        # Reusable intervention strategies (no per-call allocation).
        self._ablation = AblationIntervention()
        self._addition = AdditionIntervention()

        self.orthogonalize()

    # ── Geometric primitives ──────────────────────────────────────────────────

    def get_basis_matrix(self) -> torch.Tensor:
        """Return the (k, d) basis matrix V."""
        return torch.stack(list(self.fn_vectors), dim=0)

    @torch.no_grad()
    def orthogonalize(self) -> None:
        """Modified Gram–Schmidt (Algorithm 2, line 12).

        Re-projects V onto the Stiefel manifold after each gradient step,
        restoring V^T V = I_k. Numerically more stable than classical
        Gram–Schmidt: reorthogonalizes against each previously computed
        vector one at a time.
        """
        for i in range(len(self.fn_vectors)):
            v = self.fn_vectors[i].data
            for j in range(i):
                u = self.fn_vectors[j].data
                v = v - projection_einops(v.unsqueeze(0), u).squeeze(0)
            self.fn_vectors[i].data = v / (v.norm() + 1e-6)

    @torch.no_grad()
    def normalize(self) -> None:
        """Renormalize each basis vector to unit length (no orthogonalization).

        Used by TDO (k=1) where orthogonalization is trivial.
        """
        for v in self.fn_vectors:
            v.data /= v.data.norm() + 1e-6

    def sample_directions(self, n: int) -> torch.Tensor:
        """Sample n unit directions from the cone interior (Algorithm 2, Sample).

        Draws c ~ N(0, I_k), folds to the non-negative orthant via abs(),
        normalizes to the positive orthant of S^{k-1}, then maps to ℝ^d
        via r = V·s. This yields uniform coverage of the cone and avoids
        boundary bias from convex-combination sampling (§3.5).

        The loss is evaluated on these samples (L_sample in Algorithm 2,
        line 8) alongside the basis vectors (L_basis, line 9) to prevent
        basis collapse.
        """
        basis = self.get_basis_matrix()  # (k, d)
        k = basis.shape[0]
        # Gaussian folding: c ~ N(0, I_k), s = |c| / ||c||
        w = torch.randn(n, k, device=basis.device, dtype=basis.dtype).abs()
        w = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
        # Map to residual stream: r = V·s
        dirs = w @ basis  # (n, d)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        return dirs

    # FLAG: No paper equivalent. Not in Algorithm 1 or 2.
    # Appears to project an arbitrary weight vector through the basis.
    # If used for evaluation (e.g. sweeping specific cone directions),
    # document the use case or remove.
    def transform(self, weights: torch.Tensor) -> torch.Tensor:
        """Project a (k,) weight vector to a single unit direction in ℝ^d.

        Not part of the published method. Kept for notebook convenience.
        """
        basis = self.get_basis_matrix()
        direction = torch.matmul(weights.to(basis.device), basis)
        return direction / (direction.norm() + 1e-6)

    # ── Intervention dispatch (Strategy pattern) ──────────────────────────────

    def ablate(self, direction: torch.Tensor) -> None:
        """Directional ablation: x̃ = x − r̂ r̂ᵀx at all layers and positions.

        Implements f_ablate(r) from §3.1, Eq. 1. Call inside model.trace.
        """
        self._ablation.apply(self.module, direction)

    def add(self, direction: torch.Tensor, alpha: float, layer_idx: int) -> None:
        """Activation addition: x̃ = x + α·r at layer l* (§3.1, Eq. 2).

        Call inside model.trace. alpha calibrated to ||θ_DIM||_2 (§3.3).
        """
        self._addition.apply(self.module, direction, alpha=alpha, layer_idx=layer_idx)

    def __call__(self, direction: torch.Tensor) -> None:  # type: ignore[override]
        """Shorthand for self.ablate(direction). Backward compat only."""
        self.ablate(direction)