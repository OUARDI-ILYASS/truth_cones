"""Abstract Intervention base class — the Strategy interface.

The paper defines two intervention operations on the residual stream
(§3.1): directional ablation (Eq. 1) and activation addition (Eq. 2).
Both share the same call signature — apply(module, direction, ...) —
so training (Algorithms 1–2) and evaluation loops can dispatch by
intervention type without branching on string flags.

Concrete subclasses:
  AblationIntervention  →  f_ablate(r):  x̃ = x − r̂r̂ᵀx   ∀ l, i
  AdditionIntervention  →  f_add(r,l*,α): x̃ = x + α·r    at l*
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class Intervention(ABC):
    """Abstract intervention applied inside an nnsight trace context.

    Subclasses mutate the model's forward computation graph via nnsight's
    layer.input / layer.output proxy assignments, injecting the causal
    manipulation specified by one of the two axioms in Definition 2.1.
    """

    name: str = "intervention"

    @abstractmethod
    def apply(
        self,
        module: Any,           # model body (e.g. model.model) with .layers attribute
        direction: torch.Tensor,  # (d,) truth direction r̂ or cone sample
        **kwargs: Any,
    ) -> None:
        """Apply the intervention. Called inside ``with model.trace(...)``."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"