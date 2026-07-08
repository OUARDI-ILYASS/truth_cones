"""Activation addition at the causally selected layer l* (§3.1, Eq. 2).

Implements f_add(r, l*, α):  x̃_i^(l*) = x_i^(l*) + α · r   ∀ i

Injects a scaled truth direction into the residual stream at a single
layer, testing causal sufficiency: if a false statement flips to "True"
under addition, r is sufficient to override the original verdict
(Definition 2.1, axiom 1 — monotonic scaling).

Unlike ablation (all layers, all positions), addition operates at one
layer only (§3.1, following Arditi et al. 2024).
"""

from __future__ import annotations

from typing import Any

import torch

from src.interventions.base import Intervention


class AdditionIntervention(Intervention):
    """Add α · r̂ to the residual stream at layer l*.

    The scaling coefficient α is calibrated to ||θ_DIM||_2 (§3.3),
    matching the model's own representational scale at l*. Per-model
    α ranges from 4.07 (Llama-3.1-8B) to 106.56 (Gemma-2-9B).
    """

    name = "addition"

    def apply(
        self,
        module: Any,
        direction: torch.Tensor,
        *,
        alpha: float,
        layer_idx: int,
        **kwargs: Any,
    ) -> None:
        """Add α · r̂ to the input of module.layers[layer_idx].

        Args:
            module:    the model body (e.g. model.model).
            direction: (d,) truth direction r. Normalized internally.
            alpha:     steering magnitude, set to ||θ_DIM||_2 (§3.3).
            layer_idx: target layer index l* from activation patching (§3.2).
        """
        if alpha is None or layer_idx is None:
            raise ValueError("AdditionIntervention requires `alpha` and `layer_idx` kwargs.")

        target_layer = module.layers[layer_idx]
        # Cast to target layer's device/dtype — needed for multi-GPU
        # sharding (device_map="auto") where layers live on different GPUs.
        # Skip quantized int weights; need a float param for dtype reference.
        target_param = next(
            p for p in target_layer.parameters() if p.is_floating_point()
        )
        d = direction.to(device=target_param.device, dtype=target_param.dtype)
        d = d / (d.norm() + 1e-6)

        # applying addition at ALL token positions. Code adds to the full layer
        # input tensor (shape: batch, seq_len, d), which broadcasts α·d across
        # every position.
        activation = target_layer.input[0].clone()
        target_layer.input[0] = activation + (alpha * d)