"""Directional ablation across every layer (§3.1, Eq. 1).

Implements f_ablate(r):  x̃_i^(l) = x_i^(l) − r̂ r̂ᵀ x_i^(l)   ∀ l, i

This erases the component of the residual stream along r̂ at every
layer and token position, testing causal necessity: if the model flips
from "True" to "False" after ablation, r̂ was necessary for the truth
verdict (Definition 2.1, axiom 2-i).  Functional specificity (axiom
2-ii) is checked separately via KL retention on retain prompts.

Following Arditi et al. (2024) and Wollschläger et al. (2025), ablation
is applied across all layers and all token positions (§3.1).
"""

from __future__ import annotations

from typing import Any

import torch

from src.interventions.base import Intervention
from src.interventions.projection import projection_einops


class AblationIntervention(Intervention):
    """Project direction out of every layer's residual stream.

    The paper's Eq. 1 specifies ablation on x_i^(l) ∀ l, i — the
    residual stream between layers. This implementation is stricter:
    it ablates at three sites per layer (layer input, attention output,
    MLP output) to prevent the direction from being reintroduced by
    sub-layer computations within a single block.

    Handles model sharding (device_map="auto") by caching per-device
    copies of the direction vector.
    """

    # FLAG: The code ablates layer input + attn output + MLP output
    # (three projections per layer). The paper's Eq. 1 specifies
    # x̃_i^(l) = x_i^(l) − r̂r̂ᵀx_i^(l) ∀ l, i, which reads as
    # projecting the residual stream *between* layers. Ablating
    # sub-layer outputs is more aggressive — it also prevents attn
    # and MLP from reintroducing the direction within a layer.
    # This is arguably better but doesn't match the equation. The
    # paper should either document this or the code should match
    # the equation (layer input only). Wollschläger et al. (2025)
    # also apply ablation "across all token positions" without
    # specifying sub-layer granularity.

    name = "ablation"

    def apply(
        self,
        module: Any,
        direction: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Ablate direction from every layer in module.layers.

        Args:
            module:    the model body (e.g. model.model) whose .layers
                       attribute iterates over transformer blocks.
            direction: (d,) truth direction r̂. Normalized internally.
        """
        # Normalize once on source device. Per-layer device casts
        # happen below to handle sharded models.
        d_src = direction / (direction.norm() + 1e-6)

        # Cache per-device copies to avoid redundant transfers
        # when consecutive layers share a GPU.
        device_cache: dict[torch.device, torch.Tensor] = {}

        def _direction_for(layer: Any) -> torch.Tensor:
            param = next(p for p in layer.parameters() if p.is_floating_point())
            key = param.device
            if key not in device_cache:
                device_cache[key] = d_src.to(device=param.device, dtype=param.dtype)
            return device_cache[key]

        for layer in module.layers:
            d = _direction_for(layer)
            # Three ablation sites per layer:
            #   1. Layer input  — the residual stream entering this block
            #   2. Attn output  — prevents attn from reintroducing r̂
            #   3. MLP output   — prevents MLP from reintroducing r̂
            self._ablate_input(layer, d)
            self._ablate_tuple_output(layer.self_attn, d)
            self._ablate_tuple_output(layer.mlp, d)

    @staticmethod
    def _ablate_input(layer: Any, direction: torch.Tensor) -> None:
        """Project direction out of the layer's input activation.

        layer.input[0] is the residual stream x_i^(l) entering this
        block via the nnsight tracing API.
        """
        activation = layer.input[0].clone()
        layer.input[0] = activation - projection_einops(activation, direction)

    @staticmethod
    def _ablate_tuple_output(layer_block: Any, direction: torch.Tensor) -> None:
        """Project direction out of a sub-layer's output (attn or MLP).

        Transformer sub-layers return variable-length tuples depending
        on architecture and config (e.g. attention weights, KV cache).
        Only the first element (hidden states) is ablated; the rest
        are passed through unchanged.
        """
        out = layer_block.output
        tuple_length = len(out) if isinstance(out, tuple) else 1

        if tuple_length > 1:
            activation = out[0]
            new_act = activation - projection_einops(activation, direction)
            # Reconstruct tuple, preserving non-activation elements
            # (attention weights, KV cache, etc.)
            if tuple_length == 2:
                layer_block.output = (new_act, out[1])
            elif tuple_length == 3:
                layer_block.output = (new_act, out[1], out[2])
            else:
                layer_block.output = (new_act, *tuple(out[1:]))
        else:
            activation = out
            layer_block.output = activation - projection_einops(activation, direction)