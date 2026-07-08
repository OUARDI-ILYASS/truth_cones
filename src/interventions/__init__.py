"""Causal interventions on transformer activations.

Strategy pattern: ``Intervention`` is the abstract base class with concrete
implementations ``AblationIntervention`` and ``AdditionIntervention``. Both
expose the same ``apply(model, direction)`` method so they can be swapped
inside training and evaluation loops.

This module also contains the ``TruthCone`` nn.Module (orthonormal basis
container) and the optimization routines TDO and TCO.
"""

from src.interventions.ablation import AblationIntervention
from src.interventions.addition import AdditionIntervention
from src.interventions.base import Intervention
from src.interventions.patching import ActivationPatching
from src.interventions.projection import projection_einops
from src.interventions.tco import truth_cone_optimization
from src.interventions.tco_optimized import truth_cone_optimization as truth_cone_optimization2
from src.interventions.tdo import truth_direction_optimization
from src.interventions.truth_cone import TruthCone

__all__ = [
    "AblationIntervention",
    "ActivationPatching",
    "AdditionIntervention",
    "Intervention",
    "TruthCone",
    "projection_einops",
    "truth_cone_optimization",
    "truth_cone_optimization2",
    "truth_direction_optimization",
]
