"""Layer selection handoff between activation patching (§3.2) and all
downstream methods (DIM §3.3, TDO §3.4, TCO §3.5, evaluation §3.6).

Activation patching writes layer_selection.json with a per-model entry
specifying l* and token position. All downstream methods read from this
file, so the cross-family comparison does not rely on a fixed depth
(§3.2: "DIM, TDO, and TCO then all operate at the layer most causally
implicated for that model").

l* selection rule (§3.2): most-downstream layer in NPE group (b) with
mean NPE > 0.1. Group (b) = end-of-statement punctuation in mid layers,
where clause-level information is aggregated into a propositional
judgment. Across all six models, l* falls at normalized depth 0.44–0.69
(Tab. master).

Token position is i = −5 for all models (§3.2 footnote): the four
trailing tokens after the final word are the fixed suffix
"This statement is:", so position −5 lands on the end-of-statement
punctuation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logging_setup import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1.0"


@dataclass
class LayerSelection:
    """Per-model entry in layer_selection.json.

    Records l* and the evidence for why it was chosen, so the choice
    can be defended in §3.2 without re-running activation patching.

    Fields:
        add_layer:        l* — the causally selected layer (§3.2).
        token_position:   i = −5 (end-of-statement punctuation).
        selection_method: "most_downstream_group_b" — deepest layer in
                          NPE group (b) with mean NPE > 0.1.
        n_layers_total:   L (total transformer layers).
        normalized_depth: l* / L, reported in Tab. master (0.44–0.69).
        evidence:         NPE values, per-dataset breakdowns, etc.
    """

    add_layer: int
    token_position: int = -1
    selection_method: str = "most_downstream_group_b"
    n_layers_total: int = 0
    normalized_depth: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


def load_layer_selection(path: Path) -> Dict[str, LayerSelection]:
    """Load layer_selection.json → {model_name: LayerSelection}.

    Called by DIM (§3.3), TDO (§3.4), TCO (§3.5), and all evaluation
    scripts to retrieve l* per model.

    Raises:
        FileNotFoundError: if the file does not exist. Run activation
            patching (§3.2) first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run activation patching (run_patching.py) first."
        )

    with open(path) as f:
        raw = json.load(f)

    selections: Dict[str, LayerSelection] = {}
    for model_name, sel in raw.get("selections", {}).items():
        selections[model_name] = LayerSelection(
            add_layer=int(sel["add_layer"]),
            token_position=int(sel.get("token_position", -1)),
            selection_method=sel.get("selection_method", "most_downstream_group_b"),
            n_layers_total=int(sel.get("n_layers_total", 0)),
            normalized_depth=float(sel.get("normalized_depth", 0.0)),
            evidence=sel.get("evidence", {}),
        )
    logger.info("Loaded layer_selection.json: %d models", len(selections))
    return selections


def write_layer_selection(
    path: Path,
    selections: Dict[str, LayerSelection],
    *,
    produced_by: str = "activation_patching",
) -> None:
    """Write layer_selection.json. Idempotent — overwrites the file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "produced_by":    produced_by,
        "timestamp":      datetime.now().isoformat(),
        "selections":     {name: asdict(sel) for name, sel in selections.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("layer_selection.json written: %s  (%d models)", path, len(selections))


def upsert_layer_selection(
    path: Path,
    model_name: str,
    selection: LayerSelection,
    *,
    produced_by: str = "activation_patching",
) -> None:
    """Insert or update a single model's entry without losing others.

    Useful when running activation patching model-by-model in separate
    jobs (e.g. one RunPod instance per model).
    """
    existing: Dict[str, LayerSelection] = {}
    if path.exists():
        try:
            existing = load_layer_selection(path)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Existing %s is malformed; starting fresh.", path)
            existing = {}
    existing[model_name] = selection
    write_layer_selection(path, existing, produced_by=produced_by)