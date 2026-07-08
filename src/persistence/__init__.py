"""Persistence: manifest writers, handoff JSON contract, results summaries."""

from src.persistence.handoff import (
    LayerSelection,
    load_layer_selection,
    write_layer_selection,
)
from src.persistence.manifest import write_manifest, write_weights
from src.persistence.results_summary import (
    build_alignment_summary,
    build_patching_handoff,
    build_retention_summary,
    build_tco_summary,
    build_tdo_summary,
    write_results_summary,
)

__all__ = [
    "LayerSelection",
    "build_alignment_summary",
    "build_patching_handoff",
    "build_retention_summary",
    "build_tco_summary",
    "build_tdo_summary",
    "load_layer_selection",
    "write_layer_selection",
    "write_manifest",
    "write_results_summary",
    "write_weights",
]
