"""Per-run visualization utilities.

Produces diagnostic figures alongside manifests during pipeline runs:
  - NPE heatmaps for activation patching (§3.2, Fig. NPE)
  - Training curves for TDO (§3.4) and TCO (§3.5)

Cross-experiment paper figures (Tab. master, Tab. cone_asr, etc.) are
produced in the analysis notebooks, which read from results_summary.json.
This module covers only the per-run visuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# ── Color palette (consistent across all figures) ─────────────────────────────
class COLORS:
    LIGHT_BLUE   = "#46B1E1"
    BLUE         = "#156082"
    LIGHT_ORANGE = "#F2AA84"
    ORANGE       = "#E97132"
    PURPLE       = "#A02B93"
    GREEN        = "#4EA72E"


def plot_npe_heatmap(
    npe_matrix: np.ndarray,                # (L, n_tokens)
    token_labels: List[str],
    title: str,
    *,
    selected_layer: Optional[int] = None,
) -> go.Figure:
    """Plotly heatmap of the NPE matrix (§3.2, Fig. NPE).

    Rows = layers (0 at top), columns = token positions. Color encodes
    NPE(l, i) from Eq. 3. The red dashed line marks l* (most-downstream
    group (b) layer with NPE > 0.1).
    """
    n_layers = npe_matrix.shape[0]

    fig = px.imshow(
        npe_matrix,
        labels=dict(x="Token Position", y="Model Layer", color="Normalized Patching Effect"),
        x=token_labels,
        y=list(range(n_layers)),
        color_continuous_scale=[[0, "#FFFFFF"], [1, COLORS.BLUE]],
        zmin=0, zmax=1,
        aspect="auto",
        template="simple_white",
        title=title,
    )
    fig.update_layout(
        xaxis=dict(
            ticktext=token_labels,
            tickvals=list(range(len(token_labels))),
            tickangle=-45,
        ),
        xaxis_title="Token Position (Target Statement)",
        yaxis_title="Model Layer",
        coloraxis_colorbar=dict(title="NPE"),
        width=900,
        height=600,
    )
    if selected_layer is not None:
        fig.add_hline(
            y=selected_layer,
            line=dict(color="red", dash="dash", width=2),
            annotation_text=f"Selected: layer {selected_layer}",
            annotation_position="right",
        )
    return fig


def save_static_heatmap(fig: go.Figure, path: Path, *, scale: int = 2) -> None:
    """Save a Plotly figure as PNG (requires kaleido)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=900, height=600, scale=scale)


def save_heatmap_png(matrix: np.ndarray, path: Path, *,
                     token_labels=None, title="",
                     xlabel="Token position", ylabel="Layer") -> None:
    """Matplotlib fallback for NPE heatmaps (no kaleido needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues",
                   vmin=0, vmax=abs(matrix).max(), origin="upper")
    if token_labels is not None:
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Normalized Patching Effect")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(
    history: Dict[str, List[float]],
    title: str,
) -> go.Figure:
    """Multi-trace line plot of the composite loss during TDO/TCO training.

    Traces:
      - Total = L(r) (black)
      - Ablation = λ_abl · L_abl (blue) — necessity
      - Addition = λ_add · L_add (orange) — sufficiency
      - Retain = λ_ret · L_ret (green) — surgicality
      - Red dashed verticals = LR reduction events
    """
    steps = list(range(len(history.get("total_loss", []))))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=history.get("total_loss", []),
        name="Total", line=dict(color="black", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=history.get("abl_loss", []),
        name="Ablation", line=dict(color=COLORS.BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=history.get("add_loss", []),
        name="Addition", line=dict(color=COLORS.ORANGE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=history.get("ret_loss", []),
        name="Retain (KL)", line=dict(color=COLORS.GREEN, width=2),
    ))

    for s in history.get("lr_changes", []):
        fig.add_vline(
            x=s,
            line=dict(color="red", dash="dash", width=1),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Step (effective batch)",
        yaxis_title="Loss",
        template="simple_white",
        width=900, height=500,
    )
    return fig