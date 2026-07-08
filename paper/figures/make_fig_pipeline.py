"""
make_fig_pipeline.py — Figure 2 pipeline schematic for the paper.
Output: fig_pipeline.pdf (vector, includes cleanly via \\includegraphics)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Rectangle
from matplotlib.patches import Circle

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette (match teaser figure where relevant)
C_TRUE    = "#2E7D32"
C_FALSE   = "#C62828"
C_RETAIN  = "#616161"
C_METHOD  = "#1565C0"
C_CONE    = "#1565C0"
C_ACCENT  = "#C62828"
C_BG      = "#FAFAFA"

# Figure is full text-width; use figure* in LaTeX
fig = plt.figure(figsize=(14, 6.5))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.set_aspect("equal")
ax.axis("off")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def panel(x, y, w, h, title, fill=C_BG, edge="#888888"):
    """Draw a rounded panel box with a title at the top."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.0,rounding_size=0.6",
                         linewidth=0.8, edgecolor=edge,
                         facecolor=fill)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 1.2, title,
            ha="center", va="top", fontsize=9, fontweight="bold")

def labeled_box(cx, cy, w, h, text, fill, edge, fontsize=6.5):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.0,rounding_size=0.25",
                         linewidth=0.6, edgecolor=edge, facecolor=fill)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)

def arrow(x1, y1, x2, y2, label=None, lw=0.9, color="#444444", label_side="above"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle="-|>", mutation_scale=10,
                                  linewidth=lw, color=color))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        if label_side == "right":
            ax.text(x2 + 0.3, my, label,
                    ha="left", va="center", fontsize=6.5, style="italic")
        else:
            ax.text(mx, my + 0.6, label,
                    ha="center", va="bottom", fontsize=6.5, style="italic")


# ---------------------------------------------------------------------------
# PANEL A — Datasets (left, top)
# ---------------------------------------------------------------------------
panel(1, 27, 22, 22, "A. Datasets")

# Three domain labels
for i, name in enumerate(["cities", "animals", "elements"]):
    ax.text(4 + i*7, 45, name, ha="center", fontsize=7.5, fontweight="bold")

# Triple example (true/false/retain)
labeled_box(12, 41, 19, 1.6,
            r"$p_{\mathrm{true}}$: 'Tokyo is in Japan.'",
            fill=C_TRUE + "33", edge=C_TRUE)
labeled_box(12, 38.8, 19, 1.6,
            r"$p_{\mathrm{false}}$: 'Hanoi is in Poland.'",
            fill=C_FALSE + "33", edge=C_FALSE)
labeled_box(12, 36.6, 19, 1.6,
            r"$p_{\mathrm{retain}}$: 'Summarize the email\dots'",
            fill=C_RETAIN + "33", edge=C_RETAIN)

# Targets
labeled_box(7, 33.0, 7, 1.4,
            r"$t_{\mathrm{true}}{=}$'True'",
            fill="white", edge=C_TRUE)
labeled_box(17, 33.0, 7, 1.4,
            r"$t_{\mathrm{false}}{=}$'False'",
            fill="white", edge=C_FALSE)

ax.text(12, 29.8, "forced-choice True/False over factual triples",
        ha="center", fontsize=6.5, style="italic")

# ---------------------------------------------------------------------------
# PANEL B — Localization (left, bottom)
# ---------------------------------------------------------------------------
panel(1, 1, 22, 24, "B. Exp 1 — Localization")

# NPE mini-heatmap (cols = token positions, rows = layers, top-to-bottom)
hm_x, hm_y0 = 5.5, 8
n_cols, n_rows = 7, 5  # 7 token positions × 5 layer bands

# Synthetic NPE values mimicking the activation patching pattern:
npe = np.zeros((n_rows, n_cols))

# Group (a): country token, early-to-mid layers
npe[4, 2] = 1.0
npe[3, 2] = 0.95
npe[2, 2] = 0.65

# Group (b): summarization position, mid layers
npe[2, 3] = 0.5
npe[3, 3] = 0.35

# Group (c): colon token, mid-to-late layers
npe[0, 6] = 0.90
npe[1, 6] = 0.95
npe[2, 6] = 1

for i in range(n_rows):
    for j in range(n_cols):
        color = plt.cm.Blues(0.05 + 0.85 * npe[i, j])
        ax.add_patch(Rectangle((hm_x + j * 2, hm_y0 + i * 2), 1.9, 1.9,
                               linewidth=0.3, edgecolor='white', facecolor=color))

# Token labels below heatmap
tok_labels = ["...", "is", "in", "Iran", ".", "stmt", ":"]
for j, tok in enumerate(tok_labels):
    ax.text(hm_x + j * 2 + 0.95, hm_y0 - 0.4, tok,
            ha="center", va="top", fontsize=5.5, family="monospace",
            rotation=35)

# Layer axis labels on the left
ax.text(hm_x - 0.6, hm_y0 + 0 * 2 + 0.95, "$L$", fontsize=5.5, ha="right", va="center")
ax.text(hm_x - 0.6, hm_y0 + 4 * 2 + 0.95, "0", fontsize=6, ha="right", va="center")
ax.text(hm_x - 1.8, hm_y0 + 2 * 2 + 0.95, "Layer", fontsize=6, ha="center",
        va="center", rotation=90)

# Group annotations
ax.text(hm_x + 7 * 2 + 0.3, hm_y0 + 1 * 2, "(c)", fontsize=6.5, va="center",
        color="#1a5276")
ax.text(hm_x + 4 * 2, hm_y0 + 3.5 * 2, "(b)", fontsize=6.5, va="center",
        fontweight="bold", color="#1a5276")
ax.text(hm_x + 1 * 2, hm_y0 + 3.5 * 2, "(a)", fontsize=6.5, va="center",
        color="#1a5276")

# Highlight l* cell with a red border
l_star_row, l_star_col = 2, 3  # deepest (b) with NPE > 0.1
ax.add_patch(Rectangle((hm_x + l_star_col * 2, hm_y0 + l_star_row * 2),
                        1.9, 1.9, linewidth=1.5, edgecolor="#c0392b",
                        facecolor="none", zorder=5))

# Arrow from l* cell to annotation
lstar_cx = hm_x + l_star_col * 2 + 1.9
lstar_cy = hm_y0 + l_star_row * 2 + 0.95
arrow(lstar_cx, lstar_cy, lstar_cx + 2.5, lstar_cy,
      lw=0.8, color="#c0392b", label_side="right")
ax.text(lstar_cx + 2.8, lstar_cy, r"$l^{*}$",
        fontsize=8, fontweight="bold", color="#c0392b",
        ha="left", va="center")
ax.text(hm_x + 3 * 2, hm_y0 + 2 * 2 - 0.5,
        "deepest (b)\nNPE $> 0.1$",
        fontsize=5.5, color="#c0392b", ha="left", va="top")

# NPE > 0.1 threshold dashed line in a mini-colorbar
cb_x, cb_y, cb_w, cb_h = 2, 8, 0.8, 9
for step in range(20):
    frac = step / 19
    ax.add_patch(Rectangle((cb_x, cb_y + frac * cb_h), cb_w, cb_h / 19,
                            linewidth=0, facecolor=plt.cm.Blues(0.05 + 0.85 * frac)))
ax.text(cb_x + cb_w / 2, cb_y - 0.3, "0", fontsize=5, ha="center", va="top")
ax.text(cb_x + cb_w / 2, cb_y + cb_h + 0.5, "1", fontsize=5, ha="center", va="bottom")
ax.text(cb_x - 0.3, cb_y + cb_h / 2, "NPE", fontsize=5.5, ha="center", va="center",
        rotation=90)

# STR corruption illustration above heatmap
ax.text(hm_x + 3 * 2 + 0.95, hm_y0 + 5 * 2 + 1.2, "Japan",
        fontsize=6, ha="center", va="center", color="#27ae60", family="monospace")
arrow(hm_x + 3 * 2 + 0.95, hm_y0 + 5 * 2 + 0.7,
      hm_x + 3 * 2 + 0.95, hm_y0 + 5 * 2 - 0.2,
      lw=0.7, color="#888888")
ax.text(hm_x + 3 * 2 + 0.95, hm_y0 + 5 * 2 - 0.5, "Iran",
        fontsize=6, ha="center", va="top", color="#c0392b", family="monospace")
ax.text(hm_x + 3 * 2 + 0.95, hm_y0 + 5 * 2 + 2.0, "STR",
        fontsize=5.5, ha="center", va="bottom", style="italic", color="#888888")

ax.text(12, 2.2, "STR corruption + activation patching",
        ha="center", fontsize=6.5, style="italic")
# ---------------------------------------------------------------------------
# PANEL C — TCO: cone + algorithm steps (center, large — CENTERPIECE)
# ---------------------------------------------------------------------------
panel(26, 1, 44, 48, "C. Exp 2 & 3 — TDO / TCO")

# ---- the cone -------------------------------------------------------------
apex   = np.array([42, 40])
base_l = np.array([34, 16])
base_r = np.array([50, 16])

cone = Polygon([apex, base_l, base_r], closed=True,
               facecolor=C_CONE, alpha=0.18,
               edgecolor=C_METHOD, linewidth=1.0)
ax.add_patch(cone)

# elliptical base (perspective hint)
t = np.linspace(0, np.pi, 50)
ax.plot(42 + 8*np.cos(t), 16 - 1.0*np.sin(t),
        "--", linewidth=0.5, color=C_METHOD, alpha=0.6)

# Monte-Carlo interior sample arrows
rng = np.random.default_rng(42)
for _ in range(22):
    u, v = sorted([rng.random(), rng.random()])
    p = u*base_l + (v-u)*base_r + (1-v)*apex
    ax.add_patch(FancyArrowPatch(apex, p, arrowstyle="-|>",
                 mutation_scale=4, linewidth=0.3, color=C_METHOD, alpha=0.5))

# basis vectors v1, v2 (bold)
ax.add_patch(FancyArrowPatch(apex, base_l, arrowstyle="-|>",
             mutation_scale=10, linewidth=1.6, color=C_METHOD))
ax.text(33.0, 26, r"$v_1$", fontsize=10, color=C_METHOD, fontweight="bold")
ax.add_patch(FancyArrowPatch(apex, (42, 16), arrowstyle="-|>",
             mutation_scale=10, linewidth=1.6, color=C_METHOD))
ax.text(42.5, 26, r"$v_2$", fontsize=10, color=C_METHOD, fontweight="bold")

# DIM warm-start vector (thin red, nearly aligned with v1)
dim_end = base_l + np.array([1.6, 0.3])
ax.add_patch(FancyArrowPatch(apex, dim_end, arrowstyle="-|>",
             mutation_scale=8, linewidth=0.9, color=C_ACCENT))
ax.text(30.5, 21, r"$\theta_{\mathrm{DIM}}$", fontsize=7.5,
        color=C_ACCENT, style="italic")

# one MC sample highlighted with label
ax.text(43, 30, r"$u_j$", fontsize=8, color=C_METHOD, style="italic")

# ---- algorithm steps (right column) ---------------------------------------
sx, sy0, dy = 54, 40, 4.3
steps = [
    (r"$V \leftarrow \theta_{\mathrm{DIM}}/\|\theta_{\mathrm{DIM}}\|$",
     "warm-start ($k{=}1$)"),
    (r"$V \leftarrow [\,V^{(k-1)},\, v_{\mathrm{rand}}\,]$",
     "augment basis ($k{>}1$)"),
    (r"$u_j = V c_j / \|V c_j\|,\ \ c_j\!\sim\!\mathcal{N}(0,I_k)$",
     "sample interior dirs"),
    (r"$\mathcal{L} = \mathbb{E}_j[\,\mathcal{L}(u_j)\,]"
     r" + \frac{1}{k}\sum_i \mathcal{L}(v_i)$",
     "loss on samples + basis"),
    (r"$V \leftarrow V - \eta\,\nabla_V \mathcal{L}$",
     "gradient step"),
    (r"$V \leftarrow \mathrm{GramSchmidt}(V)$",
     "re-orthonormalize"),
]
for i, (eq, cap) in enumerate(steps):
    y = sy0 - i*dy
    ax.text(sx - 1.4, y, f"{i+1}", ha="center", va="center", fontsize=7,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.18", fc=C_METHOD, ec="none"))
    ax.text(sx, y + 0.5, eq, ha="left", va="center", fontsize=7)
    ax.text(sx, y - 1.1, cap, ha="left", va="center", fontsize=5.8,
            style="italic", color="#555")

# loop indicator
ax.annotate("", xy=(sx - 1.4, sy0 - 2*dy + 0.8),
            xytext=(sx - 1.4, sy0 - 5*dy - 0.8),
            arrowprops=dict(arrowstyle="-|>", color="#888", linewidth=0.8,
                            connectionstyle="arc3,rad=-0.45"))
ax.text(sx - 4.2, sy0 - 3.5*dy, "repeat", fontsize=5.8, rotation=90,
        ha="center", va="center", style="italic", color="#888")

# ---- bottom labels --------------------------------------------------------
ax.text(48, 9,
        r"TDO: $k{=}1$ direction $v_1$   ·   TCO: $k{\geq}2$ cone interior",
        ha="center", fontsize=7, color=C_METHOD)
ax.text(48, 6, "frozen model · loss = $\\mathcal{L}_{abl}+\\mathcal{L}_{add}"
        "+\\mathcal{L}_{ret}$ (necessity · sufficiency · surgicality)",
        ha="center", fontsize=6, style="italic", color="#555")
# ---------------------------------------------------------------------------
# PANEL D — Orthogonality (bottom-right, top)
# ---------------------------------------------------------------------------
panel(73, 27, 26, 22, "D. Exp 4 — $|\\cos(\\theta_{\\mathrm{DIM}}, v_i)|$")

# Mini heatmap: 4 rows (v_1..v_4) × 6 cols (models)
hm_x0, hm_y0 = 77, 32
cell_w, cell_h = 3, 2.3

# v_1 row bright (high alignment), v_2-4 dark
heatmap_data = np.array([
    [0.93, 0.91, 0.87, 0.86, 0.84, 0.82],   # k=1 (warm-start anchor row)
    [0.03, 0.10, 0.06, 0.02, 0.04, 0.01],   # v_2
    [0.02, 0.01, 0.06, 0.02, 0.01, 0.04],   # v_3
    [0.05, 0.07, 0.01, 0.01, 0.04, 0.02],   # v_4
])
for i in range(4):
    for j in range(6):
        val = heatmap_data[i, j]
        color = plt.cm.Blues(min(0.15 + val * 0.85, 1.0))
        ax.add_patch(Rectangle((hm_x0 + j*cell_w, hm_y0 + (3-i)*cell_h),
                                cell_w - 0.1, cell_h - 0.1,
                                linewidth=0, facecolor=color))
        ax.text(hm_x0 + j*cell_w + cell_w/2 - 0.05,
                hm_y0 + (3-i)*cell_h + cell_h/2 - 0.05,
                f"{val:.2f}", ha="center", va="center",
                fontsize=5.5, color="white" if val > 0.4 else "black")

# Row labels
for i, lbl in enumerate(["$v_1$", "$v_2$", "$v_3$", "$v_4$"]):
    ax.text(hm_x0 - 0.5, hm_y0 + (3-i)*cell_h + cell_h/2 - 0.05,
            lbl, ha="right", va="center", fontsize=7)

# Column labels (model abbreviations)
model_labels = ["Q-1.5B", "G-2B", "Q-7B", "G-9B", "L-8B", "Q-14B"]
for j, lbl in enumerate(model_labels):
    ax.text(hm_x0 + j*cell_w + cell_w/2 - 0.05, hm_y0 - 0.8,
            lbl, ha="center", va="top", fontsize=5.5, rotation=45)

ax.text(86, 28.5, "bright $v_1$, dark $v_{2..4}$ ⇒ novel structure",
        ha="center", fontsize=6.5, style="italic")


# ---------------------------------------------------------------------------
# PANEL E — KL retention (bottom-right, bottom)
# ---------------------------------------------------------------------------
panel(73, 1, 26, 24, "E. Exp 5 — KL retention")

# Heatmap: 4 rows (k=1..4) × 6 cols (models), green if pass, red if fail
# Hand-coded from Exp 5 results
kl_pass = np.array([
    # k=1
    [True,  True,  False, True,  False, False],   # Q1.5, G2, Q7, G9, L8, Q14
    # k=2
    [True,  True,  True,  True,  True,  True],
    # k=3
    [True,  True,  False, True,  False, True],
    # k=4
    [True,  True,  True,  True,  True,  True],
])
kl_y0 = 7
for i in range(4):
    for j in range(6):
        c = C_TRUE + "88" if kl_pass[i, j] else C_FALSE + "BB"
        edge = C_TRUE if kl_pass[i, j] else C_FALSE
        ax.add_patch(Rectangle((hm_x0 + j*cell_w, kl_y0 + (3-i)*cell_h),
                                cell_w - 0.1, cell_h - 0.1,
                                linewidth=0.4, edgecolor=edge, facecolor=c))
        mark = r"$\checkmark$" if kl_pass[i, j] else r"$\times$"
        ax.text(hm_x0 + j*cell_w + cell_w/2 - 0.05,
                kl_y0 + (3-i)*cell_h + cell_h/2 - 0.05,
                mark, ha="center", va="center", fontsize=7, fontweight="bold",
                color="white")

# Row labels
for i, lbl in enumerate(["$k{=}1$", "$k{=}2$", "$k{=}3$", "$k{=}4$"]):
    ax.text(hm_x0 - 0.5, kl_y0 + (3-i)*cell_h + cell_h/2 - 0.05,
            lbl, ha="right", va="center", fontsize=7)

# Column labels
for j, lbl in enumerate(model_labels):
    ax.text(hm_x0 + j*cell_w + cell_w/2 - 0.05, kl_y0 - 0.8,
            lbl, ha="center", va="top", fontsize=5.5, rotation=45)

ax.text(86, 3.2, "green ≤ 0.1 (surgical), red > 0.1",
        ha="center", fontsize=6.5, style="italic")


# ---------------------------------------------------------------------------
# Connectors between panels
# ---------------------------------------------------------------------------
# A → B (datasets feed into localization)
arrow(12, 27.3, 12, 24.6, label="contrastive triples", lw=0.8, label_side = "right")

# B → C (l* feeds into TCO)
arrow(23, 13, 26, 13, label="$l^*$", lw=0.9, color=C_ACCENT)

# C → D (V feeds into orthogonality analysis)
arrow(70, 38, 73, 38, label="$V$", lw=0.9, color=C_METHOD)

# C → E (V feeds into KL evaluation)
arrow(70, 13, 73, 13, label="$V$", lw=0.9, color=C_METHOD)


fig.savefig("fig_pipeline.pdf")
print("Wrote fig_pipeline.pdf")