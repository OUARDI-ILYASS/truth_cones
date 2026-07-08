import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Data from results_summary.json (Exp 3, Exp 5)
ks = [1, 2, 3, 4]
models = {
    "Qwen-2.5-1.5B": {
        "asr": [1.000, 1.000, 1.000, 1.000],
        "kl":  [0.042, 0.008, 0.009, 0.008],
        "color": "#90CAF9", "marker": "o",
    },
    "Gemma-2-2B": {
        "asr": [1.000, 1.000, 1.000, 1.000],
        "kl":  [0.028, 0.004, 0.007, 0.012],
        "color": "#81C784", "marker": "s",
    },
    "Llama-3.1-8B": {
        "asr": [1.000, 1.000, 0.807, 0.983],
        "kl":  [0.282, 0.017, 0.811, 0.017],
        "color": "#F57C00", "marker": "^",
    },
    "Qwen-2.5-7B": {
        "asr": [0.020, 1.000, 1.000, 1.000],
        "kl":  [0.292, 0.037, 0.202, 0.044],
        "color": "#1976D2", "marker": "o",
    },
    "Gemma-2-9B": {
        "asr": [0.774, 1.000, 1.000, 1.000],
        "kl":  [0.048, 0.001, 0.001, 0.002],
        "color": "#2E7D32", "marker": "s",
    },
    "Qwen-2.5-14B": {
        "asr": [0.658, 1.000, 1.000, 0.994],
        "kl":  [0.194, 0.009, 0.027, 0.010],
        "color": "#0D47A1", "marker": "o",
    },
}

fig, (ax_asr, ax_kl) = plt.subplots(1, 2, figsize=(7.0, 2.8))

# Left: ASR
for name, m in models.items():
    ax_asr.plot(ks, m["asr"], color=m["color"], marker=m["marker"],
                markersize=4, linewidth=1.3, label=name)
ax_asr.set_xlabel("Cone dimension $k$")
ax_asr.set_ylabel("Mean MC ASR (ablation)")
ax_asr.set_title("(a) ASR rises with $k$")
ax_asr.set_xticks(ks)
ax_asr.set_ylim(-0.05, 1.08)
ax_asr.grid(True, alpha=0.25, linewidth=0.4)
ax_asr.legend(loc="lower right", frameon=False, ncol=1)

# Right: KL with threshold
for name, m in models.items():
    ax_kl.plot(ks, m["kl"], color=m["color"], marker=m["marker"],
               markersize=4, linewidth=1.3, label=name)
ax_kl.axhline(0.1, color="#C62828", linestyle="--", linewidth=0.9, alpha=0.8)
ax_kl.text(0.85, 0.12, r"KL$=0.1$ (Arditi et al. 2024)",
           color="#C62828", fontsize=7, style="italic")
ax_kl.set_xlabel("Cone dimension $k$")
ax_kl.set_ylabel("Mean MC KL (retain)")
ax_kl.set_title("(b) KL falls below surgicality threshold at $k=2$")
ax_kl.set_xticks(ks)
ax_kl.set_ylim(0, 0.95)
ax_kl.grid(True, alpha=0.25, linewidth=0.4)

fig.tight_layout()
fig.savefig("fig_teaser.pdf")
print("Wrote fig_teaser.pdf")