
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Locate the shared path helper independently of the working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '1-Descriptor'))

from _paths import READ_RESULTS, OUTPUT_FIGURES, prepare_outputs
RES = str(READ_RESULTS)
FIG = str(OUTPUT_FIGURES)

PROPERTIES = ["Boiling Point (BP)", "Flash Point (FP)", "Molar Volume (MV)",
              "Polarizability (Pol)", "Molar Refractivity (MR)"]
PROP_SHORT = {"Boiling Point (BP)": "Boiling point (BP)", "Molar Volume (MV)": "Molar volume (MV)",
              "Flash Point (FP)": "Flash point (FP)", "Polarizability (Pol)": "Polarizability (Pol)",
              "Molar Refractivity (MR)": "Molar refractivity (MR)"}
PROP_FILL = {
    "Boiling Point (BP)":      "#E4572E",  
    "Flash Point (FP)":        "#F2A900", 
    "Molar Volume (MV)":       "#8E44AD",  
    "Polarizability (Pol)":    "#27AE60",  
    "Molar Refractivity (MR)": "#2E86DE",  
}
PROP_EDGE = {
    "Boiling Point (BP)":      "#B03A1A",
    "Flash Point (FP)":        "#B37B00",
    "Molar Volume (MV)":       "#6C3483",
    "Polarizability (Pol)":    "#1D7A43",
    "Molar Refractivity (MR)": "#1B4F91",
}
plt.rcParams.update({
    "font.size": 13, "font.family": "DejaVu Sans",
    "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

def load():
    df = pd.read_csv(f"{RES}/dataset_with_indices.csv").dropna(subset=["DegEntropy"])
    x = df["DegEntropy"].values.astype(float)
    return df, x, len(x)

def styleC_binned_violin(df, x, n):
    """Distribution of each property across ENT(G) quartile bins -- violin
    + jittered points, no curve at all."""
    n_bins = 4
    bin_idx, bin_edges = pd.qcut(x, n_bins, labels=False, retbins=True)
    positions = np.arange(1, n_bins + 1)
    bin_labels = [f"{bin_edges[k]:.1f}\u2013{bin_edges[k+1]:.1f}" for k in range(n_bins)]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    axes = axes.flatten()
    for i, prop in enumerate(PROPERTIES):
        y = df[prop].values.astype(float)
        ax = axes[i]
        fill, edge = PROP_FILL[prop], PROP_EDGE[prop]
        groups = [y[bin_idx == k] for k in range(n_bins)]
        parts = ax.violinplot(groups, positions=positions, widths=0.72,
                              showmeans=False, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(fill); body.set_edgecolor(edge)
            body.set_linewidth(1.3); body.set_alpha(0.55)
        for k, vals in enumerate(groups):
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(positions[k] + jitter, vals, s=18, facecolor=fill,
                      edgecolor=edge, linewidth=0.5, alpha=0.95, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(bin_labels, fontsize=10)
        ax.set_xlabel("Degree entropy $ENT(G)$ (quartile bins)", fontsize=12)
        ax.set_ylabel(PROP_SHORT[prop])

    axes[-1].axis("off")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = []
    for prop in PROPERTIES:
        handles.append(Patch(facecolor=PROP_FILL[prop], edgecolor=PROP_EDGE[prop],
                             alpha=0.7, linewidth=1.3, label=PROP_SHORT[prop]))
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor="#777777",
                          markeredgecolor="#333333", markeredgewidth=0.6, markersize=8,
                          label="Individual compound"))
    handles.append(Patch(facecolor="none", edgecolor="#555555", linewidth=1.3,
                         label="Violin: density per bin"))
    axes[-1].legend(handles=handles, loc="center", fontsize=12, frameon=False,
                    handletextpad=0.6, labelspacing=1.3)

    fig.tight_layout()
    fig.savefig(f"{FIG}/Figure_04_Quartile_Violins.pdf")
    plt.close(fig)

if __name__ == "__main__":
    prepare_outputs()
    df, x, n = load()
    styleC_binned_violin(df, x, n)
    print("Saved Figure_04_Quartile_Violins.pdf")
