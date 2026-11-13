
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

def fig_predicted_vs_actual():
    df = pd.read_csv(f"{RES}/dataset_with_indices.csv").dropna(subset=["DegEntropy"])
    x = df["DegEntropy"].values.astype(float)
    n = len(x)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4))
    axes = axes.flatten()

    all_z_actual, all_z_pred, all_prop = [], [], []

    for i, prop in enumerate(PROPERTIES):
        y = df[prop].values.astype(float)
        coeffs = np.polyfit(x, y, 3)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        ax = axes[i]
        fill, edge = PROP_FILL[prop], PROP_EDGE[prop]
        lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#555555", linewidth=1.4,
               linestyle="--", zorder=1, label="$y=x$")
        ax.scatter(y, y_pred, s=26, facecolor=fill, edgecolor=edge, linewidth=0.5, alpha=0.85, zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Actual {PROP_SHORT[prop]}", fontsize=12.5)
        ax.set_ylabel(f"Predicted {PROP_SHORT[prop]}", fontsize=12.5)
        mu, sd = y.mean(), y.std()
        all_z_actual.append((y - mu) / sd)
        all_z_pred.append((y_pred - mu) / sd)
        all_prop.append(np.full(n, i))

    z_actual = np.concatenate(all_z_actual)
    z_pred = np.concatenate(all_z_pred)
    pooled_r2 = 1 - np.sum((z_actual - z_pred) ** 2) / np.sum(z_actual ** 2)

    ax6 = axes[-1]
    lo, hi = -3.2, 3.2
    ax6.plot([lo, hi], [lo, hi], color="#555555", linewidth=1.4, linestyle="--", zorder=1)
    for i, prop in enumerate(PROPERTIES):
        ax6.scatter(all_z_actual[i], all_z_pred[i], s=16, facecolor=PROP_FILL[prop],
                   edgecolor=PROP_EDGE[prop], linewidth=0.4, alpha=0.75, zorder=2,
                   label=PROP_SHORT[prop].split("(")[-1].rstrip(")"))
    ax6.set_xlim(lo, hi)
    ax6.set_ylim(lo, hi)
    ax6.set_aspect("equal", adjustable="box")
    ax6.set_xlabel("Actual (standardized, all properties)", fontsize=12)
    ax6.set_ylabel("Predicted (standardized)", fontsize=12)
    ax6.legend(loc="lower right", frameon=False, fontsize=9, ncol=1, markerscale=1.4)

    fig.tight_layout()
    fig.savefig(f"{FIG}/Figure_05_Predicted_vs_Actual.pdf")
    plt.close(fig)
    print(f"Saved: Figure_05_Predicted_vs_Actual.pdf (pooled standardized R2={pooled_r2:.4f})")

if __name__ == "__main__":
    prepare_outputs()
    fig_predicted_vs_actual()
