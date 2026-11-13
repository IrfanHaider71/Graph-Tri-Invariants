
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Locate the shared path helper independently of the working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '1-Descriptor'))

from _paths import INPUT, READ_RESULTS, OUTPUT_RESULTS, OUTPUT_FIGURES, prepare_outputs

RES = str(READ_RESULTS)
FIG = str(OUTPUT_FIGURES)

PROPERTIES = ["Boiling Point (BP)", "Flash Point (FP)", "Molar Volume (MV)",
              "Polarizability (Pol)", "Molar Refractivity (MR)"]
PROP_SHORT = {"Boiling Point (BP)": "BP", "Molar Volume (MV)": "MV",
              "Flash Point (FP)": "FP", "Polarizability (Pol)": "Pol",
              "Molar Refractivity (MR)": "MR"}

INDICES = ["GraphEnergy", "Schultz", "DegEntropy"]
INDEX_LABEL = {"GraphEnergy": "Graph energy $E(G)$", "Schultz": "Schultz index $Sc(G)$",
               "DegEntropy": "Degree entropy $ENT(G)$"}

C_OBS = "#0072B2"     
C_FIT = "#D55E00"     
C_INSAMPLE = "#0072B2"
C_CV = "#D55E00"

C_ENERGY = "#2E86DE"        
C_ENERGY_EDGE = "#1B4F91"
C_SCHULTZ = "#E4572E"       
C_SCHULTZ_EDGE = "#B03A1A"
C_ENTROPY = "#27AE60"      
C_ENTROPY_EDGE = "#1D7A43"

plt.rcParams.update({
    "font.size": 13,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 900,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig2_r2_comparison():

    reg = pd.read_csv(f"{RES}/regression_summary.csv")
    cubic = reg[reg["Model"] == "Cubic"]
    x = np.arange(len(PROPERTIES))
    width = 0.25
    colors = {"GraphEnergy": C_ENERGY, "Schultz": C_SCHULTZ, "DegEntropy": C_ENTROPY}
    edges = {"GraphEnergy": C_ENERGY_EDGE, "Schultz": C_SCHULTZ_EDGE, "DegEntropy": C_ENTROPY_EDGE}
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, idx in enumerate(INDICES):
        sub = cubic[cubic["Index"] == idx].set_index("Property")
        vals = [sub.loc[p, "R2"] for p in PROPERTIES]
        ax.bar(x + (i - 1) * width, vals, width, label=INDEX_LABEL[idx], color=colors[idx],
               edgecolor=edges[idx], linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([p.split("(")[1].rstrip(")") for p in PROPERTIES], fontsize=13)
    ax.set_ylabel("Cubic regression $R^2$", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24), frameon=False,
              columnspacing=1.0, handletextpad=0.4, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/Figure_03a_Cubic_R2.png")
    plt.close(fig)

def fig3_cv_generalization():
    cv = pd.read_csv(f"{RES}/cv_degentropy_cubic.csv")
    x = np.arange(len(PROPERTIES))
    width = 0.32
    fig, ax = plt.subplots(figsize=(5.7, 3.6))
    in_sample = [cv.set_index("Property").loc[p, "InSample_R2"] for p in PROPERTIES]
    cv_mean = [cv.set_index("Property").loc[p, "CV_R2_mean"] for p in PROPERTIES]
    cv_std = [cv.set_index("Property").loc[p, "CV_R2_std"] for p in PROPERTIES]
    ax.bar(x - width / 2, in_sample, width, label="In-sample $R^2$", color=C_INSAMPLE,
           edgecolor="white", linewidth=1.2)
    ax.bar(x + width / 2, cv_mean, width, yerr=cv_std, capsize=5,
           error_kw=dict(elinewidth=2.2, capthick=2.2, ecolor="#333333"),
           label="5-fold CV $R^2$ (mean $\\pm$ SD)", color=C_CV, edgecolor="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([p.split("(")[1].rstrip(")") for p in PROPERTIES], fontsize=13)
    ax.set_ylabel("$R^2$", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=2,
              columnspacing=1.1, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/Figure_03b_Cross_Validation.png")
    plt.close(fig)


if __name__ == "__main__":
    prepare_outputs()
    fig2_r2_comparison()
    fig3_cv_generalization()
    print("Figures saved to", FIG)
