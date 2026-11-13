
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P

# Locate the shared path helper independently of the working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '1-Descriptor'))

from _paths import INPUT, READ_RESULTS, OUTPUT_RESULTS, OUTPUT_FIGURES, prepare_outputs

IN_CSV = str(READ_RESULTS / "dataset_with_indices.csv")
OUT_SUMMARY = str(OUTPUT_RESULTS / "regression_summary.csv")
OUT_BEST = str(OUTPUT_RESULTS / "best_predictor_per_property.csv")

INDICES = ["GraphEnergy", "Schultz", "DegEntropy"]
PROPERTIES = ["Boiling Point (BP)", "Molar Volume (MV)", "Flash Point (FP)",
              "Polarizability (Pol)", "Molar Refractivity (MR)"]


def fit_poly(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = len(x)
    k = degree  
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return {"degree": degree, "coeffs": coeffs, "R2": r2, "Adj_R2": adj_r2, "n": n}

def main():
    df = pd.read_csv(IN_CSV).dropna(subset=INDICES)
    print(f"n = {len(df)} (after dropping disconnected-graph compounds)")

    rows = []
    for idx in INDICES:
        x = df[idx].values.astype(float)
        for prop in PROPERTIES:
            y = df[prop].values.astype(float)
            for deg, label in [(1, "Linear"), (2, "Quadratic"), (3, "Cubic")]:
                res = fit_poly(x, y, deg)
                rows.append({
                    "Index": idx, "Property": prop, "Model": label,
                    "R2": res["R2"], "Adj_R2": res["Adj_R2"], "n": res["n"],
                    "coeffs": ";".join(f"{c:.6g}" for c in res["coeffs"]),
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_SUMMARY, index=False)

    print("\n=== R^2 by Index x Property x Model ===")
    pivot = out.pivot_table(index=["Property", "Index"], columns="Model", values="R2")
    pivot = pivot[["Linear", "Quadratic", "Cubic"]]
    print(pivot.round(4).to_string())

    cubic = out[out["Model"] == "Cubic"]
    best = cubic.loc[cubic.groupby("Property")["R2"].idxmax()]
    best = best[["Property", "Index", "R2", "Adj_R2", "coeffs"]].rename(columns={"R2": "Cubic_R2", "Adj_R2": "Cubic_AdjR2"})
    best.to_csv(OUT_BEST, index=False)
    print("\n=== Best predictor per property (by cubic R^2) ===")
    print(best[["Property", "Index", "Cubic_R2", "Cubic_AdjR2"]].round(4).to_string(index=False))


if __name__ == "__main__":
    prepare_outputs()
    main()
