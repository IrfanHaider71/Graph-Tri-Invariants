
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

# Locate the shared path helper independently of the working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '1-Descriptor'))

from _paths import INPUT, READ_RESULTS, OUTPUT_RESULTS, OUTPUT_FIGURES, prepare_outputs

IN_CSV = str(READ_RESULTS / "dataset_with_indices.csv")
OUT_STATS = str(OUTPUT_RESULTS / "full_cubic_statistics.csv")
OUT_CV = str(OUTPUT_RESULTS / "cv_degentropy_cubic.csv")

INDICES = ["GraphEnergy", "Schultz", "DegEntropy"]
PROPERTIES = ["Boiling Point (BP)", "Molar Volume (MV)", "Flash Point (FP)",
              "Polarizability (Pol)", "Molar Refractivity (MR)"]
SEED = 42


def cubic_design_matrix(x):
    return np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])


def full_ols_stats(x, y):

    X = cubic_design_matrix(x)
    n, k = X.shape[0], X.shape[1] - 1  
    scale = np.sqrt((X ** 2).sum(axis=0))
    X_scaled = X / scale
    beta_scaled, _, _, _ = np.linalg.lstsq(X_scaled, y, rcond=None)
    beta = beta_scaled / scale
    y_pred = X @ beta
    resid = y - y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    see = np.sqrt(ss_res / (n - k - 1))  
    ms_reg = (ss_tot - ss_res) / k
    ms_res = ss_res / (n - k - 1)
    f_stat = ms_reg / ms_res
    p_value = stats.f.sf(f_stat, k, n - k - 1)
    r = np.sqrt(r2)
    return {
        "n": n, "R": r, "R2": r2, "Adj_R2": adj_r2, "SEE": see,
        "F": f_stat, "p": p_value,
        "delta_cubic": beta[3], "gamma_quad": beta[2], "beta_lin": beta[1], "alpha": beta[0],
    }


def cv_r2_cubic(x, y, n_splits=5, seed=SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(x):
        X_tr = cubic_design_matrix(x[tr])
        scale = np.sqrt((X_tr ** 2).sum(axis=0))
        beta_scaled, _, _, _ = np.linalg.lstsq(X_tr / scale, y[tr], rcond=None)
        beta = beta_scaled / scale
        X_te = cubic_design_matrix(x[te])
        pred = X_te @ beta
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
        scores.append(1 - ss_res / ss_tot)
    return np.array(scores)


def main():
    df = pd.read_csv(IN_CSV).dropna(subset=INDICES)
    print(f"n = {len(df)}")

    rows = []
    for idx in INDICES:
        x = df[idx].values.astype(float)
        for prop in PROPERTIES:
            y = df[prop].values.astype(float)
            res = full_ols_stats(x, y)
            res["Index"] = idx
            res["Property"] = prop
            rows.append(res)

    stats_df = pd.DataFrame(rows)
    cols = ["Property", "Index", "n", "R", "R2", "Adj_R2", "SEE", "F", "p",
            "delta_cubic", "gamma_quad", "beta_lin", "alpha"]
    stats_df = stats_df[cols]
    stats_df.to_csv(OUT_STATS, index=False)
    print("\n=== Full cubic-model OLS statistics ===")
    print(stats_df.round(4).to_string(index=False))

    print("\n=== 5-fold CV, DegEntropy cubic model (genuine held-out R^2) ===")
    cv_rows = []
    x = df["DegEntropy"].values.astype(float)
    for prop in PROPERTIES:
        y = df[prop].values.astype(float)
        scores = cv_r2_cubic(x, y)
        in_sample_r2 = full_ols_stats(x, y)["R2"]
        cv_rows.append({
            "Property": prop, "InSample_R2": in_sample_r2,
            "CV_R2_mean": scores.mean(), "CV_R2_std": scores.std(),
            "CV_R2_min": scores.min(), "CV_R2_max": scores.max(),
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT_CV, index=False)
    print(cv_df.round(4).to_string(index=False))


if __name__ == "__main__":
    prepare_outputs()
    main()
