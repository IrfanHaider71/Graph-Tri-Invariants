# Graph-Tri Invariants

Source code for **A Graph-Based Framework for Molecular Property Prediction from Topological Descriptors**, by  Irfan Haider et al.

**Journal:** [Journal of Computer-Aided Molecular Design](https://link.springer.com/journal/10822)  
**Status:** Manuscript under revision for resubmission.

The study compares graph energy E(G), the Schultz index Sc(G), and normalized-vertex-degree Shannon entropy ENT(G) for five physicochemical properties using linear, quadratic, and cubic regressions.

## Repository layout

| Folder | Contents |
| --- | --- |
| `1-Descriptor/` | Descriptor sanity checks, molecular-graph construction, descriptor computation, and shared paths |
| `2- Analysis/` | Regression comparisons, cubic statistics, five-fold validation, and Figures 3–5 |
| `3-Conditioning Runtime and Applicability/` | Numerical conditioning, applicability diagnostics, runtime benchmarks, and verification |
| `4-Supplementary/` | Figure S1 from saved applicability diagnostics |

Keep these folder names unchanged: the scripts use relative paths to locate helpers and results. Figures 1–2 are separate illustrative manuscript artwork.

## Supplementary tables

Supplementary Tables S1–S3 supplied with the manuscript list source compounds, computed descriptors, and physicochemical-property values. Table S4 and Figure S1 report applicability diagnostics. Main-manuscript Table 4 reports raw descriptor–property Pearson correlations, distinct from observed–fitted model correlations.

## Environment

Install dependencies in your Python environment:

```text
python -m pip install numpy pandas networkx rdkit scipy scikit-learn matplotlib openpyxl threadpoolctl
```

In VS Code, choose this environment with **Python: Select Interpreter**. This list is not an exact historical version lock.

## Input

Create `1-Descriptor/inputs/` and place `dataset_combined.xlsx` there. The first worksheet must retain the original rows and these column names:

```text
Serial Number
Name
SMILES
Boiling Point (BP)
Flash Point (FP)
Molar Volume (MV)
Polarizability (Pol)
Molar Refractivity (MR)
```

Preserve additional source columns and original full-precision values. Alternatively, set the `QSPR_INPUT` environment variable to the full workbook path before running the scripts.

## Reproducing the analysis

Run from the repository root, in this order:

```text
python -B "1-Descriptor/00_sanity_check.py"
python -B "1-Descriptor/01_compute_indices.py"
python -B "2- Analysis/01_regression.py"
python -B "2- Analysis/02_full_cubic_statistics.py"
python -B "2- Analysis/03_figure03_panels.py"
python -B "2- Analysis/04_figure04_quartile_violins.py"
python -B "2- Analysis/05_figure05_predicted_actual.py"
python -B "3-Conditioning Runtime and Applicability/01_prepare_and_conditioning.py"
python -B "3-Conditioning Runtime and Applicability/02_applicability.py"
python -B "3-Conditioning Runtime and Applicability/03_benchmark.py"
python -B "3-Conditioning Runtime and Applicability/04_verify_analyses.py"
python -B "4-Supplementary/plot_figure_s1.py"
```

Descriptor and regression CSVs are written to `1-Descriptor/generated/results_csv/analysis/`, and Figures 3–5 to `1-Descriptor/generated/figures/`. Conditioning, applicability, and runtime outputs are written under `3-Conditioning Runtime and Applicability/generated/`. Figure S1 is written under the repository-root `generated/Figure_S1/` directory.

The conditioning script snapshots its inputs and hashes source files for verification. Finish code edits before running it. Runtime measurements depend on hardware and environment; reruns do not reproduce exact historical timings.

To redraw Figure S1 from another diagnostics file:

```text
python -B "4-Supplementary/plot_figure_s1.py" --diagnostics "path/to/full_fit_applicability_diagnostics.csv"
```

## Scientific contribution

Graph-Tri-Invariants brings together three complementary views of molecular connectivity: the spectral information captured by graph energy, the degree-distance relationships captured by the Schultz index, and the distribution of vertex degrees captured by Shannon entropy. Their comparison helps explain how different aspects of molecular topology relate to physicochemical properties.

The analysis covers **221 connected molecular graphs** curated from 223 source records and examines five properties: boiling point, flash point, molar volume, polarizability, and molar refractivity. A common regression framework compares **45 descriptor-property models** across linear, quadratic, and cubic forms, enabling consistent assessment of descriptor behavior and model complexity.

The computational workflow includes:

- **Descriptor verification:** molecular-graph construction and checks of the three invariant calculations, with degree entropy expressed in bits using probabilities d(v)/(2m).
- **Statistical comparison:** raw descriptor-property Pearson correlations, regression coefficients, goodness-of-fit statistics, and significance tests.
- **Cross-validation:** a reproducible shuffled five-fold assessment of the degree-entropy cubic models, using seed 42.
- **Applicability analysis:** leverage and internally studentized residual diagnostics to characterize structural influence and model-data agreement.
- **Computational assessment:** numerical conditioning checks and runtime benchmarks for evaluating implementation stability and computational cost.

Together, these analyses provide an interpretable comparison of spectral, degree-distance, and information-theoretic descriptors for molecular property analysis. The scripts reproduce the numerical analyses and generate the corresponding statistical summaries and figures. Cubic-model significance is evaluated using the F survival function for numerical stability, with significance reported in the manuscript as p < 0.001.

## License & Usage Policy

Copyright © 2026 Irfan Haider and coauthors.

This repository is provided **solely for transparency and peer review**. Reviewers may inspect and run the code to evaluate the manuscript. Any other use, reproduction, redistribution, or modification requires prior written permission. See [USAGE_NOTICE.txt](USAGE_NOTICE.txt) for the full terms.

## Contact

**First author:** Irfan Haider | [irfanhaider@mail.dlut.edu.cn](mailto:irfanhaider@mail.dlut.edu.cn)
