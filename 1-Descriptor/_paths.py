
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = Path(os.environ.get('QSPR_INPUT', str(ROOT / 'inputs' / 'dataset_combined.xlsx'))).expanduser().resolve()
BASELINE_RESULTS = ROOT / 'generated' / 'results_csv' / 'analysis'
READ_RESULTS = Path(os.environ.get('QSPR_RESULTS_DIR', str(BASELINE_RESULTS))).resolve()
OUTPUT_RESULTS = ROOT / 'generated' / 'results_csv' / 'analysis'
OUTPUT_FIGURES = ROOT / 'generated' / 'figures'


def prepare_outputs():
    OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
