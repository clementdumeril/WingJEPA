"""Print the results table of the README from results/summary.csv (no data or model needed).

python scripts/results_table.py

Each cell: mean over the seeds, with (min-max) when there are several. Horizons are medians over the flights of
each run, in seconds; position error is the RMS 3D distance after 5 s, in meters, training-range aircraft.
"""
import csv
import statistics
import sys
from pathlib import Path

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary.csv"

MODELS = [  # (label, calm-air run prefix, turbulent run prefix)
    ("A0: direct state loss", "a0_calmT100", "a0_v2T100"),
    ("A: JEPA, decoder on the frozen latent", "jepa_calmT100", "jepa_v2T100"),
    ("A: JEPA pre-training, then fine-tuning", "jepa_ft_calmT100", "jepa_ft_v2T100"),
    ("C-JEPA: physics head on the frozen JEPA trunk", "c_jepa_calmT100_ajuste", "c_jepa_v2T100_ajuste"),
    ("C-direct: physics head, free correction", "c_direct_calmT100_ajuste", "c_direct_v2T100_ajuste"),
    ("C-direct: physics head, constrained", "c_direct_calmT100_ajuste_param", "c_direct_v2T100_ajuste_param"),
]
COLUMNS = [  # (header, condition, column of summary.csv, decimals)
    ("Calm: held-out flights (s)", "calm", "model_horizon_s_test_flights", 2),
    ("Calm: extreme aircraft (s)", "calm", "model_horizon_s_extreme_aircraft", 2),
    ("Calm: position error at 5 s (m)", "calm", "model_pos_rms_5s_m_train_aircraft", 1),
    ("Turbulence: held-out flights (s)", "turbulent", "model_horizon_s_test_flights", 2),
]
BASELINE = {  # the constant-state baseline, read from the A0 runs of each condition
    "Calm: held-out flights (s)": ("a0_calmT100", "ref_constant_state_horizon_s_test_flights"),
    "Calm: extreme aircraft (s)": ("a0_calmT100", "ref_constant_state_horizon_s_extreme_aircraft"),
    "Calm: position error at 5 s (m)": ("a0_calmT100", "ref_constant_state_pos_rms_5s_m_train_aircraft"),
    "Turbulence: held-out flights (s)": ("a0_v2T100", "ref_constant_state_horizon_s_test_flights"),
}


def runs(rows, prefix):
    # exact prefix + _s<seed>, so that "c_direct_calmT100_ajuste" does not also match "..._ajuste_param"
    return [r for r in rows if r["run"].rsplit("_s", 1)[0] == prefix]


def cell(values, d):
    values = [float(v) for v in values if v != ""]
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.{d}f} (1 seed)"
    return f"{statistics.mean(values):.{d}f} ({min(values):.{d}f}-{max(values):.{d}f})"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8")))
    print("| Model | " + " | ".join(c[0] for c in COLUMNS) + " |")
    print("| --- |" + " --- |" * len(COLUMNS))
    base = [cell([runs(rows, BASELINE[h][0])[0][BASELINE[h][1]]], d).replace(" (1 seed)", "") for h, _, _, d in COLUMNS]
    print("| Baseline: constant state | " + " | ".join(base) + " |")
    for label, calm, turbulent in MODELS:
        cells = [cell([r[col] for r in runs(rows, calm if cond == "calm" else turbulent)], d) for _, cond, col, d in COLUMNS]
        print(f"| {label} | " + " | ".join(cells) + " |")
