# src/evaluation/monitoring.py

"""
Simple monitoring script:

- Compares new severity distribution vs baseline.
- Compares new RMSE vs baseline.
- Prints warnings if thresholds are exceeded.
"""

import json
from typing import Dict

import numpy as np
import pandas as pd


def load_baseline_stats(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def monitor_new_data(
    df_scored: pd.DataFrame,
    df_with_preds: pd.DataFrame,
    baseline_stats_path: str,
    severity_col: str = "severity",
    value_col: str = "Oxygen[%sat]",
    pred_col: str = "y_pred",
):
    stats = load_baseline_stats(baseline_stats_path)

    # Severity shift
    new_sev_median = float(df_scored[severity_col].median())
    base_sev_median = stats["severity_median"]
    sev_diff = new_sev_median - base_sev_median

    # RMSE shift
    residuals = df_with_preds[value_col] - df_with_preds[pred_col]
    new_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    base_rmse = stats["test_rmse"]

    print(f"Baseline severity median: {base_sev_median:.3f}")
    print(f"New severity median: {new_sev_median:.3f} (Δ={sev_diff:+.3f})")

    print(f"Baseline test RMSE: {base_rmse:.3f}")
    print(f"New RMSE:      {new_rmse:.3f} (Δ={new_rmse - base_rmse:+.3f})")

    if sev_diff > 0.1:
        print("[WARN] Severity median increased significantly. Possible data drift.")
    if new_rmse > base_rmse * 1.5:
        print("[WARN] RMSE increased >50% over baseline. Consider retraining.")
