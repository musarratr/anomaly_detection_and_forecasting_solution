"""
Monitoring helpers aligned with the model registry / promotion flow.

- Loads production baselines from `models/` (or an override path).
- Compares new severity distributions and RMSE vs the production baseline.
- Returns structured warnings instead of printing directly.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.registry import ARTIFACT_NAMES


def load_baseline_stats(
    model_dir: str | None = "models", baseline_stats_path: str | None = None
) -> Dict:
    path = baseline_stats_path or os.path.join(
        model_dir or "", ARTIFACT_NAMES["baseline_stats"]
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_production_metadata(registry_dir: str = "models/registry") -> Optional[Dict]:
    path = os.path.join(registry_dir, "production.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def monitor_new_data(
    df_scored: pd.DataFrame,
    df_with_preds: pd.DataFrame,
    model_dir: str = "models",
    registry_dir: str = "models/registry",
    severity_col: str = "severity",
    value_col: str = "oxygen",
    pred_col: str = "y_pred",
    severity_tol: float = 0.1,
    rmse_ratio_tol: float = 1.5,
) -> Dict:
    """
    Compare new batches to production baseline.

    Returns a dict with summary stats and warning messages.
    """
    stats = load_baseline_stats(model_dir)
    prod_meta = load_production_metadata(registry_dir)

    warnings: List[str] = []

    # Severity shift
    new_sev_median = float(df_scored[severity_col].median())
    base_sev_median = float(stats["severity_median"])
    sev_diff = new_sev_median - base_sev_median
    if sev_diff > severity_tol:
        warnings.append(
            "Severity median increased (possible data drift). "
            f"Δ={sev_diff:+.3f} vs tol {severity_tol}"
        )

    # RMSE shift
    residuals = df_with_preds[value_col] - df_with_preds[pred_col]
    new_rmse = float(np.sqrt(np.mean(residuals**2)))
    base_rmse = float(stats["test_rmse"])
    if new_rmse > base_rmse * rmse_ratio_tol:
        warnings.append(
            "RMSE degraded beyond tolerance. "
            f"new={new_rmse:.3f}, baseline={base_rmse:.3f}, "
            f"tol_ratio={rmse_ratio_tol}"
        )

    return {
        "production_run": prod_meta["current_run_id"] if prod_meta else None,
        "severity": {
            "baseline_median": base_sev_median,
            "new_median": new_sev_median,
            "delta": sev_diff,
            "tolerance": severity_tol,
        },
        "rmse": {
            "baseline": base_rmse,
            "new": new_rmse,
            "ratio": new_rmse / base_rmse if base_rmse else np.inf,
            "tolerance_ratio": rmse_ratio_tol,
        },
        "warnings": warnings,
    }
