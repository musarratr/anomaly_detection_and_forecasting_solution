"""
Lightweight model registry and promotion helpers.

Conventions:
- Production artefacts live under `models/` (anomaly config, forecaster, baseline stats).
- Each training run is stored under `models/registry/runs/<run_id>/`.
- Run metadata is captured in `run_info.json` and aggregate `models/registry/production.json`.

Promotion rule:
- A candidate is promoted only if it beats the current production `test_rmse`
  (lower is better). If no production model exists, the first candidate wins by default.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional


ARTIFACT_NAMES = {
    "anomaly_config": "anomaly_detector_config.json",
    "forecaster": "forecaster.pkl",
    "baseline_stats": "baseline_stats.json",
}


def _safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_dir(registry_dir: str, run_id: Optional[str] = None) -> str:
    run_id = run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(registry_dir, "runs", run_id)
    _ensure_dir(run_dir)
    return run_dir


def build_output_paths(run_dir: str) -> Dict[str, str]:
    return {
        "model_dir": run_dir,
        "anomaly_config_path": os.path.join(run_dir, ARTIFACT_NAMES["anomaly_config"]),
        "forecaster_path": os.path.join(run_dir, ARTIFACT_NAMES["forecaster"]),
        "baseline_stats_path": os.path.join(run_dir, ARTIFACT_NAMES["baseline_stats"]),
    }


def load_production_stats(model_dir: str) -> Optional[Dict[str, Any]]:
    return _safe_load_json(os.path.join(model_dir, ARTIFACT_NAMES["baseline_stats"]))


def candidate_beats_baseline(
    candidate_stats: Dict[str, Any],
    baseline_stats: Optional[Dict[str, Any]],
    metric: str = "test_rmse",
) -> bool:
    """
    Returns True if candidate wins, or if no baseline exists.
    Lower metric is better.
    """
    candidate_metric = candidate_stats.get(metric)
    if candidate_metric is None:
        return False
    if not baseline_stats:
        return True
    baseline_metric = baseline_stats.get(metric)
    if baseline_metric is None:
        return True
    return float(candidate_metric) < float(baseline_metric)


def promote_run(run_dir: str, model_dir: str) -> None:
    """Copy candidate artefacts into production model_dir."""
    _ensure_dir(model_dir)
    for fname in ARTIFACT_NAMES.values():
        src = os.path.join(run_dir, fname)
        dst = os.path.join(model_dir, fname)
        shutil.copy2(src, dst)


@dataclass
class RunInfo:
    run_id: str
    run_dir: str
    promoted: bool
    metric_used: str
    candidate_metric: float
    baseline_metric: Optional[float]
    source_config: str
    created_at: str
    output_paths: Dict[str, str]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def save_run_info(run_dir: str, info: RunInfo) -> None:
    path = os.path.join(run_dir, "run_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info.to_json(), f, indent=2)


def update_production_pointer(registry_dir: str, info: RunInfo) -> None:
    prod_path = os.path.join(registry_dir, "production.json")
    payload = {
        "current_run_id": info.run_id,
        "promoted_at": info.created_at,
        "metric_used": info.metric_used,
        "candidate_metric": info.candidate_metric,
        "baseline_metric": info.baseline_metric,
        "run_dir": info.run_dir,
    }
    _ensure_dir(registry_dir)
    with open(prod_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

