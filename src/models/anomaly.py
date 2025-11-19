# src/models/anomaly.py

"""
Rule-based anomaly detector aligned with the exploratory notebook
`oxygen_anomaly_detector_analysis.ipynb`.

Components (all operate per sensor):
- Baseline + point anomalies: rolling mean/std z-score scaled to [0,1].
- Collective anomalies: rolling mean(|z|) scaled between low/high thresholds.
- Contextual anomalies: deviation from global hour-of-day baseline.
- Sensor faults: stuck sensor, spikes/glitches, and high noise.

Severity is the max of the sub-scores (point, collective, contextual, sensor_fault).
"""

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class AnomalyConfig:
    # Rolling windows (in minutes)
    roll_window_baseline: int = 60
    roll_window_collective: int = 120
    roll_window_stuck: int = 60
    roll_window_noise: int = 60

    # Point/contextual z-score thresholds for score scaling
    z_point_low: float = 2.0
    z_point_high: float = 5.0
    z_ctx_low: float = 2.0
    z_ctx_high: float = 4.0

    # Collective anomaly thresholds (mean |z| over window)
    collective_low: float = 1.0
    collective_high: float = 3.0

    # Sensor fault parameters
    stuck_rel_std_factor: float = 0.1
    noise_factor: float = 1.5
    spike_z_threshold: float = 3.0

    # Numerical stability
    eps: float = 1e-6

    # Training-only hyperparameter (used in train_pipeline.py)
    severity_quantile_for_training_cutoff: float = 0.99


def _squash_z(z: pd.Series, low: float, high: float) -> pd.Series:
    """
    Map |z| to [0, 1] linearly between `low` and `high`.

    - |z| <= low => 0
    - |z| >= high => 1
    - low < |z| < high => linear ramp
    """
    az = z.abs()
    return ((az - low) / (high - low)).clip(lower=0.0, upper=1.0)


def _prepare_frame(
    df: pd.DataFrame, time_col: str, sensor_id_col: str, value_col: str
) -> pd.DataFrame:
    """Clean and enrich the frame with hour/dayofweek for contextual scores."""
    df_prep = df.copy()
    if sensor_id_col not in df_prep.columns:
        df_prep[sensor_id_col] = "global"

    df_prep[time_col] = pd.to_datetime(df_prep[time_col], errors="coerce")
    df_prep = df_prep[
        df_prep[time_col].notna() & df_prep[value_col].notna()
    ].copy()

    df_prep["hour"] = df_prep[time_col].dt.hour
    df_prep["dayofweek"] = df_prep[time_col].dt.dayofweek
    df_prep = df_prep.sort_values([sensor_id_col, time_col]).reset_index(drop=True)
    return df_prep


def _add_baseline_and_point_scores(
    df: pd.DataFrame, cfg: AnomalyConfig, sensor_id_col: str, value_col: str
) -> pd.DataFrame:
    """
    Rolling baseline (mean/std) per sensor and point anomaly score from z-score.
    """
    df = df.copy()
    g = df.groupby(sensor_id_col, group_keys=False)

    df["roll_mean"] = g[value_col].transform(
        lambda s: s.rolling(
            window=cfg.roll_window_baseline,
            min_periods=cfg.roll_window_baseline // 2,
        ).mean()
    )
    df["roll_std"] = g[value_col].transform(
        lambda s: s.rolling(
            window=cfg.roll_window_baseline,
            min_periods=cfg.roll_window_baseline // 2,
        ).std()
    )

    df["z_global"] = (df[value_col] - df["roll_mean"]) / (df["roll_std"] + cfg.eps)
    df["score_point"] = _squash_z(df["z_global"], cfg.z_point_low, cfg.z_point_high)
    return df


def _add_collective_scores(
    df: pd.DataFrame, cfg: AnomalyConfig, sensor_id_col: str
) -> pd.DataFrame:
    """Collective anomaly score: rolling mean(|z_global|) scaled to [0,1]."""
    df = df.copy()
    g = df.groupby(sensor_id_col, group_keys=False)

    df["roll_mean_abs_z"] = g["z_global"].transform(
        lambda s: s.abs().rolling(
            window=cfg.roll_window_collective,
            min_periods=cfg.roll_window_collective // 2,
        ).mean()
    )

    df["score_collective"] = (
        (df["roll_mean_abs_z"] - cfg.collective_low)
        / (cfg.collective_high - cfg.collective_low)
    ).clip(lower=0.0, upper=1.0)
    return df


def _add_contextual_scores(
    df: pd.DataFrame, cfg: AnomalyConfig, value_col: str
) -> pd.DataFrame:
    """Contextual score: deviation from hour-of-day baseline across all sensors."""
    df = df.copy()
    ctx_stats = (
        df.groupby("hour")[value_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "ctx_mean_hour", "std": "ctx_std_hour"})
    )

    df = df.join(ctx_stats, on="hour")

    df["z_context"] = (df[value_col] - df["ctx_mean_hour"]) / (
        df["ctx_std_hour"] + cfg.eps
    )
    df["score_context"] = _squash_z(df["z_context"], cfg.z_ctx_low, cfg.z_ctx_high)
    return df


def _add_sensor_fault_scores(
    df: pd.DataFrame, cfg: AnomalyConfig, sensor_id_col: str, value_col: str
) -> pd.DataFrame:
    """
    Sensor fault scores: stuck sensor, spikes/glitches, and high noise.
    """
    df = df.copy()
    g = df.groupby(sensor_id_col, group_keys=False)

    sensor_std = g[value_col].transform("std")
    sensor_diff_std = g[value_col].transform(lambda s: s.diff().std())

    df["roll_std_long"] = g[value_col].transform(
        lambda s: s.rolling(
            window=cfg.roll_window_stuck,
            min_periods=cfg.roll_window_stuck // 2,
        ).std()
    )

    ratio_std = df["roll_std_long"] / (sensor_std + cfg.eps)
    df["score_stuck"] = (
        (cfg.stuck_rel_std_factor - ratio_std) / cfg.stuck_rel_std_factor
    ).clip(lower=0.0, upper=1.0)

    diff_prev = g[value_col].diff()
    diff_next = -g[value_col].diff(-1)
    spike_mag = np.minimum(diff_prev.abs(), diff_next.abs())
    spike_norm = spike_mag / (sensor_diff_std + cfg.eps)
    candidate_spike = (diff_prev * diff_next < 0) & (
        spike_norm > cfg.spike_z_threshold
    )

    df["score_spike"] = 0.0
    df.loc[candidate_spike, "score_spike"] = (
        (spike_norm[candidate_spike] - cfg.spike_z_threshold)
        / cfg.spike_z_threshold
    ).clip(upper=1.0)

    df["roll_std_noise"] = g[value_col].transform(
        lambda s: s.rolling(
            window=cfg.roll_window_noise,
            min_periods=cfg.roll_window_noise // 2,
        ).std()
    )
    noise_ratio = df["roll_std_noise"] / (sensor_std + cfg.eps)
    df["score_noise"] = ((noise_ratio - 1.0) / (cfg.noise_factor - 1.0)).clip(
        lower=0.0, upper=1.0
    )

    df["score_sensor_fault"] = df[
        ["score_stuck", "score_spike", "score_noise"]
    ].max(axis=1)
    return df


def _add_severity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Severity is the max of the sub-scores (matches the notebook)."""
    df = df.copy()
    score_cols = [
        "score_point",
        "score_collective",
        "score_context",
        "score_sensor_fault",
    ]
    df["severity"] = df[score_cols].max(axis=1)
    return df


class OxygenAnomalyDetector:
    """Wrapper around the anomaly pipeline used in the notebook."""

    def __init__(self, cfg: AnomalyConfig | None = None):
        self.cfg = cfg or AnomalyConfig()
        self.fitted_ = False

    def fit(
        self,
        df: pd.DataFrame,
        time_col: str,
        sensor_id_col: str,
        value_col: str,
    ):
        """
        Rule-based detector has no trainable parameters; keep for API symmetry.
        """
        _ = _prepare_frame(df, time_col, sensor_id_col, value_col)
        self.fitted_ = True
        return self

    def score(
        self,
        df: pd.DataFrame,
        time_col: str,
        sensor_id_col: str,
        value_col: str,
    ) -> pd.DataFrame:
        """Run the full anomaly pipeline and return the scored frame."""
        if not self.fitted_:
            raise RuntimeError("Call `fit(df, ...)` before `score(df, ...)`.")

        cfg = self.cfg
        df_proc = _prepare_frame(df, time_col, sensor_id_col, value_col)

        df_scored = _add_baseline_and_point_scores(
            df_proc, cfg, sensor_id_col, value_col
        )
        df_scored = _add_collective_scores(df_scored, cfg, sensor_id_col)
        df_scored = _add_contextual_scores(df_scored, cfg, value_col)
        df_scored = _add_sensor_fault_scores(
            df_scored, cfg, sensor_id_col, value_col
        )
        df_scored = _add_severity_score(df_scored)
        return df_scored

    def to_dict(self) -> Dict:
        return self.cfg.__dict__

    @staticmethod
    def save_config(cfg: "AnomalyConfig", path: str) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, indent=2)
