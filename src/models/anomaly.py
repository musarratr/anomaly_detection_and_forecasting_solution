# src/models/anomaly.py

"""
Rule-based anomaly detection for oxygen [%sat]:

- Point anomalies: large deviation from rolling median (Z-score style)
- Collective anomalies: sustained high point-anomaly activity in a window
- Contextual anomalies: deviation from typical level by time-of-day
- Sensor fault anomalies:
    - stuck sensor: very low std over long window
    - spikes/glitches: large first-difference spikes
    - high noise: high short-window std
"""

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class AnomalyConfig:
    rolling_window_minutes: int = 60
    point_zscore_threshold: float = 3.0
    collective_window_minutes: int = 60
    collective_point_threshold: float = 0.7
    stuck_window_minutes: int = 120
    stuck_std_threshold: float = 0.05
    spike_zscore_threshold: float = 4.0
    noise_window_minutes: int = 30
    noise_std_multiplier: float = 2.0
    # Training-only hyperparameter (used in train_pipeline.py)
    severity_quantile_for_training_cutoff: float = 0.99


class OxygenAnomalyDetector:
    """Wrapper around rule-based anomaly detection pipeline."""

    def __init__(self, cfg: AnomalyConfig | None = None):
        self.cfg = cfg or AnomalyConfig()
        self.fitted_ = False
        self.context_baseline_: pd.Series | None = None  # by (sensor_id, hour_of_day)

    # ------------------------------------------------------------------ #
    # FIT: learn simple context baseline
    # ------------------------------------------------------------------ #
    def fit(
        self,
        df: pd.DataFrame,
        time_col: str,
        sensor_id_col: str,
        value_col: str,
    ):
        """
        'Fit' computes simple context baseline: median oxygen per sensor_id x hour_of_day.
        """
        df = df.copy()
        df["hour"] = df[time_col].dt.hour
        grp = df.groupby([sensor_id_col, "hour"])[value_col].median()
        self.context_baseline_ = grp
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------ #
    # Point anomalies
    # ------------------------------------------------------------------ #
    def _add_point_scores(
        self, df: pd.DataFrame, sensor_id_col: str, value_col: str
    ) -> pd.DataFrame:
        cfg = self.cfg
        df = df.copy()
        win = cfg.rolling_window_minutes

        # Rolling median & MAD per sensor
        grp = df.groupby(sensor_id_col)[value_col]
        rolling_median = (
            grp.rolling(win, min_periods=win // 2)
            .median()
            .reset_index(level=0, drop=True)
        )
        rolling_mad = grp.transform(
            lambda x: (np.abs(x - x.median()))
            .rolling(win, min_periods=win // 2)
            .median()
        )

        # Avoid division by zero
        rolling_mad = rolling_mad.replace(0, np.nan)
        z = (df[value_col] - rolling_median) / (1.4826 * rolling_mad)
        df["point_score"] = np.abs(z).fillna(0.0)

        # Normalise to [0,1] by threshold
        df["point_score_norm"] = np.clip(
            df["point_score"] / cfg.point_zscore_threshold, 0, 1
        )
        return df

    # ------------------------------------------------------------------ #
    # Collective anomalies
    # ------------------------------------------------------------------ #
    def _add_collective_scores(
        self, df: pd.DataFrame, sensor_id_col: str
    ) -> pd.DataFrame:
        cfg = self.cfg
        df = df.copy()
        win = cfg.collective_window_minutes

        # Collective score = rolling mean of (point_score_norm > some threshold)
        high_point = (df["point_score_norm"] > cfg.collective_point_threshold).astype(
            float
        )
        df["collective_score"] = (
            high_point.groupby(df[sensor_id_col])
            .rolling(win, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # Already in [0,1]
        return df

    # ------------------------------------------------------------------ #
    # Contextual anomalies
    # ------------------------------------------------------------------ #
    def _add_contextual_scores(
        self, df: pd.DataFrame, time_col: str, sensor_id_col: str, value_col: str
    ) -> pd.DataFrame:
        if self.context_baseline_ is None:
            df = df.copy()
            df["context_score"] = 0.0
            return df

        df = df.copy()
        df["hour"] = df[time_col].dt.hour

        ctx = self.context_baseline_
        baseline = []
        for sid, hour in zip(df[sensor_id_col].values, df["hour"].values):
            if (sid, hour) in ctx.index:
                baseline.append(ctx.loc[(sid, hour)])
            else:
                baseline.append(np.nan)
        df["context_baseline"] = baseline

        # Deviation from hourly median
        diff = np.abs(df[value_col] - df["context_baseline"])
        # Robust scale: median absolute deviation of diff
        scale = np.nanmedian(diff) or 1.0
        df["context_score"] = np.clip(diff / (scale * 3.0), 0, 1)  # roughly 3-MAD
        return df

    # ------------------------------------------------------------------ #
    # Sensor-fault anomalies
    # ------------------------------------------------------------------ #
    def _add_sensor_fault_scores(
        self, df: pd.DataFrame, sensor_id_col: str, time_col: str, value_col: str
    ) -> pd.DataFrame:
        cfg = self.cfg
        df = df.copy()

        # Stuck sensor: very low std over long window
        sw = cfg.stuck_window_minutes
        rolling_std_long = (
            df.groupby(sensor_id_col)[value_col]
            .rolling(sw, min_periods=sw // 2)
            .std()
            .reset_index(level=0, drop=True)
        )
        stuckness = 1.0 - np.clip(rolling_std_long / cfg.stuck_std_threshold, 0, 1)
        df["stuck_score"] = stuckness.fillna(0.0)

        # Spikes: large absolute first-difference vs std
        diff = df.groupby(sensor_id_col)[value_col].diff().fillna(0.0)
        diff_std = diff.groupby(df[sensor_id_col]).transform("std").replace(0, np.nan)
        spike_z = np.abs(diff) / diff_std
        spike_z = spike_z.replace(np.inf, np.nan).fillna(0.0)
        df["spike_score"] = np.clip(spike_z / cfg.spike_zscore_threshold, 0, 1)

        # High noise: high short-window std relative to long-window std
        nw = cfg.noise_window_minutes
        rolling_std_short = (
            df.groupby(sensor_id_col)[value_col]
            .rolling(nw, min_periods=nw // 2)
            .std()
            .reset_index(level=0, drop=True)
        )
        long_std = rolling_std_long.replace(0, np.nan)
        noise_ratio = rolling_std_short / long_std
        df["noise_score"] = np.clip(
            (noise_ratio - 1.0) / max(cfg.noise_std_multiplier - 1.0, 1e-6),
            0,
            1,
        ).fillna(0.0)

        # Aggregate sensor-fault score
        df["sensor_fault_score"] = np.clip(
            df[["stuck_score", "spike_score", "noise_score"]].max(axis=1), 0, 1
        )
        return df

    # ------------------------------------------------------------------ #
    # Combine into severity
    # ------------------------------------------------------------------ #
    def _combine_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine scores into a single severity in [0,1]."""
        df = df.copy()
        # Weighted sum, then squashed
        w_point = 0.4
        w_coll = 0.2
        w_ctx = 0.2
        w_fault = 0.2

        raw = (
            w_point * df["point_score_norm"]
            + w_coll * df["collective_score"]
            + w_ctx * df["context_score"]
            + w_fault * df["sensor_fault_score"]
        )
        df["severity"] = np.clip(raw, 0, 1)
        return df

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def score(
        self,
        df: pd.DataFrame,
        time_col: str,
        sensor_id_col: str,
        value_col: str,
    ) -> pd.DataFrame:
        """Run full anomaly pipeline and return scored frame."""
        if not self.fitted_:
            raise RuntimeError("Call `fit(df, ...)` before `score(df, ...)`.")

        df_scored = df.copy()
        df_scored = self._add_point_scores(df_scored, sensor_id_col, value_col)
        df_scored = self._add_collective_scores(df_scored, sensor_id_col)
        df_scored = self._add_contextual_scores(
            df_scored, time_col, sensor_id_col, value_col
        )
        df_scored = self._add_sensor_fault_scores(
            df_scored, sensor_id_col, time_col, value_col
        )
        df_scored = self._combine_scores(df_scored)
        return df_scored

    def to_dict(self) -> Dict:
        return self.cfg.__dict__

    @staticmethod
    def save_config(cfg: "AnomalyConfig", path: str) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, indent=2)
