"""Causal imputation + feature engineering + scaling, packaged as one fit/transform object.

Design rules (each is enforced by a test in tests/test_phase2.py):
  1. fit() sees TRAINING rows only (medians, mode baselines, scaler).
  2. Every time-based feature uses the current and PAST readings of the same machine only.
  3. transform() is deterministic and returns the same columns, in the same order, every time.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config as C

log = logging.getLogger(__name__)
IMPUTE_COLS = C.SENSORS + C.CONTEXT_NUMERIC


def _assert_chronological(df: pd.DataFrame):
    if not df.groupby(C.EQUIP)[C.TIME].apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError("Rows must be sorted chronologically within each equipment unit")


class CausalImputer:
    """Forward-fill within each machine (past values only, limited), then fall back to
    the training median of the SAME operating mode, then the global training median."""

    def __init__(self, columns=IMPUTE_COLS, max_ffill=C.MAX_FFILL):
        self.columns, self.max_ffill = list(columns), max_ffill

    def fit(self, train_df):
        self.mode_medians_ = train_df.groupby(C.MODE)[self.columns].median()
        self.global_medians_ = train_df[self.columns].median()
        return self

    def mode_baseline(self, df, col):
        """Training median of `col` for each row's operating mode (global median if unseen mode)."""
        base = df[C.MODE].map(self.mode_medians_[col])
        return base.fillna(self.global_medians_[col])

    def transform(self, df):
        out = df.copy()
        out[self.columns] = out.groupby(C.EQUIP)[self.columns].ffill(limit=self.max_ffill)
        for c in self.columns:
            out[c] = out[c].fillna(self.mode_baseline(out, c))
        return out


class FeaturePipeline:
    def __init__(self, use_mode_residuals=True, roll_windows=C.ROLL_WINDOWS):
        self.use_mode_residuals = use_mode_residuals
        self.roll_windows = list(roll_windows)

    # ------------------------------------------------------------------ build
    def _build(self, df):
        d = self.imputer_.transform(df)
        g = d.groupby(C.EQUIP, sort=False)
        cols = {c: d[c] for c in C.SENSORS + C.CONTEXT_NUMERIC}

        # physically motivated differences
        cols["temp_minus_ambient"] = d["temperature_c"] - d["ambient_temp_c"]
        cols["temp_minus_coolant"] = d["temperature_c"] - d["coolant_temp_c"]

        # causal temporal features: trailing windows & backward differences only
        for s in C.ROLL_SENSORS:
            cols[f"{s}__diff1"] = g[s].diff().fillna(0.0)
            for w in self.roll_windows:
                r = g[s].rolling(w, min_periods=1)
                cols[f"{s}__rmean{w}"] = r.mean().droplevel(0)
                cols[f"{s}__rstd{w}"] = r.std().droplevel(0).fillna(0.0)

        # deviation from the training baseline of the current operating mode
        if self.use_mode_residuals:
            for s in C.SENSORS:
                cols[f"{s}__vs_mode"] = d[s] - self.imputer_.mode_baseline(d, s)

        # drift-adaptive deviation: mode residual minus the machine's own trailing median
        # of that residual (past only). Removes slow trends; episodes (<=63 readings) are
        # shorter than the 240-reading window so they stand out.
        if self.use_mode_residuals:
            for s in C.SENSORS:
                res = d[s] - self.imputer_.mode_baseline(d, s)
                med = (res.groupby(d[C.EQUIP]).rolling(C.RECENT_WINDOW, min_periods=C.RECENT_MIN_PERIODS)
                          .median().droplevel(0))
                dev = (res - med).fillna(0.0)
                cols[f"{s}__vs_recent"] = dev
                cols[f"{s}__vs_recent_rmean5"] = (dev.groupby(d[C.EQUIP]).rolling(5, min_periods=1)
                                                     .mean().droplevel(0))

        # fixed-category one-hot of operating mode (unseen mode -> all zeros)
        for m in C.OPERATING_MODES:
            cols[f"mode_{m}"] = (d[C.MODE] == m).astype(float)

        return pd.DataFrame(cols, index=d.index)

    # -------------------------------------------------------------------- API
    def fit(self, train_df):
        """train_df: chronologically sorted TRAINING rows (labels used only to pick
        normal rows for the scaler)."""
        _assert_chronological(train_df)
        self.imputer_ = CausalImputer().fit(train_df)
        X = self._build(train_df)
        self.feature_names_ = list(X.columns)
        normal = (train_df[C.TARGET] == 0).to_numpy()
        self.scaler_ = StandardScaler().fit(X.loc[normal])
        return self

    def transform(self, df):
        _assert_chronological(df)
        unknown = set(df[C.MODE].unique()) - set(C.OPERATING_MODES)
        if unknown:
            log.warning("Unseen operating modes %s -> all-zero one-hot, global-median baseline", unknown)
        return self._build(df)[self.feature_names_]

    def scale(self, X):
        return pd.DataFrame(self.scaler_.transform(X[self.feature_names_]),
                            index=X.index, columns=self.feature_names_)

    def feature_sets(self):
        """'full' = everything; 'detrended' = drift-robust subset (no raw/rolling LEVELS of drifting sensors)."""
        keep = [f for f in self.feature_names_ if (
            "__vs_recent" in f or f.endswith("__diff1") or "__rstd" in f
            or f.startswith("mode_") or f in C.STATIONARY_RAW)]
        return {"full": self.feature_names_, "detrended": keep}

    def export_manifest(self):
        return {
            "feature_names": self.feature_names_,
            "feature_sets": self.feature_sets(),
            "n_features": len(self.feature_names_),
            "excluded_columns": C.EXCLUDED,
            "config": {"roll_windows": self.roll_windows, "max_ffill": C.MAX_FFILL,
                       "roll_sensors": C.ROLL_SENSORS, "use_mode_residuals": self.use_mode_residuals,
                       "seed": C.SEED},
            "scaler": {"type": "StandardScaler", "fit_on": "normal training rows only",
                       "mean": self.scaler_.mean_.tolist(), "scale": self.scaler_.scale_.tolist()},
        }


def scale_with_manifest(X: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Re-apply the saved scaler anywhere (e.g. Colab) without unpickling sklearn objects."""
    names = manifest["feature_names"]
    mean, scale = np.array(manifest["scaler"]["mean"]), np.array(manifest["scaler"]["scale"])
    return pd.DataFrame((X[names].to_numpy() - mean) / scale, index=X.index, columns=names)