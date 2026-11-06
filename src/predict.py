"""Reusable inference module (used by the Streamlit app and usable from a notebook/CLI).

    from src.predict import Predictor
    p = Predictor("random_forest_detrended")
    result = p.predict(raw_dataframe)          # raw sensor rows in, predictions out

Pipeline: validate schema -> causal imputation + features (saved pipeline) -> model score
          -> threshold (chosen on VALIDATION data) -> severity band.
"""
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.data_preprocessing import validate_schema

log = logging.getLogger(__name__)
SEVERITY_FILE = "severity_cutoffs.json"


class Predictor:
    def __init__(self, run="random_forest_detrended", models_dir=C.MODELS_DIR, pipeline_path=None):
        self.run, self.models_dir = run, Path(models_dir)
        self.pipeline = joblib.load(pipeline_path or self.models_dir / "preprocessing_pipeline.joblib")
        if run.startswith("autoencoder"):
            self._load_autoencoder()
        else:
            b = joblib.load(self.models_dir / f"{run}.joblib")
            self.model, self.feature_names = b["model"], b["feature_names"]
            self.threshold, self.model_type = float(b["threshold"]), b["model_type"]
        self.severity_cutoffs = self._load_cutoffs()

    # ------------------------------------------------------------------ loading
    def _load_autoencoder(self):
        import torch  # imported lazily so sklearn-only users don't need torch
        from src.train_autoencoder import DEVICE, AutoEncoder
        cfg = json.loads((self.models_dir / "autoencoder_config.json").read_text())
        net = AutoEncoder(cfg["n_in"], cfg["hidden"], cfg["latent"])
        net.load_state_dict(torch.load(self.models_dir / "autoencoder_best.pt", map_location="cpu"))
        self.model = net.to(DEVICE).eval()
        self.feature_names, self.threshold, self.model_type = cfg["feature_names"], float(cfg["threshold"]), "autoencoder"
        self._ae_mean, self._ae_scale = np.array(cfg["scaler_mean"]), np.array(cfg["scaler_scale"])

    def _load_cutoffs(self):
        f = self.models_dir / SEVERITY_FILE
        if f.exists():
            entry = json.loads(f.read_text()).get(self.run)
            if entry:
                return entry["cutoffs"]
        return None

    # ------------------------------------------------------------------ scoring
    def score_features(self, X: pd.DataFrame) -> np.ndarray:
        """Anomaly score for already-engineered features. HIGHER = MORE ANOMALOUS."""
        if self.model_type == "random_forest":
            return self.model.predict_proba(X[self.feature_names])[:, 1]
        if self.model_type == "isolation_forest":
            return -self.model.score_samples(X[self.feature_names])
        from src.train_autoencoder import recon_error
        Xs = ((X[self.feature_names].to_numpy() - self._ae_mean) / self._ae_scale).astype(np.float32)
        return recon_error(self.model, Xs)

    def severity(self, scores) -> np.ndarray:
        """Low / Medium / High for flagged readings, 'None' otherwise.
        Cutoffs = tertiles of the scores of correctly flagged VALIDATION anomalies
        (computed once by src.register_models and checked there on the test period)."""
        scores = np.asarray(scores)
        sev = np.full(len(scores), "None", dtype=object)
        flagged = scores >= self.threshold
        if self.severity_cutoffs is None:
            sev[flagged] = "Unrated"
            return sev
        c1, c2 = self.severity_cutoffs
        sev[flagged & (scores < c1)] = "Low"
        sev[flagged & (scores >= c1) & (scores < c2)] = "Medium"
        sev[flagged & (scores >= c2)] = "High"
        return sev

    # ------------------------------------------------------------------ public API
    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """raw: dataframe with the sensor schema (labels optional). Raises ValueError on bad input."""
        if raw is None or len(raw) == 0:
            raise ValueError("The input contains no rows")
        df = validate_schema(raw, require_labels=False)
        X = self.pipeline.transform(df)
        t0 = time.perf_counter()
        scores = self.score_features(X)
        self.last_inference_seconds = time.perf_counter() - t0

        out = pd.DataFrame({"equipment_id": df[C.EQUIP].values, "timestamp": df[C.TIME].values,
                            "operating_mode": df[C.MODE].values, "anomaly_score": scores})
        out["predicted_status"] = np.where(scores >= self.threshold, "Anomaly", "Normal")
        out["anomaly_severity"] = self.severity(scores)
        # first readings of each machine have <RECENT_MIN_PERIODS of history -> less reliable
        out["in_warmup"] = (df.groupby(C.EQUIP).cumcount() < C.RECENT_MIN_PERIODS).values
        if C.TARGET in df:      # historical evaluation only - NOT a prediction
            out["actual_is_anomaly"] = df[C.TARGET].values
        if "anomaly_type" in df:
            out["actual_anomaly_type"] = df["anomaly_type"].values
        return out

    def predict_file(self, path) -> pd.DataFrame:
        return self.predict(pd.read_csv(path))