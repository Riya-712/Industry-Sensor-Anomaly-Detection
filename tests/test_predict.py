import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import config as C
from src.data_preprocessing import assign_split, make_time_split
from src.feature_engineering import FeaturePipeline
from src.predict import Predictor
from tests.test_phase2 import make_df

THR = 0.3


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    df = make_df(n_ts=800)
    df["split"] = assign_split(df, make_time_split(df))
    train = df[df.split == "train"]
    pipe = FeaturePipeline().fit(train)
    joblib.dump(pipe, d / "preprocessing_pipeline.joblib")
    feats = pipe.feature_sets()["detrended"]
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(pipe.transform(train)[feats], train[C.TARGET])
    joblib.dump({"model": rf, "feature_names": feats, "threshold": THR, "model_type": "random_forest"}, d / "rf_test.joblib")
    return d, df.drop(columns=["split"])


def test_output_schema_and_row_count(setup):
    d, raw = setup
    out = Predictor("rf_test", models_dir=d).predict(raw)
    assert len(out) == len(raw)
    for col in ["equipment_id", "timestamp", "anomaly_score", "predicted_status", "anomaly_severity", "in_warmup"]:
        assert col in out
    assert set(out.predicted_status) <= {"Normal", "Anomaly"}


def test_threshold_rule_and_severity_for_normal_rows(setup):
    d, raw = setup
    out = Predictor("rf_test", models_dir=d).predict(raw)
    assert ((out.predicted_status == "Anomaly") == (out.anomaly_score >= THR)).all()
    assert (out.loc[out.predicted_status == "Normal", "anomaly_severity"] == "None").all()


def test_severity_bands_with_cutoffs(setup):
    d, _ = setup
    (d / "severity_cutoffs.json").write_text(json.dumps({"rf_test": {"cutoffs": [0.5, 0.8]}}))
    p = Predictor("rf_test", models_dir=d)
    assert p.severity([0.1, 0.35, 0.6, 0.9]).tolist() == ["None", "Low", "Medium", "High"]
    (d / "severity_cutoffs.json").unlink()


def test_row_order_does_not_change_predictions(setup):
    d, raw = setup
    p = Predictor("rf_test", models_dir=d)
    a = p.predict(raw)
    b = p.predict(raw.sample(frac=1, random_state=0))
    np.testing.assert_allclose(a.anomaly_score.values, b.anomaly_score.values)


def test_warmup_flag_and_optional_labels(setup):
    d, raw = setup
    out = Predictor("rf_test", models_dir=d).predict(raw.drop(columns=[C.TARGET, "anomaly_type"]))
    assert "actual_is_anomaly" not in out
    first = out.groupby("equipment_id").head(C.RECENT_MIN_PERIODS)
    assert first.in_warmup.all() and not out.drop(first.index).in_warmup.any()


def test_invalid_inputs_raise_value_error(setup):
    d, raw = setup
    p = Predictor("rf_test", models_dir=d)
    with pytest.raises(ValueError, match="Missing required columns"):
        p.predict(raw.drop(columns=["vibration_mm_s"]))
    with pytest.raises(ValueError, match="no rows"):
        p.predict(raw.iloc[0:0])
    bad = raw.copy(); bad["timestamp"] = bad["timestamp"].astype(str); bad.loc[0, "timestamp"] = "not a date"
    with pytest.raises(ValueError, match="timestamps"):
        p.predict(bad)
