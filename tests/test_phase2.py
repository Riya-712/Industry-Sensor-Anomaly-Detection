import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_preprocessing import assign_split, make_time_split, validate_schema
from src.feature_engineering import FeaturePipeline


def make_df(n_ts=600, n_eq=3, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_ts, freq="1min")
    parts = []
    for e in range(n_eq):
        d = pd.DataFrame({"reading_id": [f"R{e}_{i}" for i in range(n_ts)], "timestamp": ts,
                          "equipment_id": f"EQ-{e:03d}",
                          "operating_mode": rng.choice(C.OPERATING_MODES, n_ts, p=[.05, .7, .1, .1, .05])})
        for s in C.SENSORS:
            d[s] = rng.normal(50, 5, n_ts)
        d["ambient_temp_c"] = rng.normal(22, 1, n_ts)
        d["equipment_age_hours"] = 1000 + np.arange(n_ts) / 60
        d["hours_since_maintenance"] = np.arange(n_ts) / 60
        d["is_anomaly"] = (rng.random(n_ts) < 0.05).astype(int)
        d["anomaly_type"] = np.where(d.is_anomaly == 1, "x", "normal")
        d["rolling_anomaly_rate"], d["maintenance_priority_score"], d["alert_code"] = 0.1, 1.0, "NONE"
        parts.append(d)
    df = pd.concat(parts)
    for s in C.SENSORS[:4]:
        df.loc[rng.random(len(df)) < 0.03, s] = np.nan
    return df.sort_values(["timestamp", "equipment_id"], kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted():
    df = make_df()
    df["split"] = assign_split(df, make_time_split(df))
    pipe = FeaturePipeline().fit(df[df.split == "train"])
    return pipe, df


def test_schema_missing_column_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(make_df().drop(columns=["temperature_c"]))


def test_schema_bad_label_and_duplicates_raise():
    df = make_df()
    bad = df.copy(); bad.loc[0, "is_anomaly"] = 2
    with pytest.raises(ValueError, match="0 and 1"):
        validate_schema(bad)
    with pytest.raises(ValueError, match="duplicated"):
        validate_schema(pd.concat([df, df.iloc[[0]]]))


def test_split_is_chronological_and_timestamp_atomic(fitted):
    _, df = fitted
    t = {k: df[df.split == k].timestamp for k in ["train", "val", "test"]}
    assert t["train"].max() < t["val"].min() and t["val"].max() < t["test"].min()
    assert not set(t["train"]) & set(t["val"]) and not set(t["val"]) & set(t["test"])
    assert abs(len(t["train"]) / len(df) - 0.70) < 0.02


def test_no_excluded_columns_in_features(fitted):
    pipe, _ = fitted
    assert not set(pipe.feature_names_) & set(C.EXCLUDED)


def test_no_nans_after_transform(fitted):
    pipe, df = fitted
    X = pipe.transform(df)
    assert not X.isna().any().any() and np.isfinite(X.to_numpy()).all()


def test_features_are_causal(fitted):
    """Changing FUTURE readings must not change features of earlier rows."""
    pipe, df = fitted
    cut = np.sort(df.timestamp.unique())[400]
    df2 = df.copy()
    fut = df2.timestamp > cut
    for s in C.SENSORS:
        df2.loc[fut, s] = df2.loc[fut, s] * 10 + 99
    past = (df.timestamp <= cut).to_numpy()
    pd.testing.assert_frame_equal(pipe.transform(df)[past], pipe.transform(df2)[past])


def test_transform_is_split_invariant(fitted):
    pipe, df = fitted
    train = df[df.split == "train"]
    pd.testing.assert_frame_equal(pipe.transform(train), pipe.transform(df).loc[train.index])


def test_scaler_fit_on_normal_train_rows_only(fitted):
    pipe, df = fitted
    train = df[df.split == "train"]
    X = pipe.transform(train)
    normal = X[train.is_anomaly == 0]
    np.testing.assert_allclose(pipe.scaler_.mean_, normal.mean().to_numpy(), rtol=1e-6, atol=1e-9)


def test_detrended_set_excludes_levels_of_drifting_sensors(fitted):
    pipe, _ = fitted
    sets = pipe.feature_sets()
    assert set(sets["detrended"]) <= set(sets["full"])
    for s in ["temperature_c", "coolant_temp_c", "acoustic_level_db", "oil_viscosity_cst", "vibration_mm_s"]:
        assert s not in sets["detrended"] and f"{s}__rmean5" not in sets["detrended"]
    assert "temperature_c__vs_recent" in sets["detrended"]


def test_recent_deviation_removes_linear_drift():
    """Real data drifts ~0.007 C/min; use 0.01 C/min. A 240-reading trailing median lags such a
    ramp by only ~1.2 C, while the raw level moves by ~16 C between early and late periods."""
    df = make_df(n_ts=2000, n_eq=2)
    df["split"] = assign_split(df, make_time_split(df))
    t = (df.timestamp - df.timestamp.min()).dt.total_seconds() / 60
    df["temperature_c"] = df["temperature_c"] + 0.01 * t
    pipe = FeaturePipeline().fit(df[df.split == "train"])
    X = pipe.transform(df)
    late, early = X[(t > 1500).to_numpy()], X[(t < 300).to_numpy()]
    assert late["temperature_c"].mean() - early["temperature_c"].mean() > 10      # raw level drifts
    assert abs(late["temperature_c__vs_recent"].mean()) < 2.5                     # deviation does not
