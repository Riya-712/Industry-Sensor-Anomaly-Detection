import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config as C
from src.evaluate import apply_threshold, compute_metrics, episode_metrics, select_threshold


def test_threshold_rule_is_score_greater_or_equal():
    assert apply_threshold([0.1, 0.5, 0.9], 0.5).tolist() == [0, 1, 1]


def test_select_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([.1, .2, .3, .4, .8, .9])
    thr = select_threshold(y, s)
    assert apply_threshold(s, thr).tolist() == y.tolist()


def _meta(n, eq="E1"):
    return pd.DataFrame({C.EQUIP: eq, C.TIME: pd.date_range("2024-01-01", periods=n, freq="1min")})


def test_compute_metrics_counts():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    s = np.array([.1, .2, .9, .1, .8, .3, .1, .9])
    m = compute_metrics(_meta(8), y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3)
    assert np.isclose(m["false_alarm_rate"], 1 / 5) and np.isclose(m["missed_anomaly_rate"], 1 / 3)


def test_episode_metrics_detection_and_delay():
    # two episodes: rows 2-5 (alert first at row 4 -> delay 2 min) and rows 8-9 (never alerted)
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    p = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    e = episode_metrics(_meta(10), y, p)
    assert e["n_episodes"] == 2 and e["episodes_detected"] == 1
    assert e["episode_recall"] == 0.5 and e["median_delay_min"] == 2.0


def test_model_bundle_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    m = IsolationForest(n_estimators=20, random_state=0).fit(X)
    path = tmp_path / "b.joblib"
    joblib.dump({"model": m, "feature_names": list(X.columns), "threshold": 0.5}, path)
    b = joblib.load(path)
    np.testing.assert_allclose(-b["model"].score_samples(X[b["feature_names"]]), -m.score_samples(X))
