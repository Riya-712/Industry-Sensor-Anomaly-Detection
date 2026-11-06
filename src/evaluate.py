"""Shared evaluation utilities for ALL models (Isolation Forest, Random Forest, Autoencoder).

Convention: every model exposes a score where HIGHER = MORE ANOMALOUS.
  - Isolation Forest : -score_samples(X)
  - Random Forest    : predict_proba(X)[:, 1]
  - Autoencoder      : reconstruction error
A threshold is chosen on VALIDATION scores only; the test set is scored once at that threshold.
"""
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src import config as C
from src.data_preprocessing import load_processed

log = logging.getLogger(__name__)
REPORT_DIR = C.ROOT / "reports" / "phase3"
FIG_DIR = REPORT_DIR / "figures"


# ------------------------------------------------------------------ data
def load_experiment_data(feature_set="detrended", holdout=None):
    """{'train'|'val'|'test': (meta, X, y)}.
    holdout='EQ-005' -> train/val EXCLUDE that machine, test contains ONLY that machine
    (generalisation to unseen equipment). Model weights never see the machine; note the
    preprocessing statistics (mode medians) were still fitted on all five machines."""
    data = {}
    for s in ["train", "val", "test"]:
        meta, X, y = load_processed(s, feature_set)
        if holdout:
            keep = (meta[C.EQUIP] == holdout) if s == "test" else (meta[C.EQUIP] != holdout)
            meta, X, y = (meta[keep].reset_index(drop=True), X[keep].reset_index(drop=True),
                          y[keep].reset_index(drop=True))
        data[s] = (meta, X, y)
    return data


def run_tag(model, feature_set, holdout):
    return f"{model}_{feature_set}" + (f"_holdout-{holdout}" if holdout else "")


# ------------------------------------------------------------------ thresholds
def select_threshold(y_val, scores_val):
    """Threshold that maximises F1 on the VALIDATION set. Rule: predict anomaly if score >= thr."""
    p, r, thr = precision_recall_curve(y_val, scores_val)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def apply_threshold(scores, thr):
    return (np.asarray(scores) >= thr).astype(int)


# ------------------------------------------------------------------ metrics
def episode_metrics(meta, y_true, y_pred):
    """Event-level view. An episode = consecutive anomalous readings of one machine.
    Detected = at least one alert inside the episode; delay = minutes from onset to first alert."""
    d = pd.DataFrame({"eq": meta[C.EQUIP].values, "ts": pd.to_datetime(meta[C.TIME].values),
                      "y": np.asarray(y_true), "p": np.asarray(y_pred)}).sort_values(["eq", "ts"])
    n, det, delays = 0, 0, []
    for _, g in d.groupby("eq"):
        run_id = (g["y"] != g["y"].shift()).cumsum()
        for _, ep in g[g["y"] == 1].groupby(run_id[g["y"] == 1]):
            n += 1
            hit = ep[ep["p"] == 1]
            if len(hit):
                det += 1
                delays.append((hit["ts"].iloc[0] - ep["ts"].iloc[0]).total_seconds() / 60)
    return {"n_episodes": n, "episodes_detected": det,
            "episode_recall": det / n if n else float("nan"),
            "median_delay_min": float(np.median(delays)) if delays else float("nan"),
            "mean_delay_min": float(np.mean(delays)) if delays else float("nan")}


def compute_metrics(meta, y, scores, thr):
    pred = apply_threshold(scores, thr)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else float("nan"),   # FPR
        "missed_anomaly_rate": fn / (fn + tp) if (fn + tp) else float("nan"),  # FNR
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_rows": int(len(y)), "base_rate": float(np.mean(y)),
        "alerts_per_equipment": {k: int(v) for k, v in
                                 pd.Series(pred).groupby(meta[C.EQUIP].values).sum().items()},
    }
    out.update(episode_metrics(meta, y, pred))   # n_episodes, episodes_detected, episode_recall, *_delay_min
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


def measure_latency(score_fn, X, repeats=5):
    """Median wall-clock milliseconds to score 1,000 rows (batch inference)."""
    sample = X.iloc[:1000]
    times = []
    for _ in range(repeats):
        t = time.perf_counter(); score_fn(sample); times.append(time.perf_counter() - t)
    return float(np.median(times) / len(sample) * 1000 * 1000)


# ------------------------------------------------------------------ full evaluation + plots
def evaluate_model(name, score_fn, data):
    """Select threshold on val, report val & test. Returns (results_dict, scores_dict)."""
    scores = {s: score_fn(data[s][1]) for s in ["val", "test"]}
    thr = select_threshold(data["val"][2], scores["val"])
    res = {"threshold": thr, "metrics": {s: compute_metrics(data[s][0], data[s][2], scores[s], thr)
                                         for s in ["val", "test"]},
           "inference_ms_per_1000_rows": measure_latency(score_fn, data["test"][1])}
    save_plots(name, data["test"][2].values, scores["test"], thr)
    return res, scores


def save_plots(name, y, scores, thr):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    p, r, _ = precision_recall_curve(y, scores)
    ax[0, 0].plot(r, p); ax[0, 0].axhline(np.mean(y), ls="--", c="grey", label="base rate")
    ax[0, 0].set(title=f"PR curve (AP={average_precision_score(y, scores):.3f})", xlabel="Recall", ylabel="Precision")
    ax[0, 0].legend()
    fpr, tpr, _ = roc_curve(y, scores)
    ax[0, 1].plot(fpr, tpr); ax[0, 1].plot([0, 1], [0, 1], "--", c="grey")
    ax[0, 1].set(title=f"ROC (AUC={roc_auc_score(y, scores):.3f})", xlabel="FPR", ylabel="TPR")
    cm = confusion_matrix(y, apply_threshold(scores, thr), labels=[0, 1])
    ax[1, 0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1, 0].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=13)
    ax[1, 0].set(title="Confusion matrix (test)", xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"],
                 xlabel="Predicted", ylabel="Actual")
    ax[1, 1].hist(scores[y == 0], bins=60, alpha=.6, label="normal", density=True)
    ax[1, 1].hist(scores[y == 1], bins=60, alpha=.6, label="anomaly", density=True)
    ax[1, 1].axvline(thr, c="k", ls="--", label="threshold")
    ax[1, 1].set(title="Anomaly score distribution (test)", xlabel="score (higher = more anomalous)")
    ax[1, 1].legend()
    fig.suptitle(name); fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_eval.png", dpi=120); plt.close(fig)


def save_results(name, payload):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}_results.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


# ------------------------------------------------------------------ comparison table
def comparison_table():
    rows = []
    for f in sorted(REPORT_DIR.glob("*_results.json")):
        r = json.loads(f.read_text()); t, v = r["metrics"]["test"], r["metrics"]["val"]
        rows.append({"run": r["run"], "val_pr_auc": v["pr_auc"], "precision": t["precision"],
                     "recall": t["recall"], "f1": t["f1"], "pr_auc": t["pr_auc"], "roc_auc": t["roc_auc"],
                     "false_alarm_rate": t["false_alarm_rate"], "episode_recall": t["episode_recall"],
                     "median_delay_min": t["median_delay_min"],
                     "ms_per_1000_rows": r["inference_ms_per_1000_rows"],
                     "train_seconds": r["train_seconds"]})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(comparison_table().round(3).to_string(index=False))