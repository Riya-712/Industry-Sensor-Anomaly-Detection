"""Model 1 - Isolation Forest (unsupervised).
Fit on NORMAL training readings only. Hyperparameters are chosen by VALIDATION PR-AUC
(threshold-free), then the alert threshold is chosen on validation, then test is scored once.

    python -m src.train_isolation_forest --feature-set detrended
    python -m src.train_isolation_forest --feature-set full
    python -m src.train_isolation_forest --feature-set detrended --holdout EQ-005
"""
import argparse
import logging
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

from src import config as C
from src.evaluate import evaluate_model, load_experiment_data, run_tag, save_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("iforest")

# NOTE on `contamination`: it only shifts sklearn's built-in decision offset. We never use that
# offset - we rank by score_samples and pick our own threshold on validation data - so contamination
# cannot change the ranking, PR-AUC or ROC-AUC. We keep it at 'auto' and tune what matters.
GRID = {"n_estimators": [100, 300], "max_samples": [256, 1024, 4096], "max_features": [0.5, 1.0]}


def main(feature_set="detrended", holdout=None, quick=False):
    tag = run_tag("isolation_forest", feature_set, holdout)
    data = load_experiment_data(feature_set, holdout)
    (_, Xtr, ytr), (_, Xva, yva) = data["train"], data["val"]
    Xnorm = Xtr[ytr == 0]
    log.info("[%s] fitting on %d normal train rows, %d features", tag, len(Xnorm), Xnorm.shape[1])

    grid = list(ParameterGrid(GRID))[:2] if quick else list(ParameterGrid(GRID))
    best, t0 = None, time.perf_counter()
    for params in grid:
        m = IsolationForest(**params, contamination="auto", random_state=C.SEED, n_jobs=-1).fit(Xnorm)
        ap = average_precision_score(yva, -m.score_samples(Xva))
        log.info("  %s -> val PR-AUC %.4f", params, ap)
        if best is None or ap > best["val_pr_auc"]:
            best = {"params": params, "val_pr_auc": ap, "model": m}
    train_seconds = time.perf_counter() - t0

    model = best["model"]
    score_fn = lambda X: -model.score_samples(X)          # higher = more anomalous
    res, _ = evaluate_model(tag, score_fn, data)

    bundle = {"model": model, "feature_names": list(Xtr.columns), "feature_set": feature_set,
              "threshold": res["threshold"], "score_direction": "higher_is_more_anomalous",
              "model_type": "isolation_forest", "holdout": holdout}
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, C.MODELS_DIR / f"{tag}.joblib")
    save_results(tag, {"run": tag, "model": "isolation_forest", "feature_set": feature_set,
                       "holdout": holdout, "seed": C.SEED, "best_params": best["params"],
                       "search_space": GRID, "selection_metric": "val_pr_auc",
                       "train_rows": int(len(Xnorm)), "n_features": int(Xnorm.shape[1]),
                       "feature_names": list(Xtr.columns), "train_seconds": train_seconds, **res})
    t, v = res["metrics"]["test"], res["metrics"]["val"]
    print(f"\n[{tag}] best={best['params']}\n  val  PR-AUC={v['pr_auc']:.3f} F1={v['f1']:.3f}  thr={res['threshold']:.4f}"
          f"\n  test PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f} P={t['precision']:.3f} "
          f"R={t['recall']:.3f} F1={t['f1']:.3f} FAR={t['false_alarm_rate']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="detrended", choices=["detrended", "full"])
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke tests")
    a = ap.parse_args()
    main(a.feature_set, a.holdout, a.quick)