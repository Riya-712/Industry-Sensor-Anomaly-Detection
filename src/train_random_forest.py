"""Model 3 - Random Forest classifier (supervised benchmark).
Trained on is_anomaly with class weighting. Hyperparameters chosen by VALIDATION PR-AUC,
threshold chosen on validation, test scored once. Includes feature importance.

    python -m src.train_random_forest --feature-set detrended
    python -m src.train_random_forest --feature-set full
    python -m src.train_random_forest --feature-set detrended --holdout EQ-005
"""
import argparse
import logging
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from src import config as C
from src.evaluate import FIG_DIR, evaluate_model, load_experiment_data, run_tag, save_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("rf")

SPACE = {"n_estimators": [200, 300], "max_depth": [8, 16], "min_samples_leaf": [5, 20],
         "max_features": ["sqrt", 0.3], "class_weight": ["balanced_subsample", None]}
N_ITER = 8    # random search over 32 combos; a time-ordered split makes K-fold CV invalid,
              # so candidates are compared on the validation block instead


def main(feature_set="detrended", holdout=None, quick=False):
    tag = run_tag("random_forest", feature_set, holdout)
    data = load_experiment_data(feature_set, holdout)
    (_, Xtr, ytr), (_, Xva, yva) = data["train"], data["val"]
    log.info("[%s] %d train rows (%.2f%% anomalous), %d features", tag, len(Xtr), 100 * ytr.mean(), Xtr.shape[1])

    cands = list(ParameterSampler(SPACE, n_iter=2 if quick else N_ITER, random_state=C.SEED))
    best, t0 = None, time.perf_counter()
    for params in cands:
        m = RandomForestClassifier(**params, random_state=C.SEED, n_jobs=-1).fit(Xtr, ytr)
        ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        log.info("  %s -> val PR-AUC %.4f", params, ap)
        if best is None or ap > best["val_pr_auc"]:
            best = {"params": params, "val_pr_auc": ap, "model": m}
    train_seconds = time.perf_counter() - t0

    model = best["model"]
    score_fn = lambda X: model.predict_proba(X)[:, 1]
    res, _ = evaluate_model(tag, score_fn, data)

    # feature importance: impurity-based (fast, biased toward high-cardinality) AND permutation on VALIDATION
    imp = pd.Series(model.feature_importances_, index=Xtr.columns)
    perm = permutation_importance(model, Xva, yva, scoring="average_precision",
                                  n_repeats=2 if quick else 3, random_state=C.SEED, n_jobs=-1)
    perm = pd.Series(perm.importances_mean, index=Xtr.columns)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    imp.sort_values().tail(20).plot.barh(ax=ax[0], title="Impurity importance (top 20)")
    perm.sort_values().tail(20).plot.barh(ax=ax[1], title="Permutation importance on val (drop in PR-AUC)")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{tag}_importance.png", dpi=120); plt.close(fig)

    bundle = {"model": model, "feature_names": list(Xtr.columns), "feature_set": feature_set,
              "threshold": res["threshold"], "score_direction": "higher_is_more_anomalous",
              "model_type": "random_forest", "holdout": holdout}
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, C.MODELS_DIR / f"{tag}.joblib")
    save_results(tag, {"run": tag, "model": "random_forest", "feature_set": feature_set, "holdout": holdout,
                       "seed": C.SEED, "best_params": best["params"], "search_space": SPACE, "n_iter": len(cands),
                       "selection_metric": "val_pr_auc", "train_rows": int(len(Xtr)),
                       "n_features": int(Xtr.shape[1]), "feature_names": list(Xtr.columns),
                       "train_seconds": train_seconds,
                       "top_permutation_importance": perm.sort_values(ascending=False).head(10).round(5).to_dict(),
                       **res})
    t, v = res["metrics"]["test"], res["metrics"]["val"]
    print(f"\n[{tag}] best={best['params']}\n  val  PR-AUC={v['pr_auc']:.3f} F1={v['f1']:.3f}  thr={res['threshold']:.4f}"
          f"\n  test PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f} P={t['precision']:.3f} "
          f"R={t['recall']:.3f} F1={t['f1']:.3f} FAR={t['false_alarm_rate']:.4f}"
          f"\n  top permutation importance: {perm.sort_values(ascending=False).head(5).round(4).to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="detrended", choices=["detrended", "full"])
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny search for smoke tests")
    a = ap.parse_args()
    main(a.feature_set, a.holdout, a.quick)