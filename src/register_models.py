"""Phase 5 - MLflow tracking + best-model selection.

    python -m src.register_models

1. Reads every reports/phase3/*_results.json (IF, RF, Autoencoder).
2. Picks the best model by VALIDATION PR-AUC (never by test score, never by accuracy).
3. Computes severity cut-offs per model and checks them on the test period.
4. Logs every model as an MLflow run (params, metrics, figures, model files, preprocessing).
5. Registers the best scikit-learn model in the MLflow model registry (if supported)
   and writes models/best_model.json for the app.

View results:  mlflow ui --backend-store-uri sqlite:///mlflow.db     (then open http://127.0.0.1:5000)
"""
import json
import logging
import math

import mlflow
import numpy as np

from src import config as C
from src.data_preprocessing import load_json
from src.evaluate import FIG_DIR, REPORT_DIR, load_experiment_data
from src.predict import SEVERITY_FILE, Predictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("register")
EXPERIMENT = "industrial-anomaly-detection"
REGISTERED_NAME = "industrial-anomaly-detector"
METRICS = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_alarm_rate",
           "missed_anomaly_rate", "episode_recall", "median_delay_min"]


def severity_calibration(run, feature_set):
    """Cut-offs from validation; precision per severity band on test as a sanity check."""
    data = load_experiment_data(feature_set)
    pred = Predictor(run)
    sv, st = pred.score_features(data["val"][1]), pred.score_features(data["test"][1])
    yv, yt = data["val"][2].values, data["test"][2].values
    good = sv[(yv == 1) & (sv >= pred.threshold)]
    if len(good) < 6:
        return None
    c1, c2 = [float(q) for q in np.quantile(good, [1 / 3, 2 / 3])]
    bands = {"Low": (pred.threshold, c1), "Medium": (c1, c2), "High": (c2, np.inf)}
    prec = {}
    for name, (lo, hi) in bands.items():
        m = (st >= lo) & (st < hi)
        prec[name] = {"alerts": int(m.sum()), "precision": round(float(yt[m].mean()), 4) if m.sum() else None}
    return {"cutoffs": [c1, c2], "basis": "tertiles of correctly flagged validation anomalies",
            "test_precision_by_band": prec}


def clean(x):
    return None if (isinstance(x, float) and math.isnan(x)) else x


def main():
    results = {}
    for f in sorted(REPORT_DIR.glob("*_results.json")):
        r = json.loads(f.read_text()); results[r["run"]] = r
    if not results:
        raise SystemExit("No results found - run the training scripts first.")
    best_run = max(results, key=lambda k: results[k]["metrics"]["val"]["pr_auc"])
    log.info("Models found: %s", list(results))
    log.info("BEST by validation PR-AUC: %s (%.3f)", best_run, results[best_run]["metrics"]["val"]["pr_auc"])

    # severity cut-offs (needs torch only if the autoencoder is present)
    sev = {}
    for run, r in results.items():
        try:
            sev[run] = severity_calibration(run, r["feature_set"])
        except Exception as e:   # e.g. torch missing for the autoencoder
            log.warning("Severity calibration skipped for %s: %s", run, e)
    sev = {k: v for k, v in sev.items() if v}
    (C.MODELS_DIR / SEVERITY_FILE).write_text(json.dumps(sev, indent=2))

    # ---------------------------------------------------------------- MLflow
    mlflow.set_tracking_uri("sqlite:///" + (C.ROOT / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(C.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    split_manifest = load_json(C.SPLIT_MANIFEST)

    for run, r in results.items():
        with mlflow.start_run(run_name=run) as active:
            mlflow.set_tags({"model_family": r["model"], "selected": str(run == best_run).lower(),
                             "feature_set": r["feature_set"]})
            hp = r.get("best_params") or r.get("best_config") or {}
            mlflow.log_params({"model": r["model"], "feature_set": r["feature_set"], "seed": r["seed"],
                               "n_features": r["n_features"], "train_rows": r["train_rows"],
                               "selection_metric": r["selection_metric"],
                               "dataset_sha256": split_manifest["csv_sha256"][:16],
                               "val_start": split_manifest["boundaries"]["val_start"],
                               "test_start": split_manifest["boundaries"]["test_start"],
                               **{f"hp_{k}": str(v) for k, v in hp.items()}})
            mlflow.log_metric("threshold", r["threshold"])
            mlflow.log_metric("train_seconds", r["train_seconds"])
            mlflow.log_metric("inference_ms_per_1000_rows", r["inference_ms_per_1000_rows"])
            for split in ("val", "test"):
                for k in METRICS:
                    v = clean(r["metrics"][split].get(k))
                    if v is not None:
                        mlflow.log_metric(f"{split}_{k}", float(v))
            for h in r.get("loss_history", []):     # autoencoder training curves
                mlflow.log_metric("train_loss", h["train_loss"], step=h["epoch"])
                mlflow.log_metric("val_loss", h["val_loss"], step=h["epoch"])

            mlflow.log_dict({"features": r["feature_names"]}, "features.json")
            mlflow.log_artifact(str(REPORT_DIR / f"{run}_results.json"), "results")
            for fig in FIG_DIR.glob("*.png"):
                if fig.name.startswith(run) or (r["model"] == "autoencoder" and fig.name.startswith("autoencoder")):
                    mlflow.log_artifact(str(fig), "figures")
            for p in (C.PIPELINE_PATH, C.SPLIT_MANIFEST, C.FEATURE_MANIFEST):
                mlflow.log_artifact(str(p), "preprocessing")
            if run in sev:
                mlflow.log_dict(sev[run], "severity_calibration.json")

            if r["model"] == "autoencoder":
                for name in ("autoencoder_best.pt", "autoencoder_config.json"):
                    if (C.MODELS_DIR / name).exists():
                        mlflow.log_artifact(str(C.MODELS_DIR / name), "model_files")
            else:
                import joblib
                bundle = joblib.load(C.MODELS_DIR / f"{run}.joblib")
                mlflow.log_artifact(str(C.MODELS_DIR / f"{run}.joblib"), "model_files")
                try:
                    # cloudpickle: MLflow 3's default 'skops' format refuses tree models; these are
                    # our own trained models, so pickling is appropriate.
                    kw = {"serialization_format": "cloudpickle"}
                    try:
                        info = mlflow.sklearn.log_model(bundle["model"], name="model", **kw)
                    except TypeError:                       # older MLflow
                        info = mlflow.sklearn.log_model(bundle["model"], artifact_path="model", **kw)
                    if run == best_run:
                        v = mlflow.register_model(info.model_uri, REGISTERED_NAME)
                        log.info("Registered %s version %s", REGISTERED_NAME, v.version)
                except Exception as e:
                    log.warning("MLflow model logging/registry skipped for %s: %s", run, e)
            log.info("logged run %s (id %s)", run, active.info.run_id[:8])

    best = results[best_run]
    (C.MODELS_DIR / "best_model.json").write_text(json.dumps({
        "run": best_run, "selection_rule": "highest validation PR-AUC",
        "val_pr_auc": best["metrics"]["val"]["pr_auc"], "threshold": best["threshold"],
        "test_metrics": {k: best["metrics"]["test"][k] for k in METRICS if k in best["metrics"]["test"]},
        "severity": sev.get(best_run), "registered_model": REGISTERED_NAME,
        "all_models_val_pr_auc": {k: v["metrics"]["val"]["pr_auc"] for k, v in results.items()}}, indent=2, default=float))
    print(f"\nBest model: {best_run}  (val PR-AUC {best['metrics']['val']['pr_auc']:.3f})")
    if best_run in sev:
        print("Severity bands - test precision by band (should rise Low -> High):")
        for b, v in sev[best_run]["test_precision_by_band"].items():
            print(f"  {b:6s} alerts={v['alerts']:4d}  precision={v['precision']}")
    print("\nOpen MLflow UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
