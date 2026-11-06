"""Phase 2 entry point:  python -m src.build_dataset
Validates -> splits chronologically -> fits the pipeline on TRAIN -> writes processed files + manifests."""
import logging

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.data_preprocessing import (assign_split, count_straddling_episodes, file_sha256,
                                    library_versions, load_raw, make_time_split,
                                    save_json, summarise_splits)
from src.feature_engineering import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("build")


def main(csv=C.RAW_CSV):
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. load + validate + sort
    df = load_raw(csv)

    # 2. chronological split on unique timestamps
    bounds = make_time_split(df)
    df["split"] = assign_split(df, bounds)
    summary = summarise_splits(df)
    straddle = count_straddling_episodes(df, bounds)
    save_json({
        "csv_sha256": file_sha256(csv), "n_rows": int(len(df)),
        "fractions": list(C.SPLIT_FRACTIONS), "boundaries": bounds,
        "splits": summary, "anomaly_episodes_straddling_boundary": straddle,
        "unseen_equipment_scheme": {
            "holdout_equipment": C.UNSEEN_EQUIPMENT,
            "train": f"split=='train' & equipment_id!='{C.UNSEEN_EQUIPMENT}'",
            "test": f"split=='test' & equipment_id=='{C.UNSEEN_EQUIPMENT}'"},
        "seed": C.SEED, "library_versions": library_versions()}, C.SPLIT_MANIFEST)

    # 3. fit on TRAIN only, transform the whole sorted timeline (features are causal)
    train = df[df["split"] == "train"]
    pipe = FeaturePipeline().fit(train)
    X = pipe.transform(df)

    # 4. hard sanity checks
    assert not set(pipe.feature_names_) & set(C.EXCLUDED), "excluded column leaked into features"
    assert not X.isna().any().any(), "NaNs remain after preprocessing"
    assert np.isfinite(X.to_numpy()).all(), "non-finite values present"

    # 5. persist
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name]
        pd.concat([part[C.META_COLS], X.loc[part.index]], axis=1).to_csv(
            C.PROCESSED_DIR / f"{name}.csv.gz", index=False)
    # drift report: standardised mean shift of NORMAL rows vs train (report only, never used for fitting)
    sets = pipe.feature_sets()
    normal = df[C.TARGET] == 0
    mu, sd = X[(df["split"] == "train") & normal].mean(), X[(df["split"] == "train") & normal].std().replace(0, 1)
    drift = {}
    for name in ["val", "test"]:
        drift[name] = ((X[(df["split"] == name) & normal].mean() - mu) / sd).abs()
    manifest = pipe.export_manifest()
    manifest["drift_report"] = {n: {"max_shift_full": round(float(d.max()), 3),
                                    "max_shift_detrended": round(float(d[sets["detrended"]].max()), 3),
                                    "top5": {k: round(float(v), 3) for k, v in d.sort_values(ascending=False).head(5).items()}}
                                for n, d in drift.items()}
    manifest["imputation_counts_train"] = {c: int(train[c].isna().sum()) for c in
                                           pipe.imputer_.columns if train[c].isna().any()}
    save_json(manifest, C.FEATURE_MANIFEST)
    joblib.dump(pipe, C.PIPELINE_PATH)

    # 6. human-readable report
    print("\n=== SPLITS ===")
    print(pd.DataFrame(summary).T[["rows", "anomalies", "anomaly_rate", "start", "end"]].to_string())
    for k, v in summary.items():
        print(f"anomaly types in {k}: {v['anomaly_types']}")
    print("anomaly episodes straddling a boundary:", straddle)
    print(f"\n=== FEATURES: {len(pipe.feature_names_)} ===")
    print(pipe.feature_names_)

    Xtr = X[df["split"] == "train"]
    print("\nzero-variance features in train:", [c for c in Xtr if Xtr[c].std() == 0] or "none")
    lo, hi = Xtr.min(), Xtr.max()
    for name in ["val", "test"]:
        Xp = X[df["split"] == name]
        frac = ((Xp < lo) | (Xp > hi)).mean().sort_values(ascending=False)
        print(f"\nfeatures with most {name} values outside the TRAIN range (fraction of rows):")
        print(frac.head(6).round(4).to_string())
    print("\n=== DRIFT of NORMAL rows vs train (in train-std units) ===")
    for n, r in manifest["drift_report"].items():
        print(f"{n}: max|shift| full={r['max_shift_full']}  detrended={r['max_shift_detrended']}  top5={r['top5']}")
    print(f"\nfeature sets: full={len(sets['full'])}, detrended={len(sets['detrended'])}")
    print("\nDone. Files written to", C.PROCESSED_DIR, "and", C.PIPELINE_PATH)


if __name__ == "__main__":
    main()