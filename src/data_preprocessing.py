"""Loading, validation, chronological splitting and split-manifest handling."""
import hashlib
import json
import logging
import platform

import numpy as np
import pandas as pd
import sklearn

from src import config as C

log = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame, require_labels: bool = True) -> pd.DataFrame:
    """Validate and standardise a raw dataframe. Returns a sorted copy.
    Raises ValueError with a readable message (the Streamlit app will show it)."""
    required = C.INFERENCE_REQUIRED + ([C.TARGET] if require_labels else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()

    for c in C.SENSORS + C.CONTEXT_NUMERIC:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        coerced = df[c].isna().sum() - before
        if coerced:
            log.warning("%s: %d non-numeric values coerced to NaN", c, coerced)

    df[C.TIME] = pd.to_datetime(df[C.TIME], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    n_bad = int(df[C.TIME].isna().sum())
    if n_bad:
        raise ValueError(f"{n_bad} timestamps could not be parsed")

    if require_labels and not set(df[C.TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError("is_anomaly must contain only 0 and 1")

    n_dup = int(df.duplicated([C.EQUIP, C.TIME]).sum())
    if n_dup:
        raise ValueError(f"{n_dup} duplicated (equipment_id, timestamp) pairs")

    return df.sort_values([C.TIME, C.EQUIP], kind="mergesort").reset_index(drop=True)


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    df = validate_schema(pd.read_csv(path), require_labels=True)
    log.info("Loaded %s rows x %s cols from %s", *df.shape, path)
    return df


# --------------------------------------------------------------------------- split
def make_time_split(df: pd.DataFrame, fractions=C.SPLIT_FRACTIONS) -> dict:
    """Cut on UNIQUE timestamps so all machines at one instant land in the same split."""
    ts = np.sort(df[C.TIME].unique())
    n = len(ts)
    i_val = int(round(fractions[0] * n))
    i_test = int(round((fractions[0] + fractions[1]) * n))
    return {"val_start": str(pd.Timestamp(ts[i_val])),
            "test_start": str(pd.Timestamp(ts[i_test]))}


def assign_split(df: pd.DataFrame, bounds: dict) -> np.ndarray:
    vs, tst = pd.Timestamp(bounds["val_start"]), pd.Timestamp(bounds["test_start"])
    return np.where(df[C.TIME] < vs, "train", np.where(df[C.TIME] < tst, "val", "test"))


def summarise_splits(df: pd.DataFrame) -> dict:
    out = {}
    for name in ["train", "val", "test"]:
        p = df[df["split"] == name]
        out[name] = {"rows": int(len(p)), "anomalies": int(p[C.TARGET].sum()),
                     "anomaly_rate": round(float(p[C.TARGET].mean()), 5),
                     "start": str(p[C.TIME].min()), "end": str(p[C.TIME].max()),
                     "anomaly_types": {k: int(v) for k, v in
                                       p.loc[p[C.TARGET] == 1, "anomaly_type"].value_counts().items()}}
    return out


def count_straddling_episodes(df: pd.DataFrame, bounds: dict) -> dict:
    """Anomaly episodes that are 'on' both just before and just after a boundary."""
    res = {}
    for name, b in bounds.items():
        b, n = pd.Timestamp(b), 0
        for _, g in df.groupby(C.EQUIP):
            before, after = g[g[C.TIME] < b].tail(1), g[g[C.TIME] >= b].head(1)
            if len(before) and len(after) and before[C.TARGET].iloc[0] == 1 and after[C.TARGET].iloc[0] == 1:
                n += 1
        res[name] = n
    return res


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(path.read_text())


def library_versions() -> dict:
    return {"python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__}


def load_processed(split: str, feature_set: str = "detrended"):
    """Return (meta, X, y) for 'train' | 'val' | 'test'.
    feature_set: 'detrended' (drift-robust, recommended) or 'full'.
    ALWAYS select features via the manifest, never 'all other columns'."""
    feats = load_json(C.FEATURE_MANIFEST)["feature_sets"][feature_set]
    df = pd.read_csv(C.PROCESSED_DIR / f"{split}.csv.gz", parse_dates=[C.TIME])
    return df[C.META_COLS], df[feats], df[C.TARGET].astype(int)