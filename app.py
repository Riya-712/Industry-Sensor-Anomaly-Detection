"""Industrial Sensor Anomaly Detection - Streamlit dashboard.   Run:  streamlit run app/streamlit_app.py

Batch analysis of historical sensor CSV files. This is NOT real-time monitoring (no streaming layer).
Training logic lives in src/; this file only loads saved artifacts and displays results.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.data_preprocessing import assign_split, load_json, validate_schema
from src.predict import Predictor

st.set_page_config(page_title="Industrial Anomaly Detection", page_icon="🏭", layout="wide")
PAGES = ["Overview", "Sensor Monitoring", "Anomaly Detection", "Model Evaluation", "Maintenance Insights"]
REPORT_DIR = ROOT / "reports" / "phase3"
UNITS = {"temperature_c": "°C", "vibration_mm_s": "mm/s", "pressure_kpa": "kPa", "motor_rpm": "rpm",
         "flow_rate_lpm": "L/min", "power_consumption_kw": "kW", "coolant_temp_c": "°C",
         "acoustic_level_db": "dB", "oil_viscosity_cst": "cSt", "humidity_pct": "%", "ambient_temp_c": "°C"}
SEV_ORDER = ["None", "Low", "Medium", "High"]


# ------------------------------------------------------------------ helpers
def show(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                       # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def list_runs():
    runs = []
    for f in sorted(REPORT_DIR.glob("*_results.json")):
        run = json.loads(f.read_text())["run"]
        if run.startswith("autoencoder"):
            ok = all((C.MODELS_DIR / n).exists() for n in ["autoencoder_best.pt", "autoencoder_config.json"])
            try:
                import torch  # noqa: F401
            except Exception:
                ok = False
        else:
            ok = (C.MODELS_DIR / f"{run}.joblib").exists()
        if ok:
            runs.append(run)
    return runs


def default_run(runs):
    f = C.MODELS_DIR / "best_model.json"
    if f.exists():
        best = json.loads(f.read_text())["run"]
        if best in runs:
            return best
    return runs[0]


@st.cache_data(show_spinner="Loading historical data...")
def load_history():
    df = validate_schema(pd.read_csv(C.RAW_CSV), require_labels=True)
    df["split"] = assign_split(df, load_json(C.SPLIT_MANIFEST)["boundaries"])
    return df


@st.cache_resource
def get_predictor(run):
    return Predictor(run)


@st.cache_data(show_spinner="Scoring historical data with the selected model...")
def history_predictions(run):
    hist = load_history()
    out = get_predictor(run).predict(hist.drop(columns=["split"]))
    assert (out["timestamp"].values == hist[C.TIME].values).all()
    out["split"] = hist["split"].values
    return out


@st.cache_data(show_spinner="Building engineered features...")
def history_features(run):
    hist = load_history()
    return get_predictor(run).pipeline.transform(hist.drop(columns=["split"]))


def load_results():
    return {json.loads(f.read_text())["run"]: json.loads(f.read_text())
            for f in sorted(REPORT_DIR.glob("*_results.json"))}


# ------------------------------------------------------------------ pages
def page_overview(run):
    st.header("Overview")
    hist, pred = load_history(), history_predictions(run)
    n, a = len(hist), int(hist[C.TARGET].sum())
    c = st.columns(5)
    c[0].metric("Total readings", f"{n:,}")
    c[1].metric("Labelled anomalies", f"{a:,}")
    c[2].metric("Normal readings", f"{n - a:,}")
    c[3].metric("Equipment units", hist[C.EQUIP].nunique())
    c[4].metric("Anomaly percentage", f"{100 * a / n:.2f}%")

    t = pred[pred["split"] == "test"]
    alerts = t["predicted_status"] == "Anomaly"
    tp = int((alerts & (t["actual_is_anomaly"] == 1)).sum())
    st.subheader(f"Model alerts on the held-out test period - {run}")
    st.caption("The test period was never used for training or threshold selection. Earlier periods are "
               "in-sample for the supervised model, so alerts are only summarised for the test period.")
    c = st.columns(4)
    c[0].metric("Test readings", f"{len(t):,}")
    c[1].metric("Alerts raised", f"{int(alerts.sum()):,}")
    c[2].metric("Alert precision", f"{tp / max(int(alerts.sum()), 1):.1%}")
    c[3].metric("Anomalies caught (recall)", f"{tp / max(int(t['actual_is_anomaly'].sum()), 1):.1%}")

    st.subheader("Equipment-wise status")
    lab = hist.groupby(C.EQUIP)[C.TARGET].agg(readings="size", labelled_anomalies="sum")
    lab["labelled_anomaly_rate"] = (lab["labelled_anomalies"] / lab["readings"]).round(4)
    tt = t.groupby("equipment_id").agg(test_alerts=("predicted_status", lambda s: int((s == "Anomaly").sum())))
    last = t.sort_values("timestamp").groupby("equipment_id").tail(1).set_index("equipment_id")
    tbl = lab.join(tt).join(last[["predicted_status", "anomaly_severity"]].rename(
        columns={"predicted_status": "latest_status", "anomaly_severity": "latest_severity"}))
    st.dataframe(tbl.reset_index().rename(columns={C.EQUIP: "equipment_id"}), hide_index=True)
    show(px.bar(lab.reset_index(), x=C.EQUIP, y="labelled_anomaly_rate", title="Labelled anomaly rate by equipment"))

    st.subheader("Recent alerts (test period)")
    cols = ["timestamp", "equipment_id", "operating_mode", "anomaly_score", "anomaly_severity", "actual_anomaly_type"]
    st.dataframe(t[alerts].sort_values("timestamp", ascending=False).head(15)[cols], hide_index=True)


def page_monitoring(run):
    st.header("Sensor Monitoring")
    hist = load_history()
    f = st.columns(4)
    eqs = f[0].multiselect("Equipment", sorted(hist[C.EQUIP].unique()), default=[sorted(hist[C.EQUIP].unique())[0]])
    d0, d1 = hist[C.TIME].min().date(), hist[C.TIME].max().date()
    dr = f[1].date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
    modes = f[2].multiselect("Operating mode", sorted(hist[C.MODE].unique()), default=sorted(hist[C.MODE].unique()))
    sensors = f[3].multiselect("Sensors", C.SENSORS + ["ambient_temp_c"], default=C.SENSORS[:4])
    if not eqs or not sensors or len(dr) != 2:
        st.info("Select at least one equipment unit, one sensor and a complete date range.")
        return
    m = (hist[C.EQUIP].isin(eqs) & hist[C.MODE].isin(modes) &
         (hist[C.TIME].dt.date >= dr[0]) & (hist[C.TIME].dt.date <= dr[1]))
    d = hist[m]
    up = st.session_state.get("upload_raw")
    upv = validate_schema(up, require_labels=False) if up is not None else None
    show_labels = st.checkbox("Mark labelled anomalies (historical ground truth)", value=True)
    st.caption("Solid lines = historical dataset. Dashed lines = your uploaded file (from the Anomaly Detection page).")
    for s in sensors:
        fig = go.Figure()
        for e in eqs:
            g = d[d[C.EQUIP] == e]
            fig.add_trace(go.Scattergl(x=g[C.TIME], y=g[s], mode="lines", name=f"{e} (historical)", line=dict(width=1)))
            if show_labels:
                an = g[g[C.TARGET] == 1]
                fig.add_trace(go.Scattergl(x=an[C.TIME], y=an[s], mode="markers", name=f"{e} anomaly label",
                                           marker=dict(color="red", size=4)))
            if upv is not None and e in set(upv[C.EQUIP]):
                u = upv[upv[C.EQUIP] == e]
                fig.add_trace(go.Scattergl(x=u[C.TIME], y=u[s], mode="lines", name=f"{e} (uploaded)", line=dict(dash="dash")))
        fig.update_layout(title=f"{s} ({UNITS.get(s, '')})", yaxis_title=UNITS.get(s, ""), height=320)
        show(fig)
    st.caption("Note: temperature, coolant temperature, acoustic level, oil viscosity and vibration drift upward "
               "over the week in this dataset - the models use drift-robust features for that reason.")


def page_detection(run):
    st.header("Anomaly Detection")
    pred = get_predictor(run)
    with st.expander("Required CSV columns"):
        st.code(", ".join(C.INFERENCE_REQUIRED) + "   (optional: is_anomaly, anomaly_type for evaluation)")
    up = st.file_uploader("Upload a sensor CSV", type="csv")
    demo = st.button("Or use the last 6 hours of the historical data as a demo", key="sample_btn")
    raw = None
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file as CSV: {e}")
            return
        st.session_state["upload_raw"] = raw
    elif demo:
        h = load_history()
        raw = h[h[C.TIME] >= h[C.TIME].max() - pd.Timedelta(hours=6)].drop(columns=["split"])
        st.session_state["upload_raw"] = raw
    elif st.session_state.get("upload_raw") is not None:
        raw = st.session_state["upload_raw"]
    if raw is None:
        st.info("Upload a CSV to begin.")
        return
    try:
        out = pred.predict(raw)
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        return
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    flagged = out["predicted_status"] == "Anomaly"
    c = st.columns(4)
    c[0].metric("Rows scored", f"{len(out):,}")
    c[1].metric("Anomalies flagged", f"{int(flagged.sum()):,}")
    c[2].metric("Alert rate", f"{flagged.mean():.2%}")
    c[3].metric("Alert threshold", f"{pred.threshold:.4f}")
    warm = int(out["in_warmup"].sum())
    if warm:
        st.warning(f"{warm} rows are within the first {C.RECENT_MIN_PERIODS} readings of their machine in this file. "
                   "Drift-adaptive features need history, so scores there are less reliable.")
    if "actual_is_anomaly" in out:
        y, p = out["actual_is_anomaly"], flagged.astype(int)
        st.info(f"Your file contains labels. Against them: {int(((p == 1) & (y == 1)).sum())} true alerts, "
                f"{int(((p == 1) & (y == 0)).sum())} false alerts, {int(((p == 0) & (y == 1)).sum())} missed. "
                "If the rows overlap the training period these figures are optimistic.")

    eq = st.selectbox("Equipment", sorted(out["equipment_id"].unique()))
    g = out[out["equipment_id"] == eq]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=g["timestamp"], y=g["anomaly_score"], mode="lines", name="anomaly score"))
    a = g[g["predicted_status"] == "Anomaly"]
    fig.add_trace(go.Scattergl(x=a["timestamp"], y=a["anomaly_score"], mode="markers", name="flagged",
                               marker=dict(color="red", size=6)))
    fig.add_hline(y=pred.threshold, line_dash="dash", annotation_text="threshold")
    fig.update_layout(title=f"Anomaly score - {eq} ({run})", yaxis_title="score (higher = more anomalous)")
    show(fig)

    v = validate_schema(raw, require_labels=False).reset_index(drop=True)
    v["flagged"] = flagged.values
    s = st.selectbox("Sensor to inspect", C.SENSORS)
    gv = v[v[C.EQUIP] == eq]
    fig = go.Figure(go.Scattergl(x=gv[C.TIME], y=gv[s], mode="lines", name=s))
    fl = gv[gv["flagged"]]
    fig.add_trace(go.Scattergl(x=fl[C.TIME], y=fl[s], mode="markers", name="flagged", marker=dict(color="red", size=6)))
    fig.update_layout(title=f"{s} ({UNITS.get(s, '')}) with flagged readings", yaxis_title=UNITS.get(s, ""))
    show(fig)

    st.subheader("Predictions")
    st.dataframe(out, hide_index=True)
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
    st.caption("Severity is a score band (Low/Medium/High), not a fault diagnosis. The model detects THAT a reading "
               "is abnormal, not WHICH fault it is; actual_anomaly_type is only the historical label.")


def page_evaluation(run):
    st.header("Model Evaluation")
    res = load_results()
    if not res:
        st.error("No results found in reports/phase3. Run the training scripts first.")
        return
    best = default_run(list(res))
    rows = []
    for k, r in res.items():
        t, v = r["metrics"]["test"], r["metrics"]["val"]
        rows.append({"model": k + (" (selected)" if k == best else ""), "val_pr_auc": v["pr_auc"],
                     "precision": t["precision"], "recall": t["recall"], "f1": t["f1"], "pr_auc": t["pr_auc"],
                     "roc_auc": t["roc_auc"], "false_alarm_rate": t["false_alarm_rate"],
                     "episode_recall": t.get("episode_recall"), "median_delay_min": t.get("median_delay_min"),
                     "train_seconds": r["train_seconds"], "inference_ms_per_1000_rows": r["inference_ms_per_1000_rows"]})
    tbl = pd.DataFrame(rows).round(4)
    st.caption("Test-period metrics (never used for training or threshold selection). The model is selected by "
               "VALIDATION PR-AUC, not by accuracy or test score.")
    st.dataframe(tbl, hide_index=True)
    long = tbl.melt(id_vars="model", value_vars=["precision", "recall", "f1", "pr_auc", "roc_auc"])
    show(px.bar(long, x="variable", y="value", color="model", barmode="group", title="Test metrics by model"))
    st.caption("The test period contains only a few dozen anomaly episodes, so small differences between models are "
               "within noise; large gaps are meaningful. The Random Forest is supervised (uses labels); "
               "Isolation Forest and the Autoencoder are unsupervised.")
    for k, r in res.items():
        with st.expander(k, expanded=(k == best)):
            t = r["metrics"]["test"]
            c1, c2 = st.columns(2)
            cm = [[t["tn"], t["fp"]], [t["fn"], t["tp"]]]
            fig = go.Figure(go.Heatmap(z=cm, x=["Pred normal", "Pred anomaly"], y=["Actual normal", "Actual anomaly"],
                                       text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
            fig.update_layout(title="Confusion matrix (test)", yaxis_autorange="reversed", height=350)
            with c1:
                show(fig)
            with c2:
                st.write({"alert threshold": round(r["threshold"], 5), "features": r["n_features"],
                          "alerts per equipment (test)": t["alerts_per_equipment"],
                          "episodes detected": f"{t.get('episodes_detected')}/{t.get('n_episodes')}"})
            if r.get("loss_history"):
                h = pd.DataFrame(r["loss_history"])
                show(px.line(h, x="epoch", y=["train_loss", "val_loss"], title="Autoencoder training / validation loss"))
            for suffix in ("_eval.png", "_importance.png"):
                p = REPORT_DIR / "figures" / f"{k}{suffix}"
                if p.exists():
                    st.image(str(p), caption=p.name)
            if k.startswith("autoencoder"):
                p = REPORT_DIR / "figures" / "autoencoder_loss_and_error.png"
                if p.exists():
                    st.image(str(p), caption="Reconstruction-error distribution")
    sev = C.MODELS_DIR / "severity_cutoffs.json"
    if sev.exists():
        st.subheader("Severity bands - validation of the method")
        st.caption("Cut-offs come from validation data; the table shows test precision per band "
                   "(it should rise from Low to High).")
        s = json.loads(sev.read_text())
        st.dataframe(pd.DataFrame({m: {b: v["precision"] for b, v in d["test_precision_by_band"].items()}
                                   for m, d in s.items()}).T.reset_index().rename(columns={"index": "model"}),
                     hide_index=True)


def page_maintenance(run):
    st.header("Maintenance Insights")
    st.caption("Decision-support only: these are model-based indicators, not confirmed equipment diagnoses.")
    hist, pred, X = load_history(), history_predictions(run), history_features(run)
    t = pred[pred["split"] == "test"]
    hours = st.slider("Look-back window (hours, held-out test period)", 1, 24, 6)
    w = t[t["timestamp"] > t["timestamp"].max() - pd.Timedelta(hours=hours)]

    c1, c2 = st.columns(2)
    freq = hist.groupby(C.EQUIP)[C.TARGET].agg(["sum", "mean"]).reset_index()
    with c1:
        show(px.bar(freq, x=C.EQUIP, y="sum", title="Labelled anomalies by equipment (all history)"))
    with c2:
        ty = hist[hist[C.TARGET] == 1]["anomaly_type"].value_counts().reset_index()
        ty.columns = ["anomaly_type", "count"]
        show(px.bar(ty, x="anomaly_type", y="count", title="Anomaly types (historical labels)"))

    st.subheader(f"Equipment risk ranking - last {hours} h")
    wt = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    g = w.groupby("equipment_id")
    risk = pd.DataFrame({"readings": g.size(), "alerts": g["predicted_status"].apply(lambda s: int((s == "Anomaly").sum())),
                         "high": g["anomaly_severity"].apply(lambda s: int((s == "High").sum())),
                         "risk_index": g["anomaly_severity"].apply(lambda s: s.map(wt).sum() / (3 * len(s)))})
    risk["alert_rate"] = risk["alerts"] / risk["readings"]
    last = hist.sort_values(C.TIME).groupby(C.EQUIP).tail(1).set_index(C.EQUIP)
    risk["hours_since_maintenance"] = last["hours_since_maintenance"].round(1)
    risk = risk.sort_values("risk_index", ascending=False).round(4)
    risk.insert(0, "rank", range(1, len(risk) + 1))
    st.dataframe(risk.reset_index(), hide_index=True)
    st.caption("Risk index = severity-weighted share of readings in the window (Low=1, Medium=2, High=3, scaled 0-1).")

    st.subheader("Sensor patterns associated with anomalies")
    fm = load_json(C.FEATURE_MANIFEST)
    rc = [f for f in fm["feature_names"] if f.endswith("__vs_recent")]
    y = hist[C.TARGET].values == 1
    d = (X.loc[y, rc].mean() - X.loc[~y, rc].mean()) / X.loc[~y, rc].std().replace(0, 1)
    d.index = [i.replace("__vs_recent", "") for i in d.index]
    dd = d.reindex(d.abs().sort_values(ascending=False).index).head(10).reset_index()
    dd.columns = ["sensor", "standardised shift"]
    show(px.bar(dd, x="standardised shift", y="sensor", orientation="h",
                title="Deviation from recent baseline: anomalous vs normal readings (in normal-row std units)"))

    st.subheader("Decision-support recommendations")
    mean, scale = pd.Series(fm["scaler"]["mean"], index=fm["feature_names"]), pd.Series(fm["scaler"]["scale"], index=fm["feature_names"])
    shown = 0
    for eq, row in risk.iterrows():
        if row["alerts"] == 0:
            continue
        idx = w[(w["equipment_id"] == eq) & (w["predicted_status"] == "Anomaly")].index
        z = ((X.loc[idx, rc].mean() - mean[rc]) / scale[rc])
        top = [i.replace("__vs_recent", "") for i in z.abs().sort_values(ascending=False).head(3).index]
        st.markdown(f"**Priority {int(row['rank'])} - {eq}:** {int(row['alerts'])} alerts ({row['alert_rate']:.1%} of readings), "
                    f"{int(row['high'])} high-severity, in the last {hours} h. Largest deviations from the machine's recent baseline: "
                    f"**{', '.join(top)}**. Last maintenance {row['hours_since_maintenance']} h ago. "
                    "*Suggested: schedule an inspection of subsystems related to these sensors.*")
        shown += 1
        if shown == 3:
            break
    if shown == 0:
        st.success("No alerts in the selected window.")


# ------------------------------------------------------------------ main
def main():
    st.sidebar.title("🏭 Anomaly Detection")
    if not C.RAW_CSV.exists() or not C.SPLIT_MANIFEST.exists():
        st.error("Missing data/raw/data.csv or processed files. Run `python -m src.build_dataset` first.")
        return
    runs = list_runs()
    if not runs:
        st.error("No trained models found in models/. Run the training scripts first.")
        return
    page = st.sidebar.radio("Page", PAGES, key="page")
    run = st.sidebar.selectbox("Model", runs, index=runs.index(default_run(runs)), key="model")
    p = get_predictor(run)
    st.sidebar.caption(f"Alert threshold: {p.threshold:.4f}  \nSelected on validation data.")
    st.sidebar.caption("Batch analysis of historical CSV data - not real-time monitoring.")
    {"Overview": page_overview, "Sensor Monitoring": page_monitoring, "Anomaly Detection": page_detection,
     "Model Evaluation": lambda r: page_evaluation(r), "Maintenance Insights": page_maintenance}[page](run)


main()