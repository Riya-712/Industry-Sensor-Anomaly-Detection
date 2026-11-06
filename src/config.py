"""Central configuration. Every Phase 2+ module imports from here so that
VS Code, Colab and the Streamlit app all use identical definitions."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
SPLIT_MANIFEST = PROCESSED_DIR / "split_manifest.json"
FEATURE_MANIFEST = PROCESSED_DIR / "feature_manifest.json"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"

# --- column roles -----------------------------------------------------------
TIME, EQUIP, MODE, TARGET = "timestamp", "equipment_id", "operating_mode", "is_anomaly"
META_COLS = ["reading_id", TIME, EQUIP, MODE, TARGET, "anomaly_type"]

SENSORS = [
    "temperature_c", "vibration_mm_s", "pressure_kpa", "motor_rpm", "flow_rate_lpm",
    "power_consumption_kw", "coolant_temp_c", "acoustic_level_db",
    "oil_viscosity_cst", "humidity_pct",
]
CONTEXT_NUMERIC = ["ambient_temp_c", "hours_since_maintenance"]
OPERATING_MODES = ["startup", "normal", "high_load", "cooldown", "idle"]  # fixed order

# Columns that must never reach a model, with the evidence-based reason.
EXCLUDED = {
    "reading_id": "identifier",
    "anomaly_type": "label-derived: non-'normal' exactly when is_anomaly == 1",
    "rolling_anomaly_rate": "label-derived: single-feature AUC ~0.98 vs sensors <=0.61",
    "maintenance_priority_score": "label-derived: single-feature AUC ~0.99",
    "alert_code": "downstream alert logic: 5 of 8 codes are 100% anomalous",
    "equipment_age_hours": "acts as machine ID + clock (ramps per machine; test values "
                           "fall outside the training range); AUC ~0.51",
    "equipment_id": "kept for grouping/analysis only; not a predictor (cross-equipment generalisation)",
    "timestamp": "kept for ordering/analysis only",
}

# --- feature engineering ----------------------------------------------------
ROLL_SENSORS = ["temperature_c", "vibration_mm_s", "pressure_kpa",
                "motor_rpm", "power_consumption_kw", "acoustic_level_db"]
ROLL_WINDOWS = [5, 15]        # readings (1 reading = 1 minute in this dataset)
MAX_FFILL = 5                 # max consecutive readings to forward-fill
RECENT_WINDOW = 240           # trailing readings (minutes) used as a drift-adaptive baseline;
                              # must be >> typical anomaly episode (median 18, max 63 readings)
RECENT_MIN_PERIODS = 30       # warm-up: deviation is 0 until this much history exists

# Raw sensors whose NORMAL-row mean did not drift across the week (see drift report in build output).
# Level features of the other sensors are drift-contaminated, so the 'detrended' set omits them.
STATIONARY_RAW = ["motor_rpm", "flow_rate_lpm", "power_consumption_kw", "humidity_pct",
                  "ambient_temp_c", "hours_since_maintenance"]

# --- splitting --------------------------------------------------------------
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # by unique timestamp, not by row
UNSEEN_EQUIPMENT = "EQ-005"            # secondary evaluation (Phase 4)

INFERENCE_REQUIRED = [TIME, EQUIP, MODE] + SENSORS + CONTEXT_NUMERIC