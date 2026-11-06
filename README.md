# Industrial Sensor Anomaly Detection & Predictive Maintenance

An end-to-end, reproducible machine-learning project that detects abnormal operating conditions in
multivariate industrial sensor data. It compares an unsupervised baseline (Isolation Forest), an
unsupervised deep-learning model (PyTorch Autoencoder) and a supervised benchmark (Random Forest),
tracks experiments with MLflow, and serves results through a Streamlit dashboard.

> **Scope:** batch analysis of historical sensor CSV files. There is no streaming infrastructure,
> so this is **not** a real-time monitoring system.

## Business problem
Unplanned equipment failures are costly. Operators need to know *when* a machine starts behaving
abnormally, with few enough false alarms that alerts are trusted. The project measures both:
detection quality (precision, recall, PR-AUC) and operational quality (false-alarm rate,
episode-level detection, detection delay).

## Dataset
- 50,000 readings, 5 machines (EQ-001..EQ-005), 1-minute sampling over ~7 days, 11 sensors + context columns.
- Binary label `is_anomaly` (4.6% positive) with a fault category in `anomaly_type`
  (e.g. `sensor_malfunction`, `pressure_leak`, `bearing_degradation`, `thermal_runaway`).
- **Source:** _<add the dataset name/URL and licence here>_.
- The dataset has no electrical-current channel; `power_consumption_kw` is used as power, not current.
- The data shows features of simulated data (identical linear trends on every machine, label-derived columns),
  so results should not be assumed to transfer to a real plant.

## Architecture
```
Raw CSV -> validation & audit -> chronological split (by unique timestamp)
        -> causal imputation + feature engineering (fit on TRAIN only)
        -> Isolation Forest | Random Forest (VS Code)   Autoencoder (Colab)
        -> shared evaluation (threshold chosen on validation, test scored once)
        -> MLflow tracking + best-model selection -> prediction module -> Streamlit app
```

## Data audit findings that shaped the design
| Finding | Decision |
|---|---|
| `rolling_anomaly_rate`, `maintenance_priority_score` have single-feature ROC-AUC 0.98 / 0.99; `alert_code` has codes that occur only on anomalies; `anomaly_type` is `normal` exactly when `is_anomaly=0` | Excluded from all model inputs (label leakage) |
| Missing values (1-3% per sensor) are unrelated to the label | Causal imputation: past-only forward-fill, then training median of the same operating mode |
| Impossible values (e.g. negative vibration) all belong to `sensor_malfunction` | Kept - they are signal, not noise |
| Operating mode shifts sensor baselines massively (idle vs high-load RPM) | Mode one-hot + deviation from the mode baseline |
| Five sensors drift upward all week on every machine (e.g. coolant ~72 -> ~133 C) | Drift-adaptive features (deviation from each machine's own trailing 240-reading median) |
| `equipment_age_hours` only identifies the machine and counts time | Not used as a predictor |

## Preprocessing & features
- Chronological 70/15/15 split cut on **unique timestamps** (35,000 / 7,500 / 7,500 rows); no random splitting.
- All statistics (medians, baselines, scaler) are fitted on the training period only.
- Features use current and past readings only (trailing rolling mean/std, backward differences,
  trailing-median deviation). A unit test proves that changing future readings leaves past features unchanged.
- 79 engineered features; the **`detrended` set (49 features)** removes level features of drifting sensors.
  Normal-row drift on the test period: max 2.76 training-std units (`full`) vs 0.26 (`detrended`).

## Models
| Model | Type | Rationale |
|---|---|---|
| Isolation Forest | unsupervised | baseline; fitted on normal training rows only |
| Autoencoder (PyTorch) | unsupervised deep learning | learns normal sensor relationships; reconstruction error = anomaly score |
| Random Forest | supervised | label-using benchmark; `class_weight` for the 4% imbalance |

Hyperparameters are chosen on **validation PR-AUC** (threshold-free). Each model's alert threshold maximises
F1 on the validation set. The test period is scored once.

## Results (held-out test period)
| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | False-alarm rate | Episodes detected | Median delay |
|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.144 | 0.230 | 0.177 | 0.172 | 0.820 | 9.3% | 14/20 | 3.0 min |
| Autoencoder | 0.758 | 0.400 | 0.523 | 0.609 | 0.893 | 0.9% | 17/20 | 1.0 min |
| **Random Forest (selected)** | **0.916** | **0.803** | **0.856** | **0.855** | **0.937** | **0.5%** | **20/20** | **1.5 min** |

The selected model is the one with the highest validation PR-AUC (Random Forest, 0.919). Interpretation:
- Random Forest is best, but it uses labels; the Autoencoder reaches PR-AUC 0.61 with **no labels** and is far better than Isolation Forest.
- The test set contains only 20 anomaly episodes, so small gaps are within noise. Single split, single seed.
- Severity bands (Low/Medium/High) come from validation-score tertiles; on the test period precision rises across bands
  (0.78 -> 0.99 -> 1.00 for the Random Forest).

## MLflow
`python -m src.register_models` logs one run per model (parameters, validation/test metrics, threshold, training time,
dataset fingerprint, feature list, figures, model files, preprocessing artifacts), selects the best model by validation
PR-AUC and registers it in the MLflow model registry. Browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Streamlit dashboard
Overview - Sensor Monitoring - Anomaly Detection (CSV upload, validation, score + threshold chart, download) -
Model Evaluation - Maintenance Insights (risk ranking and decision-support recommendations; not diagnoses).
_<add screenshots to docs/ and link them here>_

## Install & run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# place the dataset at data/raw/data.csv
python -m src.build_dataset
python -m src.train_isolation_forest
python -m src.train_random_forest
# Autoencoder: run notebooks/03_autoencoder_colab.ipynb in Colab, copy the artifacts back (see RUN_ORDER.md)
python -m src.evaluate
python -m src.register_models
python -m pytest -q
streamlit run app/streamlit_app.py
```

## Limitations
- One week of (apparently simulated) data, 5 machines, 20 test episodes; one split and one seed - no confidence intervals.
- Detects *that* a reading is abnormal, not *which* fault; fault-type classification would be a separate task.
- Drift-adaptive features need history: the first 30 readings per machine are flagged as warm-up.
- Preprocessing statistics were fitted on all five machines; generalisation to a truly new machine is untested here.
- Thresholds maximise validation F1; a real deployment should set them from the relative cost of missed faults vs false alarms.

## Future improvements
Leave-one-machine-out evaluation; recall by fault type; cost-based threshold selection; drift monitoring with
retraining triggers; a streaming ingestion layer for genuine near-real-time use; sequence models (LSTM/TCN autoencoders).
