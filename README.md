# Cefalo Oxygen Anomaly & Forecasting

End-to-end anomaly detection and forecasting solution for minute-level dissolved oxygen readings across multiple aqua systems/customers, built for the **Cefalo Lead Data Scientist assignment**. The goal is to:

- Detect **multiple anomaly types** (point, collective, contextual, sensor-fault) and
- Train a **generic, tag-agnostic forecasting model** that can predict oxygen for the **next 1 week**,  
  while **not relying on customer-specific tags** and **minimising model duplication**.

The pipeline is fully script-based (train → deploy → monitor → retrain) and mirrors the attached notebooks:

- `notebooks/oxygen_anomaly_detector_v3_analysis.ipynb`
- `notebooks/oxygen_forecasting_model_tvt_v2.ipynb`

---

## Repository structure

```text
cefalo-oxygen-anomaly/
├─ data/
│   ├─ oxygen.csv                      # full raw dataset (assignment)
│   └─ tiny_sample.csv                 # optional small sample for tests
├─ notebooks/
│   ├─ oxygen_anomaly_detector_v3_analysis.ipynb
│   └─ oxygen_forecasting_model_tvt_v2.ipynb
├─ src/
│   ├─ data/
│   │   ├─ io.py                       # load/save CSV, parse timestamps
│   │   └─ preprocessing.py            # drop nulls, sort by sensor/time
│   ├─ features/
│   │   └─ timeseries.py               # time-of-day, day-of-week, lags, rolling
│   ├─ models/
│   │   ├─ anomaly.py                  # rule-based anomaly detector + severity
│   │   ├─ forecaster.py               # global HistGB forecaster + TVT split
│   │   └─ train_pipeline.py           # TRAIN + RETRAIN (end-to-end)
│   ├─ inference/
│   │   ├─ batch_inference.py          # DEPLOYMENT: score + 1-step forecast on CSV
│   │   └─ horizon_forecast.py         # DEPLOYMENT: 1-week horizon for a sensor
│   └─ evaluation/
│       ├─ metrics.py                  # MAE/RMSE helpers
│       └─ monitoring.py               # MONITORING: drift & performance checks
├─ configs/
│   └─ pipeline_config.yaml            # data paths, anomaly & forecast hyperparams
├─ models/
│   ├─ anomaly_detector_config.json    # saved anomaly config (traceability)
│   ├─ forecaster.pkl                  # trained sklearn model
│   └─ baseline_stats.json             # baseline metrics used for monitoring
└─ README.md
```

**Key ideas**

- **Anomaly detection** (`src/models/anomaly.py`):
  - Point anomalies via robust rolling Z-scores.
  - Collective anomalies via rolling density of point spikes.
  - Contextual anomalies via deviation from sensor × hour-of-day median.
  - Sensor-fault anomalies: stuck sensor (low variance), spikes (large diffs), high noise.
  - Combined into a single **`severity ∈ [0,1]` per minute**.
- **Forecasting** (`src/models/forecaster.py`):
  - Single **global** `HistGradientBoostingRegressor` over all sensors (tag-agnostic).
  - Features: lags, rolling mean, minute-of-day, day-of-week + cyclic encodings.
  - Proper **train/validation/test** split by time (no leakage).
  - **Anomaly-aware training**: highest-severity points are excluded before fitting.

---

## Environment setup

1. Create and activate a virtual environment (example with `venv`):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   # .venv\Scripts\activate       # Windows PowerShell
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the raw dataset is in place:

   ```bash
   ls data/oxygen.csv
   ```

   Adjust `configs/pipeline_config.yaml` if your file name or paths differ.

---

## Running the pipelines

### 1. Training / Retraining (anomaly detection + forecasting)

This runs the full **train → evaluate → save models** pipeline:

```bash
python -m src.models.train_pipeline --config configs/pipeline_config.yaml
```

What it does:

- Loads `data/oxygen.csv`.
- Cleans and sorts the data.
- Fits the **anomaly detector** and scores each minute (adds `severity`).
- Engineers time-series features (lags, rolling mean, calendar/cyclic).
- Drops the highest-severity quantile from training (anomaly-aware).
- Splits by time into **train / validation / test**.
- Trains a global `HistGradientBoostingRegressor` via small grid search.
- Saves:
  - `data/oxygen_processed_full.csv` (scored + features),
  - `models/anomaly_detector_config.json`,
  - `models/forecaster.pkl`,
  - `models/baseline_stats.json` (test MAE/RMSE + severity stats).

Re-running this command on updated `data/oxygen.csv` is the **retraining** step.

---

### 2. Batch inference (score + 1-step forecasts on a new CSV)

To score a new file and generate 1-step-ahead forecasts:

```bash
python -m src.inference.batch_inference   --config configs/pipeline_config.yaml   --input data/raw/oxygen_sample.csv   --output data/processed/oxygen_sample_forecast.csv
```

What it does:

- Rebuilds the anomaly context using the processed training data.
- Loads and cleans `oxygen_sample.csv`.
- Scores anomalies (including `severity`).
- Builds the same feature set used at training.
- Predicts 1-step-ahead oxygen (`y_pred`) for all rows with full features.
- Writes the **scored + predicted** dataset to the output path.

---

### 3. 1-week horizon forecast for a single sensor

To generate a **7-day, minute-level** forecast for a specific sensor:

```bash
python -m src.inference.horizon_forecast   --config configs/pipeline_config.yaml   --processed data/processed/oxygen_processed_full.csv   --sensor_id 'System_10|EquipmentUnit_10|SubUnit_07'   --output data/processed/forecast_1week_SubUnit_07.csv
```

Notes:

- `sensor_id` **must be quoted** because it contains `|` (pipeline character in the shell).
- The script iteratively rolls the forecaster forward 1 minute at a time for 7 days, using the same lag/rolling/calendar feature definitions as in training.

---

### 4. Monitoring & drift checks

In a notebook or script, you can monitor new data vs the training baseline:

```python
import pandas as pd
from src.evaluation.monitoring import monitor_new_data

df_new = pd.read_csv("data/processed/oxygen_sample_forecast.csv")
df_new["time"] = pd.to_datetime(df_new["time"])

monitor_new_data(
    df_scored=df_new,
    df_with_preds=df_new,
    baseline_stats_path="models/baseline_stats.json",
    severity_col="severity",
    value_col="Oxygen[%sat]",
    pred_col="y_pred",
)
```

This compares:

- **Severity distribution** (median) vs training,
- **RMSE** vs baseline test RMSE,

and prints warnings when thresholds are exceeded, signalling **potential drift** and the need to retrain.

---

## Assumptions & limitations

- **Tag-agnostic design**: the model only uses `sensor_id` and time-based features; it does **not** rely on specific metadata tags (Unit, Section, Equipment, Subunit) so it can generalise across customers with different tag schemas.
- **Rule-based anomaly detection**: anomaly detection is deterministic and interpretable, but may miss highly complex patterns that a learned model could capture.
- **Single global forecaster**: one model is shared across all sensors; this reduces duplication but may under-fit very unusual sensors or locations.
- **Minute-level, regular sampling assumed**: the pipeline expects roughly regular 1-minute intervals; large gaps or irregular sampling are not explicitly modelled.
- **No real-time infrastructure**: the implementation is **batch-oriented** (CLI scripts and notebooks). Production-grade streaming / deployment (e.g. APIs, schedulers, containers) is out of scope for this assignment but can be added on top of these pipelines.
- **Performance metrics**: MAE/RMSE are computed and stored in `models/baseline_stats.json` for the assignment; exact numbers depend on the environment and any pre-filtering choices.

For the assignment, these choices trade off **simplicity + interpretability** with **sufficient robustness** to detect and score anomalies and produce reasonable 1-week oxygen forecasts on minute-level data.
