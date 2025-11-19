# Oxygen Anomaly & Forecasting

End-to-end anomaly detection and forecasting solution for minute-level dissolved oxygen readings across multiple aqua systems/customers.

The goals are to:

- Detect **multiple anomaly types** (point, collective, contextual, sensor-fault) at minute resolution.
- Train a **generic, tag-agnostic forecasting model** that can predict dissolved oxygen for the **next 1 week**, without relying on customer-specific tag schemas or duplicating models per sensor.

The codebase is fully script-based (train → deploy → monitor → retrain) and mirrors the notebooks:

- `notebooks/oxygen_anomaly_detector_analysis.ipynb`
- `notebooks/oxygen_forecasting_model_tvt.ipynb`
- `notebooks/oxygen_forecasting_diagnostics.ipynb` (visual QA: actual vs predicted, residuals, split markers, 1-week horizon overlay)

---

## 1. Solution overview & algorithm choices

### 1.1 Anomaly detection (`src/models/anomaly.py`)

The detector now matches `notebooks/oxygen_anomaly_detector_analysis.ipynb` exactly:

- **Point anomalies** – rolling mean/std Z-score per sensor (60 min window), scaled with `z_point_low=2`, `z_point_high=5`.
- **Collective anomalies** – rolling mean(|z|) over 120 minutes, scaled between `collective_low=1`, `collective_high=3`.
- **Contextual anomalies** – deviation from hour-of-day baseline (mean/std per hour), scaled with `z_ctx_low=2`, `z_ctx_high=4`.
- **Sensor-fault anomalies**  
  - Stuck: low std vs typical (`stuck_rel_std_factor=0.1`, window 60).  
  - Spikes: sign-reversing jumps with magnitude > `spike_z_threshold=3`.  
  - High noise: short-window std >> typical (`noise_factor=1.5`, window 60).

Severity is the **max** of the sub-scores (point, collective, contextual, sensor-fault). The training pipeline applies the **severity quantile cutoff** (default 0.99) before forecasting feature engineering, mirroring the TVT notebook cleaning step.

### 1.2 Forecasting (`src/models/forecaster.py`)

The forecaster is a **single global model** shared across all sensors:

- Model: `HistGradientBoostingRegressor` (sklearn).
- Features:
  - **Lagged values**: e.g. 1, 5, 60 minute lags per sensor.
  - **Rolling statistics**: rolling mean over the last 60 minutes per sensor.
  - **Calendar/time features**:
    - `minute_of_day`, `dayofweek`
    - Cyclic encodings: `sin_time`, `cos_time`, `sin_dow`, `cos_dow`
- Split: time-based **train / validation / test** split using configurable `valid_days` and `test_days`.
- Hyperparameters: small grid search over `learning_rate` and `max_depth`.

This **global, tag-agnostic model** respects the assignment’s requirement to avoid customer-specific models and to generalise across different tag schemes.

---

## 2. Repository structure

```text
cefalo-oxygen-anomaly/
├─ data/
│   ├─ oxygen.csv                      # full raw dataset (assignment)
│   └─ tiny_sample.csv                 # optional small sample for tests
├─ notebooks/
│   ├─ oxygen_anomaly_detector_analysis.ipynb  # anomaly detection and analysis
│   ├─ oxygen_forecasting_model_tvt.ipynb      # forecaster
│   ├─ oxygen_forecasting_diagnostics.ipynb    # visual QA: actual vs pred, residuals, splits, 1-week overlay
│   └─ oxygen_eda_notebook.ipynb               # exploratory analysis
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
└─ Dockerfile
└─ README.md
└─ requirements.txt
```

---

## 3. System design: train → deploy → monitor → retrain

### 3.1 Training / retraining (`src/models/train_pipeline.py`)

**Entry point:**

```bash
python -m src.models.train_pipeline --config configs/pipeline_config.yaml
```

**Steps:**

1. **Load & clean data**
   - Read `data/oxygen.csv` via `src/data/io.py`.
   - Drop rows with null timestamps or null oxygen values.
   - Sort by `(sensor_id, time)`.

2. **Anomaly detection & scoring**
   - Fit `OxygenAnomalyDetector` on the cleaned data.
   - Compute point, collective, contextual, and sensor-fault scores.
   - Combine into a single `severity` score per minute.

3. **Feature engineering**
   - Add time-of-day and day-of-week features + cyclic encodings.
   - Add lagged oxygen values for configured `lag_minutes`.
   - Add rolling mean over a configurable window (e.g. 60 minutes).
   - Drop rows with incomplete feature sets.

4. **Anomaly-aware training subset**
   - Compute severity cutoff at a configurable quantile (e.g. 0.99).
   - Exclude rows above this cutoff from training to avoid fitting on extreme anomalies.

5. **Time-based train/validation/test split**
   - Use `valid_days` and `test_days` from `pipeline_config.yaml`.
   - Train only on the **training** window.
   - Use **validation** for hyperparameter tuning.
   - Reserve **test** for final unbiased evaluation.

6. **Model training & evaluation**
   - Train global `HistGradientBoostingRegressor` on training data.
   - Choose best hyperparameters based on validation MAE.
   - Compute MAE and RMSE on train, validation, and test sets.

7. **Artefact saving**
   - Save processed/scored dataset:
     - `data/oxygen_processed_full.csv`
   - Save anomaly config:
     - `models/anomaly_detector_config.json`
   - Save forecaster:
     - `models/forecaster.pkl`
   - Save baseline metrics & split info:
     - `models/baseline_stats.json`

Re-running the same command after updating `data/oxygen.csv` is the **retraining** step.

---

### 3.2 Deployment: batch inference (`src/inference/batch_inference.py`)

**Entry point:**

```bash
python -m src.inference.batch_inference --config configs/pipeline_config.yaml --input data/raw/oxygen_sample.csv --output data/processed/oxygen_sample_forecast.csv
```

**Steps:**

1. Load trained `forecaster.pkl` and anomaly config.
2. Rebuild **context baseline** for anomalies using `data/oxygen_processed_full.csv`.
3. Load and clean new raw data (`oxygen_sample.csv`).
4. Score anomalies (including `severity`).
5. Build the **same feature set** used at training (lags, rolling, time encodings).
6. Generate 1-step-ahead predictions `y_pred` for all rows with full features.
7. Save a CSV with both anomaly scores and forecasts to `data/processed/oxygen_sample_forecast.csv`.

---

### 3.3 Deployment: 1-week horizon forecast (`src/inference/horizon_forecast.py`)

**Entry point:**

```bash
python -m src.inference.horizon_forecast --config configs/pipeline_config.yaml --processed data/oxygen_processed_full.csv --sensor_id 'System_10|EquipmentUnit_10|SubUnit_07' --output data/processed/forecast_1week_SubUnit_07.csv
```

**Behaviour:**

- Filters processed history for a given `sensor_id`.
- Iteratively rolls the forecaster forward in **1-minute steps for 7 days**:
  - Uses the last observed/predicted values to compute lag and rolling features.
  - Adds calendar/cyclic features for each forecasted timestamp.
- Writes a dense 1-week forecast timeline for that sensor.

> Note: `sensor_id` must be **quoted** in the shell because it contains `|`, which is otherwise treated as the pipe operator.

---

### 3.4 Monitoring & drift detection (`src/evaluation/monitoring.py`)

Example (notebook or script):

```python
import pandas as pd
from src.evaluation.monitoring import monitor_new_data

df_new = pd.read_csv("data/processed/oxygen_sample_forecast.csv")
df_new["time"] = pd.to_datetime(df_new["time"])

monitor_new_data(
    df_scored=df_new,
    df_with_preds=df_new,  # same frame, already contains forecasts
    model_dir="models",
    registry_dir="models/registry",
    severity_col="severity",
    value_col="oxygen",
    pred_col="y_pred",
)
```

The monitoring step:

- Compares the **median severity** of the new data vs the production baseline in `models/baseline_stats.json`.
- Compares **RMSE** on the new data vs the production test RMSE and returns warnings if it exceeds tolerance.
- Returns a structured dict that includes the production run id from `models/registry/production.json` when present.

---

## 4. Model performance evaluation

Model performance is evaluated in the training pipeline:

- Metrics: **MAE** and **RMSE** for **train**, **validation**, and **test** splits.
- Implementation: `src/models/forecaster.py` and `src/evaluation/metrics.py`.
- Outputs:
  - Printed in the console when running `train_pipeline.py`.
  - Persisted in `models/baseline_stats.json`:
    - `test_mae`, `test_rmse`
    - `severity_median` (overall median anomaly severity)
    - `valid_start`, `test_start` (split boundaries)
    - Selected hyperparameters for the forecaster.

These metrics are directly used by the monitoring component to decide whether model performance is degrading over time.

---

## 5. Model artefact hosting

For the assignment, the key artefacts are provided via a shared Google Drive link as `models.zip`. After downloading:

1. Unzip `models.zip`.
2. Copy the following files into the local `models/` directory at the project root:
   - `models/anomaly_detector_config.json`
   - `models/forecaster.pkl`
   - `models/baseline_stats.json`

These files allow reviewers to:

- Use the **trained anomaly detector configuration** without re-fitting.
- Load the **trained forecaster** directly.
- Inspect the **baseline metrics** used by monitoring.

---

## 6. Environment setup & run instructions

### 6.1 Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scriptsctivate       # Windows PowerShell
```

### 6.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 6.3 Ensure datasets are in place

```bash
ls data/oxygen.csv
```

If the filenames or paths differ, update `configs/pipeline_config.yaml` (the `data` section) accordingly.

### 6.4 Run pipelines

- **Train + promote (registry-aware)**

  ```bash
  python -m src.models.train_and_promote --config configs/pipeline_config.yaml --model-dir models --registry-dir models/registry
  ```

- **Train only (legacy direct run)**

  ```bash
  python -m src.models.train_pipeline --config configs/pipeline_config.yaml
  ```

- **Batch inference on a new file**

  ```bash
  python -m src.inference.batch_inference --config configs/pipeline_config.yaml --input data/raw/oxygen_sample.csv --output data/processed/oxygen_sample_forecast.csv
  ```

- **1-week horizon forecast for a single sensor**

  ```bash
  python -m src.inference.horizon_forecast --config configs/pipeline_config.yaml --processed data/oxygen_processed_full.csv --sensor_id 'System_10|EquipmentUnit_10|SubUnit_07' --output data/processed/forecast_1week_SubUnit_07.csv
  ```

### 6.5 Model registry & promotion (artefact tracking)

- **Where artefacts live**
  - Production: `models/` (anomaly config, forecaster, baseline stats).
  - History: `models/registry/runs/<run_id>/` keeps every training run artefact + metadata.
  - Pointer: `models/registry/production.json` records the latest promoted run id + metric used.

- **Promotion rule**
  - Uses `test_rmse` in `baseline_stats.json` (lower is better).
  - First run wins by default; later runs must beat the current production metric.

- **How to run**
  - Locally:

    ```bash
    python -m src.models.train_and_promote --config configs/pipeline_config.yaml --model-dir models --registry-dir models/registry
    ```

  - Docker: see Section 8 (default `CMD` already runs this).

- **Outputs per run**
  - `run_info.json` summarising metrics, promotion decision, and paths.
  - Artefacts saved alongside the metadata for reproducibility.

---

## 7. Limitations & possible improvements

### 7.1 Current limitations

- **Rule-based anomaly detection**  
  Highly interpretable but may miss subtle or high-dimensional anomaly patterns that a learned model (e.g. autoencoder, isolation forest, or deep sequence model) could capture.

- **Single global forecaster**  
  Reduces model duplication and respects the assignment constraints, but:
  - May underfit very atypical sensors.
  - Does not explicitly model sensor-specific trends beyond what can be captured through lags and rolling stats.

- **Assumes roughly regular 1-minute sampling**  
  Large gaps, irregular sampling, or missing blocks are not explicitly modelled; they are implicitly handled via cleaning and feature construction.

- **Batch-oriented implementation**  
  Scripts are designed for batch runs (e.g. nightly jobs). Real-time APIs, model serving infrastructure, and CI/CD are out of scope for this assignment.

### 7.2 Possible improvements

- **Learned anomaly models**  
  Add an optional ML-based anomaly detector (e.g. isolation forest on residuals, sequence models on windows) alongside the rule-based detector to improve recall for complex behaviours.

- **Per-segment or hierarchical forecasting**  
  Introduce a hierarchical or mixture-of-experts forecaster that can specialise on different groups of sensors while retaining a shared core.

- **Better handling of irregular data**  
  Incorporate explicit gap handling, imputation strategies, or models designed for irregular time series.

- **Uncertainty estimation**  
  Extend forecasts with prediction intervals (e.g. via quantile regression or ensembles) to communicate uncertainty in oxygen predictions.

- **Productionisation**  
  Wrap the pipelines into containerised services, add scheduling (e.g. Airflow), centralised logging, and model registry integration (MLflow or similar) for a full production MLOps setup.

These enhancements would build on the current assignment-focused implementation while preserving its interpretability and tag-agnostic design.

---

## 8. Docker usage

This project includes a lightweight Docker setup so you can run **training**, **batch inference**, **horizon forecasting**, and **monitoring** without installing Python dependencies on your host.

### 8.1 Prerequisites

- Docker installed on your machine.
- Project directory structure as described above (with `data/`, `models/`, `configs/`, `src/`, etc.).
- A `requirements.txt` file in the project root containing the Python dependencies used in this project.

The Dockerfile is expected to live at the project root (`cefalo-oxygen-anomaly/Dockerfile`).

### 8.2 Build the Docker image

From the project root (`cefalo-oxygen-anomaly/`):

```bash
docker build -t cefalo-oxygen:latest .
```

This creates an image called `cefalo-oxygen:latest` with the code, dependencies, and an entrypoint that runs:

```bash
python -m <module> [args...]
```

### 8.3 Volume mounts and working directory

To make sure data, configs, and models are shared between your host and the container, mount the following directories:

- `./data` → `/app/data`
- `./models` → `/app/models`
- `./configs` → `/app/configs`

All example commands below assume you are running from the **project root** and use these volume mounts.

### 8.4 Training via Docker

Equivalent local command:

```bash
python -m src.models.train_and_promote --config configs/pipeline_config.yaml --model-dir models --registry-dir models/registry
```

Run via Docker:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/configs:/app/configs" \
  cefalo-oxygen:latest \
  src.models.train_and_promote \
    --config configs/pipeline_config.yaml \
    --model-dir models \
    --registry-dir models/registry
```

> Tip: `.dockerignore` now excludes data/models/notebooks and cache files to keep the build context small and avoid snapshot/export errors.

This will:

- Read raw data from `data/oxygen.csv`,
- Train the anomaly detector and global forecaster,
- Write processed data to `data/oxygen_processed_full.csv`,
- Save artefacts to `models/registry/runs/<run_id>/`,
- Promote artefacts to `models/` **only if** the new `test_rmse` beats the current baseline.

You can switch to a different config file just by changing the `--config` argument.

### 8.5 Batch inference via Docker

Equivalent local command:

```bash
python -m src.inference.batch_inference --config configs/pipeline_config.yaml --input data/raw/oxygen_sample.csv --output data/processed/oxygen_sample_forecast.csv
```

Run via Docker:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/configs:/app/configs" \
  cefalo-oxygen:latest \
  src.inference.batch_inference \
    --config configs/pipeline_config.yaml \
    --input data/raw/oxygen_sample.csv \
    --output data/processed/oxygen_sample_forecast.csv
```

Key points:

- `--input` and `--output` are paths **inside** the container (`/app/...`).
- Because `./data` is mounted to `/app/data`, paths like `data/raw/oxygen_sample.csv` map 1:1 between host and container.
- You can pass any other CSV path under `data/` without changing the image.

### 8.6 Horizon (1-week) forecast via Docker

Equivalent local command:

```bash
python -m src.inference.horizon_forecast --config configs/pipeline_config.yaml --processed data/oxygen_processed_full.csv --sensor_id 'System_10|EquipmentUnit_10|SubUnit_07' --output data/processed/forecast_1week_SubUnit_07.csv
```

Run via Docker (note the quotes around `sensor_id` to avoid shell pipes):

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/configs:/app/configs" \
  cefalo-oxygen:latest \
  src.inference.horizon_forecast \
    --config configs/pipeline_config.yaml \
    --processed data/processed/oxygen_processed_full.csv \
    --sensor_id 'System_10|EquipmentUnit_10|SubUnit_07' \
    --output data/processed/forecast_1week_SubUnit_07.csv
```

You can:

- Change `--processed` to point to any processed/scored dataset,
- Change `--sensor_id` to target another tag (always quote if it contains `|`),
- Change `--output` to write forecasts to a different path under `data/`.

### 8.7 Monitoring via Docker

`src/evaluation/monitoring.py` exposes a `monitor_new_data(...)` function, which you can call from a small Python snippet. The general workflow is:

1. Use **batch inference** to generate a file containing anomaly scores and forecasts (e.g. `data/processed/oxygen_sample_forecast.csv` with columns like `severity` and `y_pred`).
2. Compare this against the baseline metrics in `models/baseline_stats.json` using the monitoring function.

A simple way to do this in Docker is to override the entrypoint and run a short inline script:

```bash
docker run --rm \
  --entrypoint python \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  cefalo-oxygen:latest \
  -c "import pandas as pd; from src.evaluation.monitoring import monitor_new_data; df = pd.read_csv('data/processed/oxygen_sample_forecast.csv'); out = monitor_new_data(df_scored=df, df_with_preds=df, model_dir='models', registry_dir='models/registry'); print(out)"
```

This will:

- Load the new scored + forecasted data,
- Load the production `models/baseline_stats.json` (and production run id, if present),
- Return severity / RMSE drift warnings.

If you prefer a cleaner interface, you can add a small CLI wrapper (e.g. `src/evaluation/run_monitoring.py`) and then run it via Docker exactly like the training and inference modules:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/configs:/app/configs" \
  cefalo-oxygen:latest \
  src.evaluation.run_monitoring \
    --scored_path data/processed/oxygen_sample_forecast.csv \
    --baseline_stats_path models/baseline_stats.json
```

### 8.8 Using docker-compose (optional)

If you add the following `docker-compose.yml` to the project root:

```yaml
version: "3.9"

services:
  oxygen:
    build: .
    image: cefalo-oxygen:latest
    working_dir: /app
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./configs:/app/configs
    command: ["src.models.train_and_promote", "--config", "configs/pipeline_config.yaml", "--model-dir", "models", "--registry-dir", "models/registry"]
```

You can run:

- **Build image**

  ```bash
  docker compose build
  ```

- **Train** (default command)

  ```bash
  docker compose run --rm oxygen
  ```

- **Batch inference**

  ```bash
  docker compose run \
    --rm \
    oxygen \
    src.inference.batch_inference \
      --config configs/pipeline_config.yaml \
      --input data/raw/oxygen_sample.csv \
      --output data/processed/oxygen_sample_forecast.csv
  ```

- **Horizon forecast**

  ```bash
  docker compose run \
    --rm \
    oxygen \
    src.inference.horizon_forecast \
      --config configs/pipeline_config.yaml \
      --processed data/oxygen_processed_full.csv \
      --sensor_id 'System_10|EquipmentUnit_10|SubUnit_07' \
      --output data/processed/forecast_1week_SubUnit_07.csv
  ```

This keeps all volume mounts and image configuration in a single declarative file and lets you focus on the module + arguments only.
