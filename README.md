# Cefalo Oxygen Anomaly & Forecasting

End-to-end anomaly detection + forecasting solution for minute-level oxygen sensor data across multiple customers.

This repo implements:

- A **generic anomaly detector** with severity scoring for each minute.
- A **global forecasting model** that predicts oxygen for the next 1 week.
- A full lifecycle: **train → deploy/inference → monitor → retrain**.

---

## Repo Structure

```text
cefalo-oxygen-anomaly/
├─ data/                 # raw & processed samples (tiny, no full data)
├─ notebooks/            # EDA & experiments (supplied IPython notebooks)
├─ src/
│   ├─ data/             # loading, cleaning
│   ├─ features/         # windowing, feature engineering
│   ├─ models/           # forecasting & anomaly models + training pipeline
│   ├─ inference/        # scoring for new data & horizon forecasts
│   └─ evaluation/       # metrics, monitoring
├─ configs/              # YAML configs (window size, thresholds, etc.)
├─ models/               # saved models & scalers (for upload link)
└─ README.md
