# Data Directory

Place all oxygen CSV inputs and derived artefacts here.

- `raw/` should contain the original `oxygen.csv` file received from the field teams.
- `processed/` stores scored/cleaned outputs such as `oxygen_processed_full.csv`.

These folders are mounted into the Docker image, so keep local copies only—do not commit proprietary data.***
