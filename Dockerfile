# Dockerfile

FROM python:3.11-slim

# System deps (for scientific Python / sklearn wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir inside the container
WORKDIR /app

# Install Python dependencies
# (Assumes you have requirements.txt in repo root.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Environment tweaks
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Ensure /app is on Python path (usually already is, but explicit is fine)
ENV PYTHONPATH=/app

# Default entrypoint allows us to run any module via `docker run ... <module> <args>`
ENTRYPOINT ["python", "-m"]

# Default command trains + promotes a model using the registry workflow.
# Override in `docker run` to execute inference / monitoring modules.
CMD [
  "src.models.train_and_promote",
  "--config", "configs/pipeline_config.yaml",
  "--model-dir", "models",
  "--registry-dir", "models/registry"
]
