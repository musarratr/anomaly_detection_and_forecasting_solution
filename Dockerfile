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

# We want to run Python modules by passing module name as first argument
# e.g. docker run image src.models.train_pipeline --config ...
ENTRYPOINT ["python", "-m"]

# Default command (can be overridden in `docker run`)
CMD ["src.models.train_pipeline", "--config", "configs/pipeline_config.yaml"]
