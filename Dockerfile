# ──────────────────────────────────────────────────────────────────────────────
# BDD100K Data Analysis Container
#
# Build:
#   docker build -t bdd-analysis .
#
# Run – train split only:
#   docker run --rm \
#       -v /host/BDD100k:/data:ro \
#       -v /host/results:/results \
#       bdd-analysis \
#       /data/labels/train --output-dir /results
#
# Run – train + val (enables comparison plots and anomaly detection):
#   docker run --rm \
#       -v /host/BDD100k:/data:ro \
#       -v /host/results:/results \
#       bdd-analysis \
#       /data/labels/train \
#       --val-dir /data/labels/val \
#       --output-dir /results
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.9-slim

WORKDIR /app

# Optional proxy args (pass with --build-arg if required in your network)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY}

# ── System dependencies for matplotlib/seaborn (font rendering) ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        fontconfig \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
COPY data_analysis/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ─────────────────────────────────────────────────────
COPY data_analysis/ /app/
COPY .pylintrc /app/

# ── Default output mount-point ─────────────────────────────────────────────
RUN mkdir -p /results

# ── Entry-point ────────────────────────────────────────────────────────────
# All CLI flags are forwarded to main.py; see module docstring for examples.
ENTRYPOINT ["python", "main.py"]