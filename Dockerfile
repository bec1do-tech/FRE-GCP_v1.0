# ─────────────────────────────────────────────────────────────────────────────
# FRE GCP v1.0 — Multi-stage Dockerfile
# Builds the Cloud Run container that hosts the ADK cognitive search agent.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .

# Install wheels into a prefix we can copy cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim

# System libs:
#   libpq5          — psycopg2 runtime
#   libgl1           — PyMuPDF (libGL.so.1) on Debian Bookworm
#   libglib2.0-0     — PyMuPDF dependency
#   libgomp1         — OpenMP (used by some MuPDF builds)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libpq5 \
       libgl1 \
       libglib2.0-0 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Make entrypoint executable (needed when built on Windows — line endings fixed too)
RUN sed -i 's/\r//' entrypoint.sh && chmod +x entrypoint.sh

# Cloud Run injects PORT; default to 8080 for local testing
ENV PORT=8080
EXPOSE 8080

# Run startup smoke tests, then launch ADK web server
ENTRYPOINT ["./entrypoint.sh"]
