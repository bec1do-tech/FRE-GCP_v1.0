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

# System libs needed by psycopg2 (libpq) and Pillow
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Cloud Run injects PORT; default to 8080 for local testing
ENV PORT=8080
EXPOSE 8080

# ADK web server — serves the agent at /run, /run_sse, /list-apps
CMD ["adk", "web", "--host", "0.0.0.0", "--port", "8080"]
