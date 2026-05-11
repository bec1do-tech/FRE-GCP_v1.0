#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# FRE GCP v1.0 — Cloud Run entrypoint
# Runs startup smoke tests before launching the ADK web server.
# If tests fail (exit 1), Cloud Run aborts this instance and retries.
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "============================================="
echo " FRE Cognitive Search Agent — Startup Checks"
echo "============================================="

python test_startup.py

echo ""
echo "============================================="
echo " Starting ADK web server on port ${PORT:-8080}"
echo "============================================="

exec adk web --port "${PORT:-8080}"
