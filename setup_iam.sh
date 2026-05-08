#!/usr/bin/env bash
# =============================================================================
# setup_iam.sh — One-time IAM & infrastructure setup for Cloud Run deployment
# =============================================================================
# Run this ONCE before the first deploy_cloudrun.sh invocation.
# Requires: gcloud CLI, Owner or roles/iam.admin on the project.
#
# This is also the script to share with your GCP admin / deployment contact —
# it documents exactly what permissions the Cloud Run service account needs.
#
# Usage:
#   chmod +x setup_iam.sh
#   ./setup_iam.sh
# =============================================================================

set -euo pipefail

PROJECT_ID="rbprj-100622"
REGION="us-central1"
SA_NAME="fre-cloudrun-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
CLOUD_SQL_INSTANCE="fre-cognitive-search"
GCS_BUCKET="fre-cognitive-search-docs"

echo ""
echo "=== FRE Cloud Run — IAM Setup ==="
echo "    Project : ${PROJECT_ID}"
echo "    SA      : ${SA_EMAIL}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Create the service account
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/7] Creating service account …"
gcloud iam service-accounts create "${SA_NAME}" \
  --project="${PROJECT_ID}" \
  --display-name="FRE Cloud Run Service Account" \
  --description="Runs the FRE Cognitive Search Agent on Cloud Run" \
  2>/dev/null || echo "  (already exists — skipping)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. GCS — read/write documents + read/write session_previews
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/7] Granting GCS access …"
gsutil iam ch \
  "serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin" \
  "gs://${GCS_BUCKET}"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Cloud SQL — connect via Cloud SQL Auth Proxy / Connector
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/7] Granting Cloud SQL client role …"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/cloudsql.client" \
  --condition=None

# ─────────────────────────────────────────────────────────────────────────────
# 4. Vertex AI — embeddings + Vector Search + Gemini API
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/7] Granting Vertex AI user role …"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user" \
  --condition=None

# ─────────────────────────────────────────────────────────────────────────────
# 5. Secret Manager — read the two secrets (postgres password + ES API key)
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/7] Granting Secret Manager accessor …"
for SECRET in fre-postgres-password fre-es-api-key; do
  gcloud secrets add-iam-policy-binding "${SECRET}" \
    --project="${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    2>/dev/null || echo "  WARNING: secret '${SECRET}' not found — create it first with deploy_cloudrun.sh"
done

# ─────────────────────────────────────────────────────────────────────────────
# 6. GCS signed URLs — SA needs to sign its own tokens (for document preview)
#    The Cloud Run SA impersonates the preview signing SA via Token Creator.
# ─────────────────────────────────────────────────────────────────────────────
echo "[6/7] Granting Service Account Token Creator (for GCS signed URLs) …"
PREVIEW_SA="web-tester-local@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts add-iam-policy-binding "${PREVIEW_SA}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountTokenCreator"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Vertex AI Search (Discovery Engine) — query the data store
# ─────────────────────────────────────────────────────────────────────────────
echo "[7/7] Granting Discovery Engine viewer …"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/discoveryengine.viewer" \
  --condition=None

echo ""
echo "=== IAM setup complete ==="
echo ""
echo "Summary of roles granted to ${SA_EMAIL}:"
echo "  roles/storage.objectAdmin        — GCS bucket (read/write docs + previews)"
echo "  roles/cloudsql.client            — Cloud SQL (connect via Connector)"
echo "  roles/aiplatform.user            — Vertex AI (Gemini + embeddings + Vector Search)"
echo "  roles/secretmanager.secretAccessor — Secret Manager (postgres pw + ES key)"
echo "  roles/iam.serviceAccountTokenCreator — GCS signed URL generation"
echo "  roles/discoveryengine.viewer     — Vertex AI Search / VAIS"
echo ""
echo "Next: run ./deploy_cloudrun.sh"
