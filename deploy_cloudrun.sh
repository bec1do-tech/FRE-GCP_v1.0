#!/usr/bin/env bash
# =============================================================================
# deploy_cloudrun.sh — Deploy FRE Cognitive Search Agent to Cloud Run
# =============================================================================
# Prerequisites:
#   - gcloud CLI installed and authenticated:  gcloud auth login
#   - Docker installed and running (for local build/test)
#   - All secrets already stored in Secret Manager (see SECRETS section below)
#
# Usage:
#   chmod +x deploy_cloudrun.sh
#   ./deploy_cloudrun.sh               # build + push + deploy
#   ./deploy_cloudrun.sh --build-only  # only build & push image, skip deploy
#   ./deploy_cloudrun.sh --deploy-only # skip build, re-deploy existing image
#
# What this script does:
#   1. Builds the container image using Cloud Build (no local Docker needed)
#   2. Pushes it to Artifact Registry
#   3. Deploys to Cloud Run with all required env vars + Secret Manager wiring
#   4. Prints the service URL and runs a quick health check
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIGURE THESE VALUES FOR YOUR ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ID="rbprj-100622"
REGION="us-central1"
SERVICE_NAME="fre-cognitive-search"
IMAGE_REPO="fre"                                    # Artifact Registry repo name
IMAGE_NAME="cognitive-search-agent"
IMAGE_TAG="latest"

# Cloud SQL
CLOUD_SQL_INSTANCE="${PROJECT_ID}:${REGION}:fre-cognitive-search"

# GCS
GCS_BUCKET="fre-cognitive-search-docs"
GCS_PREFIX="Example_Dataset/"

# Elasticsearch (Elastic Cloud)
ES_URL="https://my-elasticsearch-project-a8393b.es.us-central1.gcp.elastic.cloud:443"
ES_INDEX="cognitive_search_docs"

# Vertex AI Vector Search
VERTEX_INDEX_ENDPOINT="projects/738231548859/locations/us-central1/indexEndpoints/1458579140058808320"
VERTEX_DEPLOYED_INDEX_ID="fre_cognitive_search_1776786977108"
VERTEX_INDEX_NAME="projects/738231548859/locations/us-central1/indexes/3549744702972493824"

# Vertex AI Search / VAIS
VAIS_DATA_STORE_ID="fre-database_1777369569668"
VAIS_ENGINE_ID="fileresearchengine_1777366613744"
VAIS_LOCATION="global"

# Service account that Cloud Run will run AS (needs all resource permissions)
SERVICE_ACCOUNT="fre-cloudrun-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Full image path
FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# ─────────────────────────────────────────────────────────────────────────────
# Parse flags
# ─────────────────────────────────────────────────────────────────────────────
BUILD=true
DEPLOY=true
for arg in "$@"; do
  case $arg in
    --build-only)  DEPLOY=false  ;;
    --deploy-only) BUILD=false   ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Ensure Artifact Registry repo exists
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== [0/3] Ensuring Artifact Registry repo '${IMAGE_REPO}' exists ==="
gcloud artifacts repositories describe "${IMAGE_REPO}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    > /dev/null 2>&1 \
|| gcloud artifacts repositories create "${IMAGE_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="FRE Cognitive Search Agent images"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build & push with Cloud Build (no local Docker / no proxy involved)
# ─────────────────────────────────────────────────────────────────────────────
if [ "${BUILD}" = true ]; then
  echo ""
  echo "=== [1/3] Building image with Cloud Build ==="
  echo "    Image: ${FULL_IMAGE}"
  echo ""
  gcloud builds submit . \
      --tag="${FULL_IMAGE}" \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --timeout="20m"
  echo "    Build complete."
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Store secrets in Secret Manager (idempotent — skips if exists)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== [2/3] Checking Secret Manager secrets ==="

create_secret_if_missing() {
  local name="$1"
  local prompt="$2"
  if gcloud secrets describe "${name}" --project="${PROJECT_ID}" > /dev/null 2>&1; then
    echo "    ✓ ${name} already exists"
  else
    echo ""
    echo "    Secret '${name}' not found."
    echo "    ${prompt}"
    read -r -s -p "    Value (hidden): " secret_value
    echo ""
    echo -n "${secret_value}" \
      | gcloud secrets create "${name}" \
          --data-file=- \
          --project="${PROJECT_ID}" \
          --replication-policy=automatic
    echo "    ✓ ${name} created"
  fi
}

create_secret_if_missing "fre-postgres-password" \
  "Enter the Cloud SQL postgres user password (POSTGRES_PASSWORD):"

create_secret_if_missing "fre-es-api-key" \
  "Enter the Elasticsearch Cloud API key (ELASTICSEARCH_API_KEY):"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Deploy to Cloud Run
# ─────────────────────────────────────────────────────────────────────────────
if [ "${DEPLOY}" = true ]; then
  echo ""
  echo "=== [3/3] Deploying to Cloud Run ==="

  gcloud run deploy "${SERVICE_NAME}" \
    --image="${FULL_IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --platform=managed \
    \
    --service-account="${SERVICE_ACCOUNT}" \
    \
    --add-cloudsql-instances="${CLOUD_SQL_INSTANCE}" \
    \
    --set-env-vars="^|^GCP_PROJECT=${PROJECT_ID}\
|GCP_REGION=${REGION}\
|GOOGLE_CLOUD_PROJECT=${PROJECT_ID}\
|GOOGLE_GENAI_USE_VERTEXAI=true\
|GCS_BUCKET=${GCS_BUCKET}\
|GCS_PREFIX=${GCS_PREFIX}\
|CLOUD_SQL_INSTANCE=${CLOUD_SQL_INSTANCE}\
|POSTGRES_HOST=127.0.0.1\
|POSTGRES_PORT=5432\
|POSTGRES_DB=fre_cognitive_search\
|POSTGRES_USER=postgres\
|ELASTICSEARCH_URL=${ES_URL}\
|ELASTICSEARCH_INDEX=${ES_INDEX}\
|VERTEX_AI_INDEX_ENDPOINT=${VERTEX_INDEX_ENDPOINT}\
|VERTEX_AI_DEPLOYED_INDEX_ID=${VERTEX_DEPLOYED_INDEX_ID}\
|VERTEX_AI_INDEX_NAME=${VERTEX_INDEX_NAME}\
|VAIS_DATA_STORE_ID=${VAIS_DATA_STORE_ID}\
|VAIS_ENGINE_ID=${VAIS_ENGINE_ID}\
|VAIS_LOCATION=${VAIS_LOCATION}\
|GEMINI_MODEL=gemini-2.5-flash\
|GEMINI_SYNTHESIS_MODEL=gemini-2.5-flash\
|GEMINI_ROUTER_MODEL=gemini-2.5-flash\
|GEMINI_VISION_MODEL=gemini-2.5-flash\
|VERTEX_EMBEDDING_MODEL=text-embedding-004\
|VERTEX_EMBEDDING_DIM=768\
|PREVIEW_SIGNING_SA=${SERVICE_ACCOUNT}\
|DEFAULT_TOP_K=5\
|PORT=8080" \
    \
    --set-secrets="POSTGRES_PASSWORD=fre-postgres-password:latest,ELASTICSEARCH_API_KEY=fre-es-api-key:latest" \
    \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=5 \
    --concurrency=10 \
    --timeout=300 \
    \
    --no-allow-unauthenticated

  echo ""
  echo "=== Deployment complete ==="
  SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format="value(status.url)")
  echo "  Service URL : ${SERVICE_URL}"

  # Update agent_card.json with the real URL
  if [ -f agent_card.json ]; then
    sed -i "s|PLACEHOLDER_CLOUD_RUN_URL|${SERVICE_URL}|g" agent_card.json
    echo "  agent_card.json updated with live URL"
  fi

  echo ""
  echo "=== Quick health check ==="
  HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    "${SERVICE_URL}/list-apps" || echo "000")
  if [ "${HTTP_STATUS}" = "200" ]; then
    echo "  ✓ /list-apps returned HTTP 200 — service is healthy"
  else
    echo "  ✗ /list-apps returned HTTP ${HTTP_STATUS} — check Cloud Run logs:"
    echo "    gcloud run services logs read ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID}"
  fi
fi

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  Run deployment tests:"
echo "    export CLOUD_RUN_URL=${SERVICE_URL:-'https://YOUR-SERVICE-URL'}"
echo "    python test_deployment.py"
echo ""
echo "  View logs:"
echo "    gcloud run services logs read ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID} --limit 50"
echo ""
echo "  Ingest the 4 GB dataset (run from Cloud Shell — no proxy):"
echo "    python ingest_bulk.py --workers 6"
