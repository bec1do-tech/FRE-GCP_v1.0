"""
FRE GCP v1.0 — Centralized Configuration
==========================================
All runtime parameters are loaded from environment variables.
Copy .env.example to .env and fill in your values for local development.
In Cloud Run / Cloud Functions, inject variables via the Secret Manager
or the service's environment variable settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env only in local dev (no-op when the file is absent, e.g. on Cloud Run)
load_dotenv(Path(__file__).parent / ".env")

# ── Google Cloud Platform ─────────────────────────────────────────────────────
GCP_PROJECT: str = os.environ.get("GCP_PROJECT", "")
GCP_REGION: str = os.environ.get("GCP_REGION", "us-central1")
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
# Path to a service-account JSON key (local dev / CI).  On Cloud Run / GKE
# leave unset — the workload identity / attached SA is used automatically.
GOOGLE_APPLICATION_CREDENTIALS: str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

# ── Cloud Storage ─────────────────────────────────────────────────────────────
GCS_BUCKET: str = os.environ.get("GCS_BUCKET", "")
GCS_PREFIX: str = os.environ.get("GCS_PREFIX", "documents/")
# Service account email used to sign GCS preview URLs (impersonated via ADC).
# The calling identity needs roles/iam.serviceAccountTokenCreator on this SA.
PREVIEW_SIGNING_SA: str = os.environ.get(
    "PREVIEW_SIGNING_SA", "web-tester-local@rbprj-100622.iam.gserviceaccount.com"
)

# ── Cloud SQL (PostgreSQL) ────────────────────────────────────────────────────
POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "cognitive_search")
POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "")
CLOUD_SQL_INSTANCE: str = os.environ.get("CLOUD_SQL_INSTANCE", "")

# Synchronous DSN (used by psycopg2 / SQLAlchemy create_engine)
POSTGRES_DSN: str = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── Elasticsearch (Elastic Cloud) ─────────────────────────────────────────────
ELASTICSEARCH_URL: str = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_API_KEY: str = os.environ.get("ELASTICSEARCH_API_KEY", "")
ELASTICSEARCH_INDEX: str = os.environ.get(
    "ELASTICSEARCH_INDEX", "cognitive_search_docs"
)

# ── Vertex AI Vector Search ────────────────────────────────────────────────────
VERTEX_AI_INDEX_ENDPOINT: str = os.environ.get("VERTEX_AI_INDEX_ENDPOINT", "")
VERTEX_AI_DEPLOYED_INDEX_ID: str = os.environ.get(
    "VERTEX_AI_DEPLOYED_INDEX_ID", "cognitive_search_index"
)
VERTEX_AI_INDEX_NAME: str = os.environ.get("VERTEX_AI_INDEX_NAME", "")

# ── Vertex AI Search / Gemini Enterprise ─────────────────────────────────────
VAIS_DATA_STORE_ID: str = os.environ.get("VAIS_DATA_STORE_ID", "")
# Gemini Enterprise engine ID — when set, queries route through the engine
# serving config which enables answer generation on top of the data store.
VAIS_ENGINE_ID: str = os.environ.get("VAIS_ENGINE_ID", "")
VAIS_LOCATION: str = os.environ.get("VAIS_LOCATION", "global")

# ── Gemini / Vertex AI Models ──────────────────────────────────────────────────
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL: str = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.0-flash")
VERTEX_EMBEDDING_MODEL: str = os.environ.get(
    "VERTEX_EMBEDDING_MODEL", "text-embedding-004"
)
VERTEX_EMBEDDING_DIM: int = int(os.environ.get("VERTEX_EMBEDDING_DIM", "768"))

# ── Ingestion Tuning ──────────────────────────────────────────────────────────
MAX_FILE_MB: int = int(os.environ.get("MAX_FILE_MB", "50"))
OCR_MAX_PDF_MB: int = int(os.environ.get("OCR_MAX_PDF_MB", "18"))   # Gemini inline PDF limit ~20 MB
OCR_TIMEOUT_S: int = int(os.environ.get("OCR_TIMEOUT_S", "120"))    # max seconds for OCR call
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "500"))    # approx words
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "50"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))

# ── Document Access ──────────────────────────────────────────────────────────
# When set, document citations use this as the base HTTP URL instead of
# generating GCS signed URLs.  Construct as: {DOCUMENT_BASE_URL}/{path}
# where {path} is everything after 'gs://{bucket}/' in the GCS URI.
#
# Local dev  → leave empty (signed URLs are generated automatically)
# Cloud Run  → set to your frontend route, e.g. https://fre.run.app/documents
DOCUMENT_BASE_URL: str = os.environ.get("DOCUMENT_BASE_URL", "")

# ── Search Tuning ─────────────────────────────────────────────────────────────
# DEFAULT_TOP_K: number of final RRF results returned to the synthesis agent.
# Increased to 10 so that with per-document diversity caps (max 2 chunks/doc)
# we can surface content from up to 5 different documents per query.
DEFAULT_TOP_K: int = int(os.environ.get("DEFAULT_TOP_K", "10"))
RRF_K: int = int(os.environ.get("RRF_K", "60"))
