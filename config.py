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

# ── Cloud Storage ─────────────────────────────────────────────────────────────
GCS_BUCKET: str = os.environ.get("GCS_BUCKET", "")
GCS_PREFIX: str = os.environ.get("GCS_PREFIX", "documents/")

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

# ── Gemini / Vertex AI Models ──────────────────────────────────────────────────
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL: str = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.0-flash")
VERTEX_EMBEDDING_MODEL: str = os.environ.get(
    "VERTEX_EMBEDDING_MODEL", "text-embedding-004"
)
VERTEX_EMBEDDING_DIM: int = int(os.environ.get("VERTEX_EMBEDDING_DIM", "768"))

# ── Ingestion Tuning ──────────────────────────────────────────────────────────
MAX_FILE_MB: int = int(os.environ.get("MAX_FILE_MB", "50"))
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "500"))    # approx words
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "50"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))

# ── Search Tuning ─────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = int(os.environ.get("DEFAULT_TOP_K", "5"))
RRF_K: int = int(os.environ.get("RRF_K", "60"))
