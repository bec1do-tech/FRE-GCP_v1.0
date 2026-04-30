# FRE GCP v1.0 — Cognitive Search Agent

> Transform a 10 TB enterprise document repository into a conversational knowledge base — powered by Google ADK, Gemini, Cloud SQL, Elasticsearch, and Vertex AI Vector Search.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              cognitive_search_agent  (root Agent)            │
│  Routes between: Document Q&A  │  Ingestion  │  Status      │
└──────────────────┬──────────────┬──────────────┬─────────────┘
                   │              │              │
                   ▼              ▼              ▼
          document_qa_    ingestion_manager  get_search_
          pipeline        _agent              status tool
          (SequentialAgent) (Agent)
                   │
        ┌──────────┴────────────┐
        ▼                       ▼
  parallel_search_          synthesis_
  gatherer                  agent
  (ParallelAgent)           (Agent)
   ├─ es_search_agent         │
   └─ vertex_search_agent  ◄──┘
         │                (reads both
         ▼                 result sets)
  Hybrid BM25 +
  Semantic results
```

### ADK Patterns Used
| Pattern | Where | Purpose |
|---|---|---|
| `Agent` | root, synthesis, ingestion | Conversational reasoning + tool calling |
| `SequentialAgent` | `document_qa_pipeline` | Gather results → then synthesise |
| `ParallelAgent` | `parallel_search_gatherer` | ES + Vertex AI search simultaneously |

---

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Agent Framework** | Google ADK 0.3 | Orchestration, routing, tool calling |
| **AI / Reasoning** | Gemini 2.5 Flash | Answer synthesis, routing decisions |
| **Vision** | Gemini 2.0 Flash | Image & chart understanding in PDFs/PPTX |
| **Embeddings** | Vertex AI `text-embedding-004` | 768-dim dense vectors |
| **Keyword Search** | Elasticsearch 8 (Elastic Cloud) | BM25 + metadata filtering |
| **Vector Search** | Vertex AI Vector Search | Semantic ANN retrieval |
| **Object Storage** | Cloud Storage (GCS) | Raw document source of truth (10 TB) |
| **Metadata DB** | Cloud SQL — PostgreSQL 16 | Documents, chunks, conversation history |
| **Event Trigger** | Cloud Functions (gen2) | Automated pipeline on GCS upload |
| **Compute** | Cloud Run | Hosts the ADK agent (serverless) |

---

## Project Structure

```
FRE_GCP_v1.0/
├── cognitive_search_agent/          # ADK multi-agent system
│   ├── agent.py                     # Root agent + pipeline assembly
│   └── sub_agents/
│       ├── es_search_agent/         # Elasticsearch BM25 specialist
│       ├── vertex_search_agent/     # Vertex AI semantic specialist
│       ├── synthesis_agent/         # Answer synthesis + citations
│       └── ingestion_agent/         # Ingestion manager
│
├── ingestion/                       # Document processing pipeline
│   ├── extractor.py                 # Multimodal extraction (Gemini Vision)
│   ├── chunker.py                   # Semantic text chunking with overlap
│   ├── pipeline.py                  # Orchestrates full ingestion flow
│   └── gcs_trigger.py               # Cloud Functions entry point
│
├── search/                          # Search backends
│   ├── es_index.py                  # Elasticsearch client + mapping
│   ├── vertex_vector.py             # Vertex AI Vector Search + embeddings
│   └── hybrid.py                    # Reciprocal Rank Fusion (RRF)
│
├── storage/                         # Persistence layer
│   ├── gcs.py                       # Cloud Storage helpers
│   └── postgres.py                  # Cloud SQL schema + queries
│
├── tools/                           # ADK tool wrappers
│   ├── search_tools.py              # hybrid_search, get_document_chunks
│   └── ingestion_tools.py           # trigger_*_ingestion, get_ingestion_status
│
├── config.py                        # All config from environment variables
├── requirements.txt
├── .env.example                     # Copy to .env and fill in values
├── Dockerfile                       # Cloud Run container
└── docker-compose.yml               # Local dev: postgres + elasticsearch
```

---

## Quick Start — Local Development

### 1. Prerequisites

```bash
# Python 3.12
python --version

# Google Cloud SDK (for ADC)
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Clone and install

```bash
cd FRE_GCP_v1.0
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in at minimum:
#   GOOGLE_API_KEY or set up Application Default Credentials
#   GCP_PROJECT
```

### 4. Start local services (PostgreSQL + Elasticsearch)

```bash
docker compose up -d
```

### 5. Run the ADK agent

```bash
adk web
# Open http://localhost:8000/dev-ui
# Select: cognitive_search_agent
```

---

## GCP Setup (Production)

### One-time Infrastructure

```bash
# 1. Create GCS bucket
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME

# 2. Create Cloud SQL (PostgreSQL) instance
gcloud sql instances create fre-db \
  --database-version=POSTGRES_16 \
  --tier=db-f1-micro \
  --region=us-central1

gcloud sql databases create cognitive_search --instance=fre-db
gcloud sql users set-password postgres --instance=fre-db --password=YOUR_PASSWORD

# 3. Create Vertex AI Vector Search Index
# See search/vertex_vector.py docstring for full gcloud commands

# 4. Deploy Cloud Function (auto-trigger on GCS upload)
gcloud functions deploy fre-ingest-trigger \
  --gen2 \
  --runtime=python312 \
  --region=us-central1 \
  --source=. \
  --entry-point=process_gcs_event \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=YOUR_BUCKET_NAME" \
  --set-env-vars GCP_PROJECT=YOUR_PROJECT,...

# 5. Deploy ADK agent to Cloud Run
gcloud run deploy cognitive-search-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT=YOUR_PROJECT,...
```

---

## Ingestion Pipeline

### Automated (Event-Driven)
Upload any supported file to the GCS bucket → Cloud Function triggers automatically → file is extracted, chunked, embedded, and indexed.

```bash
gsutil cp report.pdf gs://YOUR_BUCKET/documents/
# Pipeline runs automatically within ~30 seconds
```

### Manual (via Agent)
Ask the agent directly:
```
You: Index the document gs://my-bucket/documents/Q4_report.pdf
You: Index everything in gs://my-bucket/finance/2024/
You: How many documents are indexed?
```

### Programmatic
```python
from ingestion.pipeline import process_document, process_folder

# Single file
result = process_document("gs://bucket/report.pdf")

# Entire folder
results = process_folder("my-bucket", prefix="documents/", max_workers=8)
```

---

## Search: Reciprocal Rank Fusion (RRF)

Two-phase hybrid search:

1. **Elasticsearch BM25** — lightning-fast keyword + metadata filtering (author, department, date, case_id)
2. **Vertex AI Vector Search** — semantic ANN search over 768-dim embeddings
3. **RRF Fusion** — `score(d) = Σ 1 / (k + rank(d))` with k=60

This ensures:
- Fast response even on 10 TB corpora (ES pre-filters, Vertex re-ranks)
- Keyword terms AND conceptual meaning are both considered
- Both backends are independently optional (graceful degradation)

---

## Multimodal Understanding

The extractor calls Gemini Vision on every image, chart, and diagram found inside:

| File Type | Text Extraction | Visual Extraction |
|---|---|---|
| PDF | pypdf (page text) | Gemini Vision on every embedded image |
| DOCX | python-docx | Gemini Vision on inline images |
| PPTX | python-pptx (slide text) | Gemini Vision on slide images |
| XLSX | openpyxl (cell values) | — |
| TXT/MD/CSV | direct read | — |

Example: a bar chart on page 3 of a PDF becomes searchable as:
> *"[Image on page 3: Bar chart showing quarterly revenue from Q1 to Q4 2024. Q4 shows the highest revenue at €42M, representing a 15% increase over Q3.]"*

---

## Example Conversations

**Complex conceptual question:**
```
You:    What were the main supply chain risks cited in our Q4 2024 reports?

Agent:  Based on Q4 2024 reports indexed in the system:

        **Answer**: Three primary supply chain risks were identified across 
        multiple Q4 reports:

        **Key Findings**:
        • Single-supplier dependency for critical components increased 
          vulnerability [Source: Q4_Risk_Assessment.pdf — gs://bucket/…]
        • Port congestion in Southeast Asian Routes caused 3-week delays
          [Source: Supply_Chain_Report_Q4.pdf — gs://bucket/…]
        • ...

        **Sources Consulted**:
        1. gs://bucket/reports/Q4_Risk_Assessment.pdf
        2. gs://bucket/finance/Supply_Chain_Report_Q4.pdf
```

**Metadata-filtered search:**
```
You:   Show me all compliance documents from the Legal department from 2023

Agent: [Delegates to pipeline with filters: department=Legal, date_from=2023-01-01, date_to=2023-12-31]
```

---

## Deduplication

Every file is hashed (MD5) before processing. If the same content is uploaded twice — under a different name or path — the pipeline detects the duplicate and skips re-indexing. This eliminates duplicate entries and prevents wasted embedding API calls on multi-terabyte corpora.

---

## Security Notes

- All credentials are loaded from environment variables (never hardcoded).
- The Cloud Run service account should follow least-privilege:
  - `roles/storage.objectViewer` on the source GCS bucket
  - `roles/cloudsql.client` on the Cloud SQL instance
  - `roles/aiplatform.user` for Vertex AI
- Use Secret Manager for production credentials instead of environment variables.
- The `GOOGLE_API_KEY` is only used for local development — in Cloud Run, use Application Default Credentials (ADC) instead.


