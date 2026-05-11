"""
Cloud Run Startup Smoke Tests
==============================
Runs inside the container BEFORE adk web starts.
Called by entrypoint.sh — if this exits non-zero, Cloud Run aborts the
instance startup so it never receives production traffic.

Checks:
  1. Python imports (all agent modules load without error)
  2. GCS bucket reachable + at least 1 document exists
  3. Elasticsearch cluster healthy (ping + index exists)
  4. PostgreSQL connection + documents table accessible
  5. Vertex AI embedding API reachable (1 test embedding)
  6. Vertex AI Search data store reachable (1 test query)

Each check is independent — a soft warning is printed for non-critical
backends; hard exits are only on services that would make the agent
completely non-functional.

Run manually:
    python test_startup.py
"""

import os
import sys
import time

PASS = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

_failures: list[str] = []
_warnings: list[str] = []


def ok(label: str, detail: str = ""):
    print(f"  {PASS} {label}" + (f"  ({detail})" if detail else ""))


def warn(label: str, detail: str = ""):
    print(f"  {WARN} {label}" + (f"  — {detail}" if detail else ""))
    _warnings.append(label)


def fail(label: str, detail: str = "", hard: bool = False):
    print(f"  {FAIL} {label}" + (f"  — {detail}" if detail else ""))
    _failures.append(label)
    if hard:
        print(f"\n  CRITICAL: {label} is required. Aborting startup.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Python imports
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Checking Python imports …")
try:
    import config                                           # noqa: F401
    from cognitive_search_agent.agent import root_agent    # noqa: F401
    ok("all agent modules import cleanly", f"root agent: {root_agent.name}")
except Exception as exc:
    fail("agent import failed", str(exc), hard=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — GCS bucket reachable
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Checking GCS bucket …")
try:
    from storage.gcs import list_blobs
    t0 = time.perf_counter()
    blobs = list(list_blobs(config.GCS_BUCKET, prefix=config.GCS_PREFIX,
                             extensions=(".pdf", ".docx", ".pptx", ".xlsx")))
    elapsed = time.perf_counter() - t0
    if blobs:
        ok("GCS bucket reachable", f"{len(blobs)} documents found in {elapsed:.1f}s")
    else:
        warn("GCS bucket reachable but 0 documents found",
             f"bucket={config.GCS_BUCKET} prefix={config.GCS_PREFIX}")
except Exception as exc:
    # Soft-fail so the container starts even if GCS perms are not yet propagated
    fail("GCS bucket unreachable", str(exc), hard=False)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Elasticsearch
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Checking Elasticsearch …")
try:
    from search.es_index import _client as _es_client, _INDEX
    t0 = time.perf_counter()
    es = _es_client()
    elapsed = time.perf_counter() - t0
    if es is None:
        warn("Elasticsearch client returned None",
             "BM25 search will be unavailable")
    else:
        count = es.count(index=_INDEX).get("count", "?")
        ok("Elasticsearch healthy", f"{count} chunks indexed, ping {elapsed:.1f}s")
except Exception as exc:
    warn("Elasticsearch check failed", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — PostgreSQL (Cloud SQL)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Checking PostgreSQL …")
try:
    from storage.postgres import _get_connection
    t0 = time.perf_counter()
    con = _get_connection()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM documents;")
    doc_count = cur.fetchone()[0]
    cur.close()
    con.close()
    elapsed = time.perf_counter() - t0
    ok("PostgreSQL reachable", f"{doc_count} documents in DB, {elapsed:.1f}s")
except Exception as exc:
    warn("PostgreSQL check failed — chunk metadata unavailable", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — Vertex AI Embedding API
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Checking Vertex AI Embedding API …")
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    vertexai.init(project=config.GCP_PROJECT, location=config.GCP_REGION)
    t0 = time.perf_counter()
    model = TextEmbeddingModel.from_pretrained(config.VERTEX_EMBEDDING_MODEL)
    embeddings = model.get_embeddings(["startup connectivity test"])
    elapsed = time.perf_counter() - t0
    dim = len(embeddings[0].values)
    ok("Vertex AI Embedding API reachable", f"dim={dim}, {elapsed:.1f}s")
except Exception as exc:
    warn("Vertex AI Embedding API check failed — semantic search may be degraded", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — Vertex AI Search (VAIS) data store
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Checking Vertex AI Search data store …")
try:
    from google.cloud import discoveryengine_v1beta as discoveryengine
    client = discoveryengine.SearchServiceClient()
    serving_config = (
        f"projects/{config.GCP_PROJECT}/locations/{config.VAIS_LOCATION}"
        f"/collections/default_collection/engines/{config.VAIS_ENGINE_ID}"
        f"/servingConfigs/default_config"
    )
    t0 = time.perf_counter()
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query="startup test",
        page_size=1,
    )
    response = client.search(request)
    elapsed = time.perf_counter() - t0
    _ = list(response)  # consume iterator to confirm connection
    ok("Vertex AI Search reachable", f"{elapsed:.1f}s")
except Exception as exc:
    warn("Vertex AI Search check failed — enterprise search results may be unavailable", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
if _failures:
    print(f"  STARTUP FAILED: {len(_failures)} critical check(s) failed.")
    for f in _failures:
        print(f"    • {f}")
    sys.exit(1)
elif _warnings:
    print(f"  STARTUP OK with {len(_warnings)} warning(s) — non-critical backends degraded.")
    for w in _warnings:
        print(f"    • {w}")
    sys.exit(0)
else:
    print("  All checks passed. Starting ADK web server.")
    sys.exit(0)
