"""
FRE GCP v1.0 — Hybrid Search via Reciprocal Rank Fusion (RRF)
=============================================================
Merges:
  1. BM25 results from Elasticsearch  (keyword + metadata filtering)
  2. Semantic results from Vertex AI Vector Search  (conceptual meaning)

Using the standard RRF formula:
  score(d) = Σ  1 / (k + rank(d))

The two-phase design follows the project specification:
  Phase 1 — Elasticsearch narrows the search space with fast metadata filters.
  Phase 2 — Vertex AI re-ranks within that space by semantic similarity.
  Phase 3 — RRF combines both ranked lists into a single, fused result.

When either backend is unavailable, the result comes from the other alone.
When both are unavailable, an empty list is returned.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from storage import postgres
from . import es_index, vertex_vector

import config

logger = logging.getLogger(__name__)

_RRF_K: int = config.RRF_K   # 60 is the canonical constant
_pg_ready: bool = False  # schema auto-init flag
_pg_available: bool = True  # set False after first connection failure


# ─────────────────────────────────────────────────────────────────────────────
# Core hybrid search
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """
    Execute a hybrid BM25 + semantic search and return the top_k results,
    ranked by Reciprocal Rank Fusion.

    Parameters
    ----------
    query   : Natural-language search query.
    top_k   : Number of results to return (default: config.DEFAULT_TOP_K).
    filters : Optional metadata filters forwarded to Elasticsearch.
              Keys: file_type, metadata.department, metadata.case_id,
                    metadata.project, date_from, date_to

    Returns
    -------
    List of result dicts (sorted by descending RRF score):
      {gcs_uri, filename, file_type, chunk_index, text, rrf_score,
       vector_id (optional), sources}
    """
    global _pg_ready, _pg_available
    _t0 = time.perf_counter()
    if not _pg_ready:
        try:
            postgres.init_db()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not auto-init postgres schema: %s — vector hits will use ES fallback.", exc)
            _pg_available = False
        _pg_ready = True  # only attempt once per process, regardless of outcome

    top_k = top_k or config.DEFAULT_TOP_K
    fetch = top_k * 3   # over-fetch before re-ranking

    # ── Phase 1: BM25 (Elasticsearch) ────────────────────────────────────────
    _t1 = time.perf_counter()
    bm25_hits = es_index.search(query, top_k=fetch, filters=filters)
    logger.info("[TIMING] ES BM25 search: %.2fs — %d hits", time.perf_counter() - _t1, len(bm25_hits))

    # ── Phase 2: Semantic (Vertex AI) ────────────────────────────────────────
    _t2 = time.perf_counter()
    vec_hits_raw = vertex_vector.search(query, top_k=fetch)
    logger.info("[TIMING] Vertex AI search: %.2fs — %d hits", time.perf_counter() - _t2, len(vec_hits_raw))

    # Enrich vector hits with text/metadata.
    # Primary source: PostgreSQL chunks table (populated during ingest).
    # Fallback: Elasticsearch — uses a single batch query for all vector_ids
    # instead of per-hit individual queries (avoids 30× TCP round-trips to ES).
    _t3 = time.perf_counter()
    vec_hits: list[dict] = []

    # -- Postgres path (fast when running on Cloud Run with Cloud SQL) ---------
    pg_records: dict[str, dict] = {}
    if _pg_available:
        for hit in vec_hits_raw:
            try:
                record = postgres.get_chunk_by_vector_id(hit["vector_id"])
                if record:
                    pg_records[hit["vector_id"]] = record
            except Exception as exc:  # noqa: BLE001
                logger.debug("Postgres lookup skipped: %s", exc)
                _pg_available = False
                break

    # -- ES batch fallback for hits not found in postgres ----------------------
    missing_ids = [
        h["vector_id"] for h in vec_hits_raw
        if h.get("vector_id") and h["vector_id"] not in pg_records
    ]
    es_by_vid: dict[str, dict] = {}
    if missing_ids:
        es_by_vid = es_index.get_chunks_by_vector_ids(missing_ids)

    # -- Collect enriched hits --------------------------------------------------
    for hit in vec_hits_raw:
        vid = hit.get("vector_id", "")
        record = pg_records.get(vid)
        if record:
            vec_hits.append({
                "gcs_uri":     record["gcs_uri"],
                "filename":    record["filename"],
                "file_type":   record["file_type"],
                "chunk_index": record["chunk_index"],
                "text":        record["chunk_text"],
                "vector_id":   vid,
                "source":      "vertex_ai",
            })
            continue

        c = es_by_vid.get(vid)
        if c is None and hit.get("gcs_uri"):
            # Final fallback: fetch by URI (one extra ES call per distinct URI)
            es_chunks = es_index.get_chunks_by_uri(hit["gcs_uri"], limit=1)
            if es_chunks:
                c = es_chunks[0]
        if c:
            vec_hits.append({
                "gcs_uri":     c.get("gcs_uri", hit.get("gcs_uri", "")),
                "filename":    c.get("filename", ""),
                "file_type":   c.get("file_type", ""),
                "chunk_index": c.get("chunk_index", 0),
                "text":        c.get("text", ""),
                "vector_id":   vid,
                "source":      "vertex_ai",
            })

    # ── Phase 3: Reciprocal Rank Fusion ──────────────────────────────────────
    logger.info("[TIMING] Vector enrichment: %.2fs — %d enriched hits", time.perf_counter() - _t3, len(vec_hits))
    _t4 = time.perf_counter()
    rrf_scores:  dict[str, float]       = {}
    merged_docs: dict[str, dict]        = {}
    sources_map: dict[str, set[str]]    = {}

    def _key(doc: dict) -> str:
        return f"{doc.get('gcs_uri', '')}::{doc.get('chunk_index', 0)}"

    for rank, doc in enumerate(bm25_hits):
        k = _key(doc)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (_RRF_K + rank + 1)
        merged_docs[k] = doc
        sources_map.setdefault(k, set()).add("elasticsearch")

    for rank, doc in enumerate(vec_hits):
        k = _key(doc)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (_RRF_K + rank + 1)
        if k not in merged_docs:
            merged_docs[k] = doc
        sources_map.setdefault(k, set()).add("vertex_ai")

    # Apply per-document cap in the final RRF output so no single document
    # floods all result slots (e.g. 1198_T1.pdf with many matching chunks).
    _MAX_CHUNKS_PER_DOC = 2
    ranked_keys_all = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    per_doc_seen: dict[str, int] = {}
    ranked_keys: list[str] = []
    for k in ranked_keys_all:
        uri = merged_docs[k].get("gcs_uri", "")
        if per_doc_seen.get(uri, 0) < _MAX_CHUNKS_PER_DOC:
            per_doc_seen[uri] = per_doc_seen.get(uri, 0) + 1
            ranked_keys.append(k)
        if len(ranked_keys) >= top_k:
            break

    results = []
    for k in ranked_keys:
        doc = dict(merged_docs[k])
        doc["rrf_score"] = round(rrf_scores[k], 6)
        doc["sources"]   = sorted(sources_map[k])
        results.append(doc)

    logger.info(
        "[TIMING] hybrid_search total: %.2fs | ES:%d Vertex:%d RRF→%d results",
        time.perf_counter() - _t0, len(bm25_hits), len(vec_hits), len(results),
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Backend status
# ─────────────────────────────────────────────────────────────────────────────

def status() -> dict:
    """Return availability and stats for both search backends."""
    return {
        "elasticsearch": es_index.index_stats(),
        "vertex_ai":     vertex_vector.collection_stats(),
    }
