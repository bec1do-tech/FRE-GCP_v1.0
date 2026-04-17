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
from typing import Any

from storage import postgres
from . import es_index, vertex_vector

import config

logger = logging.getLogger(__name__)

_RRF_K: int = config.RRF_K   # 60 is the canonical constant


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
    top_k = top_k or config.DEFAULT_TOP_K
    fetch = top_k * 3   # over-fetch before re-ranking

    # ── Phase 1: BM25 (Elasticsearch) ────────────────────────────────────────
    bm25_hits = es_index.search(query, top_k=fetch, filters=filters)

    # ── Phase 2: Semantic (Vertex AI) ────────────────────────────────────────
    vec_hits_raw = vertex_vector.search(query, top_k=fetch)

    # Enrich vector hits with text/metadata from PostgreSQL chunk records
    vec_hits: list[dict] = []
    for hit in vec_hits_raw:
        record = postgres.get_chunk_by_vector_id(hit["vector_id"])
        if record:
            vec_hits.append(
                {
                    "gcs_uri":     record["gcs_uri"],
                    "filename":    record["filename"],
                    "file_type":   record["file_type"],
                    "chunk_index": record["chunk_index"],
                    "text":        record["chunk_text"],
                    "vector_id":   hit["vector_id"],
                    "source":      "vertex_ai",
                }
            )

    # ── Phase 3: Reciprocal Rank Fusion ──────────────────────────────────────
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

    ranked_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

    results = []
    for k in ranked_keys:
        doc = dict(merged_docs[k])
        doc["rrf_score"] = round(rrf_scores[k], 6)
        doc["sources"]   = sorted(sources_map[k])
        results.append(doc)

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
