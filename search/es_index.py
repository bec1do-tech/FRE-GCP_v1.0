"""
FRE GCP v1.0 — Elasticsearch (Elastic Cloud) BM25 backend
===========================================================
Connects to Elastic Cloud (or a local ES 8.x instance for dev).
Provides:
  • Rich document mapping with structured metadata fields for pre-filtering
  • BM25 full-text search with optional metadata filters
  • Bulk indexing with idempotent upserts

Authentication
--------------
  Local dev   : no auth (docker-compose disables xpack.security)
  Elastic Cloud: ELASTICSEARCH_URL + ELASTICSEARCH_API_KEY from config

Degradation
-----------
  Every public function returns a safe fallback value when ES is unreachable,
  so the hybrid search layer can fall back to semantic-only results.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import config

logger = logging.getLogger(__name__)

# ── Index name ────────────────────────────────────────────────────────────────
_INDEX = config.ELASTICSEARCH_INDEX

# ── Mapping  ──────────────────────────────────────────────────────────────────
# The 'metadata' object uses keyword fields so we can filter precisely on
# e.g. date ranges, file type, department, or case_id.
_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "gcs_uri":    {"type": "keyword"},
            "filename":   {"type": "keyword"},
            "file_type":  {"type": "keyword"},
            "chunk_index":{"type": "integer"},
            "text":       {"type": "text", "analyzer": "standard"},
            "indexed_at": {"type": "date"},
            "vector_id":  {"type": "keyword"},
            "metadata": {
                "properties": {
                    "author":     {"type": "keyword"},
                    "department": {"type": "keyword"},
                    "case_id":    {"type": "keyword"},
                    "project":    {"type": "keyword"},
                    "date":       {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
                }
            },
        }
    },
    # Note: number_of_shards / number_of_replicas are omitted — Elastic Cloud
    # Serverless manages these automatically and rejects explicit values.
}


# ─────────────────────────────────────────────────────────────────────────────
# Client helpers
# ─────────────────────────────────────────────────────────────────────────────

def _client():
    """Return a connected Elasticsearch client or None."""
    try:
        from elasticsearch import Elasticsearch  # type: ignore[import-untyped]
        import os
        kwargs: dict[str, Any] = {"request_timeout": 15}
        if config.ELASTICSEARCH_API_KEY:
            kwargs["api_key"] = config.ELASTICSEARCH_API_KEY
        # Use RequestsHttpNode so requests library auto-reads HTTPS_PROXY env var
        if os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"):
            from elastic_transport import RequestsHttpNode
            kwargs["node_class"] = RequestsHttpNode
        es = Elasticsearch(config.ELASTICSEARCH_URL, **kwargs)
        if es.ping():
            return es
    except Exception as exc:
        logger.debug("Elasticsearch unavailable: %s", exc)
    return None


def _ensure_index(es) -> None:
    if not es.indices.exists(index=_INDEX):
        es.indices.create(index=_INDEX, **_MAPPING)


def _doc_id(gcs_uri: str, chunk_index: int) -> str:
    """Deterministic document ID from GCS URI + chunk index."""
    return hashlib.md5(f"{gcs_uri}:{chunk_index}".encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def index_chunks(chunks: list[dict]) -> bool:
    """
    Bulk-index a list of chunk dicts into Elasticsearch.

    Each chunk dict must have:
      gcs_uri (str), filename (str), file_type (str),
      chunk_index (int), text (str)
    Optional:  metadata (dict) with keys: author, department, case_id, project, date

    Returns True on success, False if ES is unavailable.
    """
    es = _client()
    if es is None:
        return False
    if not chunks:
        return True
    try:
        _ensure_index(es)
        now = datetime.now(timezone.utc).isoformat()
        operations = []
        for chunk in chunks:
            doc_id = _doc_id(chunk["gcs_uri"], chunk.get("chunk_index", 0))
            operations.append({"index": {"_index": _INDEX, "_id": doc_id}})
            operations.append(
                {
                    "gcs_uri":    chunk["gcs_uri"],
                    "filename":   chunk.get("filename", PurePosixPath(chunk["gcs_uri"]).name),
                    "file_type":  chunk.get("file_type", ""),
                    "chunk_index":chunk.get("chunk_index", 0),
                    "text":       chunk["text"],
                    "indexed_at": now,
                    "metadata":   chunk.get("metadata", {}),
                    "vector_id":  chunk.get("vector_id", ""),
                }
            )
        resp = es.bulk(operations=operations)
        if resp.get("errors"):
            logger.warning("ES bulk index had errors: %s", resp)
        es.indices.refresh(index=_INDEX)
        return True
    except Exception as exc:
        logger.error("ES index_chunks failed: %s", exc)
        return False


def search(
    query: str,
    top_k: int = 10,
    filters: dict | None = None,
    max_per_doc: int = 2,
) -> list[dict]:
    """
    BM25 full-text search with optional structured metadata filters.

    filters example:
      {"file_type": "pdf", "metadata.department": "Finance",
       "date_from": "2024-01-01", "date_to": "2024-12-31"}

    max_per_doc: maximum chunks returned from any single GCS URI.
      Set to 0 to disable diversity cap (useful for get_document_chunks).

    Returns a list of dicts:
      {gcs_uri, filename, file_type, chunk_index, text, score}
    Returns [] when ES is unavailable or the index is empty.
    """
    es = _client()
    if es is None:
        return []
    try:
        must_clauses: list[dict] = [
            {
                "multi_match": {
                    "query":  query,
                    "fields": ["text^1", "filename^2"],
                    "type":   "best_fields",
                    "fuzziness": "AUTO",
                }
            }
        ]
        filter_clauses: list[dict] = []

        if filters:
            for key, value in filters.items():
                if key == "date_from":
                    filter_clauses.append(
                        {"range": {"metadata.date": {"gte": value}}}
                    )
                elif key == "date_to":
                    filter_clauses.append(
                        {"range": {"metadata.date": {"lte": value}}}
                    )
                elif value:
                    filter_clauses.append({"term": {key: value}})

        # Over-fetch to allow per-document diversity filtering below.
        # We request `top_k * max(max_per_doc * 4, 4)` so that after capping
        # we still end up with top_k diverse results across many documents.
        fetch_size = top_k * max(max_per_doc * 4, 4) if max_per_doc > 0 else top_k
        body: dict = {
            "query": {
                "bool": {
                    "must":   must_clauses,
                    "filter": filter_clauses,
                }
            },
            "size":    fetch_size,
            "_source": ["gcs_uri", "filename", "file_type", "chunk_index", "text", "metadata"],
        }

        resp = es.search(index=_INDEX, body=body)
        results = []
        per_doc_count: dict[str, int] = {}
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            uri = src.get("gcs_uri", "")
            if max_per_doc > 0:
                if per_doc_count.get(uri, 0) >= max_per_doc:
                    continue
                per_doc_count[uri] = per_doc_count.get(uri, 0) + 1
            results.append(
                {
                    "gcs_uri":     uri,
                    "filename":    src.get("filename", ""),
                    "file_type":   src.get("file_type", ""),
                    "chunk_index": src.get("chunk_index", 0),
                    "text":        src.get("text", ""),
                    "score":       hit["_score"],
                    "source":      "elasticsearch",
                }
            )
            if len(results) >= top_k:
                break
        return results
    except Exception as exc:
        logger.error("ES search failed: %s", exc)
        return []


def get_chunks_by_uri(gcs_uri: str, limit: int = 10) -> list[dict]:
    """
    Retrieve indexed chunks for a specific GCS URI, ordered by chunk_index.
    Used as a fallback to enrich Vertex AI hits when the local postgres
    chunk table is empty (e.g. during local development against Cloud infra).
    """
    es = _client()
    if es is None:
        return []
    try:
        resp = es.search(
            index=_INDEX,
            body={
                "query": {"term": {"gcs_uri": gcs_uri}},
                "sort":  [{"chunk_index": {"order": "asc"}}],
                "size":  limit,
                "_source": ["gcs_uri", "filename", "file_type", "chunk_index", "text"],
            },
        )
        return [hit["_source"] for hit in resp["hits"]["hits"]]
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_chunks_by_uri failed for %s: %s", gcs_uri, exc)
        return []


def get_chunk_by_vector_id(vector_id: str) -> dict | None:
    """
    Look up a single chunk by its Vertex AI vector_id.
    Used to enrich Vertex AI hits when gcs_uri is not embedded in the index
    restriction metadata (e.g. vectors indexed without restriction data).
    Returns None when not found or ES is unavailable.
    """
    if not vector_id:
        return None
    es = _client()
    if es is None:
        return None
    try:
        resp = es.search(
            index=_INDEX,
            body={
                "query": {"term": {"vector_id": vector_id}},
                "size":  1,
                "_source": ["gcs_uri", "filename", "file_type", "chunk_index", "text", "vector_id"],
            },
        )
        hits = resp["hits"]["hits"]
        return hits[0]["_source"] if hits else None
    except Exception as exc:
        logger.debug("get_chunk_by_vector_id failed for %s: %s", vector_id, exc)
        return None


def get_chunks_by_vector_ids(vector_ids: list[str]) -> dict[str, dict]:
    """
    Batch-fetch chunks for a list of Vertex AI vector_ids in a single ES query.
    Returns a dict mapping vector_id → chunk source dict.
    Much faster than calling get_chunk_by_vector_id() in a loop.
    """
    if not vector_ids:
        return {}
    es = _client()
    if es is None:
        return {}
    try:
        resp = es.search(
            index=_INDEX,
            body={
                "query": {"terms": {"vector_id": list(vector_ids)}},
                "size":  len(vector_ids),
                "_source": ["gcs_uri", "filename", "file_type", "chunk_index", "text", "vector_id"],
            },
        )
        return {
            hit["_source"]["vector_id"]: hit["_source"]
            for hit in resp["hits"]["hits"]
            if "vector_id" in hit["_source"]
        }
    except Exception as exc:
        logger.debug("get_chunks_by_vector_ids failed: %s", exc)
        return {}


def delete_index() -> bool:
    """Drop the entire index (used for full re-index operations)."""
    es = _client()
    if es is None:
        return False
    try:
        if es.indices.exists(index=_INDEX):
            es.indices.delete(index=_INDEX)
        return True
    except Exception as exc:
        logger.error("ES delete_index failed: %s", exc)
        return False


def index_stats() -> dict:
    """Return index health: {available: bool, doc_count: int}."""
    es = _client()
    if es is None:
        return {"available": False, "doc_count": 0}
    try:
        # Use count() — indices.stats() returns 410 on Elastic Cloud Serverless.
        if es.indices.exists(index=_INDEX):
            result = es.count(index=_INDEX)
            return {"available": True, "doc_count": result["count"]}
        return {"available": True, "doc_count": 0}
    except Exception:
        return {"available": True, "doc_count": 0}
