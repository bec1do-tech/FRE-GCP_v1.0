"""
FRE GCP v1.0 — Ingestion Pipeline
====================================
Orchestrates the full document processing flow for a single GCS object:

  ┌──────────────┐
  │  GCS Object  │  (PDF, DOCX, PPTX, XLSX, TXT, …)
  └──────┬───────┘
         │ download_to_bytes
         ▼
  ┌──────────────┐
  │  MD5 Hash    │  → PostgreSQL dedup check  (skip if already indexed)
  └──────┬───────┘
         │ extract()
         ▼
  ┌──────────────┐
  │  Extractor   │  text + image descriptions via Gemini Vision
  └──────┬───────┘
         │ chunk_text()
         ▼
  ┌──────────────┐
  │   Chunker    │  overlapping text chunks
  └──────┬───────┘
         │ parallel index
         ├──────────────────────────────┐
         ▼                              ▼
  ┌─────────────┐              ┌──────────────────┐
  │Elasticsearch│              │Vertex AI Vector  │
  │  BM25 Index │              │  Search Index    │
  └─────────────┘              └──────────────────┘
         │
         ▼
  ┌──────────────┐
  │  PostgreSQL  │  metadata + chunk records
  └──────────────┘

Idempotent: re-running on the same GCS URI re-indexes and updates all records.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from storage import gcs, postgres
from ingestion.extractor import extract
from ingestion.chunker import chunk_text
from search import es_index, vertex_vector

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    gcs_uri:      str
    status:       str           # "indexed" | "skipped" | "failed"
    chunk_count:  int = 0
    image_count:  int = 0
    es_ok:        bool = False
    vertex_ok:    bool = False
    error:        str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _file_type(filename: str) -> str:
    return PurePosixPath(filename).suffix.lower().lstrip(".")


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_document(gcs_uri: str, force: bool = False) -> PipelineResult:
    """
    Run the full ingestion pipeline for one GCS object.

    Parameters
    ----------
    gcs_uri : Full GCS URI, e.g.  gs://my-bucket/documents/report.pdf
    force   : If True, re-index even if content hash matches an existing record.

    Returns
    -------
    PipelineResult describing what happened.
    """
    filename = PurePosixPath(gcs_uri).name

    # ── Step 0: validate file type ────────────────────────────────────────────
    if not gcs.is_supported(gcs_uri):
        logger.info("Skipping unsupported file: %s", gcs_uri)
        return PipelineResult(gcs_uri, "skipped", error="unsupported file type")

    # ── Step 1: download ──────────────────────────────────────────────────────
    logger.info("Downloading %s …", gcs_uri)
    try:
        data = gcs.download_to_bytes(gcs_uri)
    except Exception as exc:
        logger.error("Download failed for %s: %s", gcs_uri, exc)
        return PipelineResult(gcs_uri, "failed", error=str(exc))

    # ── Step 2: deduplication (MD5 hash check) ────────────────────────────────
    content_hash = _md5(data)
    if not force and postgres.is_duplicate(content_hash):
        logger.info("Duplicate content — skipping %s (md5=%s)", filename, content_hash)
        return PipelineResult(gcs_uri, "skipped", error="duplicate content")

    # ── Step 3: register document in PostgreSQL (status=processing) ───────────
    blob_meta = gcs.get_blob_metadata(gcs_uri)
    doc_id = postgres.upsert_document(
        gcs_uri     = gcs_uri,
        filename    = filename,
        content_md5 = content_hash,
        file_size   = blob_meta.get("size", len(data)),
        file_type   = _file_type(filename),
        status      = "processing",
    )

    # ── Step 4: extract text + visual content ─────────────────────────────────
    logger.info("Extracting text from %s …", filename)
    try:
        result = extract(data, filename)
    except Exception as exc:
        logger.error("Extraction failed for %s: %s", filename, exc)
        postgres.mark_document_failed(doc_id)
        return PipelineResult(gcs_uri, "failed", error=f"extraction: {exc}")

    if not result.text.strip():
        logger.warning("No text extracted from %s — marking as failed.", filename)
        postgres.mark_document_failed(doc_id)
        return PipelineResult(gcs_uri, "failed", error="empty extraction")

    # ── Step 5: chunk ─────────────────────────────────────────────────────────
    logger.info("Chunking %s …", filename)
    chunks = chunk_text(result.text)
    logger.info("%d chunks produced for %s.", len(chunks), filename)

    if not chunks:
        postgres.mark_document_failed(doc_id)
        return PipelineResult(gcs_uri, "failed", error="no chunks produced")

    # ── Step 6: parallel indexing (ES + Vertex AI) ────────────────────────────
    es_docs = [
        {
            "gcs_uri":    gcs_uri,
            "filename":   filename,
            "file_type":  _file_type(filename),
            "chunk_index":c.chunk_index,
            "text":       c.text,
            "metadata":   result.metadata,
        }
        for c in chunks
    ]

    vertex_docs = [
        {
            "vector_id": c.vector_id,
            "text":      c.text,
            "gcs_uri":   gcs_uri,
            "filename":  filename,
            "doc_id":    doc_id,
        }
        for c in chunks
    ]

    es_ok     = False
    vertex_ok = False

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_es     = pool.submit(es_index.index_chunks, es_docs)
        future_vertex = pool.submit(vertex_vector.upsert_chunks, vertex_docs)

        for future in as_completed([future_es, future_vertex]):
            if future is future_es:
                try:
                    es_ok = future.result()
                except Exception as exc:
                    logger.error("ES indexing error for %s: %s", filename, exc)
            else:
                try:
                    vertex_ok = future.result()
                except Exception as exc:
                    logger.error("Vertex AI indexing error for %s: %s", filename, exc)

    # ── Step 7: persist chunks to PostgreSQL ──────────────────────────────────
    postgres.upsert_chunks(
        doc_id,
        [
            {
                "chunk_index":     c.chunk_index,
                "chunk_text":      c.text,
                "vertex_vector_id":c.vector_id,
            }
            for c in chunks
        ],
    )

    # ── Step 8: mark document as indexed ──────────────────────────────────────
    postgres.mark_document_indexed(doc_id, len(chunks))
    logger.info(
        "✓ Indexed %s: %d chunks | ES=%s | Vertex=%s | images=%d",
        filename, len(chunks), es_ok, vertex_ok, result.image_count,
    )

    return PipelineResult(
        gcs_uri     = gcs_uri,
        status      = "indexed",
        chunk_count = len(chunks),
        image_count = result.image_count,
        es_ok       = es_ok,
        vertex_ok   = vertex_ok,
    )


def process_folder(
    bucket: str,
    prefix: str = "",
    force: bool = False,
    max_workers: int = 4,
) -> list[PipelineResult]:
    """
    Process all supported documents under a GCS prefix in parallel.

    Parameters
    ----------
    bucket      : GCS bucket name.
    prefix      : Object name prefix to scan (e.g. "documents/finance/").
    force       : Re-index already-indexed documents.
    max_workers : Number of concurrent ingestion workers.

    Returns
    -------
    List of PipelineResult (one per blob discovered).
    """
    uris = list(
        gcs.list_blobs(bucket, prefix=prefix, extensions=gcs.SUPPORTED_EXTENSIONS)
    )
    logger.info("Found %d supported files under gs://%s/%s", len(uris), bucket, prefix)

    results: list[PipelineResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_document, uri, force): uri for uri in uris}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                uri = futures[future]
                logger.error("Unexpected error for %s: %s", uri, exc)
                results.append(PipelineResult(uri, "failed", error=str(exc)))

    indexed  = sum(1 for r in results if r.status == "indexed")
    skipped  = sum(1 for r in results if r.status == "skipped")
    failed   = sum(1 for r in results if r.status == "failed")
    logger.info(
        "Folder scan complete — indexed=%d  skipped=%d  failed=%d", indexed, skipped, failed
    )
    return results
