"""
FRE GCP v1.0 — ADK Tool Wrappers: Ingestion
=============================================
Tools that let the agent trigger and monitor the document ingestion pipeline.
These are exposed to the ingestion_manager_agent through its `tools` parameter.
"""

import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trigger ingestion for a single document
# ─────────────────────────────────────────────────────────────────────────────

def trigger_document_ingestion(gcs_uri: str, force: bool = False):
    """
    Trigger the ingestion pipeline for a single document stored in GCS.
    The pipeline will: download the file, extract text and images (via Gemini
    Vision for charts/diagrams), chunk the content, generate embeddings, and
    index everything into Elasticsearch and Vertex AI Vector Search.

    Use this tool when a user asks to index a specific document, or when a
    new document has been uploaded and needs to be made searchable.

    Parameters
    ----------
    gcs_uri : Full GCS URI of the document to index.
              Format: gs://<bucket>/<path>/<filename>
              Example: gs://my-docs/reports/Q4_2024.pdf
    force   : If True, re-index the document even if it was already processed.
              Use this to refresh stale index entries after a file is updated.

    Returns
    -------
    Dict with keys:
      gcs_uri     (str) : the URI that was processed
      status      (str) : "indexed" | "skipped" | "failed"
      chunk_count (int) : number of text chunks created
      image_count (int) : number of images processed via Gemini Vision
      es_ok       (bool): whether Elasticsearch indexing succeeded
      vertex_ok   (bool): whether Vertex AI indexing succeeded
      error       (str) : error message if status is "failed"
    """
    try:
        from storage.postgres import init_db
        from ingestion.pipeline import process_document

        init_db()
        result = process_document(gcs_uri, force=force)
        return {
            "gcs_uri":     result.gcs_uri,
            "status":      result.status,
            "chunk_count": result.chunk_count,
            "image_count": result.image_count,
            "es_ok":       result.es_ok,
            "vertex_ok":   result.vertex_ok,
            "error":       result.error,
        }
    except Exception as exc:
        logger.error("trigger_document_ingestion error: %s", exc)
        return {"gcs_uri": gcs_uri, "status": "failed", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Trigger ingestion for an entire GCS folder / prefix
# ─────────────────────────────────────────────────────────────────────────────

def trigger_folder_ingestion(
    bucket: str,
    prefix: str = "",
    force: bool = False,
    max_workers: int = 4,
):
    """
    Trigger the ingestion pipeline for all supported documents under a GCS
    bucket prefix.  Processing is parallelised across up to max_workers threads.

    Use this tool when a user asks to index an entire folder, department share,
    or project directory in GCS.

    Parameters
    ----------
    bucket      : GCS bucket name (without gs:// prefix).
    prefix      : Object name prefix, e.g. "documents/finance/" (optional).
    force       : Re-index documents that were already indexed.
    max_workers : Number of parallel ingestion workers (1–8).

    Returns
    -------
    Dict with keys:
      bucket   (str) : the bucket scanned
      prefix   (str) : the prefix scanned
      total    (int) : total documents found
      indexed  (int) : successfully indexed
      skipped  (int) : skipped (duplicates or unsupported types)
      failed   (int) : documents that failed to index
      details  (list): abbreviated PipelineResult for each document
    """
    try:
        from storage.postgres import init_db
        from ingestion.pipeline import process_folder

        init_db()
        results = process_folder(
            bucket     = bucket,
            prefix     = prefix,
            force      = force,
            max_workers= max(1, min(max_workers, 8)),
        )

        details = [
            {
                "gcs_uri":     r.gcs_uri,
                "status":      r.status,
                "chunk_count": r.chunk_count,
                "error":       r.error,
            }
            for r in results
        ]

        return {
            "bucket":  bucket,
            "prefix":  prefix,
            "total":   len(results),
            "indexed": sum(1 for r in results if r.status == "indexed"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "failed":  sum(1 for r in results if r.status == "failed"),
            "details": details,
        }
    except Exception as exc:
        logger.error("trigger_folder_ingestion error: %s", exc)
        return {"bucket": bucket, "prefix": prefix, "total": 0, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion / indexing status
# ─────────────────────────────────────────────────────────────────────────────

def get_ingestion_status():
    """
    Return a summary of the current document indexing status from the database.

    Use this tool when a user asks how many documents have been indexed,
    whether indexing is still in progress, or if any documents failed to index.

    Returns
    -------
    Dict with document counts by status:
      indexed    (int) : successfully indexed documents
      processing (int) : currently being processed
      failed     (int) : documents that encountered errors
      pending    (int) : queued but not yet started
      total      (int) : grand total
    """
    try:
        from storage.postgres import get_document_stats

        stats = get_document_stats()
        total = sum(stats.values())
        return {**stats, "total": total}
    except Exception as exc:
        logger.error("get_ingestion_status error: %s", exc)
        return {"error": str(exc)}
