"""
FRE GCP v1.0 — ADK Tool Wrappers: Search
==========================================
Plain Python functions exposed as ADK tools to the agent layer.
Each function has a precise docstring — the ADK framework passes this to the
Gemini model so it knows when and how to call each tool.

Design principles
-----------------
• Tools are pure functions: no global mutable state.
• All parameters have clear types and defaults.
• Errors are returned as structured dicts, never raised, so the agent
  can report them gracefully to the user.
• Citation links back to GCS URIs are always included in results.
"""

import json
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid search tool
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = 5,
    file_type: str = "",
    department: str = "",
    case_id: str = "",
    date_from: str = "",
    date_to: str = "",
):
    """
    Search the document repository using a combination of keyword (BM25) and
    semantic (vector) search, fused by Reciprocal Rank Fusion.

    Use this tool to answer questions about the content of indexed documents.
    It returns the most relevant text excerpts along with their source files
    and citation information.

    Parameters
    ----------
    query      : The natural-language question or search query.
    top_k      : Maximum number of results to return (1–20).
    file_type  : Optional filter by file extension, e.g. "pdf", "xlsx", "docx".
    department : Optional filter by department metadata field.
    case_id    : Optional filter by case/project ID metadata field.
    date_from  : Optional lower bound on document date (YYYY-MM-DD).
    date_to    : Optional upper bound on document date (YYYY-MM-DD).

    Returns
    -------
    Dict with keys:
      results (list): each item has {rank, filename, gcs_uri, excerpt, sources}
      total   (int) : number of results returned
      query   (str) : the original query
    """
    try:
        from search.hybrid import hybrid_search as _hybrid

        filters: dict = {}
        if file_type:
            filters["file_type"] = file_type.lower().lstrip(".")
        if department:
            filters["metadata.department"] = department
        if case_id:
            filters["metadata.case_id"] = case_id
        if date_from:
            filters["date_from"] = date_from
        if date_to:
            filters["date_to"] = date_to

        raw = _hybrid(query=query, top_k=max(1, min(top_k, 20)), filters=filters or None)

        results = []
        for rank, hit in enumerate(raw, start=1):
            results.append(
                {
                    "rank":     rank,
                    "filename": hit.get("filename", ""),
                    "gcs_uri":  hit.get("gcs_uri", ""),
                    "excerpt":  hit.get("text", "")[:1000],   # truncate for context window
                    "sources":  hit.get("sources", []),
                    "rrf_score":hit.get("rrf_score", 0.0),
                }
            )

        return {"results": results, "total": len(results), "query": query}

    except Exception as exc:
        logger.error("hybrid_search tool error: %s", exc)
        return {"results": [], "total": 0, "query": query, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Document retrieval tool
# ─────────────────────────────────────────────────────────────────────────────

def get_document_chunks(gcs_uri: str, max_chunks: int = 10):
    """
    Retrieve the stored text chunks for a specific document by its GCS URI.

    Use this tool when you need the full content of a particular document,
    or when you want to verify or expand on a specific citation returned by
    the hybrid_search tool.

    Parameters
    ----------
    gcs_uri    : The GCS URI of the document (e.g. gs://bucket/path/file.pdf).
                 This is returned in the 'gcs_uri' field of hybrid_search results.
    max_chunks : Maximum number of chunks to return (1–50).

    Returns
    -------
    Dict with keys:
      gcs_uri  (str)  : the requested document URI
      filename (str)  : the document filename
      chunks   (list) : list of {chunk_index, text} dicts
      total    (int)  : total number of chunks in the DB for this document
    """
    try:
        import psycopg2.extras
        from storage.postgres import _conn  # type: ignore[attr-defined]

        with _conn() as con:
            with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT c.chunk_index, c.chunk_text,
                           d.filename, d.gcs_uri,
                           (SELECT COUNT(*) FROM chunks WHERE doc_id = d.id) AS total
                    FROM chunks c
                    JOIN documents d ON d.id = c.doc_id
                    WHERE d.gcs_uri = %s
                    ORDER BY c.chunk_index
                    LIMIT %s
                    """,
                    (gcs_uri, max(1, min(max_chunks, 50))),
                )
                rows = cur.fetchall()

        if rows:
            return {
                "gcs_uri":  gcs_uri,
                "filename": rows[0]["filename"],
                "chunks":   [{"chunk_index": r["chunk_index"], "text": r["chunk_text"]} for r in rows],
                "total":    rows[0]["total"],
            }

    except Exception as exc:
        logger.debug("get_document_chunks postgres unavailable: %s — falling back to ES.", exc)

    # Fallback: retrieve chunks directly from Elasticsearch
    try:
        from search.es_index import get_chunks_by_uri
        es_rows = get_chunks_by_uri(gcs_uri, limit=max(1, min(max_chunks, 50)))
        if es_rows:
            return {
                "gcs_uri":  gcs_uri,
                "filename": es_rows[0].get("filename", ""),
                "chunks":   [{"chunk_index": r.get("chunk_index", i), "text": r.get("text", "")} for i, r in enumerate(es_rows)],
                "total":    len(es_rows),
            }
    except Exception as exc:
        logger.error("get_document_chunks ES fallback error: %s", exc)

    return {
        "gcs_uri": gcs_uri,
        "filename": "",
        "chunks": [],
        "total": 0,
        "error": "Document not found in index.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Search backend status tool
# ─────────────────────────────────────────────────────────────────────────────

def get_search_status():
    """
    Return the current health and statistics of the search backends.

    Use this tool when the user asks about the status of the document index,
    how many documents are indexed, or whether the search system is operational.

    Returns
    -------
    Dict with keys:
      elasticsearch (dict): {available, doc_count}
      vertex_ai     (dict): {available, endpoint_set}
      database      (dict): {indexed, processing, failed, pending}
    """
    from search.hybrid import status as _status

    backends = _status()
    result = {
        "elasticsearch": backends.get("elasticsearch", {}),
        "vertex_ai":     backends.get("vertex_ai", {}),
        "database":      {"available": False, "note": "unreachable from local machine"},
    }
    try:
        from storage.postgres import get_document_stats
        result["database"] = get_document_stats()
    except Exception as exc:
        logger.warning("get_search_status: DB unavailable: %s", exc)
    return result
