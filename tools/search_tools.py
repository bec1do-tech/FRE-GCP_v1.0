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
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — parallel GCS URL signing
# ─────────────────────────────────────────────────────────────────────────────

def _batch_sign_uris(gcs_uris: list) -> dict:
    """
    Sign a list of GCS URIs in parallel threads.
    Returns {gcs_uri: https_url} — empty string on failure for a given URI.
    """
    from pathlib import PurePosixPath
    import config as _cfg
    from storage.gcs import generate_signed_url

    base_url = getattr(_cfg, "DOCUMENT_BASE_URL", "").rstrip("/")
    if base_url:
        # Production path: no signing needed, construct URL directly
        return {
            u: f"{base_url}/{u[5:].partition('/')[2]}"
            for u in gcs_uris
            if u.startswith("gs://")
        }

    def _sign(gcs_uri):
        try:
            without = gcs_uri[5:]
            bucket, _, obj_path = without.partition("/")
            url = generate_signed_url(
                bucket=bucket,
                blob_path=obj_path,
                expiration_seconds=3600,
                signing_sa=_cfg.PREVIEW_SIGNING_SA,
            )
            return gcs_uri, url
        except Exception as exc:
            logger.debug("URL signing failed for %s: %s", gcs_uri, exc)
            return gcs_uri, ""

    result = {}
    if not gcs_uris:
        return result
    with ThreadPoolExecutor(max_workers=min(8, len(gcs_uris))) as ex:
        futures = {ex.submit(_sign, u): u for u in gcs_uris}
        for f in as_completed(futures, timeout=15):
            try:
                gcs_uri, url = f.result()
                result[gcs_uri] = url
            except Exception:
                pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid search tool
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = 10,
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

        # Resolve HTTP URLs for all unique GCS URIs in parallel
        unique_uris = list({r["gcs_uri"] for r in results if r.get("gcs_uri")})
        uri_map = _batch_sign_uris(unique_uris)
        for r in results:
            r["http_url"] = uri_map.get(r["gcs_uri"], "")

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
# Document URL tool
# ─────────────────────────────────────────────────────────────────────────────

def get_document_url(gcs_uri: str) -> dict:
    """
    Return a browser-accessible HTTPS URL for a single GCS document.
    Prefer get_document_urls() when you need URLs for multiple documents.

    Parameters
    ----------
    gcs_uri : GCS URI, e.g. gs://fre-cognitive-search-docs/.../file.pdf

    Returns
    -------
    Dict: {url, filename, gcs_uri}  or {error} on failure.
    """
    from pathlib import PurePosixPath
    import config as _cfg
    try:
        if not gcs_uri.startswith("gs://"):
            return {"url": "", "filename": "", "gcs_uri": gcs_uri,
                    "error": f"Not a valid GCS URI: {gcs_uri!r}"}
        without_scheme = gcs_uri[5:]
        bucket, _, obj_path = without_scheme.partition("/")
        filename = PurePosixPath(obj_path).name
        base_url = _cfg.DOCUMENT_BASE_URL.rstrip("/")
        if base_url:
            url = f"{base_url}/{obj_path}"
        else:
            from storage.gcs import generate_signed_url
            url = generate_signed_url(bucket=bucket, blob_path=obj_path,
                                      expiration_seconds=3600,
                                      signing_sa=_cfg.PREVIEW_SIGNING_SA)
        return {"url": url, "filename": filename, "gcs_uri": gcs_uri}
    except Exception as exc:
        logger.error("get_document_url failed for %s: %s", gcs_uri, exc)
        return {"url": "", "filename": "", "gcs_uri": gcs_uri, "error": str(exc)}


def get_document_urls(gcs_uris_json: str) -> dict:
    """
    Return browser-accessible HTTPS URLs for ALL source documents at once.

    USE THIS TOOL (not get_document_url) to build the Sources Consulted section.
    Pass ALL unique gcs_uris from the search results in a single call.

    Parameters
    ----------
    gcs_uris_json : JSON array of GCS URIs, e.g.:
        '["gs://fre-cognitive-search-docs/A.pdf",
          "gs://fre-cognitive-search-docs/B.pdf"]'

    Returns
    -------
    Dict with keys:
      documents (list): each item has {url, filename, gcs_uri, error (if any)}
      total     (int) : number of URIs processed
    """
    import json
    from pathlib import PurePosixPath
    import config as _cfg

    try:
        uris = json.loads(gcs_uris_json)
    except Exception as exc:
        return {"documents": [], "total": 0,
                "error": f"Invalid JSON: {exc}. Pass a JSON array of gs:// URIs."}

    # For production with a base URL, no signing needed — build all URLs fast
    base_url = _cfg.DOCUMENT_BASE_URL.rstrip("/")
    results = []

    for gcs_uri in uris:
        try:
            if not gcs_uri.startswith("gs://"):
                results.append({"url": "", "filename": "", "gcs_uri": gcs_uri,
                                 "error": "Not a valid GCS URI"})
                continue
            without_scheme = gcs_uri[5:]
            bucket, _, obj_path = without_scheme.partition("/")
            filename = PurePosixPath(obj_path).name
            if base_url:
                url = f"{base_url}/{obj_path}"
            else:
                from storage.gcs import generate_signed_url
                url = generate_signed_url(bucket=bucket, blob_path=obj_path,
                                          expiration_seconds=3600,
                                          signing_sa=_cfg.PREVIEW_SIGNING_SA)
            results.append({"url": url, "filename": filename, "gcs_uri": gcs_uri})
        except Exception as exc:
            logger.error("get_document_urls failed for %s: %s", gcs_uri, exc)
            results.append({"url": "", "filename": PurePosixPath(gcs_uri).name,
                            "gcs_uri": gcs_uri, "error": str(exc)})

    return {"documents": results, "total": len(results)}


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
