"""
FRE GCP v1.0 — Document Page Preview Tool
==========================================
Renders a specific page of a PDF stored in GCS as a JPEG, uploads the result
to GCS (session_previews/ prefix) and returns a V4 signed URL so the ADK web
UI can render it inline without any local HTTP server.

Signing is done by impersonating the PREVIEW_SIGNING_SA service account via
the IAM Credentials API (HTTPS → flows through px proxy automatically).
No service-account key file is required on disk.

Note: no 'from __future__ import annotations' — ADK introspects signatures
at runtime and that directive breaks type resolution.
"""

import logging
import time
import uuid

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def preview_document_page(
    gcs_uri: str,
    page_number: int = 1,
    tool_context: ToolContext = None,  # ADK injects this automatically
) -> str:
    """
    Render a specific page of a PDF document stored in GCS as an inline image.
    The rendered JPEG is uploaded to GCS (session_previews/) and a signed URL
    is returned so the browser can display it directly.

    Use this tool when:
      - The user asks to see a page from a document
      - You need to extract exact data points from a chart or graph in the document
      - You want to verify or reproduce a specific visual in the document

    Parameters
    ----------
    gcs_uri     : The GCS URI of the document
                  (e.g. gs://bucket/path/file.pdf).
                  Returned in the 'gcs_uri' field of hybrid_search results.
    page_number : The 1-based page number to render (default: 1).
    tool_context: Injected by ADK — do NOT pass from the LLM.

    Returns
    -------
    A ready-to-use markdown string. Copy the ENTIRE return value verbatim into
    your response.  The first line is an inline image tag — the UI renders it
    as a visible image.
    """
    _filename = gcs_uri.split("/")[-1]
    _error_prefix = f"ERROR previewing {_filename} page {page_number}: "
    _result_parts: list = []  # accumulated markdown lines

    # ── Step 1: Download PDF bytes from GCS ──────────────────────────────────
    _t0 = time.perf_counter()
    try:
        from storage.gcs import download_to_bytes
        pdf_bytes = download_to_bytes(gcs_uri)
    except Exception as exc:
        logger.error("preview_document_page download error: %s", exc)
        return f"{_error_prefix}Could not download from GCS: {exc}"
    logger.info("[TIMING] GCS download: %.2fs", time.perf_counter() - _t0)

    # ── Step 2: Render page with PyMuPDF ─────────────────────────────────────
    _t1 = time.perf_counter()
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        page_idx = max(0, min(page_number - 1, total_pages - 1))
        page = doc[page_idx]

        mat = fitz.Matrix(1.2, 1.2)   # 1.2x = ~1440×1100px — good quality, reasonable size
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpg", jpg_quality=60)
        doc.close()
    except ImportError:
        return f"{_error_prefix}PyMuPDF (fitz) is not installed. Run: pip install PyMuPDF"
    except Exception as exc:
        logger.error("preview_document_page render error: %s", exc)
        return f"{_error_prefix}PDF rendering failed: {exc}"
    logger.info("[TIMING] PyMuPDF render: %.2fs", time.perf_counter() - _t1)

    # ── Step 3: Upload JPEG to GCS and generate a signed URL ─────────────────
    _t2 = time.perf_counter()
    try:
        import config
        from storage.gcs import upload_bytes as _upload, generate_signed_url as _sign

        bucket = config.GCS_BUCKET or "fre-cognitive-search-docs"
        safe_name = _filename.replace(" ", "_").replace("/", "_")
        fname = f"{safe_name}_p{page_number}_{uuid.uuid4().hex[:6]}.jpg"
        blob_path = f"session_previews/{fname}"

        gcs_preview_uri = _upload(img_bytes, bucket, blob_path, "image/jpeg")
        logger.info("Preview saved to GCS: %s", gcs_preview_uri)

        preview_url = _sign(bucket, blob_path)
        _result_parts.append(f"![{_filename} — page {page_number}]({preview_url})")

    except Exception as exc:
        logger.error("GCS preview upload/sign failed: %s", exc)
        return f"{_error_prefix}Failed to upload/sign preview: {exc}"
    logger.info("[TIMING] GCS upload + sign: %.2fs", time.perf_counter() - _t2)

    # ── Step 4: Generate a signed URL for the original PDF ───────────────────
    try:
        from storage.gcs import parse_gcs_uri, generate_signed_url as _sign
        _pdf_bucket, _pdf_blob = parse_gcs_uri(gcs_uri)
        pdf_url = _sign(_pdf_bucket, _pdf_blob)
        _result_parts.append(f"[\U0001f4c4 Open full PDF]({pdf_url})")
    except Exception as exc:
        logger.warning("Could not sign PDF URL (%s): %s", gcs_uri, exc)
        # Fall back to raw GCS URI so the agent can still cite the source
        _result_parts.append(f"[\U0001f4c4 Source: {_filename}]({gcs_uri})")

    # ── Step 5: Store in session state ───────────────────────────────────────
    if tool_context is not None:
        try:
            previews: list = tool_context.state.get("session_previews", [])
            previews.append({
                "gcs_source": gcs_uri,
                "page": page_number,
                "preview_url": preview_url,
                "gcs_preview_uri": gcs_preview_uri,
            })
            tool_context.state["session_previews"] = previews
        except Exception as _state_exc:
            logger.debug("Could not update session state: %s", _state_exc)

    logger.info("[TIMING] preview_document_page total: %.2fs", time.perf_counter() - _t0)
    return "\n".join(_result_parts)


def preview_documents_batch(
    pages_json: str,
    tool_context: ToolContext = None,
) -> str:
    """
    Render multiple document pages IN PARALLEL and return all preview images
    in a single call.  Use this instead of calling preview_document_page
    repeatedly — it is much faster because all GCS downloads and uploads run
    concurrently.

    Parameters
    ----------
    pages_json : JSON array of objects, each with "gcs_uri" and "page_number".
                 Example: '[{"gcs_uri":"gs://bucket/file.pdf","page_number":4},
                             {"gcs_uri":"gs://bucket/file.pdf","page_number":7}]'
                 Maximum 6 entries; extras are silently ignored.

    Returns
    -------
    A single string containing all image markdown blocks, separated by blank
    lines. Paste the ENTIRE return value verbatim into your reply.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        pages = json.loads(pages_json)
    except Exception as exc:
        return f"ERROR: pages_json is not valid JSON — {exc}"

    if not isinstance(pages, list) or len(pages) == 0:
        return "ERROR: pages_json must be a non-empty JSON array."

    pages = pages[:6]  # cap at 6

    _t0 = time.perf_counter()

    def _one(item):
        gcs_uri = item.get("gcs_uri", "")
        page_number = int(item.get("page_number", 1))
        return preview_document_page(gcs_uri, page_number, tool_context)

    results = [None] * len(pages)
    with ThreadPoolExecutor(max_workers=min(6, len(pages))) as ex:
        future_to_idx = {ex.submit(_one, p): i for i, p in enumerate(pages)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = f"ERROR on item {idx}: {exc}"

    logger.info("[TIMING] preview_documents_batch total: %.2fs for %d pages",
                time.perf_counter() - _t0, len(pages))
    return "\n\n".join(r for r in results if r)

