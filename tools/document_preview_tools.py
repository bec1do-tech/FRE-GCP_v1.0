"""
FRE GCP v1.0 — Document Page Preview Tool
==========================================
Renders a specific page of a PDF stored in GCS as a JPEG, saves it to a local
static folder, and returns a localhost URL so ADK's web UI can render it inline.

The image is served by a lightweight background HTTP server on port 8001
(started on first use, daemon thread).  Using a URL instead of base64 means
the Gemini API request body stays small — no more proxy ServerDisconnectedError.

Additionally, passes the raw JPEG bytes to Gemini Vision to extract:
  • Exact data points from charts/graphs
  • Table contents
  • Any text that's purely visual (axis labels, legends, etc.)

Note: no 'from __future__ import annotations' — ADK introspects signatures
at runtime and that directive breaks type resolution.
"""

import http.server
import logging
import os
import threading
import time
import uuid

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

# ── Local preview image server ────────────────────────────────────────────────
_PREVIEW_DIR = os.path.join(os.path.dirname(__file__), "..", "static_previews")
_PREVIEW_PORT = 8001
_server_started = False
_server_lock = threading.Lock()


def _ensure_preview_server() -> None:
    """Start a one-shot background HTTP server to serve rendered preview images."""
    global _server_started
    with _server_lock:
        if _server_started:
            return
        os.makedirs(_PREVIEW_DIR, exist_ok=True)

        class _SilentHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.path.abspath(_PREVIEW_DIR), **kwargs)

            def log_message(self, fmt, *args):  # suppress access log noise
                pass

        def _serve():
            try:
                srv = http.server.HTTPServer(("localhost", _PREVIEW_PORT), _SilentHandler)
                logger.info("Preview image server started on http://localhost:%d/", _PREVIEW_PORT)
                srv.serve_forever()
            except OSError:
                # Port already in use (e.g. adk reloaded) — silently ignore
                pass

        t = threading.Thread(target=_serve, daemon=True, name="preview-http-server")
        t.start()
        _server_started = True


def preview_document_page(
    gcs_uri: str,
    page_number: int = 1,
    tool_context: ToolContext = None,  # ADK injects this automatically
) -> str:
    """
    Render a specific page of a PDF document stored in GCS as an inline image.
    The rendered page is saved locally (for inline display) and also uploaded
    to GCS so it persists beyond the current session.

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

    # ── Step 3: Save preview image and build image_markdown URL ──────────────
    # Uses the same localhost URL approach as generate_chart (confirmed working
    # in the ADK web UI). Base64 data URIs caused failures because the LLM
    # cannot reliably emit 90 000+ chars of base64 verbatim in its response.
    try:
        _ensure_preview_server()
        os.makedirs(os.path.abspath(_PREVIEW_DIR), exist_ok=True)
        safe_name = _filename.replace(" ", "_").replace("/", "_")

        # JPEG page preview — saved to static_previews/ and served via localhost
        fname = f"{safe_name}_p{page_number}_{uuid.uuid4().hex[:6]}.jpg"
        fpath = os.path.join(os.path.abspath(_PREVIEW_DIR), fname)
        with open(fpath, "wb") as fh:
            fh.write(img_bytes)
        logger.info("Preview saved: %s (%d bytes)", fpath, len(img_bytes))

        image_url = f"http://localhost:{_PREVIEW_PORT}/{fname}"
        _result_parts.append(f"![{_filename} — page {page_number}]({image_url})")

        # Original PDF — served so the user can open/download it directly
        pdf_fname = safe_name if safe_name.lower().endswith(".pdf") else safe_name + ".pdf"
        pdf_fpath = os.path.join(os.path.abspath(_PREVIEW_DIR), pdf_fname)
        with open(pdf_fpath, "wb") as fh:
            fh.write(pdf_bytes)
        pdf_url = f"http://localhost:{_PREVIEW_PORT}/{pdf_fname}"
        _result_parts.append(f"[\U0001f4c4 Open full PDF]({pdf_url})")
        logger.info("PDF served at: http://localhost:%d/%s", _PREVIEW_PORT, pdf_fname)

        # ── upload preview JPEG to GCS for persistence ──────────────────────────────────
        gcs_preview_uri = ""
        try:
            import config
            from storage.gcs import upload_bytes as _gcs_upload
            bucket = config.GCS_BUCKET or "fre-cognitive-search-docs"
            blob_path = f"session_previews/{fname}"
            gcs_preview_uri = _gcs_upload(img_bytes, bucket, blob_path, "image/jpeg")
            logger.info("Preview saved to GCS: %s", gcs_preview_uri)
            _result_parts.append(f"\n\U0001f4be **Saved to GCS:** `{gcs_preview_uri}`")
        except Exception as _gcs_exc:
            logger.warning("GCS preview upload skipped: %s", _gcs_exc)

        # ── store in session state so agent remembers what was previewed ──────────
        if tool_context is not None:
            try:
                previews: list = tool_context.state.get("session_previews", [])
                previews.append({
                    "gcs_source": gcs_uri,
                    "page": page_number,
                    "local_url": image_url,
                    "gcs_preview_uri": gcs_preview_uri,
                })
                tool_context.state["session_previews"] = previews
            except Exception as _state_exc:
                logger.debug("Could not update session state: %s", _state_exc)

    except Exception as exc:
        logger.error("Failed to save preview image: %s", exc)
        return f"{_error_prefix}Failed to save preview: {exc}"

    logger.info("[TIMING] preview_document_page total: %.2fs", time.perf_counter() - _t0)
    return "\n".join(_result_parts)
