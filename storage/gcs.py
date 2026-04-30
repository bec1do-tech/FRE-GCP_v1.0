"""
FRE GCP v1.0 — Cloud Storage (GCS) helpers
============================================
Thin wrapper around google-cloud-storage for downloading, streaming, and
listing blobs.  All public functions degrade gracefully to empty/None if the
GCS client cannot be initialised (e.g. during local dev without ADC).

GCS URI format:  gs://<bucket>/<object-path>
"""

from __future__ import annotations

import io
import logging
from pathlib import PurePosixPath
from typing import Iterator

logger = logging.getLogger(__name__)

# ── Lazy GCS client (avoids hard crash when credentials are absent) ───────────
_gcs_client = None


def _client():
    """
    Return a GCS client.

    When a corporate HTTPS proxy is detected (HTTPS_PROXY / HTTP_PROXY env vars),
    the proxy intercepts TLS and re-signs with the corporate CA, which is not in
    Python's certifi bundle.  We configure an AuthorizedSession with verify=False
    and pass it as the _http transport so downloads work through the proxy.
    """
    global _gcs_client
    if _gcs_client is None:
        try:
            import os
            from google.cloud import storage  # type: ignore[import-untyped]

            if os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"):
                # Corporate proxy — SSL cert is re-signed by the proxy CA.
                # Use an AuthorizedSession with cert verification disabled.
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                import google.auth  # type: ignore[import-untyped]
                from google.auth.transport.requests import AuthorizedSession  # type: ignore[import-untyped]

                creds, _ = google.auth.default()
                authed_session = AuthorizedSession(creds)
                authed_session.verify = False
                _gcs_client = storage.Client(credentials=creds, _http=authed_session)
                logger.debug("GCS client initialised with proxy-aware AuthorizedSession (verify=False).")
            else:
                _gcs_client = storage.Client()
        except Exception as exc:
            logger.warning("GCS client unavailable: %s", exc)
    return _gcs_client


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Split 'gs://bucket/path/to/file' into (bucket, object_name)."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a valid GCS URI: {gcs_uri!r}")
    without_scheme = gcs_uri[5:]
    bucket, _, obj = without_scheme.partition("/")
    return bucket, obj


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_to_bytes(gcs_uri: str) -> bytes:
    """
    Download a GCS object and return its raw bytes.
    Raises RuntimeError if the GCS client is unavailable.
    """
    client = _client()
    if client is None:
        raise RuntimeError("GCS client not initialised — check Application Default Credentials.")
    bucket_name, object_name = parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return buf.read()


def download_to_stream(gcs_uri: str) -> io.BytesIO:
    """Download a GCS object into an in-memory BytesIO stream."""
    return io.BytesIO(download_to_bytes(gcs_uri))


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_blob_metadata(gcs_uri: str) -> dict:
    """
    Return a dict of blob metadata:
      {bucket, name, size, content_type, updated, md5_hash, generation}
    Returns {} if the blob doesn't exist or GCS is unavailable.
    """
    client = _client()
    if client is None:
        return {}
    try:
        bucket_name, object_name = parse_gcs_uri(gcs_uri)
        blob = client.bucket(bucket_name).blob(object_name)
        blob.reload()
        return {
            "bucket":       bucket_name,
            "name":         object_name,
            "filename":     PurePosixPath(object_name).name,
            "size":         blob.size or 0,
            "content_type": blob.content_type or "",
            "updated":      blob.updated.isoformat() if blob.updated else "",
            "md5_hash":     blob.md5_hash or "",
            "generation":   str(blob.generation or ""),
        }
    except Exception as exc:
        logger.warning("Could not fetch metadata for %s: %s", gcs_uri, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Listing
# ─────────────────────────────────────────────────────────────────────────────

def list_blobs(bucket: str, prefix: str = "", extensions: tuple[str, ...] = ()) -> Iterator[str]:
    """
    Yield GCS URIs (gs://...) for all blobs under *prefix* in *bucket*.
    Optionally filter by file *extensions* (e.g. ('.pdf', '.docx')).
    """
    client = _client()
    if client is None:
        logger.warning("GCS unavailable — list_blobs returns nothing.")
        return
    try:
        for blob in client.list_blobs(bucket, prefix=prefix):
            filename = blob.name.rsplit("/", 1)[-1]
            if filename.startswith("~$"):  # Office lock/temp files
                continue
            if extensions and not blob.name.lower().endswith(extensions):
                continue
            yield f"gs://{bucket}/{blob.name}"
    except Exception as exc:
        logger.error("list_blobs failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# File type helpers
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".txt", ".md", ".csv",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
)


def is_supported(gcs_uri: str) -> bool:
    """Return True if the blob's extension is in SUPPORTED_EXTENSIONS."""
    return PurePosixPath(gcs_uri).suffix.lower() in SUPPORTED_EXTENSIONS


def upload_bytes(
    data: bytes,
    bucket: str,
    blob_path: str,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload raw bytes to GCS and return the gs:// URI.

    Parameters
    ----------
    data         : File content to upload.
    bucket       : Bucket name (without gs:// prefix).
    blob_path    : Destination path inside the bucket, e.g. 'reports/file.docx'.
    content_type : MIME type written to the blob metadata.

    Returns
    -------
    The full GCS URI: gs://<bucket>/<blob_path>
    """
    client = _client()
    if client is None:
        raise RuntimeError("GCS client unavailable")
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
    gcs_uri = f"gs://{bucket}/{blob_path}"
    logger.info("Uploaded %d bytes to %s", len(data), gcs_uri)
    return gcs_uri
