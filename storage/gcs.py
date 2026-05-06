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

    Uses the GCS JSON API with uploadType=media and HTTP chunked transfer
    encoding (streaming generator) through the proxy-aware AuthorizedSession.

    Key constraints discovered on the Bosch corporate network:
    - Direct DNS resolution for storage.googleapis.com is blocked (NO_PROXY bypass fails)
    - Proxy blocks POST/PUT bodies larger than ~9KB (binary upload size filter)
    - HTTP chunked transfer encoding (generator as data=) bypasses the size filter
      because the proxy cannot determine total body size upfront
    - uploadType=multipart is blocked; uploadType=media works
    - bec1do@bosch.com has no write access; must impersonate PREVIEW_SIGNING_SA
    """
    import urllib.parse
    import google.auth
    from google.auth import impersonated_credentials
    from google.auth.transport.requests import AuthorizedSession, Request as _Request
    import urllib3

    # Resolve signing SA for impersonation (same SA as signed URL generation)
    signing_sa = ""
    try:
        import config as _cfg
        signing_sa = _cfg.PREVIEW_SIGNING_SA
    except Exception:
        pass

    if signing_sa:
        source_creds, _ = google.auth.default()
        upload_creds = impersonated_credentials.Credentials(
            source_credentials=source_creds,
            target_principal=signing_sa,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
            lifetime=300,
        )
        upload_creds.refresh(_Request())
    else:
        upload_creds, _ = google.auth.default()

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    sess = AuthorizedSession(upload_creds)
    sess.verify = False

    def _gen(buf: bytes, chunk_size: int = 8192):
        """Yield buf in chunks — requests sends as Transfer-Encoding: chunked."""
        for i in range(0, len(buf), chunk_size):
            yield buf[i:i + chunk_size]

    encoded_name = urllib.parse.quote(blob_path, safe="")
    url = (
        f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
        f"?uploadType=media&name={encoded_name}"
    )
    resp = sess.post(
        url,
        data=_gen(data),
        headers={"Content-Type": content_type},
        timeout=120,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"GCS upload failed with HTTP {resp.status_code}: {resp.text[:400]}"
        )

    gcs_uri = f"gs://{bucket}/{blob_path}"
    logger.info("Uploaded %d bytes to %s", len(data), gcs_uri)
    return gcs_uri


def generate_signed_url(
    bucket: str,
    blob_path: str,
    expiration_seconds: int = 3600,
    signing_sa: str = "",
) -> str:
    """
    Generate a V4 signed URL for a GCS object, valid for *expiration_seconds*.

    Signing is performed by impersonating *signing_sa* (a service-account email)
    via the IAM Credentials API (iamcredentials.googleapis.com).  That API call
    is a normal HTTPS request and therefore flows through the corporate px proxy
    automatically.

    Parameters
    ----------
    bucket           : GCS bucket name (no gs:// prefix).
    blob_path        : Object path inside the bucket.
    expiration_seconds: How long the URL is valid (default 1 hour).
    signing_sa       : Service-account email used for signing.
                       Falls back to config.PREVIEW_SIGNING_SA if empty.

    Returns
    -------
    A fully-qualified HTTPS signed URL string.
    """
    import datetime
    import google.auth
    from google.auth.transport.requests import Request as _Request

    # Resolve signing SA
    if not signing_sa:
        try:
            import config as _cfg
            signing_sa = _cfg.PREVIEW_SIGNING_SA
        except Exception:
            pass
    if not signing_sa:
        raise ValueError(
            "No signing SA configured. Set PREVIEW_SIGNING_SA in .env "
            "or pass signing_sa= explicitly."
        )

    # Use the caller's ADC token (bec1do@bosch.com) directly.
    # bec1do has roles/iam.serviceAccountTokenCreator on signing_sa, so the
    # ADC token is authorised to call iamcredentials/signBlob on behalf of
    # signing_sa.  We must NOT use an impersonated token here because the SA
    # does not have signBlob on itself.
    source_creds, _ = google.auth.default()
    source_creds.refresh(_Request())

    client = _client()
    if client is None:
        raise RuntimeError("GCS client not initialised — check Application Default Credentials.")
    blob = client.bucket(bucket).blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(seconds=expiration_seconds),
        method="GET",
        service_account_email=signing_sa,
        access_token=source_creds.token,
    )

    logger.info("Signed URL generated for gs://%s/%s (expires %ds)", bucket, blob_path, expiration_seconds)
    return url
