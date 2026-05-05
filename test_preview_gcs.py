"""
End-to-end test: render a real PDF page → upload JPEG to GCS session_previews/
→ generate V4 signed URL → confirm URL is reachable via the system proxy.

Run from the FRE_GCP_v1.0 directory:
    .venv\Scripts\python.exe test_preview_gcs.py
"""
import os
import sys
import time

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:3128")
os.environ.setdefault("HTTP_PROXY",  "http://127.0.0.1:3128")
os.environ.setdefault("NO_PROXY",    "localhost,127.0.0.1")

import config                                      # noqa: E402 (loads .env)
from storage.gcs import list_blobs, download_to_bytes, upload_bytes, generate_signed_url


# ── 1. Pick a real PDF from the bucket ───────────────────────────────────────
print("Step 1: Listing PDFs in GCS …")
pdfs = [u for u in list_blobs(config.GCS_BUCKET, prefix=config.GCS_PREFIX,
                               extensions=(".pdf",))]
if not pdfs:
    sys.exit("No PDFs found — check GCS_BUCKET / GCS_PREFIX in .env")
target = pdfs[0]
print(f"  Using: {target}")


# ── 2. Download PDF bytes ─────────────────────────────────────────────────────
print("Step 2: Downloading PDF …")
t0 = time.perf_counter()
pdf_bytes = download_to_bytes(target)
print(f"  {len(pdf_bytes):,} bytes in {time.perf_counter()-t0:.1f}s")


# ── 3. Render page 1 with PyMuPDF ────────────────────────────────────────────
print("Step 3: Rendering page 1 with PyMuPDF …")
try:
    import fitz
except ImportError:
    sys.exit("PyMuPDF not installed: pip install PyMuPDF")

t1 = time.perf_counter()
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
img_bytes = pix.tobytes("jpg", jpg_quality=60)
doc.close()
print(f"  JPEG: {len(img_bytes):,} bytes in {time.perf_counter()-t1:.1f}s")


# ── 4. Upload JPEG to GCS session_previews/ ───────────────────────────────────
print("Step 4: Uploading JPEG to GCS session_previews/ …")
import uuid
fname = f"test_{uuid.uuid4().hex[:8]}.jpg"
blob_path = f"session_previews/{fname}"

t2 = time.perf_counter()
gcs_uri = upload_bytes(img_bytes, config.GCS_BUCKET, blob_path, "image/jpeg")
print(f"  Uploaded: {gcs_uri}  ({time.perf_counter()-t2:.1f}s)")


# ── 5. Generate signed URL ────────────────────────────────────────────────────
print("Step 5: Generating V4 signed URL …")
t3 = time.perf_counter()
signed_url = generate_signed_url(config.GCS_BUCKET, blob_path, expiration_seconds=3600)
print(f"  Done in {time.perf_counter()-t3:.1f}s")
print(f"  URL: {signed_url[:120]} …")


# ── 6. Verify URL is reachable (HEAD request through px proxy) ────────────────
print("Step 6: Verifying signed URL is reachable …")
import urllib.request
import urllib.error

proxy_support = urllib.request.ProxyHandler({
    "http":  os.environ.get("HTTP_PROXY",  ""),
    "https": os.environ.get("HTTPS_PROXY", ""),
})
opener = urllib.request.build_opener(proxy_support)
try:
    req = urllib.request.Request(signed_url, method="HEAD")
    with opener.open(req, timeout=20) as resp:
        print(f"  HTTP {resp.status} — Content-Type: {resp.headers.get('Content-Type', '?')}")
        print(f"  Content-Length: {resp.headers.get('Content-Length', '?')} bytes")
except urllib.error.HTTPError as e:
    print(f"  HTTP error {e.code}: {e.reason}")
except Exception as e:
    print(f"  Connection error: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("RESULT — paste the markdown below into the ADK chat to test inline rendering:")
print(f"  ![test page 1]({signed_url[:80]}...)")
print()
print("Full signed URL (copy into browser address bar):")
print(signed_url)
