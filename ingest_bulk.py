"""
Bulk Ingestion Script — FRE GCP v1.0
======================================
Indexes all supported documents from the GCS bucket into ES + Vertex AI +
PostgreSQL.  Designed for large datasets (4 GB+) with:

  • Progress bar and ETA
  • Resume support — skips already-indexed documents (idempotent pipeline)
  • Parallel workers (configurable)
  • Per-document error logging to a JSON report file
  • Final summary printed to console

Usage
-----
From the FRE_GCP_v1.0 directory:

    # Dry run — count documents without indexing:
    python ingest_bulk.py --dry-run

    # Index everything under GCS_PREFIX (from .env):
    python ingest_bulk.py

    # Index a specific sub-folder:
    python ingest_bulk.py --prefix "Example_Dataset/"

    # Force re-index (overwrite existing):
    python ingest_bulk.py --force

    # Tune parallelism (default 4, max 8):
    python ingest_bulk.py --workers 6

    # Write a JSON report of failures:
    python ingest_bulk.py --report ingestion_report.json

Environment
-----------
Set via .env or environment variables — same as ADK web startup.
Proxy vars (HTTPS_PROXY etc.) are picked up automatically from the shell.
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Logging — INFO to console, DEBUG to file
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,   # suppress library noise
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ingest_bulk")
log.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Bulk ingest GCS documents into FRE.")
parser.add_argument("--prefix",  default=None,
                    help="GCS prefix to scan (overrides GCS_PREFIX from .env)")
parser.add_argument("--bucket",  default=None,
                    help="GCS bucket name (overrides GCS_BUCKET from .env)")
parser.add_argument("--workers", type=int, default=4,
                    help="Parallel ingestion workers (1-8, default: 4)")
parser.add_argument("--force",   action="store_true",
                    help="Re-index documents already in the index")
parser.add_argument("--dry-run", action="store_true",
                    help="List documents and count them without indexing")
parser.add_argument("--report",  default="ingestion_report.json",
                    help="Path to write JSON failure report (default: ingestion_report.json)")
parser.add_argument("--limit",   type=int, default=0,
                    help="Index only the first N documents (0 = no limit, useful for testing)")
parser.add_argument("--failed-only", action="store_true",
                    help="Retry only documents with status=failed in the DB (ignores --prefix scan)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
import config  # noqa: E402 — loads .env
from storage import gcs, postgres  # noqa: E402
from ingestion.pipeline import process_document, PipelineResult  # noqa: E402

BUCKET = args.bucket or config.GCS_BUCKET
PREFIX = args.prefix if args.prefix is not None else config.GCS_PREFIX
WORKERS = max(1, min(args.workers, 8))
FORCE = args.force

if not BUCKET:
    sys.exit("ERROR: GCS_BUCKET is not set. Edit .env or pass --bucket.")

# ─────────────────────────────────────────────────────────────────────────────
# Helper — simple progress bar (no external deps)
# ─────────────────────────────────────────────────────────────────────────────
def _bar(done: int, total: int, width: int = 40) -> str:
    pct = done / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {done}/{total} ({pct*100:.1f}%)"


def _eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "ETA: --"
    rate = done / elapsed
    remaining = (total - done) / rate
    m, s = divmod(int(remaining), 60)
    h, m = divmod(m, 60)
    if h:
        return f"ETA: {h}h {m:02d}m"
    return f"ETA: {m}m {s:02d}s"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — List all documents
# ─────────────────────────────────────────────────────────────────────────────
if args.failed_only:
    print("\nQuerying DB for failed documents …")
    try:
        import psycopg2
        con = psycopg2.connect(config.POSTGRES_DSN, connect_timeout=5)
        cur = con.cursor()
        cur.execute("SELECT gcs_uri FROM documents WHERE status = 'failed'")
        all_uris = [row[0] for row in cur.fetchall()]
        con.close()
    except Exception as exc:
        sys.exit(f"ERROR: Could not query failed documents from DB: {exc}")
    if not all_uris:
        sys.exit("No failed documents found in DB — nothing to retry.")
    print(f"Found {len(all_uris)} failed documents to retry.")
    FORCE = True  # always force re-index for failed docs
else:
    print(f"\nScanning gs://{BUCKET}/{PREFIX} …")
    t_scan = time.perf_counter()
    all_uris = list(gcs.list_blobs(BUCKET, prefix=PREFIX, extensions=gcs.SUPPORTED_EXTENSIONS))
    scan_elapsed = time.perf_counter() - t_scan

    if not all_uris:
        sys.exit(f"No supported documents found under gs://{BUCKET}/{PREFIX}")

    print(f"Found {len(all_uris)} supported documents in {scan_elapsed:.1f}s")

if not args.failed_only:
    # Extension breakdown
    ext_counts: dict[str, int] = {}
    for u in all_uris:
        ext = PurePosixPath(u).suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"  {ext:8s}  {count:5d} files")

    if args.limit > 0:
        all_uris = all_uris[: args.limit]
        print(f"\n[--limit {args.limit}] Processing first {len(all_uris)} documents only.")

    if args.dry_run:
        print(f"\nDry run complete — {len(all_uris)} documents would be indexed.")
        print(f"Re-run without --dry-run to start ingestion.")
        sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Init DB schema
# ─────────────────────────────────────────────────────────────────────────────
print("\nInitialising database schema …")
try:
    postgres.init_db()
    print("  Database schema OK")
except Exception as exc:
    print(f"  WARNING: DB init failed ({exc}) — chunk metadata will not be stored.")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Parallel ingestion with progress
# ─────────────────────────────────────────────────────────────────────────────
total = len(all_uris)
results: list[PipelineResult] = []
t_start = time.perf_counter()

print(f"\nStarting ingestion — {total} documents, {WORKERS} workers, force={FORCE}\n")

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(process_document, uri, FORCE): uri for uri in all_uris}

    done = 0
    for future in as_completed(futures):
        uri = futures[future]
        try:
            result = future.result()
        except Exception as exc:
            result = PipelineResult(uri, "failed", error=str(exc))

        results.append(result)
        done += 1
        elapsed = time.perf_counter() - t_start

        # Status icon
        icon = {"indexed": "✓", "skipped": "–", "failed": "✗"}.get(result.status, "?")

        # Progress line (overwrite in place)
        bar_str = _bar(done, total)
        eta_str = _eta(elapsed, done, total)
        rate = done / elapsed if elapsed > 0 else 0
        status_line = f"\r  {bar_str}  {rate:.1f} doc/s  {eta_str}  "
        print(status_line, end="", flush=True)

        # Print failures immediately below progress
        if result.status == "failed":
            print(f"\n  {icon} FAILED: {PurePosixPath(uri).name}  — {result.error}")

print()  # newline after progress bar

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Summary
# ─────────────────────────────────────────────────────────────────────────────
elapsed_total = time.perf_counter() - t_start
indexed  = [r for r in results if r.status == "indexed"]
skipped  = [r for r in results if r.status == "skipped"]
failed   = [r for r in results if r.status == "failed"]
total_chunks = sum(r.chunk_count for r in indexed)
total_images = sum(r.image_count for r in indexed)

print("\n" + "=" * 60)
print(f"  Ingestion complete in {elapsed_total/60:.1f} min")
print(f"  Total documents : {total}")
print(f"  ✓ Indexed       : {len(indexed)}  ({total_chunks:,} chunks, {total_images:,} image pages)")
print(f"  – Skipped       : {len(skipped)}  (already indexed or unsupported)")
print(f"  ✗ Failed        : {len(failed)}")
if failed:
    print(f"\n  Failed documents:")
    for r in failed:
        print(f"    • {PurePosixPath(r.gcs_uri).name}  — {r.error}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Write JSON report
# ─────────────────────────────────────────────────────────────────────────────
report: dict[str, Any] = {
    "bucket":         BUCKET,
    "prefix":         PREFIX,
    "workers":        WORKERS,
    "force":          FORCE,
    "total":          total,
    "indexed":        len(indexed),
    "skipped":        len(skipped),
    "failed":         len(failed),
    "total_chunks":   total_chunks,
    "total_images":   total_images,
    "elapsed_seconds": round(elapsed_total, 1),
    "documents": [
        {
            "gcs_uri":     r.gcs_uri,
            "filename":    PurePosixPath(r.gcs_uri).name,
            "status":      r.status,
            "chunk_count": r.chunk_count,
            "image_count": r.image_count,
            "es_ok":       r.es_ok,
            "vertex_ok":   r.vertex_ok,
            "error":       r.error,
        }
        for r in results
    ],
}

report_path = Path(args.report)
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\n  Full report written to: {report_path.resolve()}")

sys.exit(1 if failed else 0)
