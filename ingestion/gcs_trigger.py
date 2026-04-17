"""
FRE GCP v1.0 — Cloud Function (GCS Event Trigger)
===================================================
This module is the entry point for a Cloud Functions (2nd gen) function that
is triggered by Eventarc whenever a new object is created in the GCS source
bucket.

Deployment
----------
  gcloud functions deploy fre-ingest-trigger \
    --gen2 \
    --runtime=python312 \
    --region=us-central1 \
    --source=. \
    --entry-point=process_gcs_event \
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
    --trigger-event-filters="bucket=YOUR_BUCKET_NAME" \
    --set-env-vars GCP_PROJECT=your-project,...

The function receives a CloudEvent whose data payload contains the GCS object
metadata (`bucket` and `name` fields).

Security: the Cloud Function service account only needs:
  - storage.objects.get on the source bucket
  - roles/cloudsql.client on the Cloud SQL instance
  - roles/aiplatform.user for Vertex AI calls
  - Secret Manager secretAccessor for any secrets
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def process_gcs_event(cloud_event) -> None:
    """
    Cloud Functions (gen2) entry point.

    The CloudEvent `data` payload follows the GCS StorageObjectData schema:
      https://cloud.google.com/storage/docs/json_api/v1/objects

    The function:
      1. Extracts (bucket, object_name) from the event.
      2. Constructs the GCS URI.
      3. Calls pipeline.process_document() to run the full ingestion pipeline.
    """
    try:
        data = cloud_event.data  # dict or JSON string
        if isinstance(data, str):
            data = json.loads(data)

        bucket = data.get("bucket", "")
        name   = data.get("name", "")

        if not bucket or not name:
            logger.error("Missing bucket or name in CloudEvent data: %s", data)
            return

        gcs_uri = f"gs://{bucket}/{name}"
        logger.info("Received GCS finalize event for: %s", gcs_uri)

        # Lazy import keeps the module importable without all dependencies
        # installed (avoids slow cold-start for non-supported files rejected early).
        from ingestion.pipeline import process_document
        from storage.postgres import init_db

        init_db()
        result = process_document(gcs_uri)

        logger.info(
            "Pipeline result for %s: status=%s, chunks=%d, images=%d",
            gcs_uri,
            result.status,
            result.chunk_count,
            result.image_count,
        )

    except Exception as exc:
        logger.exception("Unhandled error in process_gcs_event: %s", exc)
        # Re-raise so Cloud Functions marks the invocation as failed and retries.
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Local test helper
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick local test — simulate a CloudEvent for a specific GCS file.
    Usage:
      python -m ingestion.gcs_trigger gs://my-bucket/documents/report.pdf
    """
    import sys
    from types import SimpleNamespace

    uri = sys.argv[1] if len(sys.argv) > 1 else "gs://test-bucket/sample.pdf"
    parts = uri.removeprefix("gs://").split("/", 1)
    fake_event = SimpleNamespace(
        data={"bucket": parts[0], "name": parts[1] if len(parts) > 1 else ""}
    )
    process_gcs_event(fake_event)
