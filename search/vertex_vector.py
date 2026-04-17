"""
FRE GCP v1.0 — Vertex AI Vector Search backend
================================================
Provides dense-vector (semantic) search using:
  • Vertex AI Text Embedding API (text-embedding-004, 768-dim)
  • Vertex AI Vector Search (Matching Engine) for ANN retrieval

Architecture
------------
  1. Embeddings are generated in batches using the Vertex AI Embeddings API.
  2. Vectors are upserted into a Vertex AI Vector Search Index via the REST API.
  3. At query time, the user query is embedded and `find_neighbors` is called
     against the deployed IndexEndpoint.

Prerequisites (one-time GCP setup)
-----------------------------------
  1. Create a Vertex AI Vector Search Index:
       gcloud ai indexes create --display-name=cognitive-search \
         --metadata-file=index_metadata.json --region=us-central1
     index_metadata.json → {"contentsDeltaUri": "", "config": {"dimensions": 768,
       "approximateNeighborsCount": 150, "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
       "algorithmConfig": {"treeAhConfig": {}}}}

  2. Deploy the index to an IndexEndpoint:
       gcloud ai index-endpoints deploy-index <ENDPOINT_ID> \
         --deployed-index-id=cognitive_search_index \
         --index=<INDEX_ID> --region=us-central1

  3. Set VERTEX_AI_INDEX_NAME and VERTEX_AI_INDEX_ENDPOINT in .env.

Degradation
-----------
  Returns [] and logs a warning when Vertex AI is unreachable or not configured.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import config

logger = logging.getLogger(__name__)

# Module-level singletons (lazy-initialised)
_embedding_model = None
_index_endpoint  = None


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel  # type: ignore[import-untyped]

            vertexai.init(project=config.GCP_PROJECT, location=config.GCP_REGION)
            _embedding_model = TextEmbeddingModel.from_pretrained(config.VERTEX_EMBEDDING_MODEL)
        except Exception as exc:
            logger.warning("Vertex AI embedding model unavailable: %s", exc)
    return _embedding_model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings.  Returns a parallel list of float vectors.
    Returns an empty list if Vertex AI is unavailable.
    """
    model = _get_embedding_model()
    if model is None:
        return []
    try:
        from vertexai.language_models import TextEmbeddingInput  # type: ignore[import-untyped]

        results: list[list[float]] = []
        # Process in batches of BATCH_SIZE (API limit ~250 per call)
        for start in range(0, len(texts), config.BATCH_SIZE):
            batch = texts[start : start + config.BATCH_SIZE]
            inputs = [
                TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in batch
            ]
            embeddings = model.get_embeddings(
                inputs, output_dimensionality=config.VERTEX_EMBEDDING_DIM
            )
            results.extend(e.values for e in embeddings)
        return results
    except Exception as exc:
        logger.error("get_embeddings failed: %s", exc)
        return []


def get_query_embedding(query: str) -> list[float]:
    """Embed a single search query (uses RETRIEVAL_QUERY task type)."""
    model = _get_embedding_model()
    if model is None:
        return []
    try:
        from vertexai.language_models import TextEmbeddingInput  # type: ignore[import-untyped]

        inp = TextEmbeddingInput(query, "RETRIEVAL_QUERY")
        embeddings = model.get_embeddings(
            [inp], output_dimensionality=config.VERTEX_EMBEDDING_DIM
        )
        return embeddings[0].values
    except Exception as exc:
        logger.error("get_query_embedding failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Index endpoint helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_index_endpoint():
    global _index_endpoint
    if _index_endpoint is None:
        if not config.VERTEX_AI_INDEX_ENDPOINT:
            return None
        try:
            import google.cloud.aiplatform as aip  # type: ignore[import-untyped]

            aip.init(project=config.GCP_PROJECT, location=config.GCP_REGION)
            _index_endpoint = aip.MatchingEngineIndexEndpoint(
                index_endpoint_name=config.VERTEX_AI_INDEX_ENDPOINT
            )
        except Exception as exc:
            logger.warning("Vertex AI IndexEndpoint unavailable: %s", exc)
    return _index_endpoint


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict]) -> bool:
    """
    Generate embeddings and upsert datapoints into Vertex AI Vector Search.

    Each chunk dict must have:
      vector_id (str)  — unique identifier (stored in chunks table)
      text      (str)  — the text to embed
    Optional:
      gcs_uri (str), filename (str), doc_id (int)

    Returns True on success, False otherwise.
    """
    if not chunks or not config.VERTEX_AI_INDEX_NAME:
        return False
    try:
        import google.cloud.aiplatform as aip  # type: ignore[import-untyped]
        from google.cloud.aiplatform_v1.types import index as index_v1  # type: ignore[import-untyped]

        aip.init(project=config.GCP_PROJECT, location=config.GCP_REGION)

        texts    = [c["text"][: 2048] for c in chunks]  # truncate for embedding API
        vectors  = get_embeddings(texts)
        if not vectors:
            return False

        datapoints = []
        for chunk, vector in zip(chunks, vectors):
            dp = index_v1.IndexDatapoint(
                datapoint_id=chunk["vector_id"],
                feature_vector=vector,
                restricts=[
                    index_v1.IndexDatapoint.Restriction(
                        namespace="gcs_uri",
                        allow_list=[chunk.get("gcs_uri", "")],
                    )
                ],
            )
            datapoints.append(dp)

        index = aip.MatchingEngineIndex(index_name=config.VERTEX_AI_INDEX_NAME)
        index.upsert_datapoints(datapoints=datapoints)
        return True
    except Exception as exc:
        logger.error("upsert_chunks failed: %s", exc)
        return False


def search(query: str, top_k: int = 10, gcs_uri_filter: str = "") -> list[dict]:
    """
    Semantic nearest-neighbour search.
    Optionally restrict results to a specific document via gcs_uri_filter.

    Returns a list of dicts: {vector_id, gcs_uri, distance, source}
    Returns [] when Vertex AI is unavailable or not configured.
    """
    endpoint = _get_index_endpoint()
    if endpoint is None:
        return []

    query_vector = get_query_embedding(query)
    if not query_vector:
        return []

    try:
        kwargs: dict[str, Any] = {
            "deployed_index_id": config.VERTEX_AI_DEPLOYED_INDEX_ID,
            "queries":           [query_vector],
            "num_neighbors":     top_k,
        }
        if gcs_uri_filter:
            kwargs["filter"] = [
                {"namespace": "gcs_uri", "allow_tokens": [gcs_uri_filter]}
            ]

        response = endpoint.find_neighbors(**kwargs)
        results  = []
        for neighbor in response[0]:  # response is [[neighbors]] for one query
            results.append(
                {
                    "vector_id": neighbor.id,
                    "distance":  neighbor.distance,
                    "source":    "vertex_ai",
                }
            )
        return results
    except Exception as exc:
        logger.error("Vertex AI search failed: %s", exc)
        return []


def collection_stats() -> dict:
    """Return availability status: {available: bool}."""
    model_ok   = _get_embedding_model() is not None
    endpoint_ok = config.VERTEX_AI_INDEX_ENDPOINT != ""
    return {
        "available":    model_ok,
        "endpoint_set": endpoint_ok,
    }
