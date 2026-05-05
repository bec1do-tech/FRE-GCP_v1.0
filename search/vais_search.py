"""FRE GCP v1.0 — Vertex AI Search / Gemini Enterprise Backend
=============================================================
Queries the Gemini Enterprise app (FileResearchEngine) built on top of the
managed VAIS data store.  When VAIS_ENGINE_ID is set the engine-based
serving config is used, which enables Gemini Enterprise answer generation.
Falls back to the raw data-store serving config when only VAIS_DATA_STORE_ID
is configured.

Data store : fre-database_1777369569668  (unstructured, global, GCS-backed, connected to engine)
Engine     : fileresearchengine_1777366613744  (Gemini Enterprise, global)
"""

from __future__ import annotations

import logging
from typing import Any

import config

logger = logging.getLogger(__name__)


def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Query the Vertex AI Search managed data store and return ranked results.

    Returns a list of dicts, each with:
      filename  (str)  : basename of the source document
      gcs_uri   (str)  : full gs:// URI
      excerpt   (str)  : best matching snippet from the document
      score     (float): relevance score (0–1)
      source    (str)  : always "vais"
    """
    engine_id     = getattr(config, "VAIS_ENGINE_ID", "")
    data_store_id = getattr(config, "VAIS_DATA_STORE_ID", "")
    if not engine_id and not data_store_id:
        logger.warning("Neither VAIS_ENGINE_ID nor VAIS_DATA_STORE_ID configured — skipping Vertex AI Search.")
        return []

    try:
        from google.cloud import discoveryengine_v1 as discoveryengine  # type: ignore[import-untyped]
        from google.api_core.client_options import ClientOptions  # type: ignore[import-untyped]

        location = getattr(config, "VAIS_LOCATION", "global")
        client_options = (
            ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
            if location != "global"
            else None
        )
        client = discoveryengine.SearchServiceClient(client_options=client_options)

        def _do_search(serving_config: str) -> list[dict[str, Any]]:
            # Extractive answers/segments require Enterprise Edition.
            # Standard Edition rejects the field with HTTP 400.
            # Try with extractive content first; fall back to snippets-only.
            def _build_request(use_extractive: bool) -> "discoveryengine.SearchRequest":
                content_spec = discoveryengine.SearchRequest.ContentSearchSpec(
                    snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                        return_snippet=True,
                        max_snippet_count=3,
                    ),
                )
                if use_extractive:
                    content_spec.extractive_content_spec = (
                        discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                            max_extractive_answer_count=1,
                            max_extractive_segment_count=2,
                        )
                    )
                return discoveryengine.SearchRequest(
                    serving_config=serving_config,
                    query=query,
                    page_size=top_k,
                    content_search_spec=content_spec,
                )

            try:
                response = client.search(_build_request(use_extractive=True))
            except Exception as ee:
                if "extractive" in str(ee).lower() or "enterprise" in str(ee).lower() or "400" in str(ee):
                    logger.info("VAIS: extractive content not supported (Standard Edition) — retrying without it.")
                    response = client.search(_build_request(use_extractive=False))
                else:
                    raise
            out: list[dict[str, Any]] = []
            for result in response.results:
                doc = result.document
                derived = doc.derived_struct_data
                gcs_uri = derived.get("link", "") or doc.name or ""
                filename = gcs_uri.rsplit("/", 1)[-1] if gcs_uri else doc.id
                excerpt_parts: list[str] = []
                for ans in derived.get("extractive_answers", []):
                    if "content" in ans:
                        excerpt_parts.append(ans["content"])
                if not excerpt_parts:
                    for snip in derived.get("snippets", []):
                        if "snippet" in snip:
                            excerpt_parts.append(snip["snippet"])
                excerpt = " … ".join(excerpt_parts[:2]) if excerpt_parts else "(no excerpt)"
                # model_scores is a proto MapComposite — use direct key access,
                # not .get(), which can silently return a Value proto instead of float.
                score = 0.0
                try:
                    if result.model_scores:
                        score = float(result.model_scores["relevance_score"])
                except (KeyError, TypeError, ValueError):
                    pass
                out.append({
                    "filename": filename,
                    "gcs_uri": gcs_uri,
                    "excerpt": excerpt,
                    "score": score,
                    "source": "vais",
                })
            return out

        # ── Try 1: Gemini Enterprise engine serving config ────────────────────
        if engine_id:
            engine_cfg = (
                f"projects/{config.GCP_PROJECT}/locations/{location}"
                f"/collections/default_collection/engines/{engine_id}"
                f"/servingConfigs/default_search"
            )
            try:
                results = _do_search(engine_cfg)
                if results:
                    logger.info("VAIS engine returned %d results for: %s", len(results), query)
                    return results
                logger.info(
                    "VAIS engine returned 0 results (data store may not be connected yet) — "
                    "falling back to direct data store query."
                )
            except Exception as engine_exc:
                logger.warning("VAIS engine query failed (%s) — falling back to data store.", engine_exc)

        # ── Try 2: Direct data store serving config (fallback) ────────────────
        if data_store_id:
            ds_cfg = (
                f"projects/{config.GCP_PROJECT}/locations/{location}"
                f"/collections/default_collection/dataStores/{data_store_id}"
                f"/servingConfigs/default_config"
            )
            results = _do_search(ds_cfg)
            logger.info("VAIS data store returned %d results for: %s", len(results), query)
            return results

        return []

    except Exception as exc:
        logger.error("Vertex AI Search error: %s", exc)
        return []
