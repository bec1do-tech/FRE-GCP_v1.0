"""
Gemini Enterprise Search Sub-Agent
====================================
Queries the Gemini Enterprise app (FileResearchEngine) built on the managed
VAIS data store.  Uses the engine-based serving config so Gemini Enterprise
answer generation is active.  Runs in parallel with the ES and Vertex AI
vector search agents.
"""

from google.adk.agents import Agent

import config


def vais_search_tool(query: str, top_k: int = 5) -> dict:
    """
    Search the Vertex AI Search managed data store for documents relevant to
    the query.  This data store is automatically kept in sync with the GCS
    bucket — no custom ingestion pipeline required.

    Use this tool to find document excerpts using Google's fully-managed
    RAG search (Discovery Engine / Vertex AI Search).

    Parameters
    ----------
    query  : Natural-language search query.
    top_k  : Number of results to return (1–10).

    Returns
    -------
    Dict with keys:
      results (list): each item has {filename, gcs_uri, excerpt, score, source}
      total   (int) : number of results returned
      query   (str) : the original query
      backend (str) : "vertex_ai_search"
    """
    from search.vais_search import search

    results = search(query=query, top_k=top_k)

    # Resolve HTTP URLs for all unique GCS URIs in parallel
    from tools.search_tools import _batch_sign_uris
    unique_uris = list({r["gcs_uri"] for r in results if r.get("gcs_uri")})
    uri_map = _batch_sign_uris(unique_uris)
    for r in results:
        r["http_url"] = uri_map.get(r.get("gcs_uri", ""), "")

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "backend": "vertex_ai_search",
    }


vais_search_agent = Agent(
    name="vais_search_agent",
    model=config.GEMINI_MODEL,
    output_key="vais_search_results",
    description=(
        "Gemini Enterprise search agent (FileResearchEngine app). "
        "Queries documents via the managed Gemini Enterprise engine — "
        "powered by Vertex AI Search with built-in answer generation."
    ),
    instruction="""
    You are the Gemini Enterprise Search specialist — you query the
    FileResearchEngine Gemini Enterprise app backed by the managed VAIS
    data store.

    ════════════════════════════════════════════════
    ❌ ABSOLUTE TOOL CONSTRAINT — READ THIS FIRST:
    You have access to EXACTLY ONE tool: vais_search_tool.
    You MUST NEVER call any other function name.
    NEVER call: get_document_page_image, get_image, preview_document_page,
    get_document_url, get_document_chunks, hybrid_search, or ANY other tool.
    If you attempt to call a non-existent tool, the system will CRASH.
    Your ONLY allowed action is to call vais_search_tool ONCE.
    ════════════════════════════════════════════════

    Steps:
    1. Extract the INFORMATION TOPIC from the conversation.
       Strip ALL action words: never include 'draw', 'plot', 'chart',
       'summarise', 'show me', 'preview', 'display', 'visualise' etc.
       If the user's latest message is a follow-up like 'just preview them all'
       or 'show me' with no new topic, reuse the topic from the PREVIOUS turn.

    2. Call vais_search_tool with top_k=5 using the extracted topic query.

    3. Format results as:

       ## Gemini Enterprise Search Results

       **Result 1 — [filename]**
       Source: [filename](http_url)   ← use the http_url value from the result as the link href
       Score: [score]
       Excerpt: [excerpt]

       **Result 2 — [filename]**
       ...

       If http_url is empty for a result, write Source: **[filename]** (no link).
       NEVER write gs:// URIs in the Source line — the http_url is already a signed browser link.

    4. If no results, state:
       "Gemini Enterprise Search returned no results."

    Do NOT synthesise an answer — just present the raw search results.
    Do NOT attempt to display images or previews — that is handled by synthesis_agent.
    """,
    tools=[vais_search_tool],
)
