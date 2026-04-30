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
    return {
        "results": results,
        "total": len(results),
        "query": query,
        "backend": "vertex_ai_search",
    }


vais_search_agent = Agent(
    name="vais_search_agent",
    model=config.GEMINI_MODEL,
    description=(
        "Gemini Enterprise search agent (FileResearchEngine app). "
        "Queries documents via the managed Gemini Enterprise engine — "
        "powered by Vertex AI Search with built-in answer generation."
    ),
    instruction="""
    You are the Gemini Enterprise Search specialist — you query the
    FileResearchEngine Gemini Enterprise app backed by the managed VAIS
    data store (fre-database_1777367196372).

    This engine is automatically indexed directly from the GCS bucket by
    Google — with built-in OCR, parsing, chunking, embedding, and Gemini
    Enterprise answer generation. It complements the custom ES + Vertex AI
    Vector Search pipeline.

    Steps:
    1. Extract the INFORMATION TOPIC from the conversation (same as other agents).
       Strip action words: never include 'draw', 'plot', 'chart', 'summarise' etc.

    2. Call vais_search_tool with top_k=5 using the extracted topic query.

    3. Format results as:

       ## Gemini Enterprise Search Results

       **Result 1 — [filename]**
       Source: [gcs_uri]
       Score: [score]
       Excerpt: [excerpt]

       **Result 2 — [filename]**
       ...

    4. If no results, state:
       "Gemini Enterprise Search returned no results."

    Do NOT synthesise an answer — just present the raw search results.
    Do NOT answer the user's question yet.
    """,
    tools=[vais_search_tool],
)
