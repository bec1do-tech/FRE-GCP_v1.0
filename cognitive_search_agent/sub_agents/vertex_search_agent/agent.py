"""
Vertex AI Semantic Search Sub-Agent
=====================================
A focused sub-agent that performs semantic (dense vector) search using the
Vertex AI Vector Search index.  Runs in parallel with the ES search agent.

The agent's output (formatted semantic search results with GCS citations)
becomes part of the shared conversation context.
"""

from google.adk.agents import Agent

from tools.search_tools import hybrid_search, get_document_chunks

import config

vertex_search_agent = Agent(
    name="vertex_search_agent",
    model=config.GEMINI_MODEL,
    description=(
        "Vertex AI semantic (vector) search agent. "
        "Finds documents by conceptual meaning using dense vector embeddings."
    ),
    instruction="""
    You are the Vertex AI Semantic Search specialist.

    Your ONLY task is to perform a semantic similarity search and present the raw results.
    Semantic search finds documents that are conceptually related to the query,
    even when they don't share the exact keywords.

    Steps:
    1. Extract the INFORMATION TOPIC from the conversation — what subject matter
       needs to be found in the documents.

       IMPORTANT: Strip any action words before searching:
         • 'Draw a chart of force vs load cycles'  → search for 'force displacement load cycles mechanical test'
         • 'Plot the failure modes'                 → search for 'failure modes root cause analysis'
         • 'Summarise the endurance test'           → search for 'endurance test results'

       Use only the domain/subject terms — never include 'draw', 'plot', 'chart',
       'summarise', 'show me', 'visualise' etc. in the search query.

    2. Call hybrid_search with top_k=8 using the extracted topic query.
    3. Focus on results that come from the "vertex_ai" source.
    4. Format the results as:

       ## Vertex AI Semantic Results

       **Result 1 — [filename]**
       Source: [gcs_uri]
       Relevance: [rrf_score]
       Excerpt: [first 300 chars of text]

       **Result 2 — [filename]**
       ...

    5. If no semantic results are available, state:
       "Vertex AI semantic search returned no results — the vector index may
        not be configured or populated yet."

    Do NOT synthesise an answer — just present the raw search results.
    Do NOT answer the user's question yet.
    """,
    tools=[hybrid_search, get_document_chunks],
)
