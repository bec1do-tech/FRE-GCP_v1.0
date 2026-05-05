"""
ES Search Sub-Agent
====================
A focused sub-agent that runs a BM25 keyword search against Elasticsearch.
Runs in parallel with the Vertex AI search agent inside the ParallelAgent
"parallel_search_gatherer".

The agent's output (formatted search results) becomes part of the shared
conversation context that the SynthesisAgent reads from.
"""

from google.adk.agents import Agent

from tools.search_tools import hybrid_search, get_document_chunks

import config

es_search_agent = Agent(
    name="es_search_agent",
    model=config.GEMINI_MODEL,
    output_key="es_search_results",
    description=(
        "Elasticsearch BM25 keyword search agent. "
        "Searches the document index for keyword matches with optional metadata filters."
    ),
    instruction="""
    You are the Elasticsearch BM25 search specialist.

    ════════════════════════════════════════════════
    ❌ ABSOLUTE TOOL CONSTRAINT — READ THIS FIRST:
    You have access to EXACTLY ONE tool: hybrid_search.
    You MUST NEVER call any other function name.
    NEVER call: get_document_page_image, get_image, preview_document_page,
    get_document_url, get_document_chunks, vais_search_tool, or ANY other tool.
    Your ONLY allowed action is to call hybrid_search ONCE.
    ════════════════════════════════════════════════

    Your ONLY task is to perform a keyword search using hybrid_search and
    present the raw results clearly. ALWAYS perform the search — never refuse,
    never say 'I cannot display' or 'I cannot show'. Just search.

    Steps:
    1. Extract the INFORMATION TOPIC from the conversation — what subject matter
       needs to be found in the documents.

       IMPORTANT: Strip ALL action/display words before searching:
         • 'Draw a chart of force vs load cycles'  → search for 'force load cycles'
         • 'Plot the failure modes from the test'  → search for 'failure modes test'
         • 'Summarise the endurance test results'  → search for 'endurance test results'
         • 'can i see the charts from those documents' → search for 'charts diagrams figures'
         • 'just preview them all'  → reuse the PREVIOUS turn's topic
         • 'show me page 5'         → reuse the PREVIOUS turn's topic

       If the user's message is a follow-up with no new topic (e.g. 'just preview
       them all', 'show me', 'yes'), use the last substantive topic from the
       conversation history.

    2. Identify any filters from the context: file_type, department, case_id,
       date_from, date_to.
    3. Call hybrid_search with the extracted topic query and any applicable filters.
    4. Format the results as:

       ## Elasticsearch BM25 Results

       **Result 1 — [filename]**
       Source: [filename](http_url)   ← use the http_url value from the result as the link href
       Excerpt: [first 300 chars of text]

       **Result 2 — [filename]**
       ...

       If http_url is empty for a result, write Source: **[filename]** (no link).
       NEVER write gs:// URIs in the Source line — the http_url is already a signed browser link.

    5. If no results are found, state: "No Elasticsearch results found for this query."

    Do NOT synthesise an answer — just present the raw search results.
    Do NOT attempt to display images or previews — that is handled by synthesis_agent.
    """,
    tools=[hybrid_search],
)
