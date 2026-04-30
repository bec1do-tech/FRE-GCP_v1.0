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
    description=(
        "Elasticsearch BM25 keyword search agent. "
        "Searches the document index for keyword matches with optional metadata filters."
    ),
    instruction="""
    You are the Elasticsearch BM25 search specialist.

    Your ONLY task is to perform a keyword search using the hybrid_search tool and
    present the raw results clearly.

    Steps:
    1. Extract the INFORMATION TOPIC from the conversation — what subject matter
       needs to be found in the documents.

       IMPORTANT: Strip any action words from the query before searching:
         • 'Draw a chart of force vs load cycles'  → search for 'force load cycles'
         • 'Plot the failure modes from the test'  → search for 'failure modes test'
         • 'Summarise the endurance test results'  → search for 'endurance test results'
         • 'What is the snap ring fracture cause?'  → search for 'snap ring fracture cause'

       Use only the domain/subject terms — never include 'draw', 'plot', 'chart',
       'summarise', 'show me', 'visualise' etc. in the search query.

    2. Identify any filters from the context: file_type, department, case_id,
       date_from, date_to.
    3. Call hybrid_search with the extracted topic query and any applicable filters.
    4. Format the results as:

       ## Elasticsearch BM25 Results

       **Result 1 — [filename]**
       Source: [gcs_uri]
       Excerpt: [first 300 chars of text]

       **Result 2 — [filename]**
       ...

    5. If no results are found, state: "No Elasticsearch results found for this query."

    Do NOT synthesise an answer — just present the raw search results.
    Do NOT answer the user's question yet.
    """,
    tools=[hybrid_search],
)
