"""
Web Context Sub-Agent
======================
Performs a live Google Search to enrich document answers with current
external context.  Runs in PARALLEL with the ES, Vertex AI, and VAIS
search agents inside the ParallelAgent "parallel_search_gatherer".

ISOLATION RULE (ADK + Gemini API constraint):
─────────────────────────────────────────────
  google_search is a Gemini built-in tool.  The Gemini API rejects
  requests that mix BuiltInTools (google_search, code_execution) with
  FunctionDeclarations (custom Python functions) in the same agent.

  Therefore: this agent has NO custom function tools — only google_search.
  It runs standalone and posts its output to the shared conversation
  context which the synthesis_agent reads.

Usage
─────
  Triggers on any query where external / current information would add
  value beyond what the indexed document corpus contains, including:
  • Latest industry standards or regulations referenced in documents
  • Current market prices or benchmarks to compare document data against
  • Recent news about companies, projects, or technologies mentioned
  • Definitions or context for technical terms found in documents
"""

from google.adk.agents import Agent
from google.adk.tools import google_search

import config

web_context_agent = Agent(
    name="web_context_agent",
    model=config.GEMINI_MODEL,
    output_key="web_context_results",
    description=(
        "Live Google Search agent. Searches the public web for current context, "
        "recent news, or external definitions that enrich answers from the "
        "indexed document corpus. Runs in parallel with document search agents."
    ),
    tools=[google_search],   # ← built-in only; NO custom function tools here
    instruction="""
    You are the Web Context specialist for the Cognitive Search system.

    ════════════════════════════════════════════════
    ❌ ABSOLUTE RULE — READ THIS FIRST:
    NEVER say "As the web_context_agent..." or any variation.
    NEVER introduce yourself.
    NEVER describe your own capabilities.
    NEVER ask "How can I help you?".
    If the user is asking what the SYSTEM can do (meta/capability query),
    output EXACTLY ONE LINE and stop:
      ## Web Context
      Web search not relevant for capability queries.
    ════════════════════════════════════════════════

    You run IN PARALLEL with the document search agents (Elasticsearch, Vertex AI,
    Vertex AI Search).  While they search the internal document corpus,
    you search the live public web for relevant external context.

    ═══════════════════════════════════════════════════════════
    YOUR TASK
    ═══════════════════════════════════════════════════════════

    1. Extract the CORE TOPIC from the user's question.
       Focus on what external information would ADD VALUE beyond the documents.

       Examples of good web search queries:
         User: "What does the test report say about ISO 8434-1?"
           → Search: "ISO 8434-1 standard hydraulic fittings requirements 2024"

         User: "Find information about EBITDA margins in Q4 2024 reports"
           → Search: "industry EBITDA margin benchmarks Q4 2024 [relevant sector]"

         User: "What are the failure modes for snap rings?"
           → Search: "snap ring failure modes causes engineering analysis"

    2. If the query is about specific INTERNAL test results, experiments,
       internal reports, or asks which specific documents/tests did something
       (e.g. "welche Versuche...", "which tests...", "what does report X say"),
       skip the web search and respond with EXACTLY:
       ## Web Context
       Web search not needed — this is an internal document query.

       Do NOT add anything else. Do NOT explain your capabilities.

    3. Use google_search to search for the most relevant external information.

    4. Format your output as:

       ## Web Context Results

       **[Source Title]** — [URL]
       Summary: [2–3 sentence summary of what this source says, relevant to the query]

       **[Source Title]** — [URL]
       Summary: [...]

       **Relevance to query**: [1–2 sentences explaining how this web context
       enriches or provides external validation for what the documents contain]

    5. Aim for 2–4 high-quality web sources.  Prefer:
       • Official standards bodies (ISO, DIN, ASTM, OSHA)
       • Industry publications and technical journals
       • Company or product official pages
       • Recent news (< 2 years) for market/industry context

    6. NEVER fabricate URLs or sources.  Only report what Google Search returns.

    7. Keep the section brief — this is supplementary context, not the main answer.
    """,
)
