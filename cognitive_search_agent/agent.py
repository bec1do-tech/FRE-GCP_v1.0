"""
FRE GCP v1.0 — Cognitive Search Agent
=======================================
Multi-agent system built with Google ADK.

Architecture
─────────────
                        ┌────────────────────────────────────┐
                        │     cognitive_search_agent          │
                        │          (root Agent)               │
                        │                                     │
                        │  sub_agents:                        │
                        │   • document_qa_pipeline            │
                        │   • ingestion_manager_agent         │
                        │                                     │
                        │  tools:                             │
                        │   • get_search_status               │
                        └────────────┬───────────────────────┘
                                     │ delegates
              ┌──────────────────────┴─────────────────────┐
              ▼                                             ▼
  ┌───────────────────────┐               ┌────────────────────────────┐
  │  document_qa_pipeline  │               │  ingestion_manager_agent    │
  │  (SequentialAgent)     │               │  (Agent)                    │
  │                        │               │                             │
  │  Step 1:               │               │  tools:                     │
  │  parallel_search       │               │  • trigger_document_ingestion│
  │  (ParallelAgent)       │               │  • trigger_folder_ingestion  │
  │    ├─ es_search_agent  │               │  • get_ingestion_status      │
  │    └─ vertex_search    │               └────────────────────────────┘
  │         _agent         │
  │                        │
  │  Step 2:               │
  │  synthesis_agent       │
  └───────────────────────┘

ADK patterns used
─────────────────
  • Agent          — root orchestrator, ingestion manager
  • SequentialAgent — search pipeline (gather then synthesise)
  • ParallelAgent   — simultaneous ES + Vertex AI search

Run locally
───────────
  cd FRE_GCP_v1.0
  adk web
  # Open http://localhost:8000/dev-ui
"""

from google.adk.agents import Agent, ParallelAgent, SequentialAgent

from cognitive_search_agent.sub_agents.es_search_agent.agent     import es_search_agent
from cognitive_search_agent.sub_agents.vertex_search_agent.agent  import vertex_search_agent
from cognitive_search_agent.sub_agents.synthesis_agent.agent      import synthesis_agent
from cognitive_search_agent.sub_agents.ingestion_agent.agent      import ingestion_manager_agent
from tools.search_tools import get_search_status

import config

# ── Step 1: Run ES and Vertex AI search simultaneously ────────────────────────
parallel_search_gatherer = ParallelAgent(
    name="parallel_search_gatherer",
    description=(
        "Runs Elasticsearch BM25 search and Vertex AI semantic search simultaneously, "
        "producing two independent sets of ranked results."
    ),
    sub_agents=[es_search_agent, vertex_search_agent],
)

# ── Step 2: Gather → Synthesise pipeline ─────────────────────────────────────
document_qa_pipeline = SequentialAgent(
    name="document_qa_pipeline",
    description=(
        "Full document Q&A pipeline: runs ES + Vertex AI search in parallel, "
        "then synthesises a cited answer from both result sets."
    ),
    sub_agents=[parallel_search_gatherer, synthesis_agent],
)

# ── Root Master Agent ─────────────────────────────────────────────────────────
root_agent = Agent(
    name="cognitive_search_agent",
    model=config.GEMINI_MODEL,
    description=(
        "Intelligent document assistant that answers questions from a 10 TB "
        "enterprise document repository using hybrid semantic + keyword search."
    ),
    instruction="""
    You are the Cognitive Search Agent — an intelligent assistant that helps users
    find specific, actionable answers from a large enterprise document repository.

    You have access to a knowledge base of indexed documents (PDFs, DOCX, PPTX,
    XLSX, and more) that has been processed with Gemini Vision to understand charts,
    tables, and images — not just raw text.

    ═══════════════════════════════════════════════════════════
    YOUR CAPABILITIES
    ═══════════════════════════════════════════════════════════

    1. DOCUMENT Q&A  →  delegate to: document_qa_pipeline
       ─────────────────────────────────────────────────────
       Use when the user asks a question, wants to find information, or request
       a summary from the document corpus.

       Examples:
         • "What were the main supply chain risks cited in Q4 2024 reports?"
         • "Find all documents mentioning Project Aurora"
         • "Summarise our data retention policies"
         • "What does the financial report say about EBITDA margins?"

    2. DOCUMENT INDEXING  →  delegate to: ingestion_manager_agent
       ─────────────────────────────────────────────────────────────
       Use when the user wants to add or re-index documents.

       Examples:
         • "Index the file gs://bucket/reports/Q4.pdf"
         • "Scan and index everything in gs://bucket/finance/2024/"
         • "How many documents have been indexed?"
         • "Re-index this file: gs://bucket/policy.docx"

    3. SYSTEM STATUS  →  use get_search_status tool directly
       ─────────────────────────────────────────────────────
       Use when the user asks about system health, index statistics,
       or whether the backends are operational.

    ═══════════════════════════════════════════════════════════
    ROUTING RULES
    ═══════════════════════════════════════════════════════════

    • Questions about document CONTENT   → delegate to document_qa_pipeline
    • Requests to INDEX / ADD documents  → delegate to ingestion_manager_agent
    • Questions about SYSTEM STATUS      → call get_search_status tool

    ═══════════════════════════════════════════════════════════
    SEARCH FILTERS (pass to document_qa_pipeline)
    ═══════════════════════════════════════════════════════════
    If the user mentions specific constraints, note them for the search pipeline:
      • File type  (e.g. "in PDF files", "in Excel sheets")
      • Department (e.g. "from the Finance team")
      • Date range (e.g. "from Q4 2024", "between 2023 and 2024")
      • Project    (e.g. "related to Project Aurora")

    ═══════════════════════════════════════════════════════════
    COMMUNICATION STYLE
    ═══════════════════════════════════════════════════════════
    • Be direct and professional.
    • Always cite sources (the pipeline agents will provide GCS URIs).
    • If the indexed corpus does not contain the answer, say so honestly.
    • Never fabricate information — only report what the documents contain.

    When in doubt about the user's intent, ask one clarifying question.
    """,
    sub_agents=[document_qa_pipeline, ingestion_manager_agent],
    tools=[get_search_status],
)
