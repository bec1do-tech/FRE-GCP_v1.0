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
  ┌────────────────────────┐             ┌─────────────────────────────┐
  │  document_qa_pipeline   │             │  ingestion_manager_agent     │
  │  (SequentialAgent)      │             │  (Agent)                     │
  │                         │             │                              │
  │  Step 1:                │             │  tools:                      │
  │  parallel_search        │             │  • trigger_document_ingestion│
  │  (ParallelAgent)        │             │  • trigger_folder_ingestion  │
  │    ├─ es_search_agent   │             │  • get_ingestion_status      │
  │    ├─ vertex_search     │             └─────────────────────────────┘
  │    │     _agent         │
  │    ├─ vais_search_agent │  ← Vertex AI Search / Gemini Enterprise
  │    └─ web_context_agent │  ← live Google Search (isolated built-in)
  │                         │
  │  Step 2:                │
  │  synthesis_agent        │  ← merges all 4 sources + charts + model fits
  │                         │
  │  Step 3:                │
  │  code_analysis_agent    │  ← dynamic Python execution (isolated built-in)
  └────────────────────────┘

ADK patterns used
─────────────────
  • Agent            — root orchestrator, ingestion manager
  • SequentialAgent  — search pipeline (gather → synthesise → code analysis)
  • ParallelAgent    — simultaneous ES + Vertex AI + VAIS + Web search

Built-in tool isolation rule
─────────────────────────────
  google_search and built_in_code_execution are Gemini BuiltInTools.
  The Gemini API rejects requests mixing BuiltInTools with FunctionDeclarations
  in the same agent.  Therefore web_context_agent and code_analysis_agent
  each have ONLY their respective built-in tool and NO custom function tools.

Run locally
───────────
  cd FRE_GCP_v1.0
  adk web
  # Open http://localhost:8000/dev-ui
"""

from google.adk.agents import Agent, ParallelAgent, SequentialAgent

from cognitive_search_agent.sub_agents.es_search_agent.agent       import es_search_agent
from cognitive_search_agent.sub_agents.vertex_search_agent.agent    import vertex_search_agent
from cognitive_search_agent.sub_agents.vais_search_agent.agent      import vais_search_agent
from cognitive_search_agent.sub_agents.synthesis_agent.agent        import synthesis_agent
from cognitive_search_agent.sub_agents.code_analysis_agent.agent    import code_analysis_agent
from cognitive_search_agent.sub_agents.ingestion_agent.agent        import ingestion_manager_agent
from tools.search_tools import get_search_status
from tools.attachment_tools import extract_office_document_text, save_attachment_for_indexing, load_attachment_to_session

import config

# ── Step 1: Run all 4 search backends simultaneously ─────────────────────────
parallel_search_gatherer = ParallelAgent(
    name="parallel_search_gatherer",
    description=(
        "Runs Elasticsearch BM25, Vertex AI Vector Search, and Vertex AI Search "
        "(Gemini Enterprise) simultaneously, producing three independent result sets."
    ),
    sub_agents=[es_search_agent, vertex_search_agent, vais_search_agent],
)

# ── Step 2: Synthesise from all result sets ───────────────────────────────────
# ── Step 3: Optional dynamic code analysis / scenario modelling ───────────────
document_qa_pipeline = SequentialAgent(
    name="document_qa_pipeline",
    description=(
        "Full document Q&A pipeline: runs 4 search backends in parallel, "
        "synthesises a cited answer, then optionally performs code-based "
        "scenario modelling and statistical analysis."
    ),
    sub_agents=[parallel_search_gatherer, synthesis_agent, code_analysis_agent],
)

# ── Root Master Agent ─────────────────────────────────────────────────────────
root_agent = Agent(
    name="cognitive_search_agent",
    model=config.GEMINI_ROUTER_MODEL,
    planner=config.NO_THINKING_PLANNER,
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
         • "Draw a chart of the force vs displacement data from the test report"
         • "Plot the failure modes across all documents as a pie chart"
         • "Show me the cycle counts from the pulsator test as a line graph"
         • "What if the load increases from 50kN to 75kN? Plot the prediction"
         • "Fit a model to the force vs cycles data and predict at 200,000 cycles"
         • "Run an alternative scenario where supply cost increases by 20%"

    2. DOCUMENT INDEXING  → delegate to: ingestion_manager_agent
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

    • "What can you do?", "what are your capabilities?", "hi what are you
      capable of?" and similar meta/greeting questions → Answer DIRECTLY from
      the YOUR CAPABILITIES section above. DO NOT delegate to any pipeline.
      The pipeline sub-agents should never run for meta-questions.

    • Questions about document CONTENT         → delegate to document_qa_pipeline
    • Requests to VISUALISE / CHART data       → delegate to document_qa_pipeline
      (the synthesis_agent inside it has the generate_chart tool)
    • Requests to FIT A MODEL / WHAT-IF        → delegate to document_qa_pipeline
      (synthesis_agent uses analyze_and_fit_data; code_analysis_agent runs Python)
    • Requests to INDEX / ADD documents        → delegate to ingestion_manager_agent
    • Questions about SYSTEM STATUS            → call get_search_status tool
    • Attached PDF/image + question            → Gemini reads it natively; ALSO call
                                                 load_attachment_to_session so follow-up
                                                 questions work without re-uploading
    • Attached DOCX/PPTX/XLSX + question      → call load_attachment_to_session
                                                 (extracts text AND stores it for follow-ups)
    • Attached file + "save / index this"      → call save_attachment_for_indexing

    ═══════════════════════════════════════════════════════════
    FILE ATTACHMENTS (SESSION Q&A)
    ═══════════════════════════════════════════════════════════
    When the user attaches a file to the chat:

    1. ALWAYS call load_attachment_to_session(file_base64, filename) first.
       This extracts the text and stores it in session memory so every
       follow-up question in this conversation has access to the content.

    2. For PDF / image files Gemini can also read the attachment natively
       in the same turn — answer the user’s question directly from what
       you see, and the tool stores it for follow-ups.

    3. For follow-up questions (no new file attached):
       The session state contains all previously loaded documents under
       `session_documents`. Reference that content to answer questions.
       Do NOT ask the user to re-upload the file.

    4. To permanently index a file for future searches:
       Call save_attachment_for_indexing instead of (or in addition to)
       load_attachment_to_session.

    Supported upload formats: PDF, DOCX, PPTX, XLSX, TXT, MD, CSV, PNG, JPG

    ═══════════════════════════════════════════════════════════
    CHART / VISUALISATION GUIDANCE
    ═══════════════════════════════════════════════════════════
    When the user asks to draw, plot, or visualise data:
    1. Delegate to document_qa_pipeline — it will search the documents, extract
       the relevant numbers, and call generate_chart automatically.
    2. The chart will appear as an inline image in the response.
    3. Supported chart types: line, bar, scatter, pie, histogram.

    ═══════════════════════════════════════════════════════════
    PAGE PREVIEW GUIDANCE
    ═══════════════════════════════════════════════════════════
    When the user asks to "preview", "show", "see" or "look at" a page, OR
    asks to "regenerate exactly" a chart from a document:
    1. Delegate to document_qa_pipeline.
    2. The synthesis_agent will call preview_document_page(gcs_uri, page_number)
       which renders the page as an inline image AND analyses it with Gemini Vision
       to extract exact data points from any charts or tables.
    3. After seeing the page, if chart reproduction was requested, the agent
       will call generate_chart using the precise numbers extracted by Vision.

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
    tools=[get_search_status, extract_office_document_text, save_attachment_for_indexing, load_attachment_to_session],
)
