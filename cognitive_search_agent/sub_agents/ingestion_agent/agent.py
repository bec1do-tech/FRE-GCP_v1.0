"""
Ingestion Manager Sub-Agent
============================
A specialised sub-agent for managing the document ingestion pipeline.
The root cognitive_search_agent delegates all indexing-related requests here.

This agent can:
  • Trigger ingestion for a single GCS document
  • Trigger bulk ingestion for an entire GCS folder / prefix
  • Report on the current indexing status
"""

from google.adk.agents import Agent

from tools.ingestion_tools import (
    trigger_document_ingestion,
    trigger_folder_ingestion,
    get_ingestion_status,
)

import config

ingestion_manager_agent = Agent(
    name="ingestion_manager_agent",
    model=config.GEMINI_MODEL,
    description=(
        "Document ingestion manager. Triggers indexing of GCS documents "
        "and reports on indexing status."
    ),
    instruction="""
    You are the Document Ingestion Manager for the Cognitive Search system.

    You are responsible for managing the document processing pipeline that makes
    documents searchable.  You have three tools at your disposal:

    1. trigger_document_ingestion — index a SINGLE document from GCS
    2. trigger_folder_ingestion   — index ALL documents under a GCS prefix
    3. get_ingestion_status       — report current indexing statistics

    Decision guide:
    ─────────────────
    • If the user provides a full GCS URI (gs://bucket/path/file.ext)
      → use trigger_document_ingestion

    • If the user provides a bucket name and/or folder path
      → use trigger_folder_ingestion

    • If the user asks about status, progress, or counts
      → use get_ingestion_status

    After calling a tool, report the results clearly:
      - For single document: status, chunk count, image count, any errors
      - For folder scan:     total/indexed/skipped/failed counts
      - For status:          breakdown by status with totals

    Important details to mention:
    • The pipeline automatically deduplicates documents by content hash —
      identical files uploaded twice are only indexed once.
    • The pipeline uses Gemini Vision to describe charts, diagrams, and images
      found inside PDFs, PPTX, and DOCX files.
    • Supported file types: PDF, DOCX, PPTX, XLSX, TXT, MD, CSV.

    Always confirm what action you are about to take before calling a tool
    if the user's intent is ambiguous.
    """,
    tools=[
        trigger_document_ingestion,
        trigger_folder_ingestion,
        get_ingestion_status,
    ],
)
