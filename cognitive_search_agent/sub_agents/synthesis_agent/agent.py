"""
Answer Synthesis Sub-Agent
===========================
The final step of the SequentialAgent search pipeline.  It receives the
outputs of both the ES and Vertex AI search agents (available in the
conversation context) and synthesises a comprehensive, cited answer.

This agent is responsible for:
  • Merging and deduplicating results from both search backends
  • Generating a coherent, direct answer to the user's original question
  • Providing explicit citations with GCS URIs so users can verify information
  • Acknowledging when information is insufficient or contradictory
"""

from google.adk.agents import Agent

from tools.search_tools import get_document_chunks

import config

synthesis_agent = Agent(
    name="synthesis_agent",
    model=config.GEMINI_MODEL,
    description=(
        "Answer synthesis agent. Reads search results from both Elasticsearch "
        "and Vertex AI, then generates a comprehensive, cited answer."
    ),
    instruction="""
    You are the Answer Synthesis specialist for the Cognitive Search system.

    The previous agents have already performed:
      1. An Elasticsearch BM25 keyword search (results labelled "Elasticsearch BM25 Results")
      2. A Vertex AI Semantic search (results labelled "Vertex AI Semantic Results")

    Your task is to synthesise a FINAL, high-quality answer for the user, using ALL
    the search results provided in the conversation context above.

    Synthesis Rules:
    ─────────────────
    1. ANSWER FIRST: Open with a direct, concise answer to the user's question.

    2. SUPPORT WITH EVIDENCE: Follow the answer with supporting evidence drawn
       from the search results.  Quote or paraphrase relevant excerpts.

    3. CITE EVERY CLAIM: After every piece of information, include a citation:
       [Source: filename.pdf — gs://bucket/path/filename.pdf]
       If multiple documents support the same point, list all of them.

    4. MERGE SMARTLY: If the same document appears in both ES and semantic results,
       treat it as a single, highly relevant source (do not duplicate the citation).

    5. STRUCTURED OUTPUT: Format your answer with clear sections:
       - **Answer**: direct response
       - **Key Findings**: bullet points with citations
       - **Sources Consulted**: a numbered list of all unique GCS URIs referenced

    6. HANDLE GAPS HONESTLY: If the search results do not contain enough information
       to answer the question confidently, say so explicitly:
       "The indexed documents do not contain sufficient information about [topic].
        You may want to check [specific suggestion] or index additional documents."

    7. MULTIMODAL CONTENT: If any excerpt mentions "[Image on page X:]" or
       "[Visual Content]", reference that visual analysis in your answer.

    8. DO NOT HALLUCINATE: Only state information that is present in the search
       results.  If you are unsure, say you are unsure.

    Begin your synthesis now based on the search results in the conversation.
    """,
    tools=[get_document_chunks],
)
