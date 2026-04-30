"""
FRE GCP v1.0 — PostgreSQL (Cloud SQL) data layer
==================================================
Schema
------
  documents            — one row per indexed GCS object (deduplication anchor)
  chunks               — text chunks derived from each document
  conversations        — user conversation sessions
  conversation_messages— individual turns within a session

Connection
----------
  Local dev  : standard psycopg2 DSN from config.POSTGRES_DSN
  Cloud Run  : Cloud SQL Python Connector with pg8000 driver
               (set CLOUD_SQL_INSTANCE in environment to activate)

All public functions create the schema on first call (init_db()).
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import psycopg2
import psycopg2.extras

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    gcs_uri         TEXT    NOT NULL UNIQUE,
    filename        TEXT    NOT NULL,
    content_md5     TEXT    NOT NULL,
    file_size       BIGINT  NOT NULL DEFAULT 0,
    file_type       TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'pending',
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    indexed_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_md5    ON documents (content_md5);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);

CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    doc_id          INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT    NOT NULL,
    vertex_vector_id TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,             -- UUID session ID
    user_id     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON conversation_messages (session_id, created_at);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────

def _get_connection() -> psycopg2.extensions.connection:
    """
    Return a psycopg2 connection.
    Uses Cloud SQL Connector when CLOUD_SQL_INSTANCE is set, otherwise
    falls back to a plain DSN (local dev / docker compose).
    """
    if config.CLOUD_SQL_INSTANCE:
        try:
            from google.cloud.sql.connector import Connector  # type: ignore[import-untyped]
            connector = Connector()
            conn = connector.connect(
                config.CLOUD_SQL_INSTANCE,
                "pg8000",
                user=config.POSTGRES_USER,
                password=config.POSTGRES_PASSWORD,
                db=config.POSTGRES_DB,
            )
            return conn  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("Cloud SQL Connector failed (%s), falling back to DSN.", exc)

    return psycopg2.connect(config.POSTGRES_DSN, connect_timeout=1)


@contextmanager
def _conn() -> Iterator[psycopg2.extensions.connection]:
    """Context manager: yield an auto-committing connection, close on exit."""
    con = _get_connection()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables (idempotent — safe to call on every startup)."""
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(_SCHEMA)
    logger.info("Database schema initialised.")


# ─────────────────────────────────────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────────────────────────────────────

def is_duplicate(content_md5: str) -> bool:
    """
    Return True if a document with this MD5 hash has already been indexed.
    Used for deduplication before initiating the (expensive) ingestion pipeline.
    """
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM documents WHERE content_md5 = %s AND status = 'indexed' LIMIT 1",
                (content_md5,),
            )
            return cur.fetchone() is not None


def upsert_document(
    gcs_uri: str,
    filename: str,
    content_md5: str,
    file_size: int = 0,
    file_type: str = "",
    status: str = "pending",
) -> int:
    """
    Insert or update a document record.  Returns the document's integer ID.
    Status values: 'pending' → 'processing' → 'indexed' | 'failed'
    """
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (gcs_uri, filename, content_md5, file_size, file_type, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (gcs_uri) DO UPDATE SET
                    content_md5 = EXCLUDED.content_md5,
                    file_size   = EXCLUDED.file_size,
                    file_type   = EXCLUDED.file_type,
                    status      = EXCLUDED.status
                RETURNING id
                """,
                (gcs_uri, filename, content_md5, file_size, file_type, status),
            )
            row = cur.fetchone()
            return row[0]  # type: ignore[index]


def mark_document_indexed(doc_id: int, chunk_count: int) -> None:
    """Update status to 'indexed' and record the number of chunks produced."""
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET status = 'indexed', chunk_count = %s, indexed_at = NOW()
                WHERE id = %s
                """,
                (chunk_count, doc_id),
            )


def mark_document_failed(doc_id: int) -> None:
    """Update status to 'failed' after a pipeline error."""
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed' WHERE id = %s", (doc_id,)
            )


def get_document_stats() -> dict:
    """Return a summary of document counts by status."""
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM documents
                GROUP BY status
                ORDER BY status
                """
            )
            rows = cur.fetchall()
    return {row["status"]: row["count"] for row in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Chunks
# ─────────────────────────────────────────────────────────────────────────────

def upsert_chunks(doc_id: int, chunks: list[dict]) -> None:
    """
    Bulk-insert chunks for a document.  Each chunk dict must have:
      chunk_index (int), chunk_text (str), vertex_vector_id (str, optional)
    """
    if not chunks:
        return
    with _conn() as con:
        with con.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks (doc_id, chunk_index, chunk_text, vertex_vector_id)
                VALUES %s
                ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                    chunk_text       = EXCLUDED.chunk_text,
                    vertex_vector_id = EXCLUDED.vertex_vector_id
                """,
                [
                    (doc_id, c["chunk_index"], c["chunk_text"], c.get("vertex_vector_id"))
                    for c in chunks
                ],
            )


def get_chunk_by_vector_id(vector_id: str) -> dict | None:
    """Retrieve a chunk record by its Vertex AI vector ID (for citation lookup)."""
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.id, c.doc_id, c.chunk_index, c.chunk_text,
                       d.gcs_uri, d.filename, d.file_type
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE c.vertex_vector_id = %s
                LIMIT 1
                """,
                (vector_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Conversation history
# ─────────────────────────────────────────────────────────────────────────────

def ensure_conversation(session_id: str, user_id: str = "") -> None:
    """Create a conversation record if it doesn't already exist."""
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (id, user_id)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET updated_at = NOW()
                """,
                (session_id, user_id),
            )


def save_message(session_id: str, role: str, content: str) -> None:
    """Append one message turn to the conversation history."""
    ensure_conversation(session_id)
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )


def get_conversation_history(session_id: str, limit: int = 20) -> list[dict]:
    """
    Return the last *limit* messages for a session, oldest first.
    Each dict has keys: role, content, created_at.
    """
    with _conn() as con:
        with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content, created_at
                FROM conversation_messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
    return [dict(r) for r in reversed(rows)]
