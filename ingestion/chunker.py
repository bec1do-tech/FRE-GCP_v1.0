"""
FRE GCP v1.0 — Text Chunker
============================
Splits long extracted text into overlapping chunks suitable for embedding
and indexing.  The strategy:

  1. Split on paragraph / double-newline boundaries first (semantic split).
  2. If a paragraph exceeds CHUNK_SIZE words, split it on sentence boundaries.
  3. Merge consecutive small paragraphs to stay close to CHUNK_SIZE.
  4. Apply a sliding overlap of CHUNK_OVERLAP words between consecutive chunks.

This produces chunks that are:
  • Small enough for the embedding model (≤ 512 tokens ≈ 400–500 words)
  • Semantically coherent (kept at paragraph / sentence boundaries)
  • Overlapping so cross-boundary answers are not missed

All sizes are in approximate words (not tokens) for simplicity.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import config


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    vector_id:   str         # UUID — used as the Vertex AI datapoint ID
    chunk_index: int         # 0-based position within the document
    text:        str         # the actual chunk text
    word_count:  int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Internal splitting helpers
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MULTI_BLANK  = re.compile(r"\n{2,}")


def _split_by_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _words(text: str) -> list[str]:
    return text.split()


def _merge_short_paragraphs(
    paragraphs: list[str], target: int
) -> list[str]:
    """
    Greedily merge consecutive paragraphs until the merged block reaches ~target words.
    This avoids creating many tiny chunks from short paragraphs.
    """
    merged: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(_words(para))
        if current_words + para_words > target and current_parts:
            merged.append(" ".join(current_parts))
            current_parts = [para]
            current_words = para_words
        else:
            current_parts.append(para)
            current_words += para_words

    if current_parts:
        merged.append(" ".join(current_parts))

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size:    int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split *text* into overlapping Chunk objects.

    Parameters
    ----------
    text          : The full extracted document text.
    chunk_size    : Target chunk size in words (default: config.CHUNK_SIZE).
    chunk_overlap : Number of words to repeat at the start of the next chunk
                    (default: config.CHUNK_OVERLAP).

    Returns
    -------
    List of Chunk objects in document order.
    """
    size    = chunk_size    or config.CHUNK_SIZE
    overlap = chunk_overlap or config.CHUNK_OVERLAP

    if not text.strip():
        return []

    # ── 1. Split on blank lines (paragraph boundaries) ───────────────────────
    paragraphs = [p.strip() for p in _MULTI_BLANK.split(text) if p.strip()]

    # ── 2. Break oversized paragraphs on sentence boundaries ─────────────────
    fine_paragraphs: list[str] = []
    for para in paragraphs:
        if len(_words(para)) <= size:
            fine_paragraphs.append(para)
        else:
            sentences = _split_by_sentences(para)
            buffer:  list[str] = []
            buf_words = 0
            for sent in sentences:
                sw = len(_words(sent))
                if buf_words + sw > size and buffer:
                    fine_paragraphs.append(" ".join(buffer))
                    # Keep the overlap tail for context
                    tail_words = _words(" ".join(buffer))[-overlap:] if overlap else []
                    buffer     = ([" ".join(tail_words)] if tail_words else []) + [sent]
                    buf_words  = len(tail_words) + sw
                else:
                    buffer.append(sent)
                    buf_words += sw
            if buffer:
                fine_paragraphs.append(" ".join(buffer))

    # ── 3. Merge short paragraphs up to ~chunk_size ───────────────────────────
    blocks = _merge_short_paragraphs(fine_paragraphs, target=size)

    # ── 4. Build Chunk objects with sliding overlap ────────────────────────────
    chunks: list[Chunk] = []
    overlap_tail: list[str] = []  # carry-over words from previous chunk

    for idx, block in enumerate(blocks):
        block_words = _words(block)
        if overlap_tail:
            combined = overlap_tail + block_words
        else:
            combined = block_words

        chunk_text_str = " ".join(combined).strip()
        if not chunk_text_str:
            continue

        chunks.append(
            Chunk(
                vector_id   = str(uuid.uuid4()),
                chunk_index = idx,
                text        = chunk_text_str,
            )
        )
        # Slide the overlap forward
        overlap_tail = _words(chunk_text_str)[-overlap:] if overlap else []

    return chunks
