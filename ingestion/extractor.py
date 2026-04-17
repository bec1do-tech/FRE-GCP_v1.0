"""
FRE GCP v1.0 — Multimodal Document Extractor
==============================================
Extracts text content from all supported file types, including:
  • Plain text / Markdown / CSV         → direct decode
  • PDF                                  → pypdf (text) + Gemini Vision (images/charts)
  • DOCX                                 → python-docx (text + embedded images)
  • PPTX                                 → python-pptx (slide text + Gemini Vision per slide)
  • XLSX                                 → openpyxl (table data as structured text)

For binary-heavy files (PDF/PPTX/DOCX) each embedded image is sent to the
Gemini Vision model to produce a natural-language description, which is then
appended to the extracted text.  This means charts, diagrams, and infographics
become part of the searchable corpus.

All functions accept raw bytes so they work identically with GCS downloads
and local file reads.
"""

from __future__ import annotations

import io
import logging
import zipfile
import re
from pathlib import PurePosixPath
from typing import NamedTuple

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionResult(NamedTuple):
    text: str               # full extracted text (including image descriptions)
    metadata: dict          # file-level metadata (author, page_count, etc.)
    image_count: int        # how many images were processed via Vision


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Vision helper
# ─────────────────────────────────────────────────────────────────────────────

def _describe_image(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Send one image to Gemini Vision and return a textual description.
    Returns "" if Gemini is unavailable or the image is too small.
    """
    if len(image_bytes) < 1024:   # skip trivially small images (bullets, icons)
        return ""
    try:
        import google.generativeai as genai  # type: ignore[import-untyped]

        if config.GOOGLE_API_KEY:
            genai.configure(api_key=config.GOOGLE_API_KEY)

        model = genai.GenerativeModel(config.GEMINI_VISION_MODEL)
        response = model.generate_content(
            [
                "Describe the content of this image in detail. "
                "If it is a chart or graph, summarise the key data points and trends. "
                "If it is a table, transcribe the data. "
                "If it is a diagram, describe its structure and labels. "
                "Be concise but complete.",
                {"mime_type": mime_type, "data": image_bytes},
            ]
        )
        return response.text.strip()
    except Exception as exc:
        logger.debug("Gemini Vision unavailable: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Plain text / Markdown / CSV
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_bytes(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, AttributeError):
            continue
    return data.decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pdf(data: bytes) -> ExtractionResult:
    try:
        import pypdf  # type: ignore[import-untyped]
        import pypdf.errors  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pypdf not installed — PDF text extraction disabled.")
        return ExtractionResult("", {}, 0)

    text_parts: list[str] = []
    image_descriptions: list[str] = []
    metadata: dict = {}

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        metadata = {
            "page_count": len(reader.pages),
            "author":     (reader.metadata or {}).get("/Author", ""),
            "title":      (reader.metadata or {}).get("/Title", ""),
            "creator":    (reader.metadata or {}).get("/Creator", ""),
        }

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

            # Extract embedded images and describe with Gemini Vision
            for img_obj in page.images:
                try:
                    desc = _describe_image(img_obj.data, "image/png")
                    if desc:
                        image_descriptions.append(
                            f"[Image on page {page_num + 1}: {desc}]"
                        )
                except Exception:
                    pass

    except Exception as exc:
        logger.error("PDF extraction error: %s", exc)

    full_text = "\n\n".join(text_parts)
    if image_descriptions:
        full_text += "\n\n--- Visual Content ---\n" + "\n".join(image_descriptions)

    return ExtractionResult(full_text, metadata, len(image_descriptions))


# ─────────────────────────────────────────────────────────────────────────────
# DOCX
# ─────────────────────────────────────────────────────────────────────────────

def _extract_docx(data: bytes) -> ExtractionResult:
    try:
        from docx import Document  # type: ignore[import-untyped]
        from docx.oxml.ns import qn  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("python-docx not installed — DOCX extraction disabled.")
        return ExtractionResult("", {}, 0)

    text_parts: list[str] = []
    image_descriptions: list[str] = []
    metadata: dict = {}

    try:
        doc = Document(io.BytesIO(data))
        core = doc.core_properties
        metadata = {
            "author":   core.author or "",
            "title":    core.title or "",
            "subject":  core.subject or "",
            "created":  core.created.isoformat() if core.created else "",
        }

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # Extract inline images
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    img_bytes = rel.target_part.blob
                    desc = _describe_image(img_bytes)
                    if desc:
                        image_descriptions.append(f"[Document image: {desc}]")
                except Exception:
                    pass

    except Exception as exc:
        logger.error("DOCX extraction error: %s", exc)

    full_text = "\n".join(text_parts)
    if image_descriptions:
        full_text += "\n\n--- Visual Content ---\n" + "\n".join(image_descriptions)

    return ExtractionResult(full_text, metadata, len(image_descriptions))


# ─────────────────────────────────────────────────────────────────────────────
# PPTX
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pptx(data: bytes) -> ExtractionResult:
    try:
        from pptx import Presentation  # type: ignore[import-untyped]
        from pptx.util import Pt  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("python-pptx not installed — PPTX extraction disabled.")
        return ExtractionResult("", {}, 0)

    text_parts: list[str] = []
    image_descriptions: list[str] = []
    metadata: dict = {}

    try:
        prs = Presentation(io.BytesIO(data))
        metadata = {"slide_count": len(prs.slides)}

        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_texts: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = " ".join(run.text for run in para.runs if run.text.strip())
                        if line:
                            slide_texts.append(line)

                # Images on slide
                if shape.shape_type == 13:   # MSO_SHAPE_TYPE.PICTURE
                    try:
                        img_bytes = shape.image.blob
                        # Render the whole slide as PNG for richer context
                        desc = _describe_image(img_bytes)
                        if desc:
                            image_descriptions.append(
                                f"[Image on slide {slide_num}: {desc}]"
                            )
                    except Exception:
                        pass

            if slide_texts:
                text_parts.append(f"[Slide {slide_num}]\n" + "\n".join(slide_texts))

    except Exception as exc:
        logger.error("PPTX extraction error: %s", exc)

    full_text = "\n\n".join(text_parts)
    if image_descriptions:
        full_text += "\n\n--- Visual Content ---\n" + "\n".join(image_descriptions)

    return ExtractionResult(full_text, metadata, len(image_descriptions))


# ─────────────────────────────────────────────────────────────────────────────
# XLSX
# ─────────────────────────────────────────────────────────────────────────────

def _extract_xlsx(data: bytes) -> ExtractionResult:
    try:
        import openpyxl  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("openpyxl not installed — XLSX extraction disabled.")
        return ExtractionResult("", {}, 0)

    text_parts: list[str] = []
    metadata: dict = {}

    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        metadata = {
            "sheet_names": wb.sheetnames,
            "sheet_count": len(wb.sheetnames),
        }

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: list[str] = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    rows.append("\t".join(cells))
            if rows:
                text_parts.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))

        wb.close()
    except Exception as exc:
        logger.error("XLSX extraction error: %s", exc)

    return ExtractionResult("\n\n".join(text_parts), metadata, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTORS = {
    ".txt":  lambda d: ExtractionResult(_extract_text_bytes(d), {}, 0),
    ".md":   lambda d: ExtractionResult(_extract_text_bytes(d), {}, 0),
    ".csv":  lambda d: ExtractionResult(_extract_text_bytes(d), {}, 0),
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
    ".pptx": _extract_pptx,
    ".xlsx": _extract_xlsx,
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(_EXTRACTORS.keys())


def extract(data: bytes, filename: str) -> ExtractionResult:
    """
    Extract text (and visual descriptions) from raw file bytes.

    Parameters
    ----------
    data     : Raw file bytes (e.g. from GCS download).
    filename : Original filename — used only to determine the file type.

    Returns
    -------
    ExtractionResult(text, metadata, image_count)
    """
    ext = PurePosixPath(filename).suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        logger.warning("Unsupported file type: %s", filename)
        return ExtractionResult("", {}, 0)

    mb = len(data) / (1024 * 1024)
    if mb > config.MAX_FILE_MB:
        logger.warning("File %s (%.1f MB) exceeds MAX_FILE_MB=%d — skipping.", filename, mb, config.MAX_FILE_MB)
        return ExtractionResult("", {}, 0)

    return extractor(data)
