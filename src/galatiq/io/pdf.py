from __future__ import annotations

from pathlib import Path


def read_pdf(path: Path) -> str:
    """Extract text from a PDF. pdfplumber primary, PyMuPDF fallback for thin output."""
    text = _try_pdfplumber(path)
    if len(text.strip()) >= 50:
        return text
    return _try_pymupdf(path) or text


def _try_pdfplumber(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""
    chunks: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _try_pymupdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ""
    chunks: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            chunks.append(page.get_text())
    return "\n".join(chunks)
