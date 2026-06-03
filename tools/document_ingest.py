"""
Multi-format document parser using Docling.

Handles: PDF, DOCX, PPTX, XLSX, MD, TXT, HTML
Converts any supported format to clean markdown text for chunking.

Usage:
    from tools.document_ingest import ingest_document
    chunks = ingest_document("path/to/report.pdf")
    for chunk in chunks:
        save_to_kb(chunk["text"], chunk["source"])

Returns:
    List of {text, source, format, chunk_index} dicts.
    Returns [] on failure — never raises. All errors are logged.

Why Docling over pypdf/python-docx:
    Single unified API for all formats.
    Preserves table structure, heading hierarchy, and code blocks
    far better than raw text extraction.
    Output is clean markdown — same format as existing seed docs.
"""
from pathlib import Path
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")

_converter = None

SUPPORTED_FORMATS = {".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt", ".html"}

def _get_converter():
    global _converter
    if _converter is None:
        try:
            from docling.document_converter import DocumentConverter
            _converter = DocumentConverter()
        except ImportError:
            raise ImportError(
                "Docling is not installed. Run: uv add docling\n"
                "Note: Docling requires Python 3.11-3.13. "
                "If you are on Python 3.14, use .md/.txt files only."
            )
    return _converter


def _chunk(text: str, source: str, fmt: str, chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """Token based chunking — same as save_to_kb."""
    enc = _enc
    tokens = enc.encode(text)
    chunks: list[dict] = []
    start = 0
    i = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_text = enc.decode(tokens[start:end])
        chunks.append({
            "text": chunk_text,
            "source": source,
            "format": fmt,
            "chunk_index": i,
        })
        start += chunk_size - overlap
        i += 1
    return chunks


def ingest_document(path: str | Path) -> list[dict]:
    """
    Parse a document into text chunks ready for save_to_kb.

    Args:
        path: Path to document file

    Returns:
        List of {text, source, format, chunk_index} dicts.
        [] if file not found, format unsupported, or parse failed.
    """
    import tiktoken
    path = Path(path)

    if not path.exists():
        print(f"[document_ingest] File not found: {path}")
        return []

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"[document_ingest] Unsupported format: {path.suffix} ({path.name})")
        return []

    # .md and .txt: read directly, skip Docling overhead
    if path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return _chunk(text, path.stem, path.suffix.lower().lstrip("."))

    # All other formats convert via Docling
    try:
        converter = _get_converter()
        result = converter.convert(str(path))
        text = result.document.export_to_markdown()
        if not text.strip():
            print(f"[document_ingest] Empty output from Docling for: {path.name}")
            return []
        return _chunk(text, path.stem, path.suffix.lower().lstrip("."))
    except ImportError as e:
        print(f"[document_ingest] ERROR processing {path.name}: {e}")
        return []