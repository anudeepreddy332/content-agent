"""
Seed the Qdrant knowledge base with evergreen AI/ML content.

Usage:
    uv run python scripts/ingest.py --source kb/seed_docs/
    uv run python scripts/ingest.py --source kb/seed_docs/gradient_descent.md
    uv run python scripts/ingest.py --stats

What goes in the KB:
    - Definitions, formulas, core ML theory (things that don't change)
    - NOT: recent model releases, benchmark numbers, paper findings (→ Tavily)

Seed doc format:
    Plain .md or .txt files.
    Filename becomes the source identifier.
    Put new docs in kb/seed_docs/ — that directory is committed (it's just markdown
    files); the actual vector store data (kb/qdrant_data/, legacy kb/chroma_db/) is
    gitignored, since it's all derived from seed_docs/ and regenerable via this script.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from tools.save_to_kb import save_to_kb

SEED_DOCS_DIR = Path("kb/seed_docs")
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".html"}


def ingest_file(filepath: Path) -> bool:
    """
    Ingest a single file into the KB.

    Routing:
        .md / .txt  → read directly, ingest via save_to_kb
        all others  → parse via document_ingest (Docling), ingest chunk by chunk

    Returns True if at least one chunk was ingested successfully.
    """
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"  [skip] Unsupported extension: {filepath.name}")
        return False

    print(f"  → Ingesting: {filepath.name}")

    # Native path: .md and .txt (no Docling dependency)
    if filepath.suffix.lower() in {".md", ".txt"}:
        text = filepath.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  [skip] Empty file: {filepath.name}")
            return False
        source = filepath.stem
        metadata = {"filename": filepath.name, "type": "seed"}
        return save_to_kb(text, source, metadata)

    # Multi-format path: PDF, DOCX, PPTX, XLSX, HTML (requires Docling)
    try:
        from tools.document_ingest import ingest_document
    except ImportError:
        print(f"  [skip] Docling not installed — cannot parse {filepath.suffix} files")
        print(f"         Install with: uv add docling")
        return False

    chunks = ingest_document(filepath)
    if not chunks:
        print(f"  [warn] No content extracted from: {filepath.name}")
        return False

    success_count = 0
    for chunk in chunks:
        metadata = {
            "filename": filepath.name,
            "type": "seed",
            "format": chunk.get("format", ""),
            "chunk_index": chunk.get("chunk_index", 0),
        }
        ok = save_to_kb(
            text=chunk["text"],
            source=chunk["source"],
            metadata=metadata,
        )
        if ok:
            success_count += 1

    print(f"  → {success_count}/{len(chunks)} chunks ingested from: {filepath.name}")
    return success_count > 0



def ingest_directory(dirpath: Path) -> tuple[int, int]:
    """
    Ingest all supported files in a directory.
    Returns (success_count, fail_count).
    """
    files = [f for f in dirpath.iterdir() if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS]
    if not files:
        print(f"No supported files found in {dirpath}")
        return 0, 0

    print(f"Found {len(files)} files to ingest in {dirpath}\n")
    success, fail = 0, 0

    for f in sorted(files):
        if ingest_file(f):
            success += 1
        else:
            fail += 1

    return success, fail



def show_stats():
    """Print KB stats — collection count and a sample query."""
    from qdrant_client import QdrantClient
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "machinist_evergreen")
    client = QdrantClient(url=qdrant_url)

    print(f"\nKB Stats")
    print(f"{'─' * 40}")
    print(f"Backend: Qdrant at {qdrant_url}")
    print(f"Collection: {collection_name}")

    try:
        info = client.get_collection(collection_name=collection_name)
        count = info.points_count
        print(f"Total chunks: {count}")
        if count > 0:
            # Sample query to confirm retrieval works
            from tools.query_kb import query_kb
            results = query_kb("gradient descent learning rate", n_results=2)
            print(f"\nSample query: 'gradient descent learning rate'")
            for r in results:
                preview = r['text'][:120].replace('\n', ' ')
                print(f"  [{r['distance']:.4f}] {r['source']}: {preview}...")

    except Exception:
        print(f"Total chunks: 0 (collection not found — run ingest first)")
        return



def main():
    parser = argparse.ArgumentParser(
        description="Ingest seed documents into the content-agent KB"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SEED_DOCS_DIR,
        help="Path to a file or directory to ingest (default: kb/seed_docs/)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show KB stats and exit"
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    source = args.source

    if not source.exists():
        print(f"ERROR: Path does not exist: {source}")
        print(f"Create the directory and add .md or .txt files to it first.")
        sys.exit(1)

    print(f"\nContent-Agent KB Ingest")
    print(f"{'─' * 40}")

    if source.is_file():
        success = ingest_file(source)
        print(f"\n{'✓ Done' if success else '✗ Failed'}: {source.name}")

    else:
        success_count, fail_count = ingest_directory(source)
        print(f"\n{'─' * 40}")
        print(f"✓ Ingested: {success_count} files")
        if fail_count:
            print(f"✗ Failed:   {fail_count} files")

    show_stats()


if __name__ == "__main__":
    main()
