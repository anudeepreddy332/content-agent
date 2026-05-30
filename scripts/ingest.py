"""
Seed the local ChromaDB knowledge base with evergreen AI/ML content.

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
    Put them in kb/seed_docs/ — that directory is gitignored for chroma_db
    but seed_docs/ itself should be committed (it's just markdown files).
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from tools.save_to_kb import save_to_kb
from tools.query_kb import _get_collection

SEED_DOCS_DIR = Path("kb/seed_docs")
SUPPORTED_EXTENSIONS = {".md", ".txt"}


def ingest_file(filepath: Path) -> bool:
    """Ingest a single file into the KB."""
    if filepath.suffix not in SUPPORTED_EXTENSIONS:
        print(f"  [skip] Unsupported extension: {filepath.name}")
        return False

    text = filepath.read_text(encoding="utf-8").strip()
    if not text:
        print(f"  [skip] Empty file: {filepath.name}")
        return False

    source = filepath.stem  #filename w/o extension
    metadata = {
        "filename": filepath.name,
        "type": "seed",
    }
    print(f"  → Ingesting: {filepath.name}")
    success = save_to_kb(text=text, source=source, metadata=metadata)
    return success

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
    collection = _get_collection()
    count = collection.count()
    print(f"\nKB Stats")
    print(f"{'─' * 40}")
    print(f"Collection: {os.getenv('CHROMA_COLLECTION', 'machinist_evergreen')}")
    print(f"Total chunks: {count}")
    print(f"DB path: {os.getenv('CHROMA_DB_PATH', './kb/chroma_db')}")

    if count > 0:
        # Sample query to confirm retrieval works
        from tools.query_kb import query_kb
        results = query_kb("gradient descent learning rate", n_results=2)
        print(f"\nSample query: 'gradient descent learning rate'")
        for r in results:
            preview = r['text'][:120].replace('\n', ' ')
            print(f"  [{r['distance']:.4f}] {r['source']}: {preview}...")




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
