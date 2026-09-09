"""Ingest a PDF and index it into ChromaDB.

Usage: python -m scripts.build_index data/samples/sample.pdf
"""

import sys

from app.config import settings
from app.ingestion.loader import ingest_pdf
from app.retrieval.vectorstore import index_chunks


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.build_index <path-to-pdf>")
        sys.exit(1)

    chunks = ingest_pdf(sys.argv[1])
    count = index_chunks(chunks)
    print(f"✅ Indexed {count} chunks into ChromaDB ('{settings.chroma_dir}')")


if __name__ == "__main__":
    main()
