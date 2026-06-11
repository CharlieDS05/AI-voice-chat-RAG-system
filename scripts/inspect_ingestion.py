"""Ingest a PDF and inspect the resulting chunks + metadata.

Usage (from project root):
    python -m scripts.inspect_ingestion data/samples/sample.pdf
"""

import sys

from app.ingestion.loader import ingest_pdf


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.inspect_ingestion <path-to-pdf>")
        sys.exit(1)

    chunks = ingest_pdf(sys.argv[1])

    pages = {chunk.metadata["page"] for chunk in chunks}
    sizes = [len(chunk.page_content) for chunk in chunks]

    print(f"✅ Ingestion complete: {len(chunks)} chunks from {len(pages)} pages")
    print(f"   Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes) // len(sizes)}\n")

    for chunk in chunks[:3]:
        print(f"--- {chunk.metadata['chunk_id']} ---")
        print(f"metadata: {chunk.metadata}")
        print(f"text    : {chunk.page_content[:200]!r}...\n")

    # The overlap seam, made visible: the end of chunk 1 should reappear
    # at the start of chunk 2 (when both come from the same page).
    if len(chunks) >= 2:
        print("Tail of chunk 1:", repr(chunks[1].page_content[-80:]))
        print("Head of chunk 2:", repr(chunks[2].page_content[:80]))


if __name__ == "__main__":
    main()
