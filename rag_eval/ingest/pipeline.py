"""Ingestion pipeline.

Loads documents → splits into chunks → ingests into the configured vector store.
Can be run as a module: `python -m rag_eval.ingest.pipeline`
"""

from __future__ import annotations

import logging
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_eval.config import get_settings
from rag_eval.ingest.loader import load_documents_from_dir
from rag_eval.store.factory import get_vector_store

logger = logging.getLogger(__name__)
console = Console()

# Chunk size tuned for MiniLM-L6-v2's 256-token context window.
# 512 chars ≈ 100–150 tokens (conservative to avoid truncation).
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


def run_ingestion(documents_dir: str | None = None) -> int:
    """Load, split, and ingest documents into the vector store.

    Args:
        documents_dir: Override for the documents directory path.
                       Defaults to settings.documents_dir.

    Returns:
        Number of chunks ingested.
    """
    settings = get_settings()
    docs_dir = documents_dir or settings.documents_dir

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Load raw documents
        task = progress.add_task("Loading documents...", total=None)
        raw_docs = load_documents_from_dir(docs_dir)
        progress.update(task, description=f"Loaded {len(raw_docs)} documents.")

        # Step 2: Split into chunks
        progress.update(task, description="Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)

        # Propagate doc_id metadata to each chunk so precision/recall scoring works
        # (each chunk inherits its parent document's doc_id)
        for chunk in chunks:
            if "doc_id" not in chunk.metadata:
                logger.warning("Chunk missing doc_id metadata — retrieval scoring may be inaccurate.")

        progress.update(task, description=f"Split into {len(chunks)} chunks. Connecting to vector store...")

        # Step 3: Ingest
        store = get_vector_store(settings)
        progress.update(task, description=f"Ingesting {len(chunks)} chunks into {settings.vector_store_backend}...")
        store.ingest(chunks)

    console.print(
        f"[bold green]✅ Ingested {len(chunks)} chunks from {len(raw_docs)} documents "
        f"into {settings.vector_store_backend}[/bold green]"
    )
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    try:
        count = run_ingestion()
        sys.exit(0)
    except Exception as exc:
        console.print(f"[bold red]Ingestion failed: {exc}[/bold red]")
        logger.exception("Ingestion error")
        sys.exit(1)
