"""Document loading utilities.

Reads JSON knowledge base files from data/documents/ and produces
LangChain Document objects with normalized metadata including doc_id.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_documents_from_dir(documents_dir: str | Path) -> list[Document]:
    """Load all JSON documents from a directory.

    Each JSON file should contain a single document object with at least:
    - "id": unique document identifier (matched against golden_dataset.json)
    - "title": human-readable title
    - "content": the text content to embed

    Optional fields (stored in metadata):
    - "category": topic tag
    - "tags": list of keywords

    Args:
        documents_dir: Path to directory containing .json doc files.

    Returns:
        List of LangChain Document objects ready for text-splitting and ingestion.

    Raises:
        FileNotFoundError: If documents_dir does not exist.
        ValueError: If any JSON file is missing required fields.
    """
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_path}")

    json_files = sorted(docs_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No .json files found in {docs_path}")

    logger.info("Found %d document files in '%s'.", len(json_files), docs_path)

    documents: list[Document] = []
    for file_path in json_files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required fields
        for required in ("id", "title", "content"):
            if required not in data:
                raise ValueError(
                    f"Document file {file_path.name} missing required field: '{required}'"
                )

        doc = Document(
            page_content=data["content"],
            metadata={
                "doc_id": data["id"],
                "title": data["title"],
                "category": data.get("category", "general"),
                "tags": data.get("tags", []),
                "source": str(file_path),
            },
        )
        documents.append(doc)
        logger.debug("Loaded document '%s' (%d chars).", data["id"], len(data["content"]))

    logger.info("Loaded %d documents total.", len(documents))
    return documents
