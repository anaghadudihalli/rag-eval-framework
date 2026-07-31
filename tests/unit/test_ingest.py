"""Unit tests for the ingest module.

Covers:
- load_documents_from_dir(): happy path, missing fields, missing dir
- run_ingestion(): chunk count, doc propagation, vector store called
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document

from rag_eval.ingest.loader import load_documents_from_dir
from rag_eval.ingest.pipeline import run_ingestion


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_doc(directory: Path, filename: str, data: dict) -> Path:
    """Write a JSON document file into directory and return its path."""
    path = directory / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_doc(doc_id: str = "doc-001", content_len: int = 100) -> dict:
    return {
        "id": doc_id,
        "title": f"Document {doc_id}",
        "content": "A" * content_len,
    }


# ── load_documents_from_dir ──────────────────────────────────────────────────


class TestLoadDocumentsFromDir:
    def test_loads_single_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc1.json", _minimal_doc("doc-001"))
            docs = load_documents_from_dir(tmpdir)
        assert len(docs) == 1

    def test_loads_multiple_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                _write_doc(Path(tmpdir), f"doc{i}.json", _minimal_doc(f"doc-{i:03d}"))
            docs = load_documents_from_dir(tmpdir)
        assert len(docs) == 5

    def test_page_content_equals_document_content(self) -> None:
        content = "This is the expected page content for our test document."
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", {
                "id": "test-001", "title": "Test Doc", "content": content
            })
            docs = load_documents_from_dir(tmpdir)
        assert docs[0].page_content == content

    def test_doc_id_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc("my-doc-id"))
            docs = load_documents_from_dir(tmpdir)
        assert docs[0].metadata["doc_id"] == "my-doc-id"

    def test_title_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", {
                "id": "d-001", "title": "My Title", "content": "content here"
            })
            docs = load_documents_from_dir(tmpdir)
        assert docs[0].metadata["title"] == "My Title"

    def test_category_defaults_to_general(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc())
            docs = load_documents_from_dir(tmpdir)
        assert docs[0].metadata["category"] == "general"

    def test_category_read_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {**_minimal_doc(), "category": "docker"}
            _write_doc(Path(tmpdir), "doc.json", data)
            docs = load_documents_from_dir(tmpdir)
        assert docs[0].metadata["category"] == "docker"

    def test_tags_default_to_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc())
            docs = load_documents_from_dir(tmpdir)
        assert docs[0].metadata["tags"] == []

    def test_tags_read_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {**_minimal_doc(), "tags": ["git", "version-control"]}
            _write_doc(Path(tmpdir), "doc.json", data)
            docs = load_documents_from_dir(tmpdir)
        assert docs[0].metadata["tags"] == ["git", "version-control"]

    def test_source_path_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_doc(Path(tmpdir), "doc.json", _minimal_doc())
            docs = load_documents_from_dir(tmpdir)
        assert str(path) == docs[0].metadata["source"]

    def test_docs_sorted_alphabetically(self) -> None:
        """Files are loaded in sorted order — deterministic across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "z_doc.json", _minimal_doc("z-doc"))
            _write_doc(Path(tmpdir), "a_doc.json", _minimal_doc("a-doc"))
            docs = load_documents_from_dir(tmpdir)
        assert docs[0].metadata["doc_id"] == "a-doc"
        assert docs[1].metadata["doc_id"] == "z-doc"

    def test_raises_file_not_found_for_missing_dir(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_documents_from_dir("/nonexistent/path/that/does/not/exist")

    def test_raises_value_error_when_no_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a non-JSON file — should not be picked up
            (Path(tmpdir) / "readme.txt").write_text("hello")
            with pytest.raises(ValueError, match="No .json files found"):
                load_documents_from_dir(tmpdir)

    def test_raises_value_error_when_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "bad.json", {"title": "T", "content": "C"})
            with pytest.raises(ValueError, match="missing required field.*id"):
                load_documents_from_dir(tmpdir)

    def test_raises_value_error_when_title_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "bad.json", {"id": "d-001", "content": "C"})
            with pytest.raises(ValueError, match="missing required field.*title"):
                load_documents_from_dir(tmpdir)

    def test_raises_value_error_when_content_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "bad.json", {"id": "d-001", "title": "T"})
            with pytest.raises(ValueError, match="missing required field.*content"):
                load_documents_from_dir(tmpdir)

    def test_returns_langchain_document_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc())
            docs = load_documents_from_dir(tmpdir)
        assert all(isinstance(d, Document) for d in docs)

    def test_ignores_non_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc("d-001"))
            (Path(tmpdir) / "notes.txt").write_text("ignore me")
            (Path(tmpdir) / "data.csv").write_text("a,b,c")
            docs = load_documents_from_dir(tmpdir)
        assert len(docs) == 1

    def test_accepts_path_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc())
            # Pass a Path object, not a string
            docs = load_documents_from_dir(Path(tmpdir))
        assert len(docs) == 1


# ── run_ingestion ────────────────────────────────────────────────────────────


class TestRunIngestion:
    def _make_mock_store(self) -> MagicMock:
        mock = MagicMock()
        mock.ingest.return_value = None
        return mock

    def test_returns_chunk_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a doc with enough content to produce at least one chunk
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc(content_len=600))
            mock_store = self._make_mock_store()

            with patch("rag_eval.ingest.pipeline.get_vector_store", return_value=mock_store), \
                 patch("rag_eval.ingest.pipeline.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    documents_dir=tmpdir,
                    vector_store_backend="chroma",
                )
                count = run_ingestion(documents_dir=tmpdir)

        assert count > 0
        assert isinstance(count, int)

    def test_store_ingest_called_with_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc(content_len=600))
            mock_store = self._make_mock_store()

            with patch("rag_eval.ingest.pipeline.get_vector_store", return_value=mock_store), \
                 patch("rag_eval.ingest.pipeline.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    documents_dir=tmpdir,
                    vector_store_backend="chroma",
                )
                run_ingestion(documents_dir=tmpdir)

        mock_store.ingest.assert_called_once()
        chunks_passed = mock_store.ingest.call_args[0][0]
        assert len(chunks_passed) > 0
        assert all(isinstance(c, Document) for c in chunks_passed)

    def test_chunks_inherit_doc_id_from_parent(self) -> None:
        """All chunks produced from a doc must carry the parent's doc_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", {
                "id": "parent-doc-001",
                "title": "Parent",
                "content": "X" * 1200,  # long enough to produce multiple chunks
            })
            mock_store = self._make_mock_store()

            with patch("rag_eval.ingest.pipeline.get_vector_store", return_value=mock_store), \
                 patch("rag_eval.ingest.pipeline.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    documents_dir=tmpdir,
                    vector_store_backend="chroma",
                )
                run_ingestion(documents_dir=tmpdir)

        chunks = mock_store.ingest.call_args[0][0]
        for chunk in chunks:
            assert chunk.metadata.get("doc_id") == "parent-doc-001"

    def test_multiple_docs_all_ingested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                _write_doc(Path(tmpdir), f"doc{i}.json", _minimal_doc(f"doc-{i}", content_len=600))
            mock_store = self._make_mock_store()

            with patch("rag_eval.ingest.pipeline.get_vector_store", return_value=mock_store), \
                 patch("rag_eval.ingest.pipeline.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    documents_dir=tmpdir,
                    vector_store_backend="chroma",
                )
                count = run_ingestion(documents_dir=tmpdir)

        # 3 docs × at least 1 chunk each
        assert count >= 3

    def test_uses_documents_dir_from_settings_when_not_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_doc(Path(tmpdir), "doc.json", _minimal_doc(content_len=600))
            mock_store = self._make_mock_store()

            with patch("rag_eval.ingest.pipeline.get_vector_store", return_value=mock_store), \
                 patch("rag_eval.ingest.pipeline.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    documents_dir=tmpdir,
                    vector_store_backend="chroma",
                )
                # No documents_dir override — should use settings.documents_dir
                count = run_ingestion()

        assert count > 0
