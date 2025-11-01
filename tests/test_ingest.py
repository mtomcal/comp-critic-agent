"""Unit tests for the data ingestion pipeline (System 1)."""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from comp_critic.ingest import (
    chunk_documents,
    create_vector_store,
    load_transcripts,
    run_ingestion_pipeline,
)


class TestLoadTranscripts:
    """Tests for load_transcripts function."""

    def test_load_transcripts_success(
        self, mock_transcripts_dir: Path
    ) -> None:
        """Test loading transcripts from a valid directory."""
        # Arrange - mock_transcripts_dir fixture provides sample files

        # Act
        documents = load_transcripts(mock_transcripts_dir)

        # Assert
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(doc.page_content for doc in documents)

    def test_load_transcripts_nonexistent_directory(self, temp_dir: Path) -> None:
        """Test loading from a non-existent directory raises FileNotFoundError."""
        # Arrange
        nonexistent_dir = temp_dir / "does_not_exist"

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Transcripts directory not found"):
            load_transcripts(nonexistent_dir)

    def test_load_transcripts_empty_directory(self, temp_dir: Path) -> None:
        """Test loading from an empty directory returns empty list."""
        # Arrange
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        # Act
        documents = load_transcripts(empty_dir)

        # Assert
        assert len(documents) == 0


class TestChunkDocuments:
    """Tests for chunk_documents function."""

    def test_chunk_documents_basic(self, sample_documents: List[Document]) -> None:
        """Test chunking documents with default settings."""
        # Act
        chunks = chunk_documents(sample_documents)

        # Assert
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        # Should have at least as many chunks as documents (could be more if split)
        assert len(chunks) >= len(sample_documents)

    def test_chunk_documents_preserves_metadata(
        self, sample_documents: List[Document]
    ) -> None:
        """Test that chunking preserves source metadata."""
        # Act
        chunks = chunk_documents(sample_documents)

        # Assert
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"].endswith(".txt")

    def test_chunk_documents_empty_list(self) -> None:
        """Test chunking an empty list returns empty list."""
        # Arrange
        empty_docs: List[Document] = []

        # Act
        chunks = chunk_documents(empty_docs)

        # Assert
        assert len(chunks) == 0

    def test_chunk_documents_respects_size(self) -> None:
        """Test that chunks don't exceed the configured size."""
        # Arrange
        long_text = "A " * 2000  # Create text longer than chunk size
        long_doc = Document(
            page_content=long_text,
            metadata={"source": "long.txt"},
        )

        # Act
        chunks = chunk_documents([long_doc])

        # Assert
        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should be reasonably sized (allowing for overlap)
        for chunk in chunks:
            assert len(chunk.page_content) <= 1500  # Chunk size + some buffer


class TestCreateVectorStore:
    """Tests for create_vector_store function."""

    @patch("comp_critic.ingest.Chroma.from_documents")
    @patch("comp_critic.ingest.OpenAIEmbeddings")
    def test_create_vector_store_success(
        self,
        mock_embeddings: MagicMock,
        mock_chroma: MagicMock,
        sample_chunks: List[Document],
        mock_openai_api_key: str,
        mock_chroma_db_path: Path,
    ) -> None:
        """Test creating vector store with valid chunks."""
        # Arrange
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        # Act
        result = create_vector_store(sample_chunks)

        # Assert
        assert result == mock_vector_store
        mock_embeddings.assert_called_once()
        mock_chroma.assert_called_once()

        # Verify ChromaDB was called with correct parameters
        call_kwargs = mock_chroma.call_args.kwargs
        assert call_kwargs["documents"] == sample_chunks
        assert "embedding" in call_kwargs
        assert call_kwargs["collection_name"] == "landscape_photography_transcripts"

    @patch("comp_critic.ingest.Chroma.from_documents")
    @patch("comp_critic.ingest.OpenAIEmbeddings")
    def test_create_vector_store_creates_directory(
        self,
        mock_embeddings: MagicMock,
        mock_chroma: MagicMock,
        sample_chunks: List[Document],
        mock_openai_api_key: str,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that vector store creation creates the ChromaDB directory."""
        # Arrange
        from comp_critic import config as config_module

        new_chroma_path = temp_dir / "new_chroma"
        monkeypatch.setattr(config_module.Config, "CHROMA_DB_PATH", new_chroma_path)

        mock_chroma.return_value = MagicMock()

        # Act
        create_vector_store(sample_chunks)

        # Assert
        assert new_chroma_path.exists()
        assert new_chroma_path.is_dir()


class TestRunIngestionPipeline:
    """Tests for the complete ingestion pipeline."""

    @patch("comp_critic.ingest.create_vector_store")
    @patch("comp_critic.ingest.chunk_documents")
    @patch("comp_critic.ingest.load_transcripts")
    def test_run_ingestion_pipeline_success(
        self,
        mock_load: MagicMock,
        mock_chunk: MagicMock,
        mock_create: MagicMock,
        sample_documents: List[Document],
        sample_chunks: List[Document],
        mock_openai_api_key: str,
        mock_transcripts_dir: Path,
    ) -> None:
        """Test successful execution of the complete pipeline."""
        # Arrange
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        mock_create.return_value = MagicMock()

        # Act
        run_ingestion_pipeline()

        # Assert
        mock_load.assert_called_once()
        mock_chunk.assert_called_once_with(sample_documents)
        mock_create.assert_called_once_with(sample_chunks)

    @patch("comp_critic.ingest.load_transcripts")
    def test_run_ingestion_pipeline_no_documents(
        self,
        mock_load: MagicMock,
        mock_openai_api_key: str,
        mock_transcripts_dir: Path,
    ) -> None:
        """Test pipeline handles empty transcript directory gracefully."""
        # Arrange
        mock_load.return_value = []

        # Act - should not raise, just print warning
        run_ingestion_pipeline()

        # Assert
        mock_load.assert_called_once()

    def test_run_ingestion_pipeline_missing_api_key(
        self, mock_transcripts_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test pipeline fails gracefully without API key."""
        # Arrange
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Force config reload
        from comp_critic import config as config_module

        monkeypatch.setattr(config_module.Config, "OPENAI_API_KEY", None)

        # Act & Assert
        with pytest.raises(SystemExit):
            run_ingestion_pipeline()
