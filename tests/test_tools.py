"""Unit tests for the RAG tool."""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from comp_critic.tools import (
    composition_rag_tool,
    format_search_results,
    load_vector_store,
)


class TestLoadVectorStore:
    """Tests for load_vector_store function."""

    @patch("comp_critic.tools.Chroma")
    @patch("comp_critic.tools.OpenAIEmbeddings")
    def test_load_vector_store_success(
        self,
        mock_embeddings: MagicMock,
        mock_chroma: MagicMock,
        mock_openai_api_key: str,
        mock_chroma_db_path: Path,
    ) -> None:
        """Test loading an existing vector store."""
        # Arrange
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        # Act
        result = load_vector_store()

        # Assert
        assert result == mock_vector_store
        mock_embeddings.assert_called_once()
        mock_chroma.assert_called_once()

    def test_load_vector_store_not_found(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch, mock_openai_api_key: str
    ) -> None:
        """Test loading a non-existent vector store raises ValueError."""
        # Arrange
        from comp_critic import config as config_module

        nonexistent_path = temp_dir / "nonexistent_db"
        monkeypatch.setattr(config_module.Config, "CHROMA_DB_PATH", nonexistent_path)

        # Act & Assert
        with pytest.raises(ValueError, match="ChromaDB not found"):
            load_vector_store()


class TestFormatSearchResults:
    """Tests for format_search_results function."""

    def test_format_search_results_with_documents(
        self, sample_documents: List[Document]
    ) -> None:
        """Test formatting multiple search results."""
        # Act
        result = format_search_results(sample_documents)

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain content from all documents
        for doc in sample_documents:
            assert doc.page_content in result
        # Should have result headers
        assert "Result 1" in result
        assert "Source:" in result

    def test_format_search_results_empty_list(self) -> None:
        """Test formatting with no results."""
        # Arrange
        empty_docs: List[Document] = []

        # Act
        result = format_search_results(empty_docs)

        # Assert
        assert result == "No relevant advice found in the knowledge base."

    def test_format_search_results_includes_metadata(self) -> None:
        """Test that formatted results include source metadata."""
        # Arrange
        docs = [
            Document(
                page_content="Test content",
                metadata={"source": "/path/to/transcript.txt"},
            )
        ]

        # Act
        result = format_search_results(docs)

        # Assert
        assert "/path/to/transcript.txt" in result
        assert "Test content" in result

    def test_format_search_results_handles_missing_source(self) -> None:
        """Test formatting when source metadata is missing."""
        # Arrange
        docs = [Document(page_content="Test content", metadata={})]

        # Act
        result = format_search_results(docs)

        # Assert
        assert "Unknown" in result
        assert "Test content" in result


class TestCompositionRAGTool:
    """Tests for the composition_rag_tool."""

    @patch("comp_critic.tools.load_vector_store")
    def test_composition_rag_tool_success(
        self, mock_load: MagicMock, sample_documents: List[Document]
    ) -> None:
        """Test successful RAG tool execution."""
        # Arrange
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = sample_documents[:2]
        mock_load.return_value = mock_vector_store

        # Act
        result = composition_rag_tool.invoke({"query": "rule of thirds"})

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        mock_load.assert_called_once()
        mock_vector_store.similarity_search.assert_called_once_with(
            "rule of thirds", k=3
        )

    @patch("comp_critic.tools.load_vector_store")
    def test_composition_rag_tool_no_results(self, mock_load: MagicMock) -> None:
        """Test RAG tool when no relevant documents are found."""
        # Arrange
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = []
        mock_load.return_value = mock_vector_store

        # Act
        result = composition_rag_tool.invoke({"query": "nonexistent topic"})

        # Assert
        assert "No relevant advice found" in result

    @patch("comp_critic.tools.load_vector_store")
    def test_composition_rag_tool_handles_errors(
        self, mock_load: MagicMock
    ) -> None:
        """Test RAG tool error handling."""
        # Arrange
        mock_load.side_effect = Exception("Database connection failed")

        # Act
        result = composition_rag_tool.invoke({"query": "test query"})

        # Assert
        assert "Error searching knowledge base" in result
        assert "Database connection failed" in result

    @patch("comp_critic.tools.load_vector_store")
    @pytest.mark.parametrize(
        "query,expected_in_result",
        [
            ("leading lines", "Leading lines"),
            ("long exposure", "Long exposure"),
            ("foreground", "Foreground"),
        ],
    )
    def test_composition_rag_tool_parametrized_queries(
        self,
        mock_load: MagicMock,
        sample_documents: List[Document],
        query: str,
        expected_in_result: str,
    ) -> None:
        """Test RAG tool with various queries (parametrized)."""
        # Arrange
        # Filter documents that match the query
        relevant_docs = [
            doc
            for doc in sample_documents
            if expected_in_result.lower() in doc.page_content.lower()
        ]

        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = relevant_docs
        mock_load.return_value = mock_vector_store

        # Act
        result = composition_rag_tool.invoke({"query": query})

        # Assert
        if relevant_docs:
            assert expected_in_result in result or expected_in_result.lower() in result
        assert isinstance(result, str)

    @patch("comp_critic.tools.load_vector_store")
    def test_composition_rag_tool_respects_top_k(
        self, mock_load: MagicMock, sample_documents: List[Document]
    ) -> None:
        """Test that RAG tool respects the configured top_k value."""
        # Arrange
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = sample_documents[:3]
        mock_load.return_value = mock_vector_store

        # Act
        composition_rag_tool.invoke({"query": "test"})

        # Assert
        # Verify that similarity_search was called with k=3 (from config)
        call_kwargs = mock_vector_store.similarity_search.call_args.kwargs
        assert call_kwargs["k"] == 3


class TestToolIntegration:
    """Integration tests for the tool (if ChromaDB is available)."""

    @pytest.mark.integration
    @patch("comp_critic.tools.Chroma")
    @patch("comp_critic.tools.OpenAIEmbeddings")
    def test_tool_full_workflow(
        self,
        mock_embeddings: MagicMock,
        mock_chroma: MagicMock,
        mock_openai_api_key: str,
        mock_chroma_db_path: Path,
        sample_documents: List[Document],
    ) -> None:
        """Test the complete workflow from loading to searching."""
        # Arrange
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = sample_documents[:2]
        mock_chroma.return_value = mock_vector_store

        # Act
        result = composition_rag_tool.invoke({"query": "composition tips"})

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain formatted results
        assert "Result" in result
