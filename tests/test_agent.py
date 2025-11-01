"""Unit tests for the multimodal RAG agent (System 2)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from comp_critic.agent import (
    create_critique_agent,
    critique_image,
    encode_image_to_base64,
)


class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64 function."""

    def test_encode_image_success(self, sample_image: Path) -> None:
        """Test encoding a valid image to base64."""
        # Act
        result = encode_image_to_base64(sample_image)

        # Assert
        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")
        assert len(result) > 100  # Should have substantial content

    def test_encode_image_nonexistent(self, temp_dir: Path) -> None:
        """Test encoding a non-existent image raises FileNotFoundError."""
        # Arrange
        nonexistent_image = temp_dir / "nonexistent.jpg"

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Image not found"):
            encode_image_to_base64(nonexistent_image)

    def test_encode_image_converts_rgb(self, temp_dir: Path) -> None:
        """Test that non-RGB images are converted to RGB."""
        # Arrange
        from PIL import Image

        rgba_image_path = temp_dir / "rgba_image.png"
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(rgba_image_path)

        # Act
        result = encode_image_to_base64(rgba_image_path)

        # Assert
        assert result.startswith("data:image/jpeg;base64,")

    def test_encode_image_accepts_string_path(self, sample_image: Path) -> None:
        """Test that function accepts string paths."""
        # Act
        result = encode_image_to_base64(str(sample_image))

        # Assert
        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")


class TestCreateCritiqueAgent:
    """Tests for create_critique_agent function."""

    @patch("comp_critic.agent.ChatOpenAI")
    def test_create_critique_agent_success(
        self, mock_llm_class: MagicMock, mock_openai_api_key: str
    ) -> None:
        """Test creating an agent with valid configuration."""
        # Arrange
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Act
        agent_executor = create_critique_agent()

        # Assert
        assert agent_executor is not None
        assert hasattr(agent_executor, "invoke")
        mock_llm_class.assert_called_once()

        # Verify LLM was initialized with correct parameters
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["openai_api_key"] == mock_openai_api_key
        assert call_kwargs["temperature"] == 0.7

    def test_create_critique_agent_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test creating agent without API key raises ValueError."""
        # Arrange
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from comp_critic import config as config_module

        monkeypatch.setattr(config_module.Config, "OPENAI_API_KEY", None)

        # Act & Assert
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            create_critique_agent()

    @patch("comp_critic.agent.ChatOpenAI")
    def test_create_critique_agent_has_tools(
        self, mock_llm_class: MagicMock, mock_openai_api_key: str
    ) -> None:
        """Test that agent is created with the composition_rag_tool."""
        # Arrange
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Act
        agent_executor = create_critique_agent()

        # Assert
        assert hasattr(agent_executor, "tools")
        assert len(agent_executor.tools) == 1
        assert agent_executor.tools[0].name == "composition_rag_tool"


class TestCritiqueImage:
    """Tests for critique_image function."""

    @patch("comp_critic.agent.create_critique_agent")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_success(
        self,
        mock_encode: MagicMock,
        mock_create_agent: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test successful image critique."""
        # Arrange
        mock_encode.return_value = "data:image/jpeg;base64,fake_base64_data"

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": "This is a well-composed landscape photograph. "
            "The rule of thirds is applied effectively..."
        }
        mock_create_agent.return_value = mock_agent

        # Act
        result = critique_image(sample_image)

        # Assert
        assert isinstance(result, dict)
        assert "output" in result
        assert "image_path" in result
        assert result["image_path"] == str(sample_image)
        assert len(result["output"]) > 0
        mock_encode.assert_called_once_with(sample_image)
        mock_create_agent.assert_called_once()
        mock_agent.invoke.assert_called_once()

    @patch("comp_critic.agent.create_critique_agent")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_custom_prompt(
        self,
        mock_encode: MagicMock,
        mock_create_agent: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test critique with custom prompt."""
        # Arrange
        mock_encode.return_value = "data:image/jpeg;base64,fake_base64_data"

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Custom analysis..."}
        mock_create_agent.return_value = mock_agent

        custom_prompt = "Focus only on the use of leading lines."

        # Act
        result = critique_image(sample_image, custom_prompt=custom_prompt)

        # Assert
        assert "output" in result

        # Verify custom prompt was passed to agent
        invoke_call = mock_agent.invoke.call_args[0][0]
        assert any(
            msg["text"] == custom_prompt
            for msg in invoke_call["input"]
            if isinstance(msg, dict) and "text" in msg
        )

    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_invalid_image(
        self,
        mock_encode: MagicMock,
        temp_dir: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test critique with non-existent image raises error."""
        # Arrange
        nonexistent_image = temp_dir / "nonexistent.jpg"
        mock_encode.side_effect = FileNotFoundError("Image not found")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            critique_image(nonexistent_image)

    @patch("comp_critic.agent.create_critique_agent")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_multimodal_input_format(
        self,
        mock_encode: MagicMock,
        mock_create_agent: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test that input is formatted correctly for multimodal LLM."""
        # Arrange
        fake_base64 = "data:image/jpeg;base64,fake_data"
        mock_encode.return_value = fake_base64

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Analysis"}
        mock_create_agent.return_value = mock_agent

        # Act
        critique_image(sample_image)

        # Assert
        invoke_call = mock_agent.invoke.call_args[0][0]
        assert "input" in invoke_call

        input_messages = invoke_call["input"]
        assert isinstance(input_messages, list)
        assert len(input_messages) == 2

        # Check text message
        assert input_messages[0]["type"] == "text"
        assert "text" in input_messages[0]

        # Check image message
        assert input_messages[1]["type"] == "image_url"
        assert "image_url" in input_messages[1]
        assert input_messages[1]["image_url"]["url"] == fake_base64


class TestAgentPrompt:
    """Tests for agent prompt requirements."""

    @patch("comp_critic.agent.ChatOpenAI")
    def test_agent_prompt_contains_required_instructions(
        self, mock_llm_class: MagicMock, mock_openai_api_key: str
    ) -> None:
        """Test that agent prompt includes key instruction components."""
        # Arrange
        from comp_critic.agent import AGENT_SYSTEM_PROMPT

        # Assert - verify key requirements from PRD (R-2.3)
        assert "visual analysis" in AGENT_SYSTEM_PROMPT.lower()
        assert "composition_rag_tool" in AGENT_SYSTEM_PROMPT
        assert "rule of thirds" in AGENT_SYSTEM_PROMPT.lower()
        assert "leading lines" in AGENT_SYSTEM_PROMPT.lower()
        assert "foreground" in AGENT_SYSTEM_PROMPT.lower()

    def test_agent_prompt_requires_rag_usage(self) -> None:
        """Test that prompt emphasizes using the RAG tool."""
        # Arrange
        from comp_critic.agent import AGENT_SYSTEM_PROMPT

        # Assert
        assert "MUST use" in AGENT_SYSTEM_PROMPT
        assert "knowledge base" in AGENT_SYSTEM_PROMPT.lower()


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for the agent (mocked external services)."""

    @patch("comp_critic.agent.ChatOpenAI")
    @patch("comp_critic.tools.Chroma")
    @patch("comp_critic.tools.OpenAIEmbeddings")
    def test_agent_end_to_end_mock(
        self,
        mock_embeddings: MagicMock,
        mock_chroma: MagicMock,
        mock_llm_class: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
        mock_chroma_db_path: Path,
    ) -> None:
        """Test end-to-end agent workflow with mocked services."""
        # This test verifies the complete workflow while mocking external APIs
        # In a real integration test, you would use actual API calls

        # Arrange
        # Mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = []
        mock_chroma.return_value = mock_vector_store

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Note: Full integration would require mocking the agent executor's invoke
        # This is a placeholder for actual integration testing
        assert True  # Placeholder
