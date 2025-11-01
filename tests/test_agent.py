"""Unit tests for the multimodal RAG agent (System 2)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from comp_critic.agent import (
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

    def test_encode_image_resizes_large_images(self, temp_dir: Path) -> None:
        """Test that large images are resized to reduce token usage."""
        # Arrange
        from PIL import Image

        large_image_path = temp_dir / "large_image.jpg"
        # Create a large 4000x3000 image (typical high-res landscape)
        img = Image.new("RGB", (4000, 3000), color=(100, 150, 200))
        img.save(large_image_path, format="JPEG")

        # Act
        result = encode_image_to_base64(large_image_path, max_size=2048)

        # Assert
        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")

        # Verify the image was resized by checking it's smaller than original
        # Decode and check dimensions
        import base64
        from io import BytesIO

        base64_data = result.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        resized_img = Image.open(BytesIO(img_bytes))

        # Should be resized to max_size on longest dimension
        assert max(resized_img.size) == 2048
        assert resized_img.size == (2048, 1536)  # Maintains 4:3 aspect ratio

    def test_encode_image_preserves_small_images(self, temp_dir: Path) -> None:
        """Test that small images are not upscaled."""
        # Arrange
        from PIL import Image

        small_image_path = temp_dir / "small_image.jpg"
        img = Image.new("RGB", (800, 600), color=(100, 150, 200))
        img.save(small_image_path, format="JPEG")

        # Act
        result = encode_image_to_base64(small_image_path, max_size=2048)

        # Assert
        assert isinstance(result, str)

        # Verify dimensions were not changed
        import base64
        from io import BytesIO

        base64_data = result.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        processed_img = Image.open(BytesIO(img_bytes))

        # Should remain the same size (or very close due to JPEG compression)
        assert processed_img.size == (800, 600)

    def test_encode_image_maintains_aspect_ratio_portrait(self, temp_dir: Path) -> None:
        """Test that portrait images maintain aspect ratio when resized."""
        # Arrange
        from PIL import Image

        portrait_image_path = temp_dir / "portrait_image.jpg"
        # Create a tall portrait image (3000x4000)
        img = Image.new("RGB", (3000, 4000), color=(100, 150, 200))
        img.save(portrait_image_path, format="JPEG")

        # Act
        result = encode_image_to_base64(portrait_image_path, max_size=2048)

        # Assert
        import base64
        from io import BytesIO

        base64_data = result.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        resized_img = Image.open(BytesIO(img_bytes))

        # Should be resized to 1536x2048 (maintains 3:4 aspect ratio)
        assert resized_img.size == (1536, 2048)
        assert max(resized_img.size) == 2048

    def test_encode_image_custom_max_size(self, temp_dir: Path) -> None:
        """Test that custom max_size parameter works correctly."""
        # Arrange
        from PIL import Image

        large_image_path = temp_dir / "large_image.jpg"
        img = Image.new("RGB", (3000, 2000), color=(100, 150, 200))
        img.save(large_image_path, format="JPEG")

        # Act
        result = encode_image_to_base64(large_image_path, max_size=1024)

        # Assert
        import base64
        from io import BytesIO

        base64_data = result.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        resized_img = Image.open(BytesIO(img_bytes))

        # Should be resized to 1024 on longest dimension
        assert max(resized_img.size) == 1024

    def test_encode_image_uses_config_default_max_size(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that encode_image uses config default when max_size is None."""
        # Arrange
        from PIL import Image

        from comp_critic import config as config_module

        monkeypatch.setattr(config_module.Config, "VISION_MAX_SIZE", 1024)

        large_image_path = temp_dir / "large_image.jpg"
        img = Image.new("RGB", (3000, 2000), color=(100, 150, 200))
        img.save(large_image_path, format="JPEG")

        # Act - don't pass max_size, should use config default
        result = encode_image_to_base64(large_image_path)

        # Assert
        import base64
        from io import BytesIO

        base64_data = result.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        resized_img = Image.open(BytesIO(img_bytes))

        # Should be resized to config default (1024)
        assert max(resized_img.size) == 1024


class TestCritiqueImage:
    """Tests for critique_image function."""

    @patch("comp_critic.tools.composition_rag_tool")
    @patch("comp_critic.agent.ChatOpenAI")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_success(
        self,
        mock_encode: MagicMock,
        mock_llm_class: MagicMock,
        mock_tool: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test successful image critique with manual tool calling."""
        # Arrange
        mock_encode.return_value = "data:image/jpeg;base64,fake_base64_data"

        # Mock the LLM response without tool calls (direct answer)
        mock_llm = MagicMock()
        mock_response = AIMessage(
            content="This is a well-composed landscape photograph.",
            tool_calls=[],  # No tool calls
        )
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        mock_llm.invoke.return_value = mock_response

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        mock_llm_class.return_value = mock_llm

        # Act
        result = critique_image(sample_image)

        # Assert
        assert isinstance(result, dict)
        assert "output" in result
        assert "image_path" in result
        assert "token_usage" in result
        assert result["image_path"] == str(sample_image)
        assert len(result["output"]) > 0
        mock_encode.assert_called_once_with(sample_image, max_size=None)
        assert result["token_usage"]["total_tokens"] == 150

    @patch("comp_critic.tools.composition_rag_tool")
    @patch("comp_critic.agent.ChatOpenAI")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_uses_default_low_detail(
        self,
        mock_encode: MagicMock,
        mock_llm_class: MagicMock,
        mock_tool: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that critique uses default 'low' detail from config."""
        # Arrange
        from comp_critic import config as config_module

        monkeypatch.setattr(config_module.Config, "VISION_DETAIL", "low")

        mock_encode.return_value = "data:image/jpeg;base64,fake_data"

        mock_llm = MagicMock()
        mock_response = AIMessage(content="Analysis", tool_calls=[])
        mock_response.usage_metadata = None
        mock_llm.invoke.return_value = mock_response

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_class.return_value = mock_llm

        # Act
        critique_image(sample_image)

        # Assert - verify HumanMessage contains detail="low" in image_url
        call_args = mock_llm_with_tools.invoke.call_args[0][0]
        human_message = call_args[0]
        image_content = human_message.content[1]
        assert image_content["image_url"]["detail"] == "low"

    @patch("comp_critic.tools.composition_rag_tool")
    @patch("comp_critic.agent.ChatOpenAI")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_accepts_high_detail(
        self,
        mock_encode: MagicMock,
        mock_llm_class: MagicMock,
        mock_tool: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test that critique accepts 'high' detail parameter."""
        # Arrange
        mock_encode.return_value = "data:image/jpeg;base64,fake_data"

        mock_llm = MagicMock()
        mock_response = AIMessage(content="High detail analysis", tool_calls=[])
        mock_response.usage_metadata = None
        mock_llm.invoke.return_value = mock_response

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_class.return_value = mock_llm

        # Act
        critique_image(sample_image, detail="high")

        # Assert - verify detail="high" was used
        call_args = mock_llm_with_tools.invoke.call_args[0][0]
        human_message = call_args[0]
        image_content = human_message.content[1]
        assert image_content["image_url"]["detail"] == "high"

    def test_critique_image_rejects_invalid_detail(
        self,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test that invalid detail parameter raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="detail must be 'low' or 'high'"):
            critique_image(sample_image, detail="medium")

    @patch("comp_critic.tools.composition_rag_tool")
    @patch("comp_critic.agent.ChatOpenAI")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_passes_max_size(
        self,
        mock_encode: MagicMock,
        mock_llm_class: MagicMock,
        mock_tool: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test that max_size parameter is passed to encode function."""
        # Arrange
        mock_encode.return_value = "data:image/jpeg;base64,fake_data"

        mock_llm = MagicMock()
        mock_response = AIMessage(content="Analysis", tool_calls=[])
        mock_response.usage_metadata = None
        mock_llm.invoke.return_value = mock_response

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_class.return_value = mock_llm

        # Act
        critique_image(sample_image, max_size=1024)

        # Assert
        mock_encode.assert_called_once_with(sample_image, max_size=1024)

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

    @patch("comp_critic.agent.composition_rag_tool")
    @patch("comp_critic.agent.ChatOpenAI")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_with_tool_calling(
        self,
        mock_encode: MagicMock,
        mock_llm_class: MagicMock,
        mock_tool: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test critique with tool calling loop."""
        # Arrange
        mock_encode.return_value = "data:image/jpeg;base64,fake_data"

        # Mock the tool to return a result
        mock_tool_instance = MagicMock()
        mock_tool_instance.invoke.return_value = "RAG result about composition"
        mock_tool_instance.name = "composition_rag_tool"

        # Replace the composition_rag_tool with our mock
        mock_tool.return_value = mock_tool_instance

        # First call: LLM decides to use tool
        mock_response_1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "composition_rag_tool",
                    "args": {"query": "rule of thirds"},
                    "id": "call_123",
                }
            ],
        )

        # Second call: LLM provides final answer
        mock_response_2 = AIMessage(
            content="Final critique based on RAG results.",
            tool_calls=[],
        )
        mock_response_2.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        }

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.side_effect = [mock_response_1, mock_response_2]
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_class.return_value = mock_llm

        # Act
        result = critique_image(sample_image)

        # Assert
        assert "output" in result
        assert "Final critique" in result["output"]
        assert result["token_usage"]["total_tokens"] == 300
        # Verify the tool was called through invoke
        assert mock_llm_with_tools.invoke.call_count == 2

    @patch("comp_critic.tools.composition_rag_tool")
    @patch("comp_critic.agent.ChatOpenAI")
    @patch("comp_critic.agent.encode_image_to_base64")
    def test_critique_image_custom_prompt(
        self,
        mock_encode: MagicMock,
        mock_llm_class: MagicMock,
        mock_tool: MagicMock,
        sample_image: Path,
        mock_openai_api_key: str,
    ) -> None:
        """Test critique with custom prompt."""
        # Arrange
        mock_encode.return_value = "data:image/jpeg;base64,fake_data"
        custom_prompt = "Focus only on the use of leading lines."

        mock_llm = MagicMock()
        mock_response = AIMessage(content="Custom analysis", tool_calls=[])
        mock_response.usage_metadata = None
        mock_llm.invoke.return_value = mock_response

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_class.return_value = mock_llm

        # Act
        result = critique_image(sample_image, custom_prompt=custom_prompt)

        # Assert
        assert "output" in result
        # Verify custom prompt was included in the message
        call_args = mock_llm_with_tools.invoke.call_args[0][0]
        human_message = call_args[0]
        text_content = human_message.content[0]
        assert custom_prompt in text_content["text"]

    def test_critique_image_missing_api_key(
        self,
        sample_image: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test critique without API key raises ValueError."""
        # Arrange
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from comp_critic import config as config_module

        monkeypatch.setattr(config_module.Config, "OPENAI_API_KEY", None)

        # Act & Assert
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            critique_image(sample_image)


class TestAgentPrompt:
    """Tests for agent prompt requirements."""

    def test_agent_prompt_contains_required_instructions(self) -> None:
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
