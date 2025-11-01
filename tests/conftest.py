"""Pytest configuration and shared fixtures for Comp Critic Agent tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from langchain_core.documents import Document


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_transcripts_dir(temp_dir: Path) -> Path:
    """Create a temporary directory with sample transcript files."""
    transcripts_dir = temp_dir / "transcripts"
    transcripts_dir.mkdir()

    # Create sample transcript files
    transcript1 = transcripts_dir / "video1.txt"
    transcript1.write_text(
        "When composing landscape images, the rule of thirds is fundamental. "
        "Place your horizon line either in the upper or lower third of the frame. "
        "Leading lines are powerful compositional tools that guide the viewer's eye "
        "through the image to the main subject."
    )

    transcript2 = transcripts_dir / "video2.txt"
    transcript2.write_text(
        "Long exposure photography of waterfalls creates a beautiful silky effect. "
        "Use a neutral density filter to achieve shutter speeds of several seconds. "
        "Make sure to use a tripod for stability during long exposures."
    )

    transcript3 = transcripts_dir / "video3.txt"
    transcript3.write_text(
        "Foreground interest is crucial in landscape photography. "
        "Including rocks, flowers, or other elements in the foreground adds depth "
        "and creates layers in your composition. This helps draw the viewer into the scene."
    )

    return transcripts_dir


@pytest.fixture
def sample_documents() -> list[Document]:
    """Return a list of sample Document objects for testing."""
    return [
        Document(
            page_content="Rule of thirds is a fundamental composition technique.",
            metadata={"source": "test_transcript_1.txt"},
        ),
        Document(
            page_content="Leading lines guide the viewer's eye through the image.",
            metadata={"source": "test_transcript_1.txt"},
        ),
        Document(
            page_content="Long exposure creates silky water effects in waterfalls.",
            metadata={"source": "test_transcript_2.txt"},
        ),
        Document(
            page_content="Foreground interest adds depth to landscape compositions.",
            metadata={"source": "test_transcript_3.txt"},
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Return a list of sample chunked Document objects."""
    return [
        Document(
            page_content="When composing landscape images, the rule of thirds is fundamental.",
            metadata={"source": "video1.txt", "chunk": 0},
        ),
        Document(
            page_content="Place your horizon line either in the upper or lower third.",
            metadata={"source": "video1.txt", "chunk": 1},
        ),
        Document(
            page_content="Long exposure photography of waterfalls creates silky effects.",
            metadata={"source": "video2.txt", "chunk": 0},
        ),
    ]


@pytest.fixture
def mock_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Mock the OpenAI API key for tests."""
    api_key = "test-api-key-12345"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    # Also patch the Config class attribute since it's read at import time
    from comp_critic import config as config_module

    monkeypatch.setattr(config_module.Config, "OPENAI_API_KEY", api_key)

    return api_key


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample test image."""
    from PIL import Image

    image_path = temp_dir / "test_image.jpg"

    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(image_path)

    return image_path


@pytest.fixture
def mock_chroma_db_path(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary ChromaDB path for tests."""
    chroma_path = temp_dir / "chroma_db"
    chroma_path.mkdir()

    # Patch the config to use this path
    from comp_critic import config as config_module

    monkeypatch.setattr(config_module.Config, "CHROMA_DB_PATH", chroma_path)

    return chroma_path


@pytest.fixture
def mock_transcripts_dir(sample_transcripts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Mock the transcripts directory in config."""
    from comp_critic import config as config_module

    monkeypatch.setattr(config_module.Config, "TRANSCRIPTS_DIR", sample_transcripts_dir)

    return sample_transcripts_dir
