"""Configuration management for Comp Critic Agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration for the Comp Critic Agent."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    TRANSCRIPTS_DIR: Path = PROJECT_ROOT / os.getenv("TRANSCRIPTS_DIR", "transcripts")
    CHROMA_DB_PATH: Path = PROJECT_ROOT / os.getenv("CHROMA_DB_PATH", "chroma_db")

    # OpenAI Configuration
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1")

    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # RAG Configuration
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

    # Vision API Configuration
    VISION_DETAIL: str = os.getenv("VISION_DETAIL", "low")  # "low" or "high"
    VISION_MAX_SIZE: int = int(os.getenv("VISION_MAX_SIZE", "2048"))

    # ChromaDB Configuration
    COLLECTION_NAME: str = "landscape_photography_transcripts"

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment."
            )

        if not cls.TRANSCRIPTS_DIR.exists():
            raise FileNotFoundError(f"Transcripts directory not found: {cls.TRANSCRIPTS_DIR}")

    @classmethod
    def ensure_chroma_db_path(cls) -> None:
        """Ensure the ChromaDB directory exists."""
        cls.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)


# Convenience instance
config = Config()
