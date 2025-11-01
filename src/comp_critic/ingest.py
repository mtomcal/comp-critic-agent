"""System 1: Data Ingestion Pipeline for landscape photography transcripts."""

import sys
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from comp_critic.config import config


def load_transcripts(transcripts_dir: Path) -> List[Document]:
    """
    Load all .txt transcript files from the specified directory.

    Args:
        transcripts_dir: Path to the directory containing transcript files

    Returns:
        List of Document objects containing transcript content and metadata
    """
    print(f"Loading transcripts from: {transcripts_dir}")

    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_dir}")

    # Use DirectoryLoader to load all .txt files
    loader = DirectoryLoader(
        str(transcripts_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} transcript files")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks for better retrieval.

    Args:
        documents: List of Document objects to chunk

    Returns:
        List of chunked Document objects
    """
    print(
        f"Chunking documents (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    return chunks


def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Create and persist a ChromaDB vector store with embedded document chunks.

    Args:
        chunks: List of chunked Document objects

    Returns:
        Chroma vector store instance
    """
    print(f"Creating embeddings using {config.EMBEDDING_MODEL}")

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )

    # Ensure the ChromaDB directory exists
    config.ensure_chroma_db_path()

    print(f"Creating ChromaDB vector store at: {config.CHROMA_DB_PATH}")

    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DB_PATH),
    )

    print(f"Vector store created with {len(chunks)} chunks")

    return vector_store


def run_ingestion_pipeline() -> None:
    """
    Execute the complete data ingestion pipeline.

    Steps:
        1. Load transcript documents from directory
        2. Chunk documents into smaller pieces
        3. Create embeddings and store in ChromaDB
    """
    print("=" * 60)
    print("Comp Critic Agent - Data Ingestion Pipeline")
    print("=" * 60)

    try:
        # Validate configuration
        config.validate()

        # Step 1: Load documents
        documents = load_transcripts(config.TRANSCRIPTS_DIR)

        if not documents:
            print("Warning: No transcript files found!")
            return

        # Step 2: Chunk documents
        chunks = chunk_documents(documents)

        # Step 3: Create and persist vector store
        vector_store = create_vector_store(chunks)

        print("=" * 60)
        print("Ingestion pipeline completed successfully!")
        print(f"Vector store location: {config.CHROMA_DB_PATH}")
        print("=" * 60)

    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Entry point for the ingestion script."""
    run_ingestion_pipeline()


if __name__ == "__main__":
    main()
