#!/usr/bin/env python3
"""Test script for querying ChromaDB collection interactively.

Usage:
    poetry run python scripts/test_query.py "your search query"
    poetry run python scripts/test_query.py "rule of thirds" --top-k 5
"""

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comp_critic.config import Config


def query_collection(query_text: str, top_k: int = 3) -> None:
    """Query the ChromaDB collection and print results.

    Args:
        query_text: The search query text
        top_k: Number of results to return (default: 3)
    """
    # Validate config
    Config.validate()

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(Config.CHROMA_DB_PATH))

    # Get embedding function
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=Config.OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

    # Get collection
    try:
        collection = client.get_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=embedding_function,
        )
    except Exception as e:
        print(f"Error: Could not load collection '{Config.COLLECTION_NAME}'")
        print(f"Make sure you've run ingestion first: poetry run poe ingest")
        print(f"Details: {e}")
        sys.exit(1)

    # Query collection
    print(f"\n{'=' * 80}")
    print(f"Query: '{query_text}'")
    print(f"Top-K: {top_k}")
    print(f"{'=' * 80}\n")

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
    )

    # Print results
    if not results["documents"] or not results["documents"][0]:
        print("No results found.")
        return

    for idx, (doc, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        1,
    ):
        print(f"Result {idx}:")
        print(f"  Distance: {distance:.4f}")
        print(f"  Source: {metadata.get('source', 'Unknown')}")
        print(f"  Content: {doc}")
        print(f"{'-' * 80}\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test semantic search queries against ChromaDB collection"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return (default: 3)",
    )

    args = parser.parse_args()

    query_collection(args.query, args.top_k)


if __name__ == "__main__":
    main()
