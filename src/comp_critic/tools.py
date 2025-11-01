"""RAG tool for searching landscape photography video transcripts."""

from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from comp_critic.config import config


def load_vector_store() -> Chroma:
    """
    Load the persistent ChromaDB vector store.

    Returns:
        Chroma vector store instance

    Raises:
        ValueError: If the vector store doesn't exist
    """
    if not config.CHROMA_DB_PATH.exists():
        raise ValueError(
            f"ChromaDB not found at {config.CHROMA_DB_PATH}. "
            "Please run the ingestion pipeline first."
        )

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,  # type: ignore[call-arg]
    )

    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DB_PATH),
    )

    return vector_store


def format_search_results(documents: list[Document]) -> str:
    """
    Format retrieved documents into a readable string.

    Args:
        documents: List of retrieved Document objects

    Returns:
        Formatted string containing all document content
    """
    if not documents:
        return "No relevant advice found in the knowledge base."

    formatted_results = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.strip()
        formatted_results.append(f"--- Result {i} (Source: {source}) ---\n{content}\n")

    return "\n".join(formatted_results)


@tool
def composition_rag_tool(query: str) -> str:
    """
    Searches the landscape photography video transcripts for advice.

    Use this tool to find specific techniques, tips, or critiques related
    to a compositional query. The tool performs semantic search on a knowledge
    base of expert video transcripts.

    Args:
        query: The search query describing what compositional advice to find
               (e.g., "leading lines", "rule of thirds", "long exposure waterfalls")

    Returns:
        A formatted string containing the most relevant advice from the transcripts
    """
    try:
        # Load the vector store
        vector_store = load_vector_store()

        # Perform similarity search
        documents = vector_store.similarity_search(
            query,
            k=config.RAG_TOP_K,
        )

        # Format and return results
        return format_search_results(documents)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


# Export the tool for use in the agent
__all__ = ["composition_rag_tool", "load_vector_store", "format_search_results"]
