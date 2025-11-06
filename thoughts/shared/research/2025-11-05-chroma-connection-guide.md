---
date: 2025-11-05T05:16:45+0000
researcher: AI Assistant
git_commit: 8b1c5ca6b54059d9307388e5b4ed395471ec4910
branch: main
repository: comp-critic-agent
topic: "How to connect to Chroma"
tags: [research, chroma, vector-store, rag, langchain]
status: complete
last_updated: 2025-11-05
last_updated_by: AI Assistant
---

# Research: How to Connect to Chroma

**Date**: 2025-11-05T05:16:45+0000  
**Researcher**: AI Assistant  
**Git Commit**: 8b1c5ca6b54059d9307388e5b4ed395471ec4910  
**Branch**: main  
**Repository**: comp-critic-agent

## Research Question
How to connect to and use Chroma vector database in the comp-critic-agent project?

## Summary
This project uses ChromaDB as a persistent vector store for landscape photography transcript embeddings. The connection is established through the LangChain integration (`langchain_community.vectorstores.Chroma`) with OpenAI embeddings. The database stores chunked transcript documents in a local directory configured via environment variables. Two main operations are supported: creating/ingesting data (`ingest.py`) and retrieving/querying (`tools.py`).

## Detailed Findings

### Configuration Management

**File**: `src/comp_critic/config.py:1-60`

The `Config` class centrally manages all Chroma connection settings:

- **Collection Name** (`COLLECTION_NAME`): `"landscape_photography_transcripts"` - identifies the specific collection within Chroma (line 39)
- **Database Path** (`CHROMA_DB_PATH`): Configurable via `CHROMA_DB_PATH` env var, defaults to `./chroma_db` (line 18)
- **Embedding Model** (`EMBEDDING_MODEL`): Uses `"text-embedding-3-small"` from OpenAI (line 25)
- **Persistence**: Directory-based persistence to `persist_directory` parameter
- **Helper Method** `ensure_chroma_db_path()`: Creates the directory if it doesn't exist (lines 53-55)

**Key Configuration Values**:
- Chunk size: 1000 characters (configurable via `CHUNK_SIZE`)
- Chunk overlap: 200 characters (configurable via `CHUNK_OVERLAP`)
- RAG top-k: 3 results (configurable via `RAG_TOP_K`)

### Creating and Persisting Connection (Ingestion)

**File**: `src/comp_critic/ingest.py:69-102`

The `create_vector_store()` function establishes the initial Chroma connection:

```python
def create_vector_store(chunks: list[Document]) -> Chroma:
    # 1. Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
    
    # 2. Ensure directory exists
    config.ensure_chroma_db_path()
    
    # 3. Create and persist vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DB_PATH),
    )
    return vector_store
```

**Connection Details**:
- **Method**: `Chroma.from_documents()` - used during initial creation with embedding documents
- **Embedding Integration**: OpenAI embeddings provide the embedding function
- **Persistence**: Stored locally in `config.CHROMA_DB_PATH` directory
- **Collection**: All documents stored under `"landscape_photography_transcripts"` collection

### Loading Existing Connection (Retrieval)

**File**: `src/comp_critic/tools.py:11-38`

The `load_vector_store()` function connects to an existing persisted Chroma database:

```python
def load_vector_store() -> Chroma:
    # 1. Validate database exists
    if not config.CHROMA_DB_PATH.exists():
        raise ValueError(
            f"ChromaDB not found at {config.CHROMA_DB_PATH}. "
            "Please run the ingestion pipeline first."
        )
    
    # 2. Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
    
    # 3. Connect to existing vector store
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DB_PATH),
    )
    return vector_store
```

**Connection Details**:
- **Method**: `Chroma()` constructor (not `.from_documents()`) for loading existing database
- **Embedding Function**: Must be the same model (`text-embedding-3-small`) used during ingestion
- **Persistence Directory**: Points to the saved database location
- **Collection Access**: References the same collection name created during ingestion
- **Error Handling**: Validates database exists before attempting connection

### Query/Search Usage

**File**: `src/comp_critic/tools.py:64-93`

The `composition_rag_tool()` demonstrates querying the connected Chroma database:

```python
def composition_rag_tool(query: str) -> str:
    vector_store = load_vector_store()
    
    # Perform semantic search
    documents = vector_store.similarity_search(
        query,
        k=config.RAG_TOP_K,  # Returns top 3 results by default
    )
    
    return format_search_results(documents)
```

**Retrieval Method**: `similarity_search()` - finds semantically similar documents

## Code References

- `src/comp_critic/config.py:18` - CHROMA_DB_PATH configuration
- `src/comp_critic/config.py:39` - COLLECTION_NAME definition
- `src/comp_critic/config.py:53-55` - Directory creation helper
- `src/comp_critic/ingest.py:93-98` - Creating new vector store with `Chroma.from_documents()`
- `src/comp_critic/tools.py:32-36` - Loading existing vector store with `Chroma()` constructor
- `src/comp_critic/tools.py:84-87` - Semantic search query execution

## Architecture Insights

### Two-Step Connection Pattern

1. **Ingestion Phase** (first run): Uses `Chroma.from_documents()` to create and persist database
2. **Runtime Phase** (subsequent runs): Uses `Chroma()` constructor to load existing database

### Dependency Chain

```
Config (credentials + paths) 
  ↓
OpenAI Embeddings (embedding model)
  ↓
Chroma Connection (local persistence)
  ↓
LangChain Tools (RAG queries)
```

### Key Design Decisions

- **Local Persistence**: Database persists to disk in `chroma_db/` directory, enabling offline-after-ingestion queries
- **Collection Isolation**: All transcript data stored in single named collection for logical separation
- **Lazy Loading**: Vector store only loaded when tools are invoked (not at agent initialization)
- **Embedding Consistency**: Same embedding model must be used for all operations (creation, loading, querying)

## Environment Setup

**Required Environment Variables** (from `.env.example`):
- `OPENAI_API_KEY` - Required for embedding generation
- `CHROMA_DB_PATH` - Optional, defaults to `./chroma_db`
- `TRANSCRIPTS_DIR` - Optional, defaults to `./transcripts`

**Optional RAG Configuration**:
- `RAG_TOP_K=3` - Number of results to retrieve per query
- `CHUNK_SIZE=1000` - Document chunk size during ingestion
- `CHUNK_OVERLAP=200` - Overlap between chunks

## Quick Start for Connection

### Step 1: Prepare Environment
```bash
# Create .env file
OPENAI_API_KEY=sk-...
CHROMA_DB_PATH=./chroma_db
TRANSCRIPTS_DIR=./transcripts
```

### Step 2: Ingest Data (First Time)
```python
from comp_critic.ingest import run_ingestion_pipeline
run_ingestion_pipeline()  # Creates chroma_db/ with persisted vectors
```

### Step 3: Connect and Query
```python
from comp_critic.tools import load_vector_store, composition_rag_tool

# Method 1: Direct vector store access
vector_store = load_vector_store()
results = vector_store.similarity_search("leading lines composition", k=3)

# Method 2: Through RAG tool (preferred in agent)
answer = composition_rag_tool("rule of thirds landscape photography")
```

## Dependencies

- `langchain-community>=0.0.x` - Provides `Chroma` vector store integration
- `langchain-core>=0.0.x` - Document type definitions
- `langchain-openai>=0.0.x` - OpenAI embeddings integration
- `chromadb` - Underlying vector database (via langchain-community)

## Related Research

None yet - this is the primary research document for Chroma integration.

## Open Questions

- Is the Chroma database currently populated with transcript data?
- Are there performance considerations for scaling beyond current chunk count?
- Is there a need for remote Chroma server vs. local persistence?
