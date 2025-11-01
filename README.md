# Comp Critic Agent

An AI-powered agent that provides expert-level compositional critiques of landscape photographs by combining multimodal visual analysis with retrieval-augmented generation (RAG) from a curated knowledge base of professional photography video transcripts.

## Features

- **Multimodal Analysis**: Uses GPT-4's vision capabilities to analyze compositional elements in your landscape photos
- **RAG-Enhanced Feedback**: Retrieves relevant expert advice from video transcripts to ground critiques in professional knowledge
- **Comprehensive Critiques**: Synthesizes visual analysis with expert advice for actionable, educational feedback
- **Modern Python Tooling**: Built with Poetry, Poe the Poet, pytest, and type hints

## Architecture

The system consists of two main components:

1. **System 1: Data Ingestion Pipeline** (`ingest.py`)
   - Loads `.txt` video transcripts from a directory
   - Chunks documents for optimal retrieval
   - Creates embeddings and stores them in ChromaDB

2. **System 2: Multimodal RAG Agent** (`agent.py`)
   - Accepts landscape photograph images
   - Analyzes composition using GPT-4 vision
   - Searches knowledge base for relevant techniques
   - Synthesizes comprehensive critiques

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key (for GPT-4 and embeddings)
- Video transcript files (`.txt` format) from landscape photography educational content

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd comp-critic-agent
```

### 2. Install Poetry

If you don't have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Install dependencies

```bash
poetry install
```

This will create a virtual environment and install all required dependencies including:
- LangChain for agent orchestration
- ChromaDB for vector storage
- OpenAI SDK for LLM and embeddings
- pytest for testing
- Ruff for linting/formatting

### 4. Configure environment variables

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 5. Prepare your transcripts

Place your video transcript `.txt` files in the `transcripts/` directory:

```bash
mkdir -p transcripts
# Copy your transcript files to transcripts/
cp /path/to/your/transcripts/*.txt transcripts/
```

## Usage

### Step 1: Ingest Transcripts

Before using the agent, you must run the ingestion pipeline to build the knowledge base:

```bash
poetry run poe ingest
```

This will:
- Load all `.txt` files from `transcripts/`
- Chunk them into optimal sizes
- Create embeddings using OpenAI
- Store them in `chroma_db/`

### Step 2: Critique Images

#### Using the example script

```bash
poetry run python examples/sample_critique.py path/to/your/landscape.jpg
```

#### Using the agent directly in Python

```python
from comp_critic.agent import critique_image

result = critique_image("path/to/landscape.jpg")
print(result["output"])
```

#### Using as a module

```bash
poetry run python -m comp_critic.agent path/to/landscape.jpg
```

#### Custom prompts

```python
from comp_critic.agent import critique_image

result = critique_image(
    "photo.jpg",
    custom_prompt="Focus specifically on the use of leading lines and foreground interest."
)
print(result["output"])
```

## Development

### Running Tests

Run all tests with coverage:

```bash
poetry run poe test
```

Run tests without coverage (faster):

```bash
poetry run poe test-fast
```

### Code Quality

Format code with Ruff:

```bash
poetry run poe format
```

Check formatting without modifying:

```bash
poetry run poe format-check
```

Lint code:

```bash
poetry run poe lint
```

Run type checking:

```bash
poetry run poe typecheck
```

Run all quality checks at once:

```bash
poetry run poe check-all
```

### Available Poe Tasks

View all available tasks:

```bash
poetry run poe --help
```

- `poe ingest` - Run the transcript ingestion pipeline
- `poe test` - Run tests with coverage
- `poe test-fast` - Run tests without coverage
- `poe lint` - Run linter
- `poe format` - Format code
- `poe format-check` - Check code formatting
- `poe typecheck` - Run static type checking
- `poe check-all` - Run all quality checks

## Project Structure

```
comp-critic-agent/
├── src/comp_critic/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── ingest.py          # System 1: Data ingestion
│   ├── tools.py           # RAG tool definition
│   └── agent.py           # System 2: Multimodal agent
├── tests/
│   ├── conftest.py        # Pytest fixtures
│   ├── test_ingest.py     # Ingestion tests
│   ├── test_tools.py      # RAG tool tests
│   └── test_agent.py      # Agent tests
├── examples/
│   └── sample_critique.py # Example usage
├── transcripts/           # Your video transcript files
├── chroma_db/            # Vector database (auto-generated)
├── pyproject.toml        # Poetry config + Poe tasks
├── .env                  # Environment variables (not in git)
└── README.md
```

## Configuration

You can customize behavior via environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
OPENAI_MODEL=gpt-4o              # Default: gpt-4o
TRANSCRIPTS_DIR=./transcripts    # Default: ./transcripts
CHROMA_DB_PATH=./chroma_db       # Default: ./chroma_db
CHUNK_SIZE=1000                  # Default: 1000
CHUNK_OVERLAP=200                # Default: 200
RAG_TOP_K=3                      # Default: 3
```

## How It Works

1. **Ingestion Phase** (one-time setup):
   - Transcripts are loaded from the `transcripts/` directory
   - Text is split into overlapping chunks for better retrieval
   - Each chunk is embedded using OpenAI's embedding model
   - Embeddings are stored in ChromaDB for fast similarity search

2. **Critique Phase** (per image):
   - Your landscape photo is encoded and sent to GPT-4 vision
   - The agent analyzes compositional elements (rule of thirds, leading lines, etc.)
   - Based on its analysis, it formulates search queries
   - The `composition_rag_tool` retrieves relevant expert advice from the knowledge base
   - The agent synthesizes its visual analysis with retrieved advice
   - A comprehensive critique is returned

## Testing Strategy

The project follows pytest best practices:

- **AAA Pattern**: Arrange, Act, Assert in all tests
- **Fixtures**: Reusable test components in `conftest.py`
- **Mocking**: External dependencies (OpenAI API, ChromaDB) are mocked
- **Parametrized Tests**: Multiple test cases with different inputs
- **Coverage**: Aim for >80% code coverage

Run tests with coverage report:

```bash
poetry run poe test
```

## Troubleshooting

### "OPENAI_API_KEY not found"

Make sure you've created a `.env` file with your API key:

```bash
cp .env.example .env
# Edit .env and add your key
```

### "ChromaDB not found"

Run the ingestion pipeline first:

```bash
poetry run poe ingest
```

### "Transcripts directory not found"

Create the transcripts directory and add your `.txt` files:

```bash
mkdir -p transcripts
cp /path/to/transcripts/*.txt transcripts/
```

### Import errors

Make sure you're running commands through Poetry:

```bash
poetry run python examples/sample_critique.py image.jpg
```

## Contributing

Contributions are welcome! Please:

1. Run `poe check-all` before committing
2. Add tests for new features
3. Update documentation as needed

## License

[Add your license here]

## Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/) - LLM application framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) - GPT-4 and embeddings
- [Poetry](https://python-poetry.org/) - Dependency management
- [Poe the Poet](https://poethepoet.natn.io/) - Task runner
- [pytest](https://pytest.org/) - Testing framework
- [Ruff](https://github.com/astral-sh/ruff) - Linting and formatting
