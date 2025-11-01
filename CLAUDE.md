# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Comp Critic Agent** is a multimodal RAG (Retrieval-Augmented Generation) system that critiques landscape photographs by combining GPT-4 vision analysis with a knowledge base of expert photography video transcripts.

**Two-System Architecture:**

1. **System 1 (Ingestion Pipeline)** - `src/comp_critic/ingest.py`
   - One-time setup: Loads `.txt` video transcripts → chunks → embeds → stores in ChromaDB
   - Creates persistent vector store at `./chroma_db/`

2. **System 2 (Multimodal Agent)** - `src/comp_critic/agent.py`
   - Runtime: Accepts image → GPT-4 vision analysis → queries ChromaDB via RAG tool → synthesizes critique
   - Uses LangChain's tool-calling agent pattern with a single tool: `composition_rag_tool`

**Key Dependencies:** LangChain (agent framework), ChromaDB (vector DB), OpenAI (GPT-4 + embeddings), Pillow (image processing)

## Development Philosophy

This project follows **Test-Driven Development (TDD)**:

1. Write tests first (or update existing tests)
2. Run tests frequently
3. Maintain >80% test coverage

## Essential Commands

### Quick Development Workflow

```bash
# Run single test (fastest iteration)
poetry run pytest tests/test_<module>.py::TestClass::test_name -v

# Run tests without coverage (fast)
poetry run poe test-fast

# Run all quality checks (before commits)
poetry run poe check-all
```

### All Available Commands

```bash
poe ingest        # Ingest transcripts into ChromaDB (System 1)
poe test          # Tests with coverage report
poe test-fast     # Tests without coverage (faster iteration)
poe lint          # Ruff linter
poe format        # Auto-format with Ruff
poe format-check  # Check formatting without changes
poe typecheck     # mypy type checking
poe check-all     # All checks: format + lint + typecheck + test
```

### Running the Agent

```bash
# Using example script
poetry run python examples/sample_critique.py path/to/image.jpg

# Direct module usage
poetry run python -m comp_critic.agent path/to/image.jpg

# In Python code
from comp_critic.agent import critique_image
result = critique_image("image.jpg")
```

## Architecture Deep Dive

### Multimodal Input Pattern (agent.py)

The agent accepts a specific multimodal message format required by GPT-4 vision:

```python
input_message = {
    "input": [
        {"type": "text", "text": "Critique this photo"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
}
```

Images are preprocessed via `encode_image_to_base64()` which:
- Resizes to max 2048px (reduces tokens while maintaining quality)
- Converts to JPEG with 85% quality
- Returns base64 data URI

### RAG Tool Pattern (tools.py)

The `composition_rag_tool` is defined using LangChain's `@tool` decorator:

```python
@tool
def composition_rag_tool(query: str) -> str:
    """Docstring becomes the tool description shown to the LLM."""
    # Load ChromaDB → similarity_search → format results
```

**Critical:** The tool's docstring is used by the LLM to understand when/how to invoke it. Be precise and instructive when editing tool descriptions.

### Configuration Management (config.py)

All config flows through `Config` class:
- Environment variables loaded from `.env` via `python-dotenv`
- `Config.validate()` checks required values (called before agent/ingestion runs)
- `Config.ensure_chroma_db_path()` creates DB directory if needed

**When adding config:**
1. Add to `.env.example`
2. Add as class variable in `Config`
3. Update validation if required

### ChromaDB Persistence Model

- **Collection name:** `landscape_photography_transcripts` (hardcoded in config)
- **Persistence:** Files stored in `./chroma_db/` (gitignored)
- **Embedding model:** `text-embedding-3-small` (OpenAI)
- **Loading pattern:** Tools/agent load from disk each invocation (no in-memory cache)

## Testing Standards

### Key Testing Patterns

**AAA Structure:** Arrange → Act → Assert

**Test Naming:** `test_<function>_<scenario>_<expected_result>`

**Fixtures (conftest.py):**
- `temp_dir` - Temporary directory for file operations
- `sample_transcripts_dir` - Pre-populated test transcripts
- `mock_openai_api_key` - Mocks API key in environment and Config
- `sample_image` - Creates test JPEG
- `mock_chroma_db_path` - Temporary ChromaDB location

**Critical Testing Pattern - Config Isolation:**

Use `monkeypatch` to override `Config` class attributes (not just env vars):

```python
def test_something(monkeypatch):
    from comp_critic import config as config_module
    monkeypatch.setattr(config_module.Config, "OPENAI_API_KEY", "test-key")
    # Config is loaded at import time, so env vars alone won't work
```

**Mocking External Dependencies:**

- Mock OpenAI API calls in all tests (use `pytest-mock`)
- Mock ChromaDB operations when testing non-storage logic
- Mock file I/O when not testing actual file operations

**Test Docstrings:**

Every test needs a docstring explaining *what* it verifies:

```python
def test_encode_image_resizes_large_images(temp_dir: Path) -> None:
    """Test that images >2048px are resized to reduce token usage."""
```

## Code Quality Standards

### Type Hints (Strict Enforcement)

- **Required:** All function parameters and return values
- Use `from typing import` for complex types
- `mypy` runs in strict mode (`disallow_untyped_defs = true`)
- Ignore missing imports for third-party libs (see pyproject.toml overrides)

Example:
```python
from pathlib import Path
from typing import Dict, Union

def critique_image(
    image_path: Union[str, Path],
    custom_prompt: str = "..."
) -> Dict[str, str]:
    """..."""
```

### Code Style

- **Ruff** enforces PEP 8, pyupgrade, flake8-bugbear, etc.
- Line length: 100 chars (not 88)
- Import sorting: automatic via Ruff
- Docstrings: Required for all public functions/classes

### Documentation Requirements

When adding features:
- Update function/class docstrings
- Update `.env.example` for new config
- Update `README.md` if user-facing
- Add usage examples in docstrings for complex functions

## Common Development Scenarios

### Adding a New Feature

1. **Plan:** Identify test files, decide on test cases
2. **Write tests:** In `tests/test_<module>.py`
3. **Implement:** In `src/comp_critic/<module>.py`
4. **Type hints:** Add comprehensive type annotations
5. **Quality check:** `poetry run poe check-all`
6. **Documentation:** Update docstrings, README if needed

### Modifying the Agent's Behavior

**Agent System Prompt:** `AGENT_SYSTEM_PROMPT` in `agent.py:19`

This prompt instructs the LLM on:
- What to analyze in images
- How to use the RAG tool
- How to structure output

**Important:** Changes here affect agent reasoning. Test thoroughly with real images.

### Changing ChromaDB Configuration

If modifying chunk size, embedding model, or collection name:

1. Update `config.py`
2. Delete `./chroma_db/` directory
3. Re-run `poetry run poe ingest`
4. Update tests that mock ChromaDB

**Warning:** ChromaDB schema changes require full re-ingestion.

### Bug Fixing Workflow

1. Write a test reproducing the bug
2. Verify test fails
3. Fix the bug
4. Verify test passes
5. Run `poetry run poe check-all`

## Pre-Commit Checklist

- [ ] `poetry run poe check-all` passes (all quality checks)
- [ ] New features have tests
- [ ] Test coverage >80% (check coverage report)
- [ ] Type hints on all new functions
- [ ] Docstrings updated
- [ ] `.env.example` updated if new config added

**One command to rule them all:** `poetry run poe check-all`

## Project-Specific Gotchas

### Config Import Timing

`Config` class attributes are set at module import time. In tests, use `monkeypatch.setattr()` on the `Config` class itself, not just environment variables:

```python
# ❌ Won't work (too late)
monkeypatch.setenv("OPENAI_API_KEY", "test")

# ✅ Correct
from comp_critic import config as config_module
monkeypatch.setattr(config_module.Config, "OPENAI_API_KEY", "test")
```

### Image Size vs Token Usage

Large images consume massive tokens with GPT-4 vision. The `encode_image_to_base64()` function resizes to 2048px max. Don't bypass this unless you understand token implications.

### ChromaDB Persistence

ChromaDB creates a `chroma.sqlite3` file. If you see weird search results after schema changes, delete `./chroma_db/` and re-ingest.

### Agent Verbose Mode

`AgentExecutor(verbose=True)` in `agent.py:146` - this prints agent reasoning to stdout. Useful for debugging, but noisy in tests.

## Resources

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/) - Tool-calling pattern
- [ChromaDB Docs](https://docs.trychroma.com/) - Vector store operations
- [GPT-4 Vision Guide](https://platform.openai.com/docs/guides/vision) - Multimodal input format
- [pytest](https://docs.pytest.org/) - Testing framework
- [Ruff](https://docs.astral.sh/ruff/) - Linting/formatting
- [mypy](https://mypy.readthedocs.io/) - Type checking

## Quick Reference

**File Structure:**
```
src/comp_critic/
├── config.py    # Config management (env vars, paths, validation)
├── ingest.py    # System 1: Load transcripts → ChromaDB
├── tools.py     # RAG tool definition (@tool decorator)
├── agent.py     # System 2: Multimodal agent + image encoding
tests/
├── conftest.py  # Shared fixtures (temp dirs, mocks, samples)
├── test_*.py    # Test modules (mirror src structure)
```

**Key Files to Check:**
- `PRD.md` - Product requirements and acceptance criteria
- `README.md` - User-facing usage and setup
- `.env.example` - Configuration template
- `pyproject.toml` - Dependencies and Poe tasks
