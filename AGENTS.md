# AI Agent Instructions


<!-- AI-COMMANDS:START -->
## Custom AI Command Workflows

This project has access to reusable AI command workflows from your dotfiles.

### How to Execute Commands

When the user requests any of these workflows, use the Bash tool to retrieve the command instructions:

```bash
ai-commands get <command-name>
```

The command will output the complete workflow instructions. Read the output carefully and follow all instructions exactly as written.

### Available Commands

- `create_plan`
- `implement_plan`
- `research_codebase`
- `save-session` - Create a detailed summary of the conversation and save it to ./sessions/
- `validate_plan`


### Usage Example

When user says "save the session" or "create a summary":
1. Run: `ai-commands get save-session`
2. Read the complete output
3. Follow all instructions in the returned content exactly

When user says "create a plan":
1. Run: `ai-commands get create-plan`
2. Follow the returned workflow instructions

### Command Location

All commands are stored in: `~/dotfiles/claude/commands/`

You can also run `ai-commands list` to see all available commands.

<!-- AI-COMMANDS:END -->

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
   - Uses **manual tool calling** with `llm.bind_tools()` (NOT AgentExecutor) for token efficiency
   - Token optimized: ~2-3K tokens per critique vs 60K+ with traditional agent frameworks

**Key Dependencies:** LangChain (tool binding, messages), ChromaDB (vector DB), OpenAI (GPT-4 + embeddings), Pillow (image processing)

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

The agent uses LangChain's multimodal message format with HumanMessage:

```python
messages = [
    HumanMessage(
        content=[
            {"type": "text", "text": f"{AGENT_SYSTEM_PROMPT}\n\n{custom_prompt}"},
            {
                "type": "image_url",
                "image_url": {"url": image_base64, "detail": detail}  # "low" or "high"
            },
        ]
    )
]
```

Images are preprocessed via `encode_image_to_base64()` which:
- Resizes to configurable max size (default 2048px) to reduce token usage
- Converts to JPEG with 85% quality for optimal balance
- Returns base64 data URI
- **Critical**: Vision API `detail` parameter controls token usage:
  - `detail="low"`: Fixed 85 tokens per image (fast, cheap)
  - `detail="high"`: Variable tokens based on image complexity (slower, expensive)

### RAG Tool Pattern (tools.py)

The `composition_rag_tool` is defined using LangChain's `@tool` decorator:

```python
@tool
def composition_rag_tool(query: str) -> str:
    """Docstring becomes the tool description shown to the LLM."""
    # Load ChromaDB → similarity_search → format results
```

**Critical:** The tool's docstring is used by the LLM to understand when/how to invoke it. Be precise and instructive when editing tool descriptions.

### Manual Tool Calling Pattern (agent.py)

**IMPORTANT: This project does NOT use AgentExecutor.** The previous implementation used LangChain's `AgentExecutor`, which caused massive token overhead (~60K+ tokens). The refactored implementation uses manual tool calling for 97.5% token reduction.

**How it works:**

```python
# 1. Bind tools to LLM
llm_with_tools = llm.bind_tools([composition_rag_tool])

# 2. Iterative tool calling loop (max 3 iterations)
messages: list[BaseMessage] = [HumanMessage(...)]

for iteration in range(3):
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Check if LLM wants to call tools
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        break  # No tool calls, we have final response

    # Execute each tool call
    for tool_call in response.tool_calls:
        tool_result = composition_rag_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

# 3. Extract final response from last message
final_response = messages[-1].content if isinstance(messages[-1], AIMessage) else llm.invoke(messages).content
```

**Why manual tool calling?**
- AgentExecutor adds massive overhead: agent_scratchpad, full tool schemas, conversation history
- Manual approach gives full control over message structure
- Allows token usage tracking via `response.usage_metadata`
- Simpler debugging with visible message flow

**Trade-offs:**
- ✅ 97.5% token reduction (~2-3K vs 60K+ tokens)
- ✅ Full control over prompt structure
- ✅ Easy token usage monitoring
- ❌ Manual iteration limit management (3 iterations hardcoded)
- ❌ No built-in error recovery from AgentExecutor

### Configuration Management (config.py)

All config flows through `Config` class:
- Environment variables loaded from `.env` via `python-dotenv`
- `Config.validate()` checks required values (called before agent/ingestion runs)
- `Config.ensure_chroma_db_path()` creates DB directory if needed

**Key configuration options:**
- `OPENAI_MODEL`: GPT model to use (default: "gpt-4.1")
- `VISION_DETAIL`: "low" (85 tokens) or "high" (variable tokens) - default: "low"
- `VISION_MAX_SIZE`: Max image dimension in pixels (default: 2048)
- `RAG_TOP_K`: Number of chunks to retrieve (default: 3)
- `CHUNK_SIZE`: Characters per chunk during ingestion (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

**When adding config:**
1. Add to `.env.example` with documentation
2. Add as class variable in `Config` with `os.getenv()` for user-configurable options
3. Update validation if required
4. Update README.md Configuration section

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

**Testing Manual Tool Calling Pattern:**

When testing the agent's manual tool calling, mock both the LLM and tool responses:

```python
@patch("comp_critic.agent.composition_rag_tool")
@patch("comp_critic.agent.ChatOpenAI")
def test_critique_with_tool_calling(mock_llm_class, mock_tool, sample_image):
    # Mock first response: LLM wants to call tool
    mock_response_1 = AIMessage(
        content="",
        tool_calls=[{
            "name": "composition_rag_tool",
            "args": {"query": "rule of thirds"},
            "id": "call_123"
        }]
    )

    # Mock second response: LLM provides final answer
    mock_response_2 = AIMessage(content="Final critique", tool_calls=[])
    mock_response_2.usage_metadata = {"total_tokens": 300}

    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.invoke.side_effect = [mock_response_1, mock_response_2]

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm_class.return_value = mock_llm

    result = critique_image(sample_image)
    assert mock_llm_with_tools.invoke.call_count == 2  # Two iterations
```

**Test Docstrings:**

Every test needs a docstring explaining *what* it verifies:

```python
def test_encode_image_resizes_large_images(temp_dir: Path) -> None:
    """Test that images >2048px are resized to reduce token usage."""
```

## Code Quality Standards

### Type Hints (Strict Enforcement)

- **Required:** All function parameters and return values
- Use **Python 3.10+ syntax** for type unions: `str | Path` instead of `Union[str, Path]`
- Use **lowercase generics**: `dict[str, str]` instead of `Dict[str, str]`
- `mypy` runs in strict mode (`disallow_untyped_defs = true`)
- Ignore missing imports for third-party libs (see pyproject.toml overrides)

Example:
```python
from pathlib import Path

def critique_image(
    image_path: str | Path,
    custom_prompt: str = "...",
    detail: str | None = None,
) -> dict[str, str | dict[str, int]]:
    """Returns dict with 'output', 'image_path', and 'token_usage' keys."""
```

**Important:** Only import from `typing` for types not available as builtins (e.g., `Any`, `Callable`)

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

**Token Optimization Considerations:**
- Keep the system prompt concise - every character counts
- Use `detail="low"` by default unless high-res analysis is needed
- Consider reducing `RAG_TOP_K` if context size is an issue
- Monitor token usage via the returned `token_usage` dict
- Max 3 tool calling iterations to prevent runaway token usage

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

### Vision API Token Usage

**Critical for cost control:**
- `detail="low"`: Always 85 tokens regardless of image size (recommended default)
- `detail="high"`: Variable tokens based on image complexity (can be thousands)
- The `encode_image_to_base64()` function resizes to configurable max size (default 2048px)
- Don't bypass resizing unless you understand token/cost implications

**Why this matters:** At `detail="high"`, a single high-res image can consume 5,000+ tokens. At scale, this becomes expensive quickly.

### ChromaDB Persistence

ChromaDB creates a `chroma.sqlite3` file. If you see weird search results after schema changes, delete `./chroma_db/` and re-ingest.

### Manual Tool Calling Debug Output

The agent prints debug messages during execution (e.g., "Iteration 1...", "Tool call: composition_rag_tool(...)"). This is helpful for understanding the tool calling flow but can be noisy. To disable, remove the `print()` statements in `critique_image()` function.

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
