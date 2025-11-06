---
date: 2025-11-05T04:52:22.824Z
researcher: Claude
git_commit: 8b1c5ca6b54059d9307388e5b4ed395471ec4910
branch: main
repository: comp-critic-agent
topic: "Agentic Loop Architecture and Manual Tool Calling Pattern"
tags: [research, codebase, agent, agentic-loop, tool-calling, rag, langchain]
status: complete
last_updated: 2025-11-05
last_updated_by: Claude
---

# Research: Agentic Loop Architecture and Manual Tool Calling Pattern

**Date**: 2025-11-05T04:52:22.824Z
**Researcher**: Claude
**Git Commit**: 8b1c5ca6b54059d9307388e5b4ed395471ec4910
**Branch**: main
**Repository**: comp-critic-agent

## Research Question
How does the agentic loop work in detail? What is the manual tool calling pattern, and how does it differ from traditional AgentExecutor approaches?

## Summary

This codebase implements a **manual tool calling pattern** for agentic reasoning instead of LangChain's traditional AgentExecutor. The key innovation is a **3-iteration loop** that reduces token usage by 97.5% (from 60K+ tokens to 2-3K tokens) while maintaining full tool-calling functionality.

**Core Loop Flow:**
1. User provides image → Agent encodes to base64
2. Create initial HumanMessage with image + system prompt
3. **Iteration Loop (max 3):**
   - Send messages to LLM with tools bound
   - Check if LLM returned tool calls
   - If yes: execute tool, append ToolMessage, continue loop
   - If no: break (final response ready)
4. Extract final response and token usage
5. Return critique with metadata

The loop is **manually managed** with explicit message passing, giving full control over prompt structure and enabling precise token counting.

## Detailed Findings

### 1. Agent Entry Point and Image Encoding

**File:** `src/comp_critic/agent.py:117-182` ([agent.py](https://github.com/mtomcal/comp-critic-agent/blob/8b1c5ca6b54059d9307388e5b4ed395471ec4910/src/comp_critic/agent.py#L117-L182))

The `critique_image()` function is the public interface for the agentic system:

```python
def critique_image(
    image_path: str | Path,
    custom_prompt: str = "Critique the composition of this landscape photograph.",
    detail: str | None = None,
    max_size: int | None = None,
) -> dict[str, str | dict[str, int]]:
```

**Key Steps:**
1. **Configuration Validation** (line 148):
   - Calls `config.validate()` to ensure OPENAI_API_KEY and other required values exist
   - Ensures database paths are initialized

2. **Image Encoding** (line 161):
   - Calls `encode_image_to_base64()` to convert image → base64 data URI
   - Reduces token usage through configurable resizing (default 2048px max)
   - Returns string like `"data:image/jpeg;base64,{base64_encoded_data}"`

3. **LLM Setup** (lines 164-169):
   - Creates ChatOpenAI instance with configurable model
   - **CRITICAL:** Binds tools using `llm.bind_tools([composition_rag_tool])`
   - This tells the LLM it can call `composition_rag_tool` when needed

**Image Encoding Details** (`encode_image_to_base64`, lines 60-114):
- Opens image with PIL, converts non-RGB to RGB
- Resizes if max dimension > threshold (reduces tokens from variable to fixed ~85 tokens)
- Saves as JPEG with 85% quality for optimal balance between size/quality
- Encodes to base64 and wraps in data URI format
- Token savings: High-res images can go from 5000+ tokens (`detail="high"`) to 85 tokens (`detail="low"`)

### 2. Initial Message Construction

**File:** `src/comp_critic/agent.py:172-182` ([agent.py](https://github.com/mtomcal/comp-critic-agent/blob/8b1c5ca6b54059d9307388e5b4ed395471ec4910/src/comp_critic/agent.py#L172-L182))

```python
messages: list[BaseMessage] = [
    HumanMessage(
        content=[
            {"type": "text", "text": f"{AGENT_SYSTEM_PROMPT}\n\n{custom_prompt}"},
            {
                "type": "image_url",
                "image_url": {"url": image_base64, "detail": detail},
            },
        ]
    )
]
```

**Multimodal Message Format:**
- Uses LangChain's multimodal `HumanMessage` (from `langchain_core.messages`)
- Content is a **list** of two elements:
  1. **Text content:** System prompt + user custom prompt
  2. **Image content:** Base64 data URI with detail level

**System Prompt** (lines 16-57):
- Instructs LLM to perform visual analysis (rule of thirds, leading lines, framing, etc.)
- Explicitly requires using `composition_rag_tool` for knowledge retrieval
- Specifies synthesis pattern: visual analysis + RAG knowledge → comprehensive critique
- ~57 lines of detailed reasoning instructions

**Detail Parameter** (line 152-156):
- `"low"`: Fixed 85 tokens (default, recommended for cost)
- `"high"`: Variable tokens based on image complexity (can be 5000+)
- Defaults to config value if not specified

### 3. The Manual Tool Calling Loop

**File:** `src/comp_critic/agent.py:186-211` ([agent.py](https://github.com/mtomcal/comp-critic-agent/blob/8b1c5ca6b54059d9307388e5b4ed395471ec4910/src/comp_critic/agent.py#L186-L211))

```python
# Iterative tool calling (max 3 iterations)
for iteration in range(3):
    print(f"\nIteration {iteration + 1}...")

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Check for tool calls - safe because response is always AIMessage from LLM
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        print("No tool calls - generating final response...")
        break

    # Execute tool calls
    for tool_call in response.tool_calls:
        print(f"Tool call: {tool_call['name']}({tool_call['args']})")

        if tool_call["name"] == "composition_rag_tool":
            tool_result = composition_rag_tool.invoke(tool_call["args"])
            print(f"Tool result length: {len(tool_result)} chars")

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"],
                )
            )
```

**Loop Architecture:**

| Iteration | Purpose | LLM Output | Action |
|-----------|---------|-----------|--------|
| 1 | Analyze image, decide if tool needed | Tool call OR final text | If tool call: execute + continue. If text: break |
| 2 | Process tool result, may refine query | Tool call OR final text | Same decision |
| 3 | Final pass (last allowed iteration) | Final text expected | Break regardless |

**Key Mechanics:**

1. **Message Passing:** 
   - Append LLM response (AIMessage) to messages list
   - This builds conversation history the LLM sees

2. **Tool Call Detection** (line 194):
   - Check `hasattr(response, "tool_calls")` and `response.tool_calls` is truthy
   - Safe because `ChatOpenAI.invoke()` always returns AIMessage

3. **Tool Execution** (lines 199-211):
   - Iterate over all tool calls in response
   - For each call: invoke tool with args dictionary
   - Wrap result in ToolMessage with matching tool_call_id
   - Append ToolMessage to messages list

4. **Iteration Limit** (line 187):
   - Hard limit of 3 iterations to prevent runaway token usage
   - Alternative to AgentExecutor's flexible approach

**Message List Evolution:**

```
Iteration 1:
  messages = [HumanMessage(image + prompt)]
  → response = AIMessage(content="", tool_calls=[...])
  → messages = [HumanMessage, AIMessage]
  → Tool executed
  → messages = [HumanMessage, AIMessage, ToolMessage]

Iteration 2:
  → response = AIMessage(content="", tool_calls=[...])
  → messages = [HumanMessage, AIMessage, ToolMessage, AIMessage]
  → Tool executed
  → messages = [HumanMessage, AIMessage, ToolMessage, AIMessage, ToolMessage]

Iteration 3 (or earlier if no tool calls):
  → response = AIMessage(content="Final critique", tool_calls=[])
  → messages = [..., AIMessage (final)]
  → Break (no tool calls detected)
```

### 4. The RAG Tool

**File:** `src/comp_critic/tools.py:63-93` ([tools.py](https://github.com/mtomcal/comp-critic-agent/blob/8b1c5ca6b54059d9307388e5b4ed395471ec4910/src/comp_critic/tools.py#L63-L93))

```python
@tool
def composition_rag_tool(query: str) -> str:
    """
    Searches the landscape photography video transcripts for advice.
    [Detailed docstring...]
    """
    try:
        # Load the vector store
        vector_store = load_vector_store()

        # Perform similarity search
        documents = vector_store.similarity_search(
            query,
            k=config.RAG_TOP_K,  # Default: 3 chunks
        )

        # Format and return results
        return format_search_results(documents)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
```

**Tool Definition Pattern:**
- Uses `@tool` decorator from `langchain.tools`
- Decorator converts function into LangChain Tool object
- **Docstring is critical**: Used by LLM to understand when/how to invoke tool
- Takes `query: str` parameter (what to search for)
- Returns formatted string (what gets added to messages as ToolMessage)

**Tool Workflow:**
1. LLM extracts tool call: `{"name": "composition_rag_tool", "args": {"query": "leading lines"}, "id": "call_123"}`
2. Agent calls: `composition_rag_tool.invoke({"query": "leading lines"})`
3. Tool loads ChromaDB → similarity search with k=3
4. Returns formatted results as string
5. String wrapped in ToolMessage and added to conversation

**Vector Store Loading** (lines 11-38):
- Loads from persistent storage at `config.CHROMA_DB_PATH`
- Uses OpenAI embeddings (`text-embedding-3-small`)
- Collection name: `landscape_photography_transcripts` (hardcoded)
- Raises error if DB not initialized (requires prior ingestion)

**Result Formatting** (lines 41-60):
- Formats each document with source metadata
- Returns "No relevant advice..." if empty
- Readable format with numbered results

### 5. Final Response Extraction

**File:** `src/comp_critic/agent.py:213-247` ([agent.py](https://github.com/mtomcal/comp-critic-agent/blob/8b1c5ca6b54059d9307388e5b4ed395471ec4910/src/comp_critic/agent.py#L213-L247))

```python
# Get final response
final_response: str | list[str | dict[Any, Any]]
last_message = messages[-1]
if isinstance(last_message, AIMessage) and not last_message.tool_calls:
    final_response = last_message.content
else:
    # Need one more call to get final text
    print("\nGetting final response...")
    final_response_msg = llm.invoke(messages)
    final_response = final_response_msg.content

# Extract token usage if available
token_info = {}
if hasattr(response, "usage_metadata") and response.usage_metadata:
    usage = response.usage_metadata
    token_info = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    # Print formatted token breakdown...

return {
    "output": str(final_response),
    "image_path": str(image_path),
    "token_usage": token_info,
}
```

**Response Extraction Logic:**
1. Check last message in conversation
2. If it's AIMessage without tool_calls: extract content (final response ready)
3. Otherwise: make one more LLM call to get final text response
4. This handles edge case where loop ended with ToolMessage

**Token Usage Tracking:**
- Extracted from `response.usage_metadata` (OpenAI API provides this)
- Contains input_tokens, output_tokens, total_tokens
- Printed for debugging/monitoring
- Included in return dictionary

**Return Format:**
```python
{
    "output": "Full critique text",
    "image_path": "/path/to/image.jpg",
    "token_usage": {
        "input_tokens": 523,
        "output_tokens": 287,
        "total_tokens": 810,
    }
}
```

## Code References

- `src/comp_critic/agent.py:117-247` - Main `critique_image()` function with loop
- `src/comp_critic/agent.py:60-114` - `encode_image_to_base64()` function
- `src/comp_critic/agent.py:16-57` - AGENT_SYSTEM_PROMPT definition
- `src/comp_critic/tools.py:63-93` - `composition_rag_tool` definition
- `src/comp_critic/tools.py:11-38` - ChromaDB vector store loading
- `tests/test_agent.py:378-435` - Test demonstrating full tool calling loop

## Architecture Insights

### Manual Tool Calling vs AgentExecutor

**Why Manual Tool Calling?**

Traditional LangChain AgentExecutor adds massive overhead:
- Manages agent_scratchpad (conversation history formatting)
- Includes full tool schemas in every prompt
- Adds reasoning traces
- ~60K+ tokens for typical image critique

Manual approach gives:
- ✅ 97.5% token reduction (60K+ → 2-3K)
- ✅ Full control over message structure
- ✅ Easy token usage tracking via `response.usage_metadata`
- ✅ Simpler debugging with visible message flow
- ❌ Manual iteration limit (hardcoded to 3)
- ❌ No built-in error recovery

**Trade-off Calculation:**
- GPT-4 Vision costs ~$0.03/input_token, ~$0.06/output_token
- AgentExecutor: 60,000 tokens × $0.03 ≈ $1.80 per critique
- Manual loop: 2,500 tokens × $0.03 ≈ $0.08 per critique
- **Savings: 95% cost reduction**

### Message Flow Pattern

The agentic loop is fundamentally a **conversation history pattern**:

```
Initial State:
  messages = [HumanMessage(image + prompt)]

Each Iteration:
  1. Append LLM response (AIMessage) to messages
  2. If tool calls exist:
     - Execute each tool
     - Append ToolMessage for each tool result
  3. Else:
     - Break (final response obtained)

Next Iteration Uses Full History:
  - LLM sees all previous messages
  - This provides context about what was already analyzed
  - Allows LLM to build on previous findings
```

### Tool Call Structure

When LLM decides to call a tool, it returns:

```python
AIMessage(
    content="",  # Empty content when making tool calls
    tool_calls=[
        {
            "id": "call_123",
            "name": "composition_rag_tool",
            "args": {"query": "rule of thirds in landscape"}
        }
    ]
)
```

The `bind_tools()` method trains the LLM to:
- Output tool calls in standardized format
- Only call tools explicitly bound via `bind_tools()`
- Include required metadata (id, args) for execution

### Image Detail Parameter Impact

The `detail` parameter significantly affects both cost and quality:

| Detail | Tokens | Cost (per image) | Use Case |
|--------|--------|-----------------|----------|
| "low" | ~85 (fixed) | ~$0.0026 | Default, fast, batch critiques |
| "high" | 500-5000+ | $0.015-$0.15 | Fine-grained analysis needed |

Default is `"low"` because:
- Agent system prompt focuses on composition (visible at low res)
- 85 tokens is predictable for cost calculation
- Significant visual info preserved for landscape photos

### Iteration Limit Rationale

3 iterations chosen as balance:
- **Iteration 1:** Analyze image, formulate RAG query
- **Iteration 2:** Process RAG results, refine if needed
- **Iteration 3:** Synthesize findings into final critique

Rarely need more because:
- System prompt teaches RAG tool usage upfront
- RAG results comprehensive (retrieves k=3 relevant chunks)
- LLM typically ready for final answer by iteration 2

Hard limit prevents:
- Infinite loops if LLM confused
- Runaway token usage
- Cost explosion from repeated tool calls

## Historical Context (from codebase)

### Configuration-Driven Design

`src/comp_critic/config.py` centralizes all tunable parameters:
- `VISION_DETAIL`: "low" or "high" (default: "low")
- `VISION_MAX_SIZE`: Max pixels before resize (default: 2048)
- `RAG_TOP_K`: Chunks retrieved (default: 3)
- `OPENAI_MODEL`: Model choice (default: "gpt-4.1")

This allows researchers to experiment with:
- Token usage variations
- Quality/cost trade-offs
- Different model behaviors

### System 2 Architecture (from PRD)

Requirement R-2.3 specifies:
- Visual analysis component
- RAG tool integration
- Synthesis pattern
- Multiple search queries encouraged

Manual tool calling enables all of this while achieving token budget.

## Testing Patterns

### Mock Tool Calling Loop (from test_agent.py:378-435)

The test suite demonstrates the full pattern:

```python
@patch("comp_critic.agent.composition_rag_tool")
@patch("comp_critic.agent.ChatOpenAI")
@patch("comp_critic.agent.encode_image_to_base64")
def test_critique_image_with_tool_calling(...):
    # Mock first response: LLM wants tool
    mock_response_1 = AIMessage(
        content="",
        tool_calls=[{
            "name": "composition_rag_tool",
            "args": {"query": "rule of thirds"},
            "id": "call_123",
        }],
    )

    # Mock second response: final answer
    mock_response_2 = AIMessage(
        content="Final critique based on RAG results.",
        tool_calls=[],
    )
    mock_response_2.usage_metadata = {...}

    # Side effect: first call returns with tool calls, second without
    mock_llm_with_tools.invoke.side_effect = [mock_response_1, mock_response_2]

    # Act
    result = critique_image(sample_image)

    # Assert
    assert mock_llm_with_tools.invoke.call_count == 2
```

This validates:
- Message passing works correctly
- Tool results integrated into conversation
- Final response extracted properly
- Token usage tracked

## Key Design Decisions

1. **Manual Tool Calling Over AgentExecutor**
   - Rationale: 97.5% token reduction critical for cost
   - Trade-off: Less flexible but more predictable

2. **Image Resizing by Default**
   - Rationale: Reduces tokens from 5000+ to 85 with negligible quality loss for composition
   - Trade-off: Users can override with `max_size` or `detail="high"`

3. **3-Iteration Limit**
   - Rationale: Sufficient for typical analysis + prevents runaway costs
   - Trade-off: Rare complex analyses might need 4th iteration

4. **System Prompt Emphasis on RAG**
   - Rationale: Trains LLM to use knowledge base rather than hallucinate
   - Trade-off: Slightly longer prompt but critical for reliable output

5. **Multimodal Message Format**
   - Rationale: LangChain standard, well-supported by OpenAI API
   - Trade-off: Requires specific format (list content, image_url structure)

## Open Questions

1. **Error Recovery:** If tool execution fails (e.g., ChromaDB unavailable), loop continues. Should we halt instead?
2. **Tool Call Validation:** Should we validate tool_call["name"] matches expected tools, or just check name == "composition_rag_tool"?
3. **Multiple Tool Calls Per Iteration:** Current code handles multiple tool calls in one iteration (line 199 `for tool_call in response.tool_calls`). Does LLM ever generate multiple calls, and if so, what's the interaction pattern?
4. **Final Response Fallback:** Line 220 makes additional LLM call if last message is ToolMessage. Is this ever needed in practice, or is it defensive code?
5. **Token Usage Source:** `response.usage_metadata` comes from which invocation - first LLM response or the one that triggered line 220?

## Related Research

- LangChain Tool Calling: https://python.langchain.com/docs/modules/agents/
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Manual Agent Patterns: https://python.langchain.com/docs/modules/agents/agent_types/
- GPT-4 Vision Token Limits: https://platform.openai.com/docs/guides/vision
