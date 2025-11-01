"""System 2: The Multimodal RAG Agent for compositional critique."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from PIL import Image

from comp_critic.config import config
from comp_critic.tools import composition_rag_tool

# Agent system prompt (R-2.3 from PRD)
AGENT_SYSTEM_PROMPT = """You are an expert landscape photography composition critic.
Your role is to provide detailed, constructive feedback on landscape photographs by
combining your visual analysis with expert advice from a curated knowledge base.

**Your Process:**

1. **Visual Analysis**: First, carefully examine the provided image and identify key
compositional elements:
   - Rule of thirds and grid placement
   - Subject placement and focal points
   - Leading lines and visual flow
   - Foreground, midground, and background elements
   - Framing and negative space
   - Light, exposure, and tonal balance
   - Color harmony and contrast
   - Any special techniques (e.g., long exposure, HDR, panorama)

2. **Knowledge Retrieval**: Based on your visual analysis, formulate specific search
queries to find relevant expert advice. Use the `composition_rag_tool` to search for:
   - Techniques you've identified in the image
   - Potential improvements or common pitfalls
   - Expert tips related to the scene type or conditions

3. **Synthesis**: Combine your own visual understanding with the retrieved expert
advice to create a comprehensive critique that:
   - Acknowledges what works well in the composition
   - Identifies areas for improvement
   - References specific expert advice from the knowledge base
   - Provides actionable recommendations

**IMPORTANT REQUIREMENTS:**

- You MUST use the `composition_rag_tool` to search the knowledge base. Never rely
solely on general knowledge.
- Make multiple searches if needed to cover different compositional aspects.
- Always cite when you're referencing retrieved expert advice (e.g., "According to
the transcripts...")
- If the knowledge base has no relevant information for a particular element, state
this explicitly but still provide your visual analysis.
- Be constructive, specific, and actionable in your feedback.

Your critique should be well-structured, professional, and educational."""


def encode_image_to_base64(
    image_path: str | Path,
    max_size: int | None = None,
) -> str:
    """
    Encode an image file to a base64 data URI for multimodal LLM input.

    Resizes large images to reduce token usage while maintaining quality
    for visual analysis. Images are resized to fit within max_size while
    maintaining aspect ratio.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) in pixels.
                  If None, uses config.VISION_MAX_SIZE (default: None)

    Returns:
        Base64-encoded data URI string

    Raises:
        FileNotFoundError: If the image file doesn't exist
    """
    if max_size is None:
        max_size = config.VISION_MAX_SIZE
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Open and convert image to RGB if needed
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")  # type: ignore[assignment]

    # Resize image if it's too large to avoid token limits
    # This significantly reduces token usage while maintaining visual quality
    width, height = image.size
    if max(width, height) > max_size:
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # type: ignore[assignment]

    # Encode to base64 with quality optimization
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85, optimize=True)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/jpeg;base64,{img_base64}"


def critique_image(
    image_path: str | Path,
    custom_prompt: str = "Critique the composition of this landscape photograph.",
    detail: str | None = None,
    max_size: int | None = None,
) -> dict[str, str | dict[str, int]]:
    """
    Analyze and critique a landscape photograph using a multimodal RAG agent.

    This implementation uses manual tool calling instead of LangChain's AgentExecutor
    to minimize token overhead. The simplified approach reduces token usage by ~97.5%
    compared to AgentExecutor while maintaining full functionality.

    Args:
        image_path: Path to the image file to critique
        custom_prompt: Optional custom prompt (default asks for composition critique)
        detail: Vision API detail level - "low" (85 tokens) or "high" (variable tokens).
                If None, uses config.VISION_DETAIL (default: None)
        max_size: Maximum image dimension in pixels. If None, uses config.VISION_MAX_SIZE.

    Returns:
        Dictionary containing:
            - 'output': The agent's final critique text
            - 'image_path': Path to the analyzed image
            - 'token_usage': Dictionary with input_tokens, output_tokens, total_tokens

    Raises:
        FileNotFoundError: If the image doesn't exist
        ValueError: If configuration is invalid or detail not "low" or "high"
    """
    # Validate configuration
    config.validate()

    # Use config defaults if not specified
    if detail is None:
        detail = config.VISION_DETAIL

    # Validate detail parameter
    if detail not in ("low", "high"):
        raise ValueError(f"detail must be 'low' or 'high', got: {detail}")

    print(f"\n[DEBUG] Using detail='{detail}' for image encoding")

    # Encode the image
    image_base64 = encode_image_to_base64(image_path, max_size=max_size)

    # Create the LLM with tool binding
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,  # type: ignore[call-arg]
        temperature=0.7,
    )
    llm_with_tools = llm.bind_tools([composition_rag_tool])

    # Initial message with image
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

    print("\nCalling LLM with vision...")

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
        print("\n" + "=" * 60)
        print("TOKEN USAGE BREAKDOWN")
        print("=" * 60)
        print(f"Input tokens:  {token_info['input_tokens']:,}")
        print(f"Output tokens: {token_info['output_tokens']:,}")
        print(f"Total tokens:  {token_info['total_tokens']:,}")
        print(f"Detail level:  {detail}")
        print(f"Image size:    {max_size or config.VISION_MAX_SIZE}px max")
        print("=" * 60 + "\n")

    return {
        "output": str(final_response),
        "image_path": str(image_path),
        "token_usage": token_info,
    }


def main() -> None:
    """
    Entry point for testing the agent with a sample image.

    This is primarily for development/testing purposes.
    In production, use the critique_image function directly.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m comp_critic.agent <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Analyzing image...")
    result = critique_image(image_path)

    print("\n" + "=" * 60)
    print("COMPOSITION CRITIQUE")
    print("=" * 60)
    print(f"Image: {result['image_path']}")
    print("-" * 60)
    print(result["output"])
    print("=" * 60)


if __name__ == "__main__":
    main()
