"""System 2: The Multimodal RAG Agent for compositional critique."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from PIL import Image

from comp_critic.config import config
from comp_critic.tools import composition_rag_tool


# Agent system prompt (R-2.3 from PRD)
AGENT_SYSTEM_PROMPT = """You are an expert landscape photography composition critic. Your role is to provide detailed, constructive feedback on landscape photographs by combining your visual analysis with expert advice from a curated knowledge base.

**Your Process:**

1. **Visual Analysis**: First, carefully examine the provided image and identify key compositional elements:
   - Rule of thirds and grid placement
   - Subject placement and focal points
   - Leading lines and visual flow
   - Foreground, midground, and background elements
   - Framing and negative space
   - Light, exposure, and tonal balance
   - Color harmony and contrast
   - Any special techniques (e.g., long exposure, HDR, panorama)

2. **Knowledge Retrieval**: Based on your visual analysis, formulate specific search queries to find relevant expert advice. Use the `composition_rag_tool` to search for:
   - Techniques you've identified in the image
   - Potential improvements or common pitfalls
   - Expert tips related to the scene type or conditions

3. **Synthesis**: Combine your own visual understanding with the retrieved expert advice to create a comprehensive critique that:
   - Acknowledges what works well in the composition
   - Identifies areas for improvement
   - References specific expert advice from the knowledge base
   - Provides actionable recommendations

**IMPORTANT REQUIREMENTS:**

- You MUST use the `composition_rag_tool` to search the knowledge base. Never rely solely on general knowledge.
- Make multiple searches if needed to cover different compositional aspects.
- Always cite when you're referencing retrieved expert advice (e.g., "According to the transcripts...")
- If the knowledge base has no relevant information for a particular element, state this explicitly but still provide your visual analysis.
- Be constructive, specific, and actionable in your feedback.

Your critique should be well-structured, professional, and educational."""


def encode_image_to_base64(image_path: Union[str, Path], max_size: int = 2048) -> str:
    """
    Encode an image file to a base64 data URI for multimodal LLM input.

    Resizes large images to reduce token usage while maintaining quality
    for visual analysis. Images are resized to fit within max_size while
    maintaining aspect ratio.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) in pixels (default: 2048)

    Returns:
        Base64-encoded data URI string

    Raises:
        FileNotFoundError: If the image file doesn't exist
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Open and convert image to RGB if needed
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

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

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Encode to base64 with quality optimization
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85, optimize=True)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/jpeg;base64,{img_base64}"


def create_critique_agent() -> AgentExecutor:
    """
    Create the multimodal RAG agent with the composition critique tool.

    Returns:
        AgentExecutor configured for image critique

    Raises:
        ValueError: If OPENAI_API_KEY is not configured
    """
    # Validate configuration
    config.validate()

    # Initialize the multimodal LLM (R-2.1)
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.7,  # Slightly creative but grounded
    )

    # Define tools (R-2.4)
    tools = [composition_rag_tool]

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    return agent_executor


def critique_image(
    image_path: Union[str, Path],
    custom_prompt: str = "Critique the composition of this landscape photograph.",
) -> Dict[str, str]:
    """
    Analyze and critique a landscape photograph using the multimodal RAG agent.

    Args:
        image_path: Path to the image file to critique
        custom_prompt: Optional custom prompt (default asks for composition critique)

    Returns:
        Dictionary containing:
            - 'output': The agent's final critique text
            - 'image_path': Path to the analyzed image

    Raises:
        FileNotFoundError: If the image doesn't exist
        ValueError: If configuration is invalid
    """
    # Encode the image
    image_base64 = encode_image_to_base64(image_path)

    # Create the agent
    agent_executor = create_critique_agent()

    # Prepare the multimodal input (R-2.2)
    input_message = {
        "input": [
            {"type": "text", "text": custom_prompt},
            {"type": "image_url", "image_url": {"url": image_base64}},
        ]
    }

    # Execute the agent (R-2.5)
    result = agent_executor.invoke(input_message)

    return {
        "output": result["output"],
        "image_path": str(image_path),
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
