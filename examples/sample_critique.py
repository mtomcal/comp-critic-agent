#!/usr/bin/env python3
"""
Sample script demonstrating how to use the Comp Critic Agent.

This script shows how to:
1. Load and critique a single image
2. Customize the critique prompt
3. Handle the agent's output

Usage:
    python examples/sample_critique.py path/to/your/image.jpg
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comp_critic.agent import critique_image


def main() -> None:
    """Run the sample critique workflow."""
    # Check for command-line argument
    if len(sys.argv) < 2:
        print("Usage: python examples/sample_critique.py <image_path>")
        print("\nExample:")
        print("    python examples/sample_critique.py photos/landscape.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    # Validate image exists
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("Comp Critic Agent - Landscape Photography Composition Critique")
    print("=" * 70)
    print(f"\nAnalyzing: {image_path}")
    print("\nThis may take a moment as the agent:")
    print("  1. Analyzes the visual composition")
    print("  2. Searches the knowledge base for relevant advice")
    print("  3. Synthesizes a comprehensive critique")
    print("\n" + "-" * 70)

    try:
        # Example 1: Basic critique with default prompt
        result = critique_image(image_path)

        print("\n[CRITIQUE]")
        print("-" * 70)
        print(result["output"])
        print("-" * 70)

        # Example 2: Custom prompt for specific feedback
        # Uncomment the following to try a custom prompt:
        """
        print("\n\nNow requesting focused feedback on lighting...")
        custom_result = critique_image(
            image_path,
            custom_prompt="Focus your critique specifically on the lighting and "
                         "tonal balance in this landscape photograph."
        )

        print("\n[LIGHTING-FOCUSED CRITIQUE]")
        print("-" * 70)
        print(custom_result["output"])
        print("-" * 70)
        """

        print("\n" + "=" * 70)
        print("Critique complete!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nPlease ensure:")
        print("  1. You have set OPENAI_API_KEY in your .env file")
        print("  2. You have run the ingestion pipeline: poe ingest")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
