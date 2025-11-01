# AI Assistant Guidelines for Comp Critic Agent

This document provides guidelines for AI assistants (Claude, Copilot, etc.) when helping with this project.

## Development Philosophy

This project follows **Test-Driven Development (TDD)** principles. When making changes:

1. **Write tests first** (or update existing tests) before implementing features
2. **Run tests frequently** to ensure changes don't break existing functionality
3. **Maintain high test coverage** (aim for >80%)

## Before Making Changes

### 1. Understand the Context

- Read the relevant code and existing tests
- Check the PRD.md for requirements and specifications
- Review the README.md for architecture and project structure

### 2. Plan Your Changes

- Identify which tests need to be written or updated
- Consider edge cases and error conditions
- Think about how changes affect existing functionality

## Development Workflow

### Test-Driven Development Process

1. **Write the test first**
   ```bash
   # Write test in appropriate file (tests/test_*.py)
   poetry run pytest tests/test_<module>.py::TestClass::test_new_feature -v
   ```

2. **Implement the feature**
   - Make the minimum changes needed to pass the test
   - Follow existing code patterns and style

3. **Run all tests**
   ```bash
   poetry run poe test-fast
   ```

4. **Run quality checks**
   ```bash
   poetry run poe check-all
   ```

### Quality Assurance Commands

Always run these before considering work complete:

```bash
# Run all quality checks at once (recommended)
poetry run poe check-all

# Or run individually:
poetry run poe lint          # Check code quality with Ruff
poetry run poe format-check  # Verify code formatting
poetry run poe typecheck     # Run mypy type checking
poetry run poe test          # Run tests with coverage
```

### Individual Check Commands

```bash
# Linting
poetry run poe lint

# Format checking (without modifying files)
poetry run poe format-check

# Auto-format code
poetry run poe format

# Type checking
poetry run poe typecheck

# Tests with coverage
poetry run poe test

# Tests without coverage (faster for iteration)
poetry run poe test-fast
```

## Code Quality Standards

### Type Hints

- **Always use type hints** for function parameters and return values
- Use `from typing import` for complex types (Dict, List, Union, Optional, etc.)
- Example:
  ```python
  def process_image(path: Union[str, Path], max_size: int = 2048) -> str:
      """Process an image file."""
      ...
  ```

### Testing Standards

- Follow **AAA pattern**: Arrange, Act, Assert
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Mock external dependencies (OpenAI API, file system when appropriate)
- Use pytest fixtures from `conftest.py` for common test setup
- Add docstrings to test functions explaining what they verify

Example:
```python
def test_encode_image_resizes_large_images(self, temp_dir: Path) -> None:
    """Test that large images are resized to reduce token usage."""
    # Arrange
    large_image_path = temp_dir / "large_image.jpg"
    img = Image.new("RGB", (4000, 3000), color=(100, 150, 200))
    img.save(large_image_path)

    # Act
    result = encode_image_to_base64(large_image_path, max_size=2048)

    # Assert
    assert isinstance(result, str)
    assert max(resized_img.size) == 2048
```

### Code Style

- Follow **PEP 8** standards (enforced by Ruff)
- Use **clear, descriptive variable names**
- Add **docstrings** to all functions, classes, and modules
- Keep functions **focused and single-purpose**
- Maximum line length: 88 characters (Black default)

### Documentation

When adding new features:

- Update relevant docstrings
- Add usage examples if appropriate
- Update README.md if user-facing functionality changes
- Document configuration options in .env.example

## Project-Specific Guidelines

### Configuration Management

- All configuration should go through `src/comp_critic/config.py`
- Use environment variables via `.env` file
- Provide sensible defaults
- Validate required configuration in `Config.validate()`

### Error Handling

- Raise specific exceptions (FileNotFoundError, ValueError, etc.)
- Include helpful error messages
- Test error conditions with pytest.raises()

### Dependencies

- Add new dependencies via Poetry:
  ```bash
  poetry add <package>           # Runtime dependency
  poetry add --group dev <package>  # Development dependency
  ```
- Run `poetry lock` after adding dependencies
- Commit both `pyproject.toml` and `poetry.lock`

## Common Scenarios

### Adding a New Feature

1. Write tests in appropriate `tests/test_*.py` file
2. Implement the feature in `src/comp_critic/`
3. Update type hints and docstrings
4. Run `poetry run poe check-all`
5. Update documentation if needed

### Fixing a Bug

1. Write a test that reproduces the bug
2. Verify the test fails
3. Fix the bug
4. Verify the test passes
5. Run full test suite
6. Run quality checks

### Refactoring

1. Ensure existing tests pass
2. Make refactoring changes
3. Verify tests still pass
4. Run `poetry run poe check-all`
5. Update tests if behavior intentionally changed

## Pre-Commit Checklist

Before committing changes, ensure:

- [ ] All tests pass: `poetry run poe test`
- [ ] No linting errors: `poetry run poe lint`
- [ ] Code is formatted: `poetry run poe format-check` (or run `poetry run poe format`)
- [ ] Type checking passes: `poetry run poe typecheck`
- [ ] New features have tests
- [ ] Docstrings are updated
- [ ] README.md updated if needed

**Quick check:** Run `poetry run poe check-all` to verify all of the above.

## Available Poe Tasks

Quick reference for all available tasks:

```bash
poe ingest        # Run transcript ingestion pipeline
poe test          # Run tests with coverage report
poe test-fast     # Run tests without coverage (faster)
poe lint          # Run Ruff linter
poe format        # Auto-format code with Ruff
poe format-check  # Check formatting without changes
poe typecheck     # Run mypy type checker
poe check-all     # Run all quality checks (lint + format + typecheck)
```

## Resources

- **Testing Framework**: [pytest](https://docs.pytest.org/)
- **Type Checking**: [mypy](https://mypy.readthedocs.io/)
- **Linting/Formatting**: [Ruff](https://docs.astral.sh/ruff/)
- **Dependency Management**: [Poetry](https://python-poetry.org/docs/)
- **LangChain Docs**: [LangChain Python](https://python.langchain.com/)

## Questions?

- Check the README.md for usage and architecture
- Review PRD.md for requirements and specifications
- Look at existing tests for examples
- Follow the patterns established in the codebase
