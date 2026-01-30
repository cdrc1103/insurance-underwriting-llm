# Claude Code Instructions

## Development Environment

### Setup

This project uses **Python 3.12+** with virtual environment management.

```bash
# Install uv (first time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup blank project if needed
uv init && uv venv --python 3.12
source .venv/bin/activate
```

---

## Code Standards &

### General Philosophy

**Fail Fast, Fail Loud**
- Raise exceptions immediately when errors occur
- No silent fallbacks or default behaviors that mask issues
- Use explicit error messages with context

**Explicit Over Implicit**
- Type hints are required for all functions
- Configuration via explicit parameters, not hidden globals
- Document non-obvious design decisions

**Minimal Abstractions**
- Only abstract when pattern repeats 3+ times
- Prefer composition over inheritance
- Keep classes focused and single-purpose

**Minimal Code Changes**
- Make only the changes necessary to achieve the specific goal
- Avoid refactoring, reformatting, or "improving" code that's not directly related to the task
- Don't add extra features, error handling, or abstractions beyond what's required
- If fixing a bug, change only what's needed for the fix - don't clean up surrounding code

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Variables/Functions | `snake_case` | `parse_pdf()`, `chunk_size` |
| Classes | `PascalCase` | `PDFParser`, `VectorStore` |
| Constants | `UPPER_CASE` | `MAX_CHUNK_SIZE`, `DEFAULT_MODEL` |
| Private methods | `_snake_case` | `_validate_input()` |
| Type variables | `PascalCase` | `DocumentType`, `EmbeddingT` |

### Type Hints

**Required for all public functions:**

```python
from typing import List, Dict, Optional, Union
from pathlib import Path

def parse_pdf(
    file_path: Path,
    extract_tables: bool = True,
    page_range: Optional[tuple[int, int]] = None
) -> Dict[str, List[str]]:
    """
    Parse PDF and extract structured content.

    Args:
        file_path: Path to PDF file
        extract_tables: Whether to extract tables
        page_range: Optional (start, end) page range

    Returns:
        Dictionary with 'text', 'tables', 'images' keys

    Raises:
        FileNotFoundError: If PDF doesn't exist
        PDFParsingError: If parsing fails
    """
    ...
```

## Definition of Done

A user story or feature is considered **done** when all of the following criteria are met:

### Required Checklist

- [ ] **Unit tests implemented** - New functionality has corresponding unit tests with meaningful coverage of both happy paths and edge cases
- [ ] **Code formatted** - Code has been formatted with `ruff format .`
- [ ] **Type hints complete** - All public functions have proper type annotations
- [ ] **Docstrings added** - Public functions include docstrings with Args, Returns, and Raises sections
- [ ] **No regressions** - Existing tests continue to pass
- [ ] **Code reviewed** - Use the code-reviewer subagent to review changes for quality, security vulnerabilities, and adherence to best practices

### Verification Commands

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Run unit tests (using uv)
uv run pytest tests/ -v

# Check formatting and linting
uv run ruff format --check .
uv run ruff check .
```

---
