# AGENTS.md — Aether

AI-powered viral short-form content generation engine built on
LangGraph with Ollama for local inference. Python 3.13,
managed with `uv`.

## Build & Run

```bash
uv sync                    # Install dependencies
uv run python main.py      # Run the application
uv add <package>           # Add a dependency
uv add --dev <package>     # Add a dev dependency
```

## Linting & Formatting (Ruff)

Ruff is the sole linter and formatter. Line length is 80 characters.

```bash
uv run ruff check .        # Lint
uv run ruff check --fix .  # Lint and auto-fix
uv run ruff format .       # Format
uv run ruff format --check .  # Check formatting without writing
```

Configure in `pyproject.toml` under `[tool.ruff]`. Always run
`ruff check` and `ruff format` before committing.

## Testing

Test framework: pytest (with `pytest-asyncio` for async tests).

```bash
uv run pytest                                       # All tests
uv run pytest tests/test_llm.py                     # Single file
uv run pytest tests/test_llm.py::test_load_model    # Single function
uv run pytest tests/test_llm.py::TestLlm::test_load_model  # Class method
uv run pytest -k "keyword"                          # Match keyword
uv run pytest -v                                    # Verbose output
```

Tests live in `tests/` mirroring `src/`. Files: `test_<module>.py`.
Functions: `test_<behavior>`.

## Project Structure

```
aether/
├── main.py              # Entry point (if __name__ guard)
├── src/                 # Application logic
│   ├── __init__.py      # Facade re-exports with __all__
│   ├── models.py           # Models loading/configuration
│   ├── state.py         # Workflow state types (TypedDict)
│   ├── workflow.py      # LangGraph workflow construction
│   └── agents/          # Individual agent node modules
├── prompts/             # Prompt templates (one file per niche)
│   ├── __init__.py      # Re-exports prompt strings
│   └── psychology.py    # Psychology niche prompts
├── tests/               # Test files mirror src/ structure
└── docs/                # Research & reference documentation
```

`src/` holds logic; `prompts/` holds data. Keep them separate.

## Code Style

- **DRY** — Extract shared logic; never duplicate prompt
  fragments, config values, or business logic.
- **Line length** — 80 characters max. Ruff enforces this.
- **Trailing commas** — Use in multi-line constructs. Omit in
  single-line constructs.
- **Blank lines** — Two between top-level definitions. One
  before `if __name__` guard.
- **Strings** — Double quotes for all strings.

### Imports

Order: 1) stdlib, 2) third-party, 3) local (ruff isort).

```python
import os
from pathlib import Path

from langchain_ollama import ChatOllama

from src.models import load_chat_model
```

- **Cross-package**: absolute imports (`from src.models import ...`).
- **Intra-package**: relative imports (`from .psychology import ...`).
- **Multi-symbol imports**: parenthesized, one symbol per line.

### Naming Conventions

| Element            | Convention    | Example                  |
|--------------------|---------------|--------------------------|
| Functions/methods  | `snake_case`  | `load_chat_model()`           |
| Variables          | `snake_case`  | `model_name`             |
| Constants          | `UPPER_SNAKE` | `DEFAULT_TEMPERATURE`    |
| Classes            | `PascalCase`  | `WorkflowState`          |
| Modules/packages   | `snake_case`  | `models.py`, `agents/`      |
| Test functions     | `test_*`      | `test_load_model`        |

Prompt template variables use `snake_case` (they are data, not
traditional constants).

### Type Annotations

Use type hints on all function signatures. Python 3.13 syntax:

- Prefer `X | None` over `Optional[X]`.
- Use `TypedDict` for structured state (LangGraph pattern).
- Use `list[str]` / `dict[str, int]` (lowercase builtins).

### Docstrings

Google style. Opening `"""` on its own line for multi-line:

```python
def load_chat_model(model_name: str = "llama3.1") -> ChatOllama:
    """
    Load and configure the ChatOllama language model.

    Args:
        model_name: The Ollama model identifier.

    Returns:
        Configured ChatOllama instance.
    """
```

Every public function and class must have a docstring.

### Error Handling

- Raise specific exceptions, never bare `except:`.
- Catch the narrowest exception type possible.
- Use custom exception classes in `src/exceptions.py` when
  domain-specific errors are needed.
- Always include context in error messages.
- Chain exceptions with `raise ... from e`.

### Package `__init__.py` Pattern

Use `__init__.py` as a facade. Re-export public symbols with
`__all__`. Consumers import from the package, not internal modules.

## Commit Messages

Follow conventional commits with optional ticket IDs:

```
feat(AET-12): add content review agent
fix(AET-15): handle Ollama connection timeout
refactor: extract prompt loading into factory
chore: update ruff to 0.16
docs: add workflow architecture diagram
```

Types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `ci`.
