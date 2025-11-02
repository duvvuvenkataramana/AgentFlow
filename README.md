# AgentFlow

Prototype scaffolding for the AgentFlow planner and Codex CLI adapter described in the PRD.

## Layout

- `src/agentflow/` – core package with configuration helpers and adapters.
- `tests/` – unit and live tests for the Codex adapter.

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

The Codex CLI must be installed globally:

```bash
npm.cmd install -g @openai/codex
```

Set `OPENAI_API_KEY` in `.env` (already provided). Tests automatically load it.

## Running Tests

- `pytest -k unit` runs fast unit tests.
- `pytest tests/live -m live` runs the live Codex integration test (uses real API calls).
