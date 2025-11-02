import os

import pytest

from agentflow.adapters import CodexCLIAdapter


pytestmark = pytest.mark.live


def _can_run_live() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _can_run_live(), reason="OPENAI_API_KEY is required for live test.")
def test_codex_adapter_live_round_trip(settings):
    adapter = CodexCLIAdapter(settings)
    result = adapter.run("Return the word ok.")
    assert "ok" in result.message.lower()
    assert result.usage
