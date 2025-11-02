from pathlib import Path

import pytest

from agentflow.config import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Load shared settings from the project .env file."""

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        return Settings.from_env(env_file)
    return Settings.from_env()
