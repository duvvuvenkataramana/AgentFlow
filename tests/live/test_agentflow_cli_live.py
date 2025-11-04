import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.live


def _can_run_live() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _can_run_live(), reason="OPENAI_API_KEY is required for live test.")
def test_agentflow_cli_generates_plan_from_real_run():
    repo_root = Path(__file__).resolve().parents[2]
    sandbox_dir = repo_root / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    for artifact in sandbox_dir.glob("agentflow-*.yaml"):
        artifact.unlink()

    db_path = sandbox_dir / "workspace.db"
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, body TEXT)")
            connection.execute(
                "INSERT INTO notes (body) VALUES (?)",
                ("MCP servers can expose this SQLite content.",),
            )
    finally:
        connection.close()

    prompt = (
        "Develop a minimal MCP server in Python using FastAPI that exposes the contents of the SQLite "
        f"database located at {db_path}. Provide the server implementation and explain how it serves the data."
    )

    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-m", "agentflow.cli", prompt],
        cwd=str(sandbox_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr

    plan_files = sorted(sandbox_dir.glob("agentflow-*.yaml"))
    assert plan_files, "Expected agentflow CLI to write a plan artifact in sandbox directory."

    payload = yaml.safe_load(plan_files[-1].read_text(encoding="utf-8"))
    assert payload["status"] == "completed"

    node = payload["nodes"][0]
    message = node["outputs"]["message"]

    assert "mcp" in message.lower()
    assert "sqlite" in message.lower()

    server_code = _extract_python_code_block(message)
    assert server_code, "Expected the assistant response to include a Python code block."
    assert "FastAPI" in server_code or "fastapi" in server_code

    server_path = sandbox_dir / "mcp_server.py"
    server_path.write_text(server_code, encoding="utf-8")

    file_text = server_path.read_text(encoding="utf-8")
    assert "FastAPI" in file_text or "fastapi" in file_text

    evaluation = node["outputs"].get("evaluation", {})
    assert evaluation, "Expected evaluation data to be present in node outputs."
    assert "score" in evaluation or "error" in evaluation
    metrics = node["metrics"]
    assert "evaluation_score" in metrics or "evaluation_error" in metrics
    if "eval_metrics" in payload:
        assert "self_evaluation_score" in payload["eval_metrics"] or "self_evaluation_error" in payload["eval_metrics"]


def _extract_python_code_block(markdown_text: str) -> str | None:
    pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    match = pattern.search(markdown_text)
    if not match:
        return None
    return match.group(1).strip()
