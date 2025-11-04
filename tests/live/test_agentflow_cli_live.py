import json
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


def _prepare_sandbox_with_sample_data() -> tuple[Path, Path]:
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
            connection.execute("DELETE FROM notes")
            connection.execute(
                "INSERT INTO notes (body) VALUES (?)",
                ("MCP servers can expose this SQLite content.",),
            )
    finally:
        connection.close()

    return sandbox_dir, db_path


@pytest.mark.skipif(not _can_run_live(), reason="OPENAI_API_KEY is required for live test.")
def test_agentflow_cli_generates_plan_from_real_run():
    sandbox_dir, db_path = _prepare_sandbox_with_sample_data()

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


@pytest.mark.skipif(not _can_run_live(), reason="OPENAI_API_KEY is required for live test.")
def test_agentflow_cli_branches_and_iterations_live():
    sandbox_dir, db_path = _prepare_sandbox_with_sample_data()

    prompt = (
        "You are preparing a showcase AgentFlow run. Deliver the answer in three segments.\n"
        "1. Present the FastAPI MCP server for the SQLite database at "
        f"{db_path} inside a ```python``` block. Keep it production-ready and include comments for demo clarity.\n"
        "2. Produce a ```json``` block named flow_spec that defines a DAG with at least six nodes. "
        "Ensure one node models a branch with explicit on_true/on_false targets and another node represents an iterative loop "
        "with a max_iterations property. Include edges that illustrate the control flow across all nodes.\n"
        "3. Conclude with a numbered list titled 'Live test walkthrough' summarizing two passes through the DAG, "
        "highlighting how the branch and loop behave.\n"
        "All JSON must be valid with double-quoted keys."
    )

    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-m", "agentflow.cli", prompt],
        cwd=str(sandbox_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert completed.returncode == 0, completed.stderr

    plan_files = sorted(sandbox_dir.glob("agentflow-*.yaml"))
    assert plan_files, "Expected agentflow CLI to write a plan artifact in sandbox directory."

    payload = yaml.safe_load(plan_files[-1].read_text(encoding="utf-8"))
    assert payload["status"] == "completed"

    node = payload["nodes"][0]
    message = node["outputs"]["message"]

    server_code = _extract_python_code_block(message)
    assert server_code, "Expected the assistant response to include a Python code block."
    assert "FastAPI" in server_code or "fastapi" in server_code

    flow_spec = _extract_json_block(message)
    assert flow_spec is not None, "Expected a JSON flow_spec block describing the DAG."
    flow_data = flow_spec["flow_spec"] if isinstance(flow_spec.get("flow_spec"), dict) else flow_spec

    nodes = flow_data.get("nodes", [])
    assert isinstance(nodes, list) and len(nodes) >= 6, "Expected at least six nodes in the flow specification."
    branch_nodes = [
        node_data
        for node_data in nodes
        if isinstance(node_data, dict)
        and any(key in node_data for key in ("on_true", "on_false", "branch_true", "branch_false"))
    ]
    assert branch_nodes, "Expected the DAG to include a branching node with on_true/on_false style targets."

    loop_nodes = [
        node_data
        for node_data in nodes
        if isinstance(node_data, dict)
        and (
            (isinstance(node_data.get("type"), str) and node_data["type"].lower() in {"loop", "iteration", "for_each"})
            or any(key in node_data for key in ("max_iterations", "iterations", "loop_body"))
        )
    ]
    assert loop_nodes, "Expected the DAG to include a loop node with an iteration budget."

    edges = flow_data.get("edges", [])
    assert isinstance(edges, list) and edges, "Expected the flow_spec to describe edges."

    assert "Live test walkthrough" in message
    assert re.search(r"\n1\.\s", message) and re.search(r"\n2\.\s", message), "Expected a numbered walkthrough list."

    evaluation = node["outputs"].get("evaluation", {})
    assert evaluation, "Expected evaluation data to be present in node outputs."
    assert "score" in evaluation or "error" in evaluation
    metrics = node["metrics"]
    assert "evaluation_score" in metrics or "evaluation_error" in metrics


def _extract_python_code_block(markdown_text: str) -> str | None:
    pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    match = pattern.search(markdown_text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_json_block(markdown_text: str) -> dict | None:
    fence_pattern = re.compile(r"```json\s*(\{.*?\})```", re.IGNORECASE | re.DOTALL)
    match = fence_pattern.search(markdown_text)

    candidate: str | None
    if match:
        candidate = match.group(1).strip()
    else:
        trimmed = markdown_text.strip()
        candidate = trimmed if trimmed.startswith("{") and trimmed.endswith("}") else None

    if not candidate:
        return None

    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(loaded, dict):
        return None
    return loaded
