"""
AgentFlow CLI entry point.

Supports two workflows:
1. `agentflow "<prompt>"` — execute the prompt through the Codex CLI adapter and persist a
   single-node plan YAML artifact capturing the run.
2. `agentflow view` — launch a lightweight Flask viewer over previously generated artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agentflow.adapters import CodexCLIAdapter, CodexCLIError
from agentflow.config import ConfigurationError, Settings
from agentflow.viewer import run_viewer


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI dispatcher.

    When invoked without explicit subcommands, treats the arguments as a free-form prompt.
    """

    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        _print_usage()
        return 1

    if args[0] == "view":
        return _handle_view_command(args[1:])

    prompt = " ".join(args).strip()
    if not prompt:
        _print_usage()
        return 1

    return _handle_prompt(prompt)


def _print_usage() -> None:
    print(
        "Usage:\n"
        '  agentflow "<prompt text>"        Execute prompt via Codex and capture YAML artifact.\n'
        "  agentflow view [options]         Launch local viewer for AgentFlow artifacts.\n"
    )


def _handle_view_command(args: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="agentflow view",
        description="Launch the AgentFlow artifact viewer.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface for the Flask server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        default=5050,
        type=int,
        help="Port for the Flask server (default: 5050).",
    )
    parser.add_argument(
        "--directory",
        default=".",
        help="Directory containing AgentFlow-generated YAML artifacts (default: current directory).",
    )

    namespace = parser.parse_args(args)
    directory = Path(namespace.directory).resolve()

    if not directory.exists():
        print(f"Directory not found: {directory}", file=sys.stderr)
        return 1

    try:
        run_viewer(directory=directory, host=namespace.host, port=namespace.port)
    except KeyboardInterrupt:
        print("\nViewer stopped.")
        return 0
    return 0


def _handle_prompt(prompt: str) -> int:
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc)
    base_name = timestamp.strftime("agentflow-%Y%m%d%H%M%S")
    target_path, plan_id = _resolve_plan_path(base_name)

    adapter = CodexCLIAdapter(settings)

    node_status = "succeeded"
    plan_status = "completed"
    outputs: Dict[str, Any] = {}
    usage: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []
    error_payload: Optional[Dict[str, Any]] = None
    notes = "Codex invocation succeeded."
    evaluation_payload: Optional[Dict[str, Any]] = None

    run_started = datetime.now(timezone.utc)
    try:
        result = adapter.run(prompt)
        events = result.events
        outputs = {"message": result.message, "events": events}
        usage = result.usage or {}
        evaluation_payload = _perform_self_evaluation(adapter, prompt, result.message)
        if evaluation_payload:
            outputs["evaluation"] = _build_evaluation_outputs(evaluation_payload)
    except CodexCLIError as exc:
        node_status = "failed"
        plan_status = "failed"
        error_payload = {"message": str(exc)}
        notes = f"Codex invocation failed: {exc}"
    except Exception as exc:  # pragma: no cover - defensive catch
        node_status = "failed"
        plan_status = "failed"
        error_payload = {"message": f"Unexpected error: {exc.__class__.__name__}: {exc}"}
        notes = f"Unexpected error: {exc.__class__.__name__}"
    run_finished = datetime.now(timezone.utc)

    if "events" not in outputs:
        outputs = {**outputs, "events": events}

    duration = (run_finished - run_started).total_seconds()
    summary = prompt[:80].replace("\n", " ").strip() or "Ad-hoc Codex execution"

    plan_document = _build_plan_document(
        plan_id=plan_id,
        prompt=prompt,
        summary=summary,
        plan_status=plan_status,
        node_status=node_status,
        outputs=outputs,
        usage=usage,
        events=events,
        error_payload=error_payload,
        run_started=run_started,
        run_finished=run_finished,
        duration_seconds=duration,
        notes=notes,
        evaluation_payload=evaluation_payload,
    )

    _write_plan(target_path, plan_document)

    print(f"Wrote plan artifact: {target_path}")
    if plan_status == "failed":
        print("Execution failed; inspect the YAML artifact for details.", file=sys.stderr)
        return 1
    return 0


def _resolve_plan_path(base_name: str) -> Tuple[Path, str]:
    directory = Path.cwd()
    candidate = directory / f"{base_name}.yaml"
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{base_name}-{suffix}.yaml"
        suffix += 1

    filename = candidate.stem
    plan_id = f"plan-{filename.split('-', 1)[-1]}"
    return candidate, plan_id


def _build_plan_document(
    *,
    plan_id: str,
    prompt: str,
    summary: str,
    plan_status: str,
    node_status: str,
    outputs: Dict[str, Any],
    usage: Dict[str, Any],
    events: List[Dict[str, Any]],
    error_payload: Optional[Dict[str, Any]],
    run_started: datetime,
    run_finished: datetime,
    duration_seconds: float,
    notes: str,
    evaluation_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    created_iso = run_started.isoformat()
    finished_iso = run_finished.isoformat()

    node: Dict[str, Any] = {
        "id": "codex_execution",
        "type": "agent",
        "summary": summary or "Codex execution",
        "depends_on": [],
        "status": "succeeded" if node_status == "succeeded" else "failed",
        "attempt": 1,
        "inputs": {
            "prompt": prompt,
        },
        "outputs": outputs,
        "artifacts": [],
        "metrics": _build_metrics(usage, evaluation_payload),
        "timeline": {
            "queued_at": created_iso,
            "started_at": run_started.isoformat(),
            "ended_at": finished_iso,
            "duration_seconds": round(duration_seconds, 3),
        },
        "history": [
            {
                "attempt_id": 1,
                "timestamp": finished_iso,
                "status": "succeeded" if node_status == "succeeded" else "failed",
                "notes": notes,
            }
        ],
    }

    if error_payload:
        node["error"] = error_payload

    plan_document: Dict[str, Any] = {
        "schema_version": "1.0",
        "plan_id": plan_id,
        "name": summary or "Codex execution",
        "description": prompt,
        "created_at": created_iso,
        "last_updated": finished_iso,
        "created_by": "agentflow-cli@local",
        "version": 1,
        "status": plan_status,
        "tags": [],
        "context": {},
        "nodes": [node],
        "rollup": {
            "completion_percentage": 100 if plan_status == "completed" else 0,
            "counts": {
                "succeeded": 1 if node_status == "succeeded" else 0,
                "failed": 1 if node_status != "succeeded" else 0,
            },
            "last_writer": "agentflow-cli@local",
        },
    }

    if events:
        plan_document["metadata"] = {"codex_events_count": len(events)}

    if evaluation_payload:
        eval_metrics: Dict[str, Any] = {}
        score = evaluation_payload.get("score") if isinstance(evaluation_payload, dict) else None
        if score is not None:
            eval_metrics["self_evaluation_score"] = score
        error = evaluation_payload.get("error") if isinstance(evaluation_payload, dict) else None
        if error:
            eval_metrics["self_evaluation_error"] = error
        if eval_metrics:
            plan_document["eval_metrics"] = eval_metrics

    return plan_document


def _write_plan(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _perform_self_evaluation(
    adapter: CodexCLIAdapter,
    prompt: str,
    response: str,
) -> Optional[Dict[str, Any]]:
    evaluation_prompt = (
        "You are an impartial self-evaluation judge. Score the assistant's response from 0 to 1, "
        "where 1 represents a perfect answer and 0 represents a useless answer.\n"
        "Return ONLY a JSON object with keys `score` (float between 0 and 1) and `justification` (string). "
        "Do not wrap the JSON in code fences or add commentary.\n"
        f"Original prompt:\n{prompt}\n\nAssistant response:\n{response}\n"
    )

    try:
        evaluation_result = adapter.run(evaluation_prompt)
    except CodexCLIError as exc:
        return {"error": f"Self-evaluation failed: {exc}"}

    parsed = _parse_evaluation_payload(evaluation_result.message)
    payload: Dict[str, Any] = {
        "raw_message": evaluation_result.message,
        "events": evaluation_result.events,
        "usage": evaluation_result.usage or {},
    }
    if parsed:
        payload.update(parsed)
    else:
        payload["error"] = "Self-evaluation response was not valid JSON."
    return payload


def _parse_evaluation_payload(message: str) -> Optional[Dict[str, Any]]:
    candidate = message.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if "\n" in candidate:
            candidate = candidate.split("\n", 1)[-1]

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    score = payload.get("score")
    try:
        if score is not None:
            score = float(score)
    except (TypeError, ValueError):
        score = None

    justification = payload.get("justification") or payload.get("reasoning")
    if justification is not None:
        justification = str(justification).strip()

    return {"score": score, "justification": justification}


def _build_metrics(usage: Dict[str, Any], evaluation_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"usage": usage}
    if evaluation_payload:
        score = evaluation_payload.get("score")
        if score is not None:
            metrics["evaluation_score"] = score
        eval_usage = evaluation_payload.get("usage")
        if eval_usage:
            metrics["evaluation_usage"] = eval_usage
        error = evaluation_payload.get("error")
        if error:
            metrics["evaluation_error"] = error
    return metrics


def _build_evaluation_outputs(evaluation_payload: Dict[str, Any]) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    if evaluation_payload.get("score") is not None:
        outputs["score"] = evaluation_payload["score"]
    if evaluation_payload.get("justification"):
        outputs["justification"] = evaluation_payload["justification"]
    if evaluation_payload.get("error"):
        outputs["error"] = evaluation_payload["error"]
    outputs["raw_message"] = evaluation_payload.get("raw_message", "")
    outputs["events"] = evaluation_payload.get("events", [])
    outputs["usage"] = evaluation_payload.get("usage", {})
    return outputs


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
