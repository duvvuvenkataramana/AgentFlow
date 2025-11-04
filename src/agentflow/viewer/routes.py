"""
Flask route registrations and data serialization for the AgentFlow viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from flask import Flask, abort, jsonify, render_template, send_from_directory


@dataclass
class PlanArtifact:
    plan_id: str
    name: str
    status: str
    filename: str
    created_at: Optional[str]
    last_updated: Optional[str]
    path: Path


def register_routes(app: Flask, root: Path) -> None:
    """
    Attach HTTP routes to the provided Flask application.
    """

    root = root.resolve()
    app.config["VIEWER_ROOT"] = root

    @app.route("/")
    def index():
        return render_template("index.html", root_directory=str(root))

    @app.route("/api/plans")
    def api_list_plans():
        summaries = [_summary_payload(artifact) for artifact in _discover_plans(root)]
        return jsonify(summaries)

    @app.route("/api/plans/<plan_id>")
    def api_view_plan(plan_id: str):
        artifact = _find_plan(root, plan_id)
        if not artifact:
            abort(404, description=f"Plan {plan_id} not found.")
        return jsonify(_plan_detail_payload(artifact))

    @app.route("/plans")  # legacy
    def legacy_list_plans():
        summaries = [_summary_payload(artifact) for artifact in _discover_plans(root)]
        return jsonify(summaries)

    @app.route("/plans/<plan_id>")  # legacy
    def legacy_view_plan(plan_id: str):
        artifact = _find_plan(root, plan_id)
        if not artifact:
            abort(404, description=f"Plan {plan_id} not found.")
        payload = _load_payload(artifact.path)
        return jsonify(payload)

    @app.route("/files/<path:filename>")
    def download_file(filename: str):
        target = (root / filename).resolve()
        if not str(target).startswith(str(root)):
            abort(403)
        if not target.exists():
            abort(404)
        return send_from_directory(root, filename)


def _discover_plans(root: Path) -> List[PlanArtifact]:
    artifacts: List[PlanArtifact] = []
    for path in sorted(root.glob("agentflow-*.yaml")):
        try:
            payload = _load_payload(path)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        plan_id = payload.get("plan_id")
        if not plan_id:
            continue

        if payload.get("schema_version") != "1.0":
            continue

        artifacts.append(
            PlanArtifact(
                plan_id=plan_id,
                name=payload.get("name", plan_id),
                status=payload.get("status", "unknown"),
                filename=path.name,
                created_at=payload.get("created_at"),
                last_updated=payload.get("last_updated"),
                path=path,
            )
        )
    return artifacts


def _find_plan(root: Path, plan_id: str) -> Optional[PlanArtifact]:
    for artifact in _discover_plans(root):
        if artifact.plan_id == plan_id:
            return artifact
    return None


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Invalid plan structure.")
    return data


def _summary_payload(artifact: PlanArtifact) -> Dict[str, Any]:
    return {
        "plan_id": artifact.plan_id,
        "name": artifact.name,
        "status": artifact.status,
        "filename": artifact.filename,
        "created_at": artifact.created_at,
        "last_updated": artifact.last_updated,
    }


def _plan_detail_payload(artifact: PlanArtifact) -> Dict[str, Any]:
    payload = _load_payload(artifact.path)
    raw_nodes = payload.get("nodes") or []
    nodes_index: Dict[str, Dict[str, Any]] = {}
    graph_elements: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    prompt_count = 0
    response_count = 0
    evaluation_count = 0

    for node in raw_nodes:
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        status = str(node.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        node_type = node.get("type") or "task"
        depends_on = list(node.get("depends_on") or [])
        inputs = node.get("inputs") or {}
        outputs = node.get("outputs") or {}
        artifacts = node.get("artifacts") or []
        metrics = node.get("metrics") or {}
        timeline = node.get("timeline") or {}
        history = node.get("history") or []

        prompt_text = _extract_prompt_text(node)
        response_text = _extract_response_text(node)
        evaluation = _extract_evaluation(node)
        evaluation_class = _evaluation_css_class(evaluation.get("score"))
        evaluation["css_class"] = evaluation_class

        group_id = f"group::{node_id}"
        group_label = _truncate(node.get("summary") or node_id, limit=110)
        graph_elements.append(
            {
                "data": {
                    "id": group_id,
                    "label": group_label,
                    "status": status,
                    "role": "group",
                },
                "classes": f"task-group {_status_css_class(status)}".strip(),
            }
        )
        nodes_index[group_id] = {
            "id": group_id,
            "role": "group",
            "role_label": "Node Group",
            "display_title": node.get("summary") or node_id,
            "status": status,
            "summary": node.get("summary"),
            "parent_id": None,
            "depends_on": depends_on,
            "metrics": metrics,
            "timeline": timeline,
            "history": history,
            "artifacts": artifacts,
            "outputs": outputs,
            "inputs": inputs,
        }

        prompt_id = f"{node_id}::prompt"
        prompt_label_body = _truncate(
            prompt_text.replace("\n", " ") if prompt_text else (node.get("summary") or node_id),
            limit=90,
        )
        prompt_label = f"{node_id} prompt\\n{prompt_label_body}"
        prompt_classes = f"node-prompt {_status_css_class(status)}".strip()
        graph_elements.append(
            {
                "data": {
                    "id": prompt_id,
                    "label": prompt_label,
                    "title": prompt_label_body,
                    "parent": group_id,
                    "status": status,
                    "role": "prompt",
                },
                "classes": prompt_classes,
            }
        )
        nodes_index[prompt_id] = {
            "id": prompt_id,
            "role": "prompt",
            "role_label": "Prompt",
            "parent_id": group_id,
            "status": status,
            "summary": node.get("summary"),
            "display_title": node.get("summary") or node_id,
            "type": f"{node_type} prompt",
            "prompt_text": prompt_text,
            "inputs": inputs,
            "depends_on": depends_on,
        }
        prompt_count += 1

        response_id = f"{node_id}::response"
        response_summary = _truncate(
            response_text.replace("\n", " ") if response_text else (outputs.get("synopsis") or node.get("summary") or "Response"),
            limit=90,
        )
        score_value = evaluation.get("score")
        score_text = "Eval: --"
        if score_value is not None:
            score_text = f"Eval: {score_value:.2f}"
        response_label = f"{node_id} response\n{response_summary}\n{score_text}"
        response_classes = f"node-response {evaluation_class}".strip()
        graph_elements.append(
            {
                "data": {
                    "id": response_id,
                    "label": response_label,
                    "title": response_summary,
                    "parent": group_id,
                    "status": status,
                    "role": "response",
                },
                "classes": response_classes,
            }
        )
        nodes_index[response_id] = {
            "id": response_id,
            "role": "response",
            "role_label": "Response",
            "parent_id": group_id,
            "status": status,
            "summary": node.get("summary"),
            "display_title": node.get("summary") or node_id,
            "type": f"{node_type} response",
            "response_text": response_text,
            "outputs": outputs,
            "artifacts": artifacts,
            "metrics": metrics,
            "timeline": timeline,
            "history": history,
            "evaluation": evaluation,
            "depends_on": depends_on,
        }
        response_count += 1

        show_evaluation = evaluation.get("score") is not None or evaluation.get("justification") or evaluation.get("raw_message")
        if show_evaluation:
            evaluation_id = f"{node_id}::evaluation"
            justification = evaluation.get("justification") or ""
            summary_source = justification or evaluation.get("raw_message") or ""
            if summary_source:
                summary_source = summary_source.replace("\n", " ").replace("\\n", " ")
            evaluation_summary = _truncate(summary_source, limit=80) if summary_source else "Self-evaluation"
            score_for_label = score_text.split(":", 1)[-1].strip()
            evaluation_label = f"{node_id} evaluation\nScore: {score_for_label}\n{evaluation_summary}"
            evaluation_classes = f"node-evaluation {evaluation_class}".strip()
            graph_elements.append(
                {
                    "data": {
                        "id": evaluation_id,
                        "label": evaluation_label,
                        "title": evaluation_summary,
                        "parent": group_id,
                        "status": status,
                        "role": "evaluation",
                    },
                    "classes": evaluation_classes,
                }
            )
            nodes_index[evaluation_id] = {
                "id": evaluation_id,
                "role": "evaluation",
                "role_label": "Evaluation",
                "parent_id": group_id,
                "status": status,
                "display_title": node.get("summary") or node_id,
                "type": f"{node_type} evaluation",
                "evaluation": evaluation,
                "source_node_id": node_id,
                "depends_on": [response_id],
            }
            evaluation_count += 1
            graph_elements.append(
                {
                    "data": {
                        "id": f"{response_id}->{evaluation_id}",
                        "source": response_id,
                        "target": evaluation_id,
                    }
                }
            )

        graph_elements.append(
            {
                "data": {
                    "id": f"{prompt_id}->{response_id}",
                    "source": prompt_id,
                    "target": response_id,
                }
            }
        )

    for node in raw_nodes:
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        target_prompt_id = f"{node_id}::prompt"
        for dependency in node.get("depends_on") or []:
            dep_id = str(dependency).strip()
            if not dep_id:
                continue
            source_response_id = f"{dep_id}::response"
            if source_response_id not in nodes_index or target_prompt_id not in nodes_index:
                continue
            edge_id = f"{source_response_id}->{target_prompt_id}"
            graph_elements.append(
                {
                    "data": {
                        "id": edge_id,
                        "source": source_response_id,
                        "target": target_prompt_id,
                    }
                }
            )

    return {
        "plan_id": artifact.plan_id,
        "name": artifact.name,
        "status": artifact.status,
        "filename": artifact.filename,
        "created_at": artifact.created_at,
        "last_updated": artifact.last_updated,
        "tags": payload.get("tags") or [],
        "metrics": payload.get("metrics") or {},
        "status_counts": status_counts,
        "graph_elements": graph_elements,
        "nodes_index": nodes_index,
        "graph_stats": {
            "total": prompt_count + response_count + evaluation_count,
            "prompts": prompt_count,
            "responses": response_count,
            "evaluations": evaluation_count,
        },
    }


def _status_css_class(status: str) -> str:
    normalized = (status or "unknown").lower()
    if normalized in {"completed", "succeeded"}:
        return "status-completed"
    if normalized in {"running", "in_progress"}:
        return "status-running"
    if normalized in {"failed", "error"}:
        return "status-failed"
    if normalized in {"pending", "blocked", "queued"}:
        return "status-pending"
    return "status-other"


def _truncate(text: str, *, limit: int) -> str:
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 1].rstrip()}..."


def _extract_prompt_text(node: Dict[str, Any]) -> str:
    inputs = node.get("inputs") or {}
    if isinstance(inputs, dict):
        for key in ("prompt", "message", "command"):
            value = inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(inputs, str):
        return inputs.strip()
    return ""


def _extract_response_text(node: Dict[str, Any]) -> str:
    outputs = node.get("outputs") or {}
    if isinstance(outputs, dict):
        for key in ("message", "synopsis", "summary", "raw"):
            value = outputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(outputs, str):
        return outputs.strip()
    return ""


def _extract_evaluation(node: Dict[str, Any]) -> Dict[str, Any]:
    outputs = node.get("outputs") or {}
    metrics = node.get("metrics") or {}
    evaluation: Dict[str, Any] = {}
    candidate: Dict[str, Any] = {}
    if isinstance(outputs, dict):
        if isinstance(outputs.get("evaluation"), dict):
            candidate = outputs["evaluation"]
        else:
            candidate = outputs
    score = candidate.get("score")
    if score is None:
        score = candidate.get("evaluation_score") or metrics.get("evaluation_score")
    evaluation["score"] = _coerce_float(score)
    evaluation["justification"] = candidate.get("justification") or candidate.get("reasoning") or candidate.get("notes")
    evaluation["raw_message"] = candidate.get("raw_message") or candidate.get("raw")
    return evaluation


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluation_css_class(score: Optional[float]) -> str:
    if score is None:
        return "score-unknown"
    normalized = score
    if normalized > 1.0:
        if normalized <= 10.0:
            normalized = normalized / 10.0
        else:
            normalized = min(normalized / 100.0, 1.0)
    if normalized >= 0.75:
        return "score-high"
    if normalized >= 0.4:
        return "score-medium"
    if normalized >= 0:
        return "score-low"
    return "score-unknown"










