"""Stable JSON/CSV export for modality and knowledge ablation experiments."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

EXPORT_SCHEMA = "multisensory-experiment-export-v1"


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _scene_row(index: int, result: dict) -> dict:
    portfolio = result.get("knowledge_portfolio") or {}
    metrics = portfolio.get("metrics", {})
    timing = result.get("timing", {})
    tokens = timing.get("token_usage", {})
    availability = portfolio.get("availability", {})
    return {
        "scene_index": index,
        "valid": bool(result.get("valid")),
        "task_completed": bool(result.get("task_completed")),
        "score": result.get("score"),
        "executions_used": result.get("executions_used"),
        "native_steps_used": result.get("native_steps_used"),
        "agent_process_sec": timing.get("agent_process_sec"),
        "total_tokens": tokens.get("total_tokens", 0),
        "retrieval": portfolio.get("retrieval", "disabled"),
        "selected_skills": metrics.get("selected_skills", 0),
        "selected_principles": metrics.get("selected_principles", 0),
        "rejected_records": metrics.get("rejected", 0),
        "availability": availability,
    }


def build_experiment_export(run: Path) -> tuple[dict, list[dict]]:
    """Read one completed or partial run without opening private scene identities."""
    run = Path(run)
    report = _read(run / "test.json")
    resolved = _read(run / "resolved.json")
    indices = report.get("scene_indices", list(range(len(report.get("results", [])))))
    rows = [
        _scene_row(index, result)
        for index, result in zip(indices, report.get("results", []), strict=True)
    ]
    knowledge = resolved.get("knowledge", {"enabled": False, "retrieval": "none"})
    availability_counts: dict[str, Counter] = {}
    for row in rows:
        for modality, status in row.pop("availability").items():
            availability_counts.setdefault(modality, Counter())[status] += 1
    valid_rows = [row for row in rows if row["valid"]]
    completed = [row for row in valid_rows if row["task_completed"]]
    export = {
        "schema": EXPORT_SCHEMA,
        "complete": bool(report.get("complete")),
        "ablation": {
            "baseline": bool(resolved.get("baseline", False)),
            "knowledge_enabled": bool(knowledge.get("enabled", False)),
            "retrieval": knowledge.get("retrieval", "none"),
            "modalities": knowledge.get("modalities", []),
            "include_instances": bool(knowledge.get("include_instances", False)),
            "use_exceptions": bool(knowledge.get("use_exceptions", False)),
        },
        "aggregate": {
            "scenes": report.get("scenes", len(rows)),
            "reported_scenes": len(rows),
            "valid_scenes": len(valid_rows),
            "successes": len(completed),
            "success_rate": report.get("success_rate"),
            "score": report.get("score"),
            "mean_selected_skills": (
                sum(row["selected_skills"] for row in valid_rows) / len(valid_rows)
                if valid_rows
                else None
            ),
            "mean_selected_principles": (
                sum(row["selected_principles"] for row in valid_rows) / len(valid_rows)
                if valid_rows
                else None
            ),
            "total_tokens": sum(row["total_tokens"] for row in rows),
            "modality_availability": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(availability_counts.items())
            },
        },
    }
    return export, rows


def export_run(run: Path, output: Path) -> dict:
    """Write machine-readable aggregate JSON and one-row-per-scene CSV."""
    export, rows = build_experiment_export(run)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(
        json.dumps(export, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    fields = list(rows[0]) if rows else [
        "scene_index",
        "valid",
        "task_completed",
        "score",
        "executions_used",
        "native_steps_used",
        "agent_process_sec",
        "total_tokens",
        "retrieval",
        "selected_skills",
        "selected_principles",
        "rejected_records",
    ]
    with (output / "scenes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return export
