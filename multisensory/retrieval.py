"""Deterministic signal-aware retrieval for frozen structured knowledge."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .knowledge import (
    MANIFEST_SCHEMA,
    RECORD_FOLDERS,
    validate_canonical_skill,
    validate_principle,
)
from .skill_schema import validate_skill_instance

PORTFOLIO_SCHEMA = "multisensory-task-portfolio-v1"
USABLE_STATUS = {"measured": 3, "derived": 2, "proxy": 1}
ROOT_MODALITIES = {
    "vision": "vision_rgb",
    "vision_rgb": "vision_rgb",
    "proprioception": "proprioception",
    "spatial": "spatial",
    "force": "force",
    "tactile": "tactile",
    "contact": "contact",
    "temporal": "temporal",
    "progress": "temporal",
}


def _settings(value: Any) -> dict:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        raise TypeError("retrieval settings must be a dictionary or dataclass")
    return value


def _tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)
    return set(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


def signal_availability(evidence: dict, enabled_modalities: list[str] | tuple[str, ...]) -> dict:
    """Map public evidence into conservative measured/derived/proxy availability."""
    enabled = set(enabled_modalities)
    unknown = enabled - set(ROOT_MODALITIES.values())
    if unknown:
        raise ValueError(f"unknown knowledge modalities: {sorted(unknown)}")
    modalities = evidence.get("modalities", {}) if isinstance(evidence, dict) else {}
    result = {}
    for name in ROOT_MODALITIES.values():
        if name not in enabled:
            result[name] = "disabled"
            continue
        if name == "temporal":
            result[name] = "derived" if evidence.get("temporal") else "unavailable"
            continue
        status = modalities.get(name, {}).get("status")
        result[name] = {
            "measured": "measured",
            "derivable": "derived",
            "proxy_only": "proxy",
        }.get(status, "unavailable")
    return result


def _evidence_path(evidence: dict, path: str) -> Any:
    value: Any = evidence
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def evaluate_contract(contract: dict, availability: dict, evidence: dict | None = None) -> dict:
    """Evaluate required and optional signals without inventing absent modalities."""
    required, optional = [], []
    for group, destination in (("required", required), ("optional", optional)):
        for item in contract.get(group, []):
            path = item["path"]
            modality = ROOT_MODALITIES.get(path.split(".", 1)[0])
            actual = availability.get(modality, "unavailable")
            value = _evidence_path(evidence or {}, path)
            if path.startswith(("proprioception.", "temporal.", "progress.")):
                insufficient = (
                    isinstance(value, dict) and value.get("status") == "insufficient_data"
                )
                if value is None or value == {} or value == [] or insufficient:
                    actual = "unavailable"
            expected = item["status"]
            satisfied = actual in USABLE_STATUS and USABLE_STATUS[actual] >= USABLE_STATUS[expected]
            destination.append(
                {
                    "path": path,
                    "modality": modality,
                    "expected": expected,
                    "actual": actual,
                    "satisfied": satisfied,
                }
            )
    missing = [item for item in required if not item["satisfied"]]
    return {
        "eligible": not missing,
        "required": required,
        "optional": optional,
        "missing_required": missing,
    }


def _load_records(knowledge: Path) -> dict[str, dict[str, dict]]:
    validators = {
        "instances": validate_skill_instance,
        "skills": validate_canonical_skill,
        "principles": validate_principle,
    }
    records = {kind: {} for kind in RECORD_FOLDERS}
    for kind, validator in validators.items():
        root = knowledge / kind
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.json")):
            record = validator(json.loads(path.read_text()))
            records[kind][record["id"]] = record
    return records


def _record_text(record: dict) -> set[str]:
    fields = (
        "id",
        "title",
        "goal",
        "vertical_capability",
        "task",
        "task_family",
        "trigger",
        "preconditions",
        "stop_conditions",
        "when",
        "decision",
        "invariant",
        "expected_effects",
    )
    return _tokens([record[name] for name in fields if name in record])


def _exception_warnings(record: dict, query_tokens: set[str], enabled: bool) -> list[str]:
    if not enabled:
        return []
    warnings = []
    for text in [*record.get("known_failures", []), *record.get("exceptions", [])]:
        if query_tokens & _tokens(text):
            warnings.append(text)
    return warnings


def _rank(record: dict, query_tokens: set[str], warnings: list[str]) -> tuple[float, list[str]]:
    overlap = sorted(query_tokens & _record_text(record))
    score = float(len(overlap) * 2)
    reasons = [f"query_overlap:{token}" for token in overlap]
    if record.get("status") == "validated":
        score += 2
        reasons.append("validated")
    elif record.get("status") == "candidate":
        score += 1
        reasons.append("candidate")
    evidence_count = len(record.get("evidence_refs", []))
    score += min(evidence_count, 3) * 0.25
    if evidence_count:
        reasons.append(f"evidence_refs:{evidence_count}")
    if warnings:
        score -= len(warnings) * 3
        reasons.append(f"exception_overlap:{len(warnings)}")
    return score, reasons


def _entry(
    record: dict,
    contract: dict,
    score: float,
    reasons: list[str],
    warnings: list[str],
) -> dict:
    return {
        "id": record["id"],
        "title": record.get("title", record.get("goal", record["id"])),
        "vertical_capability": record["vertical_capability"],
        "score": score,
        "reasons": reasons,
        "contract": contract,
        "warnings": warnings,
        "implementation_refs": record.get("implementation_refs", []),
        "preconditions": record.get("preconditions", []),
        "stop_conditions": record.get("stop_conditions", []),
        "known_failures": record.get("known_failures", []),
    }


def build_task_portfolio(
    knowledge: Path,
    evidence: dict,
    query: str,
    settings: Any,
    *,
    evidence_ref: str | None = None,
) -> dict:
    """Select an auditable task-time portfolio from one frozen library."""
    settings = _settings(settings)
    mode = settings.get("retrieval", "signal_aware")
    if mode not in {"signal_aware", "all", "none"}:
        raise ValueError("knowledge.retrieval must be signal_aware, all, or none")
    maximum_skills = int(settings.get("max_skills", 8))
    maximum_principles = int(settings.get("max_principles", 4))
    if maximum_skills < 0 or maximum_principles < 0:
        raise ValueError("knowledge retrieval limits must be non-negative")
    knowledge = Path(knowledge)
    manifest_path = knowledge / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("structured retrieval requires a host-generated frozen manifest")
    records = _load_records(knowledge)
    enabled_modalities = settings.get("modalities", [])
    availability = signal_availability(evidence, enabled_modalities)
    query_tokens = _tokens(query)
    rejected, ranked_skills = [], []

    if mode != "none":
        for record in records["skills"].values():
            contract = evaluate_contract(record["signal_contract"], availability, evidence)
            if record["status"] == "retired":
                rejected.append({"id": record["id"], "reason": "retired"})
                continue
            if mode == "signal_aware" and not contract["eligible"]:
                rejected.append(
                    {
                        "id": record["id"],
                        "reason": "missing_required_signals",
                        "missing_required": contract["missing_required"],
                    }
                )
                continue
            warnings = _exception_warnings(
                record, query_tokens, bool(settings.get("use_exceptions", True))
            )
            score, reasons = _rank(record, query_tokens, warnings)
            ranked_skills.append(
                _entry(record, contract, score, reasons, warnings)
            )
    ranked_skills.sort(key=lambda item: (-item["score"], item["id"]))
    selected_skills = ranked_skills[:maximum_skills]
    selected_ids = {item["id"] for item in selected_skills}

    ranked_principles = []
    if mode != "none":
        for record in records["principles"].values():
            if record["status"] == "retired":
                rejected.append({"id": record["id"], "reason": "retired"})
                continue
            supporting = selected_ids & set(record["supporting_skill_ids"])
            if not supporting:
                rejected.append({"id": record["id"], "reason": "no_selected_supporting_skill"})
                continue
            contract = evaluate_contract(record["signal_contract"], availability, evidence)
            if mode == "signal_aware" and not contract["eligible"]:
                rejected.append(
                    {
                        "id": record["id"],
                        "reason": "missing_required_signals",
                        "missing_required": contract["missing_required"],
                    }
                )
                continue
            warnings = _exception_warnings(
                record, query_tokens, bool(settings.get("use_exceptions", True))
            )
            score, reasons = _rank(record, query_tokens, warnings)
            score += len(supporting)
            reasons.append(f"selected_supporting_skills:{len(supporting)}")
            entry = _entry(record, contract, score, reasons, warnings)
            entry.update(
                decision=record["decision"],
                invariant=record["invariant"],
                exceptions=record["exceptions"],
                falsifiers=record["falsifiers"],
                supporting_skill_ids=sorted(supporting),
            )
            ranked_principles.append(entry)
    ranked_principles.sort(key=lambda item: (-item["score"], item["id"]))

    instances = []
    if mode != "none" and settings.get("include_instances", False):
        supported = {
            identifier
            for record in records["skills"].values()
            if record["id"] in selected_ids
            for identifier in record["supporting_instance_ids"]
        }
        for identifier, record in sorted(records["instances"].items()):
            if identifier not in supported:
                continue
            instance_contract = evaluate_contract(
                record["signal_contract"], availability, evidence
            )
            if mode == "signal_aware" and not instance_contract["eligible"]:
                rejected.append(
                    {
                        "id": record["id"],
                        "reason": "missing_required_signals",
                        "missing_required": instance_contract["missing_required"],
                    }
                )
                continue
            instances.append(
                {
                    "id": record["id"],
                    "task": record["task"],
                    "goal": record["goal"],
                    "outcome": record["outcome"],
                    "evidence_refs": record["evidence_refs"],
                    "contract": instance_contract,
                }
            )

    selected_principles = ranked_principles[:maximum_principles]
    return {
        "schema": PORTFOLIO_SCHEMA,
        "query": query,
        "evidence_ref": evidence_ref,
        "retrieval": mode,
        "settings": settings,
        "availability": availability,
        "skills": selected_skills,
        "principles": selected_principles,
        "instances": instances,
        "rejected": sorted(rejected, key=lambda item: item["id"]),
        "metrics": {
            "library_instances": len(records["instances"]),
            "library_skills": len(records["skills"]),
            "library_principles": len(records["principles"]),
            "selected_skills": len(selected_skills),
            "selected_principles": len(selected_principles),
            "selected_instances": len(instances),
            "rejected": len(rejected),
        },
    }


def portfolio_summary(portfolio: dict, *, record: str = "knowledge-portfolio.json") -> dict:
    """Return the compact result payload used by both execution backends."""
    return {
        "schema": portfolio["schema"],
        "retrieval": portfolio["retrieval"],
        "settings": portfolio["settings"],
        "availability": portfolio["availability"],
        "metrics": portfolio["metrics"],
        "record": record,
    }
