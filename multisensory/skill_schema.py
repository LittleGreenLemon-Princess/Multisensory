"""Minimal structured records for evidence-grounded skill evolution.

This is intentionally smaller than ASPIRE's full consolidation repository. It
establishes a stable interchange format first; admission, canonicalization, and
principle promotion can then be added without changing execution semantics.
"""

from __future__ import annotations

from typing import Any

SKILL_INSTANCE_SCHEMA = "multisensory-skill-instance-v1"
VERTICAL_CAPABILITIES = frozenset(
    {"localization", "grasp", "transport", "manipulation", "long-horizon"}
)
SIGNAL_STATUSES = frozenset({"measured", "derived", "proxy", "unavailable"})


def validate_skill_instance(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the stable boundary of a task-specific executed skill record."""
    if not isinstance(record, dict):
        raise TypeError("skill instance must be a dictionary")
    required = {
        "schema",
        "id",
        "task",
        "task_family",
        "vertical_capability",
        "goal",
        "trigger",
        "code_ref",
        "code_sha256",
        "signal_contract",
        "evidence_refs",
        "outcome",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"missing skill-instance fields: {missing}")
    if record["schema"] != SKILL_INSTANCE_SCHEMA:
        raise ValueError(f"unsupported skill-instance schema: {record['schema']!r}")
    if record["vertical_capability"] not in VERTICAL_CAPABILITIES:
        raise ValueError("unknown vertical capability")
    for name in ("id", "task", "task_family", "goal", "trigger", "code_ref"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
    digest = record["code_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("code_sha256 must be a lowercase SHA-256 digest")
    contract = record["signal_contract"]
    if not isinstance(contract, dict) or not isinstance(contract.get("required"), list):
        raise ValueError("signal_contract.required must be a list")
    for item in contract["required"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or item.get("status") not in SIGNAL_STATUSES - {"unavailable"}
        ):
            raise ValueError("each required signal needs path and a usable status")
    references = record["evidence_refs"]
    if not isinstance(references, list) or not references or not all(
        isinstance(reference, str) and reference for reference in references
    ):
        raise ValueError("evidence_refs must contain at least one path")
    outcome = record["outcome"]
    if not isinstance(outcome, dict) or type(outcome.get("success")) is not bool:
        raise ValueError("outcome.success must be a boolean")
    return record
