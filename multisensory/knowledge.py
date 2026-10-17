"""Structured skill records and immutable frozen-knowledge manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .skill_schema import VERTICAL_CAPABILITIES, validate_skill_instance

CANONICAL_SKILL_SCHEMA = "multisensory-canonical-skill-v1"
PRINCIPLE_SCHEMA = "multisensory-principle-v1"
MANIFEST_SCHEMA = "multisensory-knowledge-manifest-v1"
SIGNAL_STATUSES = frozenset({"measured", "derived", "proxy"})
RECORD_FOLDERS = {
    "instances": validate_skill_instance,
    "skills": lambda value: validate_canonical_skill(value),
    "principles": lambda value: validate_principle(value),
}


def _strings(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a {'possibly empty ' if allow_empty else ''}list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} entries must be non-empty strings")
    return value


def validate_signal_contract(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("signal_contract must be an object")
    required = value.get("required")
    optional = value.get("optional", [])
    if not isinstance(required, list) or not isinstance(optional, list):
        raise ValueError("signal_contract required/optional must be lists")
    for group, entries in (("required", required), ("optional", optional)):
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"signal_contract.{group} entries require a path")
            if entry["path"] in seen:
                raise ValueError(f"duplicate signal path: {entry['path']}")
            seen.add(entry["path"])
            if entry.get("status") not in SIGNAL_STATUSES:
                raise ValueError(f"invalid signal status for {entry['path']}")
    return value


def _base_record(record: Any, schema: str) -> dict:
    if not isinstance(record, dict) or record.get("schema") != schema:
        raise ValueError(f"record must use schema {schema}")
    for name in ("id", "version", "title", "vertical_capability"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
    if record["vertical_capability"] not in VERTICAL_CAPABILITIES:
        raise ValueError("unknown vertical capability")
    validate_signal_contract(record.get("signal_contract"))
    return record


def validate_canonical_skill(record: Any) -> dict:
    record = _base_record(record, CANONICAL_SKILL_SCHEMA)
    for name in ("goal", "preconditions", "stop_conditions", "known_failures"):
        if name == "goal":
            if not isinstance(record.get(name), str) or not record[name].strip():
                raise ValueError("goal must be a non-empty string")
        else:
            _strings(record.get(name), name, allow_empty=name == "known_failures")
    _strings(record.get("implementation_refs"), "implementation_refs")
    _strings(record.get("supporting_instance_ids"), "supporting_instance_ids")
    _strings(record.get("evidence_refs"), "evidence_refs")
    if record.get("status") not in {"candidate", "validated", "retired"}:
        raise ValueError("canonical skill status must be candidate, validated, or retired")
    return record


def validate_principle(record: Any) -> dict:
    record = _base_record(record, PRINCIPLE_SCHEMA)
    for name in ("when", "decision", "invariant"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
    supporting = _strings(record.get("supporting_skill_ids"), "supporting_skill_ids")
    _strings(record.get("expected_effects"), "expected_effects")
    _strings(record.get("exceptions"), "exceptions", allow_empty=True)
    _strings(record.get("falsifiers"), "falsifiers")
    _strings(record.get("evidence_refs"), "evidence_refs")
    status = record.get("status")
    if status not in {"candidate", "validated", "retired"}:
        raise ValueError("principle status must be candidate, validated, or retired")
    if status == "validated" and len(set(supporting)) < 2:
        raise ValueError("a validated principle requires at least two canonical skills")
    return record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError("artifact reference must be a non-empty relative path")
    root = root.resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError(f"invalid or missing artifact reference: {reference}")
    return path


def _load_records(knowledge: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    records = {}
    by_kind = {name: [] for name in RECORD_FOLDERS}
    for folder, validator in RECORD_FOLDERS.items():
        root = knowledge / folder
        root.mkdir(exist_ok=True)
        for path in sorted(root.glob("*.json")):
            value = validator(json.loads(path.read_text()))
            identifier = value["id"]
            if identifier in records:
                raise ValueError(f"duplicate structured knowledge id: {identifier}")
            records[identifier] = value
            by_kind[folder].append(identifier)
    return records, by_kind


def _validate_relations(records: dict[str, dict], by_kind: dict[str, list[str]]) -> None:
    instance_ids = set(by_kind["instances"])
    skill_ids = set(by_kind["skills"])
    for identifier in skill_ids:
        missing = set(records[identifier]["supporting_instance_ids"]) - instance_ids
        if missing:
            raise ValueError(
                f"canonical skill {identifier} has missing instances: {sorted(missing)}"
            )
    for identifier in by_kind["principles"]:
        missing = set(records[identifier]["supporting_skill_ids"]) - skill_ids
        if missing:
            raise ValueError(f"principle {identifier} has missing skills: {sorted(missing)}")


def build_frozen_manifest(workspace: Path, frozen_harness: Path) -> dict:
    """Validate and bind frozen knowledge to code, evidence, and file hashes."""
    workspace = Path(workspace)
    frozen_harness = Path(frozen_harness)
    knowledge = frozen_harness / "knowledge"
    if not all((frozen_harness / name).is_dir() for name in ("skills", "lessons", "knowledge")):
        raise ValueError("frozen harness requires skills, lessons, and knowledge directories")
    manifest_path = knowledge / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    (knowledge / "portfolio.json").unlink(missing_ok=True)
    records, by_kind = _load_records(knowledge)
    _validate_relations(records, by_kind)

    evidence_bindings = []
    for identifier, record in sorted(records.items()):
        for reference in record.get("implementation_refs", []):
            _inside(frozen_harness, reference)
        for reference in record.get("evidence_refs", []):
            source = _inside(workspace, reference)
            evidence_bindings.append(
                {"record_id": identifier, "ref": reference, "sha256": _sha256(source)}
            )
        code_ref = record.get("code_ref")
        if code_ref:
            source = _inside(workspace, code_ref)
            if _sha256(source) != record["code_sha256"]:
                raise ValueError(f"code hash mismatch for {identifier}")

    files = {}
    for component in ("skills", "lessons", "knowledge"):
        root = frozen_harness / component
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"frozen knowledge symlink is forbidden: {path}")
            if path.is_file() and path != manifest_path:
                files[str(path.relative_to(frozen_harness))] = _sha256(path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_boundary": "playground_public_evidence",
        "files": files,
        "records": by_kind,
        "evidence_bindings": evidence_bindings,
        "compatibility": {
            "legacy_skills_supported": True,
            "legacy_lessons_supported": True,
            "structured_records_optional": True,
        },
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest
