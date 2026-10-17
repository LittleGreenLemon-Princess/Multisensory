"""Structured knowledge validation, provenance, and legacy compatibility."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multisensory.knowledge import (
    CANONICAL_SKILL_SCHEMA,
    MANIFEST_SCHEMA,
    PRINCIPLE_SCHEMA,
    build_frozen_manifest,
    validate_principle,
)
from multisensory.skill_schema import SKILL_INSTANCE_SCHEMA


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract():
    return {
        "required": [{"path": "proprioception.end_effector_pose", "status": "measured"}],
        "optional": [{"path": "contact.gripper_proxy", "status": "proxy"}],
    }


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.frozen = self.root / "frozen_harness"
        for name in ("skills", "lessons", "knowledge"):
            (self.frozen / name).mkdir(parents=True)
        write(self.frozen / "skills/move.py", "def move():\n    pass\n")
        write(self.frozen / "lessons/contact.md", "# Contact\n")
        write(self.workspace / "observations/000001/code.py", "step(action)\n")
        write(self.workspace / "observations/000001/evidence.json", {"schema": "test"})

    def tearDown(self):
        self.temporary.cleanup()

    def instance(self):
        code = self.workspace / "observations/000001/code.py"
        return {
            "schema": SKILL_INSTANCE_SCHEMA,
            "id": "grasp.instance-1",
            "task": "stack_bowls",
            "task_family": "pickup-and-place",
            "vertical_capability": "grasp",
            "goal": "move while preserving the grasp",
            "trigger": "after grasp verification",
            "code_ref": "observations/000001/code.py",
            "code_sha256": digest(code),
            "signal_contract": {"required": contract()["required"]},
            "evidence_refs": ["observations/000001/evidence.json"],
            "outcome": {"success": True},
        }

    def skill(self):
        return {
            "schema": CANONICAL_SKILL_SCHEMA,
            "id": "grasp.preserve",
            "version": "1.0.0",
            "title": "Preserve a verified grasp",
            "vertical_capability": "grasp",
            "goal": "transport without losing the object",
            "implementation_refs": ["skills/move.py"],
            "supporting_instance_ids": ["grasp.instance-1"],
            "signal_contract": contract(),
            "preconditions": ["grasp is verified"],
            "stop_conditions": ["gripper proxy indicates loss"],
            "known_failures": [],
            "evidence_refs": ["observations/000001/evidence.json"],
            "status": "validated",
        }

    def principle(self, *, status="candidate", supporting=None):
        return {
            "schema": PRINCIPLE_SCHEMA,
            "id": "grasp.measure-not-command",
            "version": "1.0.0",
            "title": "Trust measured physical effects",
            "vertical_capability": "grasp",
            "when": "a command claims completion",
            "decision": "require a measured postcondition",
            "invariant": "commands do not prove physical state",
            "expected_effects": ["fewer false-positive grasps"],
            "exceptions": [],
            "falsifiers": ["command and measurement are equivalent in all supported controllers"],
            "supporting_skill_ids": supporting or ["grasp.preserve"],
            "signal_contract": contract(),
            "evidence_refs": ["observations/000001/evidence.json"],
            "status": status,
        }

    def test_legacy_skills_and_lessons_freeze_without_structured_records(self):
        manifest = build_frozen_manifest(self.workspace, self.frozen)
        self.assertEqual(manifest["schema"], MANIFEST_SCHEMA)
        self.assertEqual(manifest["records"], {"instances": [], "skills": [], "principles": []})
        self.assertTrue(manifest["compatibility"]["legacy_skills_supported"])
        self.assertTrue((self.frozen / "knowledge/manifest.json").is_file())

    def test_manifest_binds_records_to_exact_public_evidence_and_code(self):
        write(self.frozen / "knowledge/instances/one.json", self.instance())
        write(self.frozen / "knowledge/skills/one.json", self.skill())
        write(self.frozen / "knowledge/principles/one.json", self.principle())
        manifest = build_frozen_manifest(self.workspace, self.frozen)
        self.assertEqual(manifest["records"]["instances"], ["grasp.instance-1"])
        self.assertEqual(manifest["records"]["skills"], ["grasp.preserve"])
        bindings = manifest["evidence_bindings"]
        self.assertEqual(
            {item["record_id"] for item in bindings},
            {"grasp.instance-1", "grasp.preserve", "grasp.measure-not-command"},
        )
        expected = digest(self.workspace / "observations/000001/evidence.json")
        self.assertTrue(all(item["sha256"] == expected for item in bindings))

    def test_freeze_rejects_missing_evidence_and_code_hash_mismatch(self):
        instance = self.instance()
        instance["evidence_refs"] = ["observations/999999/evidence.json"]
        write(self.frozen / "knowledge/instances/one.json", instance)
        with self.assertRaisesRegex(ValueError, "missing artifact"):
            build_frozen_manifest(self.workspace, self.frozen)
        instance["evidence_refs"] = ["observations/000001/evidence.json"]
        instance["code_sha256"] = "0" * 64
        write(self.frozen / "knowledge/instances/one.json", instance)
        with self.assertRaisesRegex(ValueError, "code hash mismatch"):
            build_frozen_manifest(self.workspace, self.frozen)

    def test_relations_must_resolve_inside_the_frozen_library(self):
        skill = self.skill()
        write(self.frozen / "knowledge/skills/one.json", skill)
        with self.assertRaisesRegex(ValueError, "missing instances"):
            build_frozen_manifest(self.workspace, self.frozen)

    def test_freeze_replaces_an_agent_authored_manifest(self):
        write(self.frozen / "knowledge/manifest.json", {"trusted": False})
        manifest = build_frozen_manifest(self.workspace, self.frozen)
        stored = json.loads((self.frozen / "knowledge/manifest.json").read_text())
        self.assertEqual(stored, manifest)
        self.assertNotIn("trusted", stored)

    def test_validated_principle_requires_multiple_canonical_skills(self):
        with self.assertRaisesRegex(ValueError, "at least two canonical skills"):
            validate_principle(self.principle(status="validated"))


if __name__ == "__main__":
    unittest.main()
