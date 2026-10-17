"""Validation tests for task-specific structured skill records."""

import unittest

from multisensory.skill_schema import SKILL_INSTANCE_SCHEMA, validate_skill_instance


def record():
    return {
        "schema": SKILL_INSTANCE_SCHEMA,
        "id": "grasp.verify.attempt-1",
        "task": "stack_bowls",
        "task_family": "pickup-and-place",
        "vertical_capability": "grasp",
        "goal": "verify a grasp",
        "trigger": "after closing the gripper",
        "code_ref": "observations/000001/code.py",
        "code_sha256": "a" * 64,
        "signal_contract": {
            "required": [{"path": "proprioception.gripper_state", "status": "measured"}]
        },
        "evidence_refs": ["observations/000001/evidence.json"],
        "outcome": {"success": True},
    }


class SkillSchemaTests(unittest.TestCase):
    def test_accepts_evidence_bound_instance(self):
        self.assertEqual(validate_skill_instance(record())["vertical_capability"], "grasp")

    def test_rejects_unavailable_required_signal(self):
        value = record()
        value["signal_contract"]["required"][0]["status"] = "unavailable"
        with self.assertRaisesRegex(ValueError, "required signal"):
            validate_skill_instance(value)

    def test_rejects_unknown_capability(self):
        value = record()
        value["vertical_capability"] = "magic"
        with self.assertRaisesRegex(ValueError, "vertical capability"):
            validate_skill_instance(value)


if __name__ == "__main__":
    unittest.main()
