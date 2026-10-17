"""Signal-aware retrieval, ablation, exception, and export tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multisensory.experiments import EXPORT_SCHEMA, export_run
from multisensory.knowledge import MANIFEST_SCHEMA
from multisensory.retrieval import PORTFOLIO_SCHEMA, build_task_portfolio
from orchestrator.config import Knowledge
from orchestrator.session import Session


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def contract(path, status):
    return {"required": [{"path": path, "status": status}], "optional": []}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.knowledge = self.root / "knowledge"
        for name in ("instances", "skills", "principles"):
            (self.knowledge / name).mkdir(parents=True)
        write(self.knowledge / "manifest.json", {"schema": MANIFEST_SCHEMA})
        self.evidence = {
            "modalities": {
                "vision_rgb": {"status": "measured"},
                "proprioception": {"status": "measured"},
                "spatial": {"status": "limited"},
                "force": {"status": "unavailable"},
                "tactile": {"status": "unavailable"},
                "contact": {"status": "proxy_only"},
            },
            "proprioception": {"gripper_state": {"left_ee_joint_state": {"values": [0.4]}}},
            "temporal": {"native_step_window": {}},
        }
        self.settings = {
            "retrieval": "signal_aware",
            "modalities": [
                "vision_rgb",
                "proprioception",
                "spatial",
                "force",
                "tactile",
                "contact",
                "temporal",
            ],
            "max_skills": 8,
            "max_principles": 4,
            "include_instances": False,
            "use_exceptions": True,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def skill(self, identifier, path, status, *, failure=""):
        return {
            "schema": "multisensory-canonical-skill-v1",
            "id": identifier,
            "version": "1.0.0",
            "title": identifier.replace(".", " "),
            "vertical_capability": "grasp",
            "goal": "grasp and stack bowls safely",
            "implementation_refs": ["skills/grasp.py"],
            "supporting_instance_ids": ["instance.one"],
            "signal_contract": contract(path, status),
            "preconditions": ["object localized"],
            "stop_conditions": ["contact lost"],
            "known_failures": [failure] if failure else [],
            "evidence_refs": ["observations/000001/evidence.json"],
            "status": "validated",
        }

    def principle(self):
        return {
            "schema": "multisensory-principle-v1",
            "id": "grasp.verify",
            "version": "1.0.0",
            "title": "Verify grasp before transport",
            "vertical_capability": "grasp",
            "when": "after closing the gripper",
            "decision": "verify contact before transport",
            "invariant": "commands do not prove physical state",
            "expected_effects": ["fewer false grasps"],
            "exceptions": ["transparent bowls hide contact"],
            "falsifiers": ["verification never predicts retention"],
            "supporting_skill_ids": ["grasp.proxy", "grasp.proprio"],
            "signal_contract": contract("contact.gripper_proxy", "proxy"),
            "evidence_refs": ["observations/000001/evidence.json"],
            "status": "validated",
        }

    def populate(self):
        write(
            self.knowledge / "skills/proxy.json",
            self.skill("grasp.proxy", "contact.gripper_proxy", "proxy"),
        )
        write(
            self.knowledge / "skills/proprio.json",
            self.skill("grasp.proprio", "proprioception.gripper_state", "measured"),
        )
        write(
            self.knowledge / "skills/force.json",
            self.skill("grasp.force", "force.normal_force", "measured"),
        )
        write(self.knowledge / "principles/verify.json", self.principle())

    def test_signal_gate_selects_proxy_and_rejects_missing_force(self):
        self.populate()
        portfolio = build_task_portfolio(
            self.knowledge, self.evidence, "stack bowls grasp", self.settings
        )
        self.assertEqual(portfolio["schema"], PORTFOLIO_SCHEMA)
        self.assertEqual(
            {item["id"] for item in portfolio["skills"]},
            {"grasp.proxy", "grasp.proprio"},
        )
        rejected = {item["id"]: item for item in portfolio["rejected"]}
        self.assertEqual(rejected["grasp.force"]["reason"], "missing_required_signals")
        missing = rejected["grasp.force"]["missing_required"][0]
        self.assertEqual(missing["actual"], "unavailable")
        self.assertEqual([item["id"] for item in portfolio["principles"]], ["grasp.verify"])

    def test_modality_ablation_disables_otherwise_available_signal(self):
        self.populate()
        settings = {**self.settings, "modalities": ["vision_rgb", "contact"]}
        portfolio = build_task_portfolio(
            self.knowledge, self.evidence, "stack bowls grasp", settings
        )
        self.assertEqual([item["id"] for item in portfolio["skills"]], ["grasp.proxy"])
        self.assertEqual(portfolio["availability"]["proprioception"], "disabled")

    def test_knowledge_ablation_modes_are_explicit(self):
        self.populate()
        none = build_task_portfolio(
            self.knowledge, self.evidence, "stack bowls", {**self.settings, "retrieval": "none"}
        )
        self.assertEqual(none["skills"], [])
        all_records = build_task_portfolio(
            self.knowledge, self.evidence, "stack bowls", {**self.settings, "retrieval": "all"}
        )
        self.assertIn("grasp.force", {item["id"] for item in all_records["skills"]})

    def test_insufficient_temporal_window_does_not_satisfy_frequency_contract(self):
        write(
            self.knowledge / "skills/oscillation.json",
            self.skill(
                "grasp.oscillation",
                "temporal.native_step_window.oscillation",
                "derived",
            ),
        )
        evidence = json.loads(json.dumps(self.evidence))
        evidence["temporal"]["native_step_window"]["oscillation"] = {
            "status": "insufficient_data"
        }
        rejected = build_task_portfolio(
            self.knowledge, evidence, "grasp", self.settings
        )
        self.assertEqual(rejected["skills"], [])
        evidence["temporal"]["native_step_window"]["oscillation"] = {
            "status": "detected",
            "detected_channels": ["left_arm_joint_state[0]"],
        }
        selected = build_task_portfolio(
            self.knowledge, evidence, "grasp", self.settings
        )
        self.assertEqual([item["id"] for item in selected["skills"]], ["grasp.oscillation"])

    def test_exception_overlap_is_visible_and_penalized(self):
        write(
            self.knowledge / "skills/clear.json",
            self.skill("grasp.clear", "contact.gripper_proxy", "proxy"),
        )
        write(
            self.knowledge / "skills/warned.json",
            self.skill(
                "grasp.warned",
                "contact.gripper_proxy",
                "proxy",
                failure="transparent bowls hide contact",
            ),
        )
        portfolio = build_task_portfolio(
            self.knowledge, self.evidence, "transparent bowls", self.settings
        )
        self.assertEqual(portfolio["skills"][0]["id"], "grasp.clear")
        warned = next(item for item in portfolio["skills"] if item["id"] == "grasp.warned")
        self.assertEqual(warned["warnings"], ["transparent bowls hide contact"])


class ExperimentExportTests(unittest.TestCase):
    def test_exports_ablation_aggregate_and_scene_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, output = root / "run", root / "export"
            write(
                run / "resolved.json",
                {
                    "baseline": False,
                    "knowledge": {
                        "enabled": True,
                        "retrieval": "signal_aware",
                        "modalities": ["proprioception", "contact"],
                        "include_instances": False,
                        "use_exceptions": True,
                    },
                },
            )
            result = {
                "valid": True,
                "task_completed": True,
                "score": 80,
                "executions_used": 3,
                "native_steps_used": 12,
                "timing": {"agent_process_sec": 2.5, "token_usage": {"total_tokens": 100}},
                "knowledge_portfolio": {
                    "retrieval": "signal_aware",
                    "availability": {"proprioception": "measured", "contact": "proxy"},
                    "metrics": {"selected_skills": 2, "selected_principles": 1, "rejected": 1},
                },
            }
            write(
                run / "test.json",
                {
                    "complete": True,
                    "scenes": 1,
                    "scene_indices": [0],
                    "results": [result],
                    "success_rate": 1.0,
                    "score": 80,
                },
            )
            exported = export_run(run, output)
            self.assertEqual(exported["schema"], EXPORT_SCHEMA)
            self.assertEqual(exported["aggregate"]["mean_selected_skills"], 2)
            self.assertEqual(
                exported["aggregate"]["modality_availability"]["contact"], {"proxy": 1}
            )
            with (output / "scenes.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["retrieval"], "signal_aware")
            self.assertTrue((output / "experiment.json").is_file())


class SessionPortfolioTests(unittest.TestCase):
    def test_test_session_builds_portfolio_before_agent_start(self):
        class Environment:
            max_steps = 20
            native_step_limit = 20
            control_steps = 0
            success = False
            terminated = False
            truncated = False

            def reset(self, *, seed):
                self.seed = seed

        def export(env, folder, **kwargs):
            write(folder / "evidence.json", {"modalities": {}, "temporal": {}})

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in ("instances", "skills", "principles"):
                (run / "workspace/knowledge" / name).mkdir(parents=True)
            write(run / "workspace/knowledge/manifest.json", {"schema": MANIFEST_SCHEMA})
            with patch("orchestrator.session.export_observation", side_effect=export):
                session = Session(
                    Environment(),
                    run,
                    phase="test",
                    seed=7,
                    budget=3,
                    seconds=None,
                    execution_seconds=5,
                    knowledge=Knowledge(enabled=True),
                    task_query="stack bowls",
                )
            workspace_portfolio = run / "workspace/knowledge/portfolio.json"
            host_portfolio = run / "knowledge-portfolios/000000.json"
            self.assertTrue(workspace_portfolio.is_file())
            self.assertEqual(
                json.loads(workspace_portfolio.read_text()),
                json.loads(host_portfolio.read_text()),
            )
            self.assertTrue((run / "workspace/observations/000000/portfolio.json").is_file())
            next_observation = run / "workspace/observations/000001"
            write(next_observation / "evidence.json", {"modalities": {}, "temporal": {}})
            session._refresh_portfolio(next_observation)
            refreshed = json.loads((run / "knowledge-portfolios/000001.json").read_text())
            self.assertEqual(refreshed["evidence_ref"], "observations/000001/evidence.json")


if __name__ == "__main__":
    unittest.main()
