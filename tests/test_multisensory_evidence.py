"""CPU tests for public-only multisensory evidence extraction."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multisensory.evidence import EvidenceTracker, build_evidence
from simulators.observations import export_observation


def observation(*, x=0.0, joints=0.0, action_x=None):
    state = {
        "left_arm_joint_state": np.full(6, joints),
        "left_ee_joint_state": np.array([0.4]),
        "left_ee_pose": np.array([x, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
    }
    result = {
        "state": state,
        "vision": {"cam_head": {"color": np.zeros((8, 8, 3), dtype=np.uint8)}},
        "additional_info": {"frequency": 25},
    }
    if action_x is not None:
        result["action"] = {
            "left_ee_pose": np.array([action_x, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
            "left_ee_joint_state": np.array([0.4]),
        }
    return result


class EvidenceTests(unittest.TestCase):
    def test_absent_modalities_are_not_fabricated_as_zero_measurements(self):
        evidence = build_evidence(observation(), timestamp=7.0)
        self.assertEqual(evidence["timestamp_unix"], 7.0)
        self.assertEqual(evidence["modalities"]["proprioception"]["status"], "measured")
        self.assertEqual(evidence["modalities"]["force"]["status"], "unavailable")
        self.assertEqual(evidence["modalities"]["tactile"]["status"], "unavailable")
        self.assertEqual(evidence["modalities"]["contact"]["status"], "proxy_only")
        self.assertEqual(evidence["progress"]["state"], "unknown")

    def test_tracker_derives_motion_without_using_private_state(self):
        tracker = EvidenceTracker()
        tracker.observe(observation(x=0.0, joints=0.0), timestamp=1.0)
        evidence = tracker.observe(observation(x=0.02, joints=0.1), timestamp=1.04)
        self.assertEqual(evidence["source_boundary"], "public_observation_only")
        self.assertEqual(evidence["progress"]["state"], "moving")
        self.assertAlmostEqual(
            evidence["temporal"]["ee_translation_delta_m"]["left_ee_pose"], 0.02
        )

    def test_command_state_tracking_error_is_explicit(self):
        evidence = build_evidence(observation(x=0.1, action_x=0.13))
        self.assertAlmostEqual(
            evidence["proprioception"]["tracking_error_l2"]["left_ee_pose"], 0.03
        )

    def test_depth_without_calibration_is_not_metric_spatial_evidence(self):
        value = observation()
        value["vision"]["cam_head"]["depth"] = np.ones((8, 8))
        evidence = build_evidence(value)
        self.assertEqual(evidence["modalities"]["spatial"]["status"], "limited")
        value["vision"]["cam_head"]["intrinsics"] = np.eye(3)
        self.assertEqual(build_evidence(value)["modalities"]["spatial"]["status"], "derivable")

    def test_export_writes_agent_visible_evidence_next_to_public_observation(self):
        class Task:
            def public_observation(self):
                return observation()

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            folder = workspace / "observations/000000"
            with patch(
                "simulators.observations.camera_images",
                return_value={"cam_head": np.zeros((8, 8, 3), dtype=np.uint8)},
            ):
                export_observation(
                    Task(), folder, workspace=workspace, evidence_tracker=EvidenceTracker()
                )
            evidence = json.loads((folder / "evidence.json").read_text())
            self.assertEqual(evidence["schema"], "multisensory-evidence-v1")
            self.assertTrue((folder / "observation.json").is_file())
            self.assertTrue((folder / "current_cam_head.png").is_file())


if __name__ == "__main__":
    unittest.main()
