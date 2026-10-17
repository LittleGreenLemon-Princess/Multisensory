"""Derive auditable evidence from public observations without privileged state.

The extractor deliberately distinguishes measured signals, derived quantities,
and unavailable modalities.  It never fills a missing force or tactile signal
with zeros, because that would make sensor absence indistinguishable from a
real zero measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA = "multisensory-evidence-v1"
POSITION_EPSILON_M = 1e-4
JOINT_EPSILON_RAD = 1e-4
NO_PROGRESS_MIN_SAMPLES = 4
NO_PROGRESS_MOTION_M = 1e-3
TRACKING_ERROR_THRESHOLD = 1e-2
OSCILLATION_MIN_SAMPLES = 8
OSCILLATION_MIN_AMPLITUDE = 1e-3
OSCILLATION_MIN_PROMINENCE = 0.5


def _array(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.ndim != 1 or not np.isfinite(result).all():
        return None
    return result


def _summary(value: Any) -> dict | None:
    array = _array(value)
    if array is None:
        return None
    return {
        "shape": list(array.shape),
        "values": array.tolist(),
        "l2_norm": float(np.linalg.norm(array)),
    }


def _state(observation: dict) -> dict:
    value = observation.get("state", {})
    return value if isinstance(value, dict) else {}


def _frequency(observation: dict) -> float | None:
    info = observation.get("additional_info", {})
    value = info.get("frequency") if isinstance(info, dict) else None
    if isinstance(value, int | float) and np.isfinite(value) and value > 0:
        return float(value)
    return None


def _matching(state: dict, suffix: str) -> dict[str, np.ndarray]:
    result = {}
    for key, value in state.items():
        if key.endswith(suffix) and (array := _array(value)) is not None:
            result[key] = array
    return result


def _modality(status: str, *, sources=(), note: str | None = None) -> dict:
    result = {"status": status, "sources": list(sources)}
    if note:
        result["note"] = note
    return result


def _delta(current: dict[str, np.ndarray], previous: dict[str, np.ndarray]) -> dict:
    values = {}
    for key in sorted(current.keys() & previous.keys()):
        if current[key].shape == previous[key].shape:
            values[key] = float(np.linalg.norm(current[key] - previous[key]))
    return values


def _tracking_errors(observation: dict) -> dict[str, float]:
    state = _state(observation)
    action = observation.get("action", {})
    if not isinstance(action, dict):
        return {}
    errors = {}
    for key, target in action.items():
        measured = _array(state.get(key))
        commanded = _array(target)
        if measured is not None and commanded is not None and measured.shape == commanded.shape:
            errors[key] = float(np.linalg.norm(commanded - measured))
    return errors


def _has_depth(observation: dict) -> bool:
    vision = observation.get("vision", {})
    if not isinstance(vision, dict):
        return False
    return any(
        isinstance(camera, dict)
        and any(
            name in camera
            for name in ("depth", "distance_to_camera", "distance_to_image_plane")
        )
        for camera in vision.values()
    )


def _has_calibration(observation: dict) -> bool:
    vision = observation.get("vision", {})
    if not isinstance(vision, dict):
        return False
    return "calibration" in vision or any(
        isinstance(camera, dict)
        and any(name in camera for name in ("intrinsics", "extrinsics", "pose"))
        for camera in vision.values()
    )


def _first_present(observation: dict, state: dict, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if name in observation or name in state]


def _window_rate(window: list[dict]) -> float | None:
    rates = [rate for item in window if (rate := _frequency(item)) is not None]
    if not rates:
        return None
    return rates[0] if all(abs(rate - rates[0]) < 1e-9 for rate in rates) else None


def _state_series(window: list[dict]) -> dict[str, np.ndarray]:
    if not window:
        return {}
    common = set(_state(window[0]))
    for item in window[1:]:
        common &= set(_state(item))
    result = {}
    for key in sorted(common):
        values = [_array(_state(item).get(key)) for item in window]
        if any(value is None for value in values):
            continue
        if len({value.shape for value in values}) != 1:
            continue
        result[key] = np.stack(values)
    return result


def _window_tracking_errors(window: list[dict]) -> list[float]:
    return [
        max(
            (
                value
                for key, value in _tracking_errors(item).items()
                if not key.endswith("ee_joint_state")
            ),
            default=0.0,
        )
        for item in window
    ]


def _no_progress(window: list[dict], series: dict[str, np.ndarray]) -> dict:
    if len(window) < NO_PROGRESS_MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "minimum_samples": NO_PROGRESS_MIN_SAMPLES,
            "observed_samples": len(window),
        }
    translations = {
        key: values[:, :3]
        for key, values in series.items()
        if key.endswith("ee_pose") and values.shape[1] >= 3
    }
    path_lengths = {
        key: float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())
        for key, values in translations.items()
    }
    joint_paths = {
        key: float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())
        for key, values in series.items()
        if key.endswith("arm_joint_state")
    }
    tracking_errors = _window_tracking_errors(window)
    maximum_tracking_error = max(tracking_errors, default=0.0)
    maximum_motion = max(path_lengths.values(), default=0.0)
    maximum_joint_motion = max(joint_paths.values(), default=0.0)
    demanded = maximum_tracking_error >= TRACKING_ERROR_THRESHOLD
    detected = (
        demanded
        and maximum_motion <= NO_PROGRESS_MOTION_M
        and maximum_joint_motion <= JOINT_EPSILON_RAD
    )
    return {
        "status": "detected" if detected else "not_detected",
        "command_state_disagreement": demanded,
        "max_tracking_error_l2": maximum_tracking_error,
        "ee_translation_path_m": path_lengths,
        "max_ee_translation_path_m": maximum_motion,
        "joint_path_rad": joint_paths,
        "max_joint_path_rad": maximum_joint_motion,
        "thresholds": {
            "tracking_error_l2": TRACKING_ERROR_THRESHOLD,
            "max_ee_translation_path_m": NO_PROGRESS_MOTION_M,
            "max_joint_path_rad": JOINT_EPSILON_RAD,
        },
    }


def _oscillation(window: list[dict], series: dict[str, np.ndarray], rate: float | None) -> dict:
    if rate is None or len(window) < OSCILLATION_MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "minimum_samples": OSCILLATION_MIN_SAMPLES,
            "observed_samples": len(window),
            "sampling_rate_hz": rate,
        }
    channels = {}
    detected = []
    for key, values in series.items():
        for index in range(values.shape[1]):
            signal = values[:, index]
            detrended = signal - np.linspace(signal[0], signal[-1], len(signal))
            amplitude = float(np.ptp(detrended))
            spectrum = np.abs(np.fft.rfft(detrended)) ** 2
            frequencies = np.fft.rfftfreq(len(signal), d=1.0 / rate)
            non_dc = spectrum[1:]
            if not non_dc.size or float(non_dc.sum()) <= 0:
                continue
            peak = int(np.argmax(non_dc)) + 1
            prominence = float(spectrum[peak] / non_dc.sum())
            centered = detrended - float(detrended.mean())
            zero_crossings = int(np.count_nonzero(np.diff(np.signbit(centered))))
            is_oscillating = (
                amplitude >= OSCILLATION_MIN_AMPLITUDE
                and prominence >= OSCILLATION_MIN_PROMINENCE
                and zero_crossings >= 3
            )
            name = f"{key}[{index}]"
            channels[name] = {
                "dominant_frequency_hz": float(frequencies[peak]),
                "spectral_prominence": prominence,
                "detrended_peak_to_peak": amplitude,
                "zero_crossings": zero_crossings,
                "detected": is_oscillating,
            }
            if is_oscillating:
                detected.append(name)
    return {
        "status": "detected" if detected else "not_detected",
        "sampling_rate_hz": rate,
        "window_samples": len(window),
        "detected_channels": detected,
        "channels": channels,
        "thresholds": {
            "detrended_peak_to_peak": OSCILLATION_MIN_AMPLITUDE,
            "spectral_prominence": OSCILLATION_MIN_PROMINENCE,
            "zero_crossings": 3,
        },
    }


def _window_features(window: list[dict] | None) -> dict:
    samples = list(window or [])
    rate = _window_rate(samples)
    series = _state_series(samples)
    return {
        "source": "public_native_step_observations",
        "samples": len(samples),
        "sampling_rate_hz": rate,
        "duration_sec": (
            None if rate is None or len(samples) < 2 else float((len(samples) - 1) / rate)
        ),
        "no_progress": _no_progress(samples, series),
        "oscillation": _oscillation(samples, series, rate),
    }


def build_evidence(
    observation: dict,
    previous_observation: dict | None = None,
    *,
    timestamp: float | None = None,
    signal_window: list[dict] | None = None,
) -> dict:
    """Build one JSON-serializable evidence record from public data only."""
    if not isinstance(observation, dict):
        raise TypeError("public observation must be a dictionary")
    state = _state(observation)
    joints = _matching(state, "arm_joint_state")
    ee_poses = _matching(state, "ee_pose")
    grippers = _matching(state, "ee_joint_state")
    frequency = _frequency(observation)
    vision = observation.get("vision")
    rgb_available = isinstance(vision, dict) and bool(vision)

    force_names = _first_present(
        observation,
        state,
        ("ee_wrench", "wrench", "joint_effort", "joint_torque", "gripper_force"),
    )
    tactile_names = _first_present(
        observation, state, ("tactile", "tactile_state", "contact_sensor")
    )

    proprioception = {
        "joint_position": {key: _summary(value) for key, value in joints.items()},
        "end_effector_pose": {key: _summary(value) for key, value in ee_poses.items()},
        "gripper_state": {key: _summary(value) for key, value in grippers.items()},
        "tracking_error_l2": _tracking_errors(observation),
    }

    temporal: dict[str, Any] = {
        "sampling_rate_hz": frequency,
        "window_samples": 1 if previous_observation is None else 2,
        "native_step_window": _window_features(signal_window),
    }
    progress = {"state": "unknown", "reason": "no previous public observation"}
    if previous_observation is not None:
        previous_state = _state(previous_observation)
        joint_delta = _delta(joints, _matching(previous_state, "arm_joint_state"))
        pose_delta = {}
        for key, value in _delta(ee_poses, _matching(previous_state, "ee_pose")).items():
            # Pose vectors include quaternion components. Keep the full delta for audit,
            # while progress uses translation only below.
            pose_delta[key] = value
        translation_delta = {}
        previous_poses = _matching(previous_state, "ee_pose")
        for key in sorted(ee_poses.keys() & previous_poses.keys()):
            if ee_poses[key].size >= 3 and previous_poses[key].size >= 3:
                translation_delta[key] = float(
                    np.linalg.norm(ee_poses[key][:3] - previous_poses[key][:3])
                )
        gripper_delta = _delta(grippers, _matching(previous_state, "ee_joint_state"))
        temporal.update(
            joint_delta_l2=joint_delta,
            ee_pose_delta_l2=pose_delta,
            ee_translation_delta_m=translation_delta,
            gripper_delta_l2=gripper_delta,
        )
        moving = any(value > JOINT_EPSILON_RAD for value in joint_delta.values()) or any(
            value > POSITION_EPSILON_M for value in translation_delta.values()
        )
        progress = {
            "state": "moving" if moving else "stationary",
            "thresholds": {
                "joint_delta_rad": JOINT_EPSILON_RAD,
                "ee_translation_delta_m": POSITION_EPSILON_M,
            },
        }

    modalities = {
        "vision_rgb": _modality(
            "measured" if rgb_available else "unavailable", sources=("vision",)
        ),
        "proprioception": _modality(
            "measured" if joints or ee_poses or grippers else "unavailable",
            sources=tuple(sorted(joints | ee_poses | grippers)),
        ),
        "spatial": _modality(
            "derivable" if _has_depth(observation) and _has_calibration(observation) else "limited",
            sources=("vision.depth", "vision.calibration")
            if _has_depth(observation) and _has_calibration(observation)
            else ("vision.rgb",),
            note="metric 3D spatial evidence requires public depth and camera calibration",
        ),
        "force": _modality(
            "measured" if force_names else "unavailable",
            sources=force_names,
            note=(
                None
                if force_names
                else "no public wrench, effort, torque, or gripper-force signal"
            ),
        ),
        "tactile": _modality(
            "measured" if tactile_names else "unavailable",
            sources=tactile_names,
            note=(
                None
                if tactile_names
                else "gripper position is not labeled as tactile measurement"
            ),
        ),
        "contact": _modality(
            "proxy_only" if grippers else "unavailable",
            sources=tuple(sorted(grippers)),
            note="contact is not asserted without a validated detector or measured contact signal",
        ),
    }

    return {
        "schema": SCHEMA,
        "timestamp_unix": float(time.time() if timestamp is None else timestamp),
        "source_boundary": "public_observation_only",
        "modalities": modalities,
        "proprioception": proprioception,
        "temporal": temporal,
        "progress": progress,
        "limitations": [
            "availability is not equivalent to semantic validity",
            "contact proxies must not be reported as measured tactile or force",
            "no simulator-private object pose, contact graph, reward, or scene identity is used",
        ],
    }


@dataclass
class EvidenceTracker:
    """Keep only the preceding detached public observation for temporal deltas."""

    previous: dict | None = None

    def observe(
        self,
        observation: dict,
        *,
        timestamp: float | None = None,
        signal_window: list[dict] | None = None,
    ) -> dict:
        evidence = build_evidence(
            observation,
            self.previous,
            timestamp=timestamp,
            signal_window=signal_window,
        )
        # RGB arrays are large and are not used for two-sample proprioceptive
        # deltas. Keep the smallest detached snapshot that preserves semantics.
        self.previous = {
            "state": {
                key: np.array(value, copy=True)
                for key, value in _state(observation).items()
                if _array(value) is not None
            }
        }
        return evidence

    def reset_history(self) -> None:
        self.previous = None
