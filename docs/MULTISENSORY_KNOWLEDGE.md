# Multisensory evidence and structured knowledge

This extension keeps the `workspace-session-v1` execution boundary and its four
primitives unchanged. It adds host-derived, agent-visible evidence built only
from the same public observation returned by `get_observation()`.

The versioned JSON Schemas are packaged under `multisensory/schemas/`:

- `public-signal-sample-v1.schema.json` defines the public, non-image native-step window;
- `evidence-v1.schema.json` defines every generated `evidence.json` record.

Each observation directory now includes `evidence.json`. The record distinguishes:

- measured signals, such as joints, end-effector poses, and gripper state;
- derived signals, such as inter-observation motion and command-state error;
- proxy-only signals, such as gripper position as a possible contact cue;
- unavailable signals, including force and tactile sensing when the native task
  does not expose them.

`additional_info.frequency` records the native policy rate. During one submitted
program, the host retains a bounded 256-sample, non-image window of the same public observation
after each native action. The window supports command-state tracking error,
no-progress detection, and detrended FFT-based oscillation features. Frequency
features are emitted only when the window contains at least eight samples with a
consistent positive sampling rate; policy-rate metadata by itself is not treated as
frequency evidence.

Missing force or tactile data is never represented by zero. Simulator-private
object poses, contacts, scene identities, shaped rewards, and evaluator state are
not inputs to evidence extraction.

## Knowledge progression

The intended knowledge lifecycle follows ASPIRE's evidence-preserving direction:

```text
executed skill-code instance
        -> reviewed canonical skill
        -> reviewed principle
        -> capability tree and cross-tree relations
```

The first implemented boundary is `multisensory-skill-instance-v1`, validated by
`multisensory.skill_schema.validate_skill_instance`. An instance binds executed
code to task identity, vertical capability, required signal semantics, evidence
files, and outcome. A later consolidation stage will create canonical skills and
principles only from repeated, reviewed instances; a single success must not
directly create a general principle.

PR 2 adds `multisensory-canonical-skill-v1`, `multisensory-principle-v1`, and
`multisensory-knowledge-manifest-v1`. The host validates record relations and exact
code hashes when Playground freezes. Public evidence referenced by a record remains
in the Playground audit trail; its path and SHA-256 digest are carried into the
manifest without copying episode observations into Test.

Vertical capabilities are `localization`, `grasp`, `transport`, `manipulation`,
and `long-horizon`. Spatial information is treated as a derived representation:
metric 3D spatial evidence is available only when both public depth and camera
calibration are present.

## Example skill instance

```yaml
schema: multisensory-skill-instance-v1
id: grasp.verify-after-lift.attempt-004
task: stack_bowls
task_family: pickup-and-place
vertical_capability: grasp
goal: verify that the object remains held after lifting
trigger: gripper closed around a visually localized object
code_ref: observations/000004/code.py
code_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
signal_contract:
  required:
    - path: proprioception.gripper_state
      status: measured
    - path: spatial.object_motion
      status: derived
evidence_refs:
  - observations/000004/evidence.json
outcome:
  success: true
  expected_effect: object and gripper moved together during lift
```

The `status` in a required signal is semantic. A `proxy` must remain labeled as
a proxy through retrieval and reporting; it cannot be promoted to measured force
or tactile evidence.
