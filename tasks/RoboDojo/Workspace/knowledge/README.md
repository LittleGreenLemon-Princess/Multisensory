# Structured knowledge

This directory holds optional machine-readable knowledge created during Playground.
It is frozen with `skills/` and `lessons/` and mounted read-only in Test.

```text
knowledge/
  instances/*.json   executed task-specific skill records
  skills/*.json      reviewed canonical reusable skills
  principles/*.json  evidence-grounded decision rules
  manifest.json      host-generated only when Playground freezes
```

Do not create or edit `manifest.json`; the host regenerates it from exact file,
code, and evidence hashes. Existing Python helpers in `skills/` and Markdown in
`lessons/` remain valid even when there are no structured records.

Use these required fields:

- Instance (`multisensory-skill-instance-v1`): `id`, `task`, `task_family`,
  `vertical_capability`, `goal`, `trigger`, `code_ref`, `code_sha256`,
  `signal_contract`, `evidence_refs`, and `outcome.success`. `code_ref` must name
  the exact executed `observations/NNNNNN/code.py`.
- Canonical skill (`multisensory-canonical-skill-v1`): `id`, `version`, `title`,
  `vertical_capability`, `goal`, `implementation_refs`,
  `supporting_instance_ids`, `signal_contract`, `preconditions`,
  `stop_conditions`, `known_failures`, `evidence_refs`, and `status`.
- Principle (`multisensory-principle-v1`): `id`, `version`, `title`,
  `vertical_capability`, `when`, `decision`, `invariant`, `expected_effects`,
  `exceptions`, `falsifiers`, `supporting_skill_ids`, `signal_contract`,
  `evidence_refs`, and `status`.

Statuses are `candidate`, `validated`, or `retired`. A validated principle needs
at least two canonical skills. A signal contract contains `required` and optional
`optional` arrays; each item has a public signal `path` and one of `measured`,
`derived`, or `proxy`. Required signals cannot be unavailable.

Evidence references must be workspace-relative paths to real public execution
artifacts such as `observations/000004/evidence.json`. Never cite private
simulator state, scores, scene identities, or host ledgers.
