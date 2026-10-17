"""How to start exploration and testing.

    .venv/bin/python -m orchestrator explore --task <dir> --run-dir <dir> --gpu N
    .venv/bin/python -m orchestrator test    --task <dir> --run-dir <dir> --gpu N
    .venv/bin/python -m orchestrator run     --task <dir> --run-dir <dir> --gpu N

``--task`` selects a task definition such as ``RoboDojo``; ``--env-task``
selects the concrete native task. The interpreter comes from task.toml.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from orchestrator import session_runner
from orchestrator.config import load_task


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="task dir, e.g. RoboDojo")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for the simulator")
    parser.add_argument("--env-task", help="native task name; overrides [simulator].env_task")
    parser.add_argument("--test-scenes", type=int, help="override Test scene count for this run")
    parser.add_argument(
        "--test-gpus",
        help="comma-separated physical GPUs for concurrent Test scenes",
    )
    parser.add_argument(
        "--test-gpu-slots",
        type=int,
        default=1,
        help="maximum concurrent Test simulators per GPU (default: 1)",
    )
    parser.add_argument(
        "--knowledge-retrieval",
        choices=("signal_aware", "all", "none"),
        help="override structured-knowledge retrieval for an ablation run",
    )
    parser.add_argument(
        "--knowledge-modalities",
        help="comma-separated enabled modalities; use an empty value for none",
    )
    parser.add_argument(
        "--knowledge-instances",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include or exclude supporting skill instances in the task portfolio",
    )
    parser.add_argument(
        "--knowledge-exceptions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable exception/counterexample ranking penalties",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="embodiedevo",
        description="Test with fresh agents; baselines skip Playground and knowledge transfer.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("explore", help="PlayGround: fixed scene, agent, execution budget")
    _common(ex)
    ex.add_argument("--timeout-sec", type=float, default=None)

    te = sub.add_parser("test", help="test with a fresh agent per scene; baselines start directly")
    _common(te)

    ru = sub.add_parser("run", help="explore, then test; baselines run Test only")
    _common(ru)
    ru.add_argument("--timeout-sec", type=float, default=None)

    export = sub.add_parser("export", help="export one run's ablation metrics as JSON and CSV")
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)

    args = ap.parse_args(argv)
    if args.command == "export":
        from multisensory.experiments import export_run

        export_run(args.run_dir, args.output_dir)
        return 0
    test_gpus = None
    if args.test_gpu_slots <= 0:
        ap.error("--test-gpu-slots must be positive")
    if args.test_gpus is not None:
        test_gpus = [item.strip() for item in args.test_gpus.split(",")]
        if any(not item.isdigit() for item in test_gpus) or len(set(test_gpus)) != len(test_gpus):
            ap.error("--test-gpus must contain unique physical GPU indices")
    elif args.test_gpu_slots > 1:
        if args.gpu is None or not args.gpu.isdigit():
            ap.error("--test-gpu-slots requires --gpu or --test-gpus")
        test_gpus = [args.gpu]
    cfg = load_task(args.task, env_task=args.env_task)
    knowledge = cfg.knowledge
    if args.knowledge_retrieval is not None:
        knowledge = replace(knowledge, enabled=True, retrieval=args.knowledge_retrieval)
    if args.knowledge_modalities is not None:
        modalities = tuple(
            item.strip() for item in args.knowledge_modalities.split(",") if item.strip()
        )
        knowledge = replace(knowledge, enabled=True, modalities=modalities)
    if args.knowledge_instances is not None:
        knowledge = replace(knowledge, enabled=True, include_instances=args.knowledge_instances)
    if args.knowledge_exceptions is not None:
        knowledge = replace(knowledge, enabled=True, use_exceptions=args.knowledge_exceptions)
    cfg = replace(cfg, knowledge=knowledge)
    if cfg.baseline and cfg.knowledge.enabled:
        ap.error("Baseline cannot enable structured knowledge retrieval")
    if args.test_scenes is not None:
        if args.test_scenes <= 0:
            ap.error("--test-scenes must be positive")
        cfg = replace(cfg, test=replace(cfg.test, scenes=args.test_scenes))
    if cfg.baseline and args.command == "explore":
        ap.error("Baseline has no Playground; use test or run")
    if args.command != "test" and not cfg.baseline:
        rc = session_runner.explore(
            cfg, args.run_dir, gpu=args.gpu, timeout_sec=args.timeout_sec
        )
        if rc or args.command == "explore":
            return rc
    return session_runner.test(
        cfg,
        args.run_dir,
        gpu=args.gpu,
        test_gpus=test_gpus,
        gpu_slots=args.test_gpu_slots,
    )


if __name__ == "__main__":
    raise SystemExit(main())
