# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Launcher for all pure-rendering R-series experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys

from scripts.benchmark.core.artifacts import create_experiment_directory
from scripts.benchmark.core.contracts import Budget, ExperimentDefinition
from scripts.benchmark.core.execution import run_experiment
from scripts.benchmark.core.planning import build_run_plan
from scripts.benchmark.rendering.suite import (
    EXPERIMENT_IDS,
    case_config,
    experiment_catalog,
    expand_experiment_cases,
    metric_definitions,
    scene_spec,
)

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    """Build the suite CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("all", *EXPERIMENT_IDS),
        default="all",
        help="R-series experiment to run; all creates one directory per track.",
    )
    parser.add_argument(
        "--report-only",
        type=Path,
        help="Rebuild one stored suite directory without loading a simulator.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List frozen R-series matrices."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plans without starting workers."
    )
    parser.add_argument(
        "--backend", choices=("both", "embodichain", "isaaclab"), default="both"
    )
    parser.add_argument("--embodichain-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--isaaclab-root", type=Path)
    parser.add_argument("--isaaclab-python", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--measured-frames", type=int, default=50)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--smoke", action="store_true", help="Use one repeat and three measured frames."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/rendering-suite")
    )
    return parser


def _print_catalog() -> None:
    """Print the stable experiment catalog without simulator imports."""
    for experiment_id in EXPERIMENT_IDS:
        experiment = experiment_catalog()[experiment_id]
        print(f"{experiment_id}: {experiment.title}")
        print(f"  question: {experiment.question}")
        for case in expand_experiment_cases(experiment_id):
            print(f"  {case.case_id}: {json.dumps(case.parameters, sort_keys=True)}")


def _runtime_environment(repo: Path) -> dict[str, str]:
    """Build the source import environment for an isolated worker."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _run_one(
    args: argparse.Namespace,
    experiment_id: str,
    *,
    platforms: Sequence[str],
    repeats: int,
    warmup_frames: int,
    measured_frames: int,
) -> Path:
    """Freeze and execute one R-series matrix."""
    experiment = experiment_catalog()[experiment_id]
    cases = expand_experiment_cases(experiment_id)
    configs = {
        case.case_id: case_config(
            experiment_id,
            case.parameters,
            warmup_frames=warmup_frames,
            measured_frames=measured_frames,
        ).to_dict()
        for case in cases
    }
    definition = ExperimentDefinition(
        experiment_id=experiment_id,
        definition_version="1.0",
        parameter_matrix=experiment.parameter_matrix,
        budget=Budget(
            max_runs=len(cases) * len(platforms) * repeats,
            max_attempts=len(cases) * len(platforms) * repeats,
            wall_time_s=args.timeout_s * len(cases) * len(platforms) * repeats,
        ),
        quality_protocol={
            "freshness_probe": "camera_eye_x_plus_0.4m_then_restore",
            "sample_nonempty": True,
            "status": "not_qualified",
        },
        comparison_invariants=(
            "config_sha256",
            "hardware_id",
            "metrics.boundary",
            "metrics.completion",
        ),
        metric_definitions=metric_definitions(),
    )
    root = create_experiment_directory(
        args.output / experiment_id,
        experiment_id=experiment_id,
        config={
            "experiment_id": experiment_id,
            "cases": configs,
            "scene": scene_spec(),
            "backends": list(platforms),
            "repeats": repeats,
        },
        definition=definition,
        assets_manifest={"assets": [], "scene": "procedural_table_three_boxes"},
    )
    repo = Path(__file__).resolve().parents[3]
    worker = Path(__file__).with_name("suite_worker.py")
    lab_python = None
    if "isaaclab" in platforms:
        lab_python = (
            args.isaaclab_python or args.isaaclab_root / "env_isaaclab/bin/python"
        )

    def command_factory(
        backend: str, case: object, repeat: int, run_dir: Path
    ) -> Sequence[str]:
        """Build one isolated worker command from a frozen matrix case."""
        environment = _runtime_environment(repo)
        if backend == "embodichain":
            command = [str(args.embodichain_python.absolute()), str(worker)]
        else:
            assert lab_python is not None
            environment["VIRTUAL_ENV"] = str(lab_python.absolute().parent.parent)
            environment.pop("CONDA_PREFIX", None)
            command = [
                str(args.isaaclab_root.resolve() / "isaaclab.sh"),
                "-p",
                str(worker),
            ]
        # The environment itself stays in RunSpec and is never written to the manifest.
        command.extend(
            [
                "--backend",
                backend,
                "--experiment-id",
                experiment_id,
                "--case-id",
                case.case_id,
                "--config",
                str(root / "config.json"),
                "--output",
                str(run_dir),
            ]
        )
        command_factory.environments[(backend, case.case_id, repeat)] = environment
        return command

    command_factory.environments = {}
    plan = build_run_plan(
        definition,
        backends=platforms,
        repeats=repeats,
        output_root=root,
        command_factory=command_factory,
        timeout_s=args.timeout_s,
    )
    runs = tuple(
        run.__class__(
            backend=run.backend,
            repeat=run.repeat,
            command=run.command,
            output=run.output,
            case_id=run.case_id,
            timeout_s=run.timeout_s,
            env=command_factory.environments[(run.backend, run.case_id, run.repeat)],
            attempt=run.attempt,
        )
        for run in plan.runs
    )
    (root / "plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Run directory: {root}", flush=True)
    if args.dry_run:
        print(f"Planned {len(runs)} runs for {experiment_id}", flush=True)
        return root
    rows = run_experiment(
        root,
        runs,
        experiment_id=experiment_id,
        budget=definition.budget,
    )
    from .suite_report import rebuild_report

    print(rebuild_report(root), flush=True)
    if any(row["status"] != "completed" for row in rows):
        raise SystemExit(1)
    return root


def main(argv: Sequence[str] | None = None) -> None:
    """List, run or rebuild the complete pure-rendering benchmark suite."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        _print_catalog()
        return
    from .suite_report import rebuild_report

    if args.report_only is not None:
        print(rebuild_report(args.report_only.resolve()))
        return
    if args.repeats < 1 or args.warmup_frames < 0 or args.measured_frames < 1:
        parser.error(
            "repeats must be positive, warmup nonnegative and measured frames positive"
        )
    if not 0 < args.timeout_s < float("inf"):
        parser.error("timeout-s must be positive and finite")
    if args.smoke:
        args.repeats, args.warmup_frames, args.measured_frames = 1, 1, 3
    platforms = (
        ("embodichain", "isaaclab") if args.backend == "both" else (args.backend,)
    )
    if "embodichain" in platforms and not args.embodichain_python.is_file():
        parser.error("--embodichain-python must point to an installed interpreter")
    if "isaaclab" in platforms:
        if (
            args.isaaclab_root is None
            or not (args.isaaclab_root / "isaaclab.sh").is_file()
        ):
            parser.error("--isaaclab-root must contain the installed isaaclab.sh")
        if args.isaaclab_python is None:
            args.isaaclab_python = args.isaaclab_root / "env_isaaclab/bin/python"
        if not args.isaaclab_python.is_file():
            parser.error("--isaaclab-python must point to the installed environment")
    experiment_ids = EXPERIMENT_IDS if args.experiment == "all" else (args.experiment,)
    for experiment_id in experiment_ids:
        _run_one(
            args,
            experiment_id,
            platforms=platforms,
            repeats=args.repeats,
            warmup_frames=args.warmup_frames,
            measured_frames=args.measured_frames,
        )


if __name__ == "__main__":
    main()
