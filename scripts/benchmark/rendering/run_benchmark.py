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

"""Camera experiment launcher; process lifecycle belongs to benchmark.core.

Run: python -m scripts.benchmark camera-pilot --help
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import os
from pathlib import Path
import sys

from scripts.benchmark.core.artifacts import create_experiment_directory
from scripts.benchmark.core.contracts import Budget, ExperimentDefinition
from scripts.benchmark.core.execution import (
    execute_worker,
    repeat_schedule,
    run_experiment,
)
from scripts.benchmark.core.records import RunSpec

# Retain the v0.1 import surface while the shared implementation has one owner.
__all__ = ["execute_worker", "main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve the two installed runtimes and execute the shared run plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        type=Path,
        help="Rebuild an existing result directory without simulation",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("camera_pilot.json")
    )
    parser.add_argument(
        "--backend", choices=("both", "embodichain", "isaaclab"), default="both"
    )
    parser.add_argument("--embodichain-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--isaaclab-root", type=Path)
    parser.add_argument("--isaaclab-python", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/camera-pilot")
    )
    args = parser.parse_args(argv)
    from .report import rebuild_report

    if args.report_only is not None:
        print(rebuild_report(args.report_only.resolve()))
        return
    if args.repeats < 1 or not 0 < args.timeout_s < float("inf"):
        parser.error("repeats and timeout-s must be positive")
    platforms = (
        ["embodichain", "isaaclab"] if args.backend == "both" else [args.backend]
    )
    if "embodichain" in platforms and not args.embodichain_python.is_file():
        parser.error("--embodichain-python must point to an installed interpreter")
    if "isaaclab" in platforms:
        if (
            args.isaaclab_root is None
            or not (args.isaaclab_root / "isaaclab.sh").is_file()
        ):
            parser.error("--isaaclab-root must contain the installed isaaclab.sh")
        lab_python = (
            args.isaaclab_python or args.isaaclab_root / "env_isaaclab/bin/python"
        )
        if not lab_python.is_file():
            parser.error("--isaaclab-python must point to the installed environment")
    from .workload import PilotCfg, config_hash

    cfg = PilotCfg.load(args.config)
    definition = ExperimentDefinition(
        experiment_id="camera-pilot",
        definition_version="1.0",
        parameter_matrix={
            "backend": tuple(platforms),
            "repeat": tuple(range(args.repeats)),
        },
        budget=Budget(
            max_runs=len(platforms) * args.repeats,
            max_attempts=len(platforms) * args.repeats,
            wall_time_s=args.timeout_s * len(platforms) * args.repeats,
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
    )
    root = create_experiment_directory(
        args.output,
        experiment_id="camera-pilot",
        config=asdict(cfg),
        definition=definition,
        assets_manifest={"assets": [], "scene": "procedural_table_three_boxes"},
    )
    repo = Path(__file__).resolve().parents[3]
    worker = Path(__file__).with_name("worker.py")
    plans = []
    for backend, repeat in repeat_schedule(platforms, args.repeats):
        run_dir = root / f"r{repeat:02d}_{backend}"
        worker_env = os.environ.copy()
        worker_env["PYTHONPATH"] = (
            str(repo) + os.pathsep + worker_env.get("PYTHONPATH", "")
        )
        worker_env["PYTHONUNBUFFERED"] = "1"
        if backend == "embodichain":
            # Preserve the venv symlink path so Python selects its pyvenv.cfg.
            command = [str(args.embodichain_python.absolute()), str(worker)]
        else:
            worker_env["VIRTUAL_ENV"] = str(lab_python.absolute().parent.parent)
            worker_env.pop("CONDA_PREFIX", None)
            command = [
                str(args.isaaclab_root.resolve() / "isaaclab.sh"),
                "-p",
                str(worker),
            ]
        command += [
            "--backend",
            backend,
            "--config",
            str(root / "config.json"),
            "--output",
            str(run_dir),
        ]
        plans.append(
            RunSpec(
                backend,
                repeat,
                tuple(command),
                run_dir,
                case_id=config_hash(cfg),
                timeout_s=args.timeout_s,
                env=worker_env,
            )
        )
    print(f"Run directory: {root}", flush=True)
    try:
        rows = run_experiment(
            root,
            plans,
            experiment_id="camera-pilot",
            budget=definition.budget,
        )
    except KeyboardInterrupt:
        if (root / "runs.json").exists():
            print(rebuild_report(root), flush=True)
        raise SystemExit(130)
    print(rebuild_report(root), flush=True)
    if any(row["status"] != "completed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
