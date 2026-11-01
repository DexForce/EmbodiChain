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

"""Build real-motion prototypes and collect their variants with the official task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from embodichain.data_pipeline.real2sim.replay import export_real2sim_replay_manifest
from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
    load_real2sim_tabletop_profile,
)
from embodichain.utils.config_paths import resolve_config_path
from examples.expansion.trajectory.real2sim_carton._audit import _check_case
from examples.expansion.trajectory.real2sim_carton.prepare import (
    acquire,
    write_json,
    write_yaml,
)

__all__: list[str] = []
_RECIPES = ("nominal", "joint_7", "joint_11", "task_space", "time_115", "time_130")


def _deployment(
    prepared: Path, output: Path, layouts: list[str] | None, recipes: list[str] | None
) -> tuple[Path, Path, int]:
    if not json.loads((prepared / "qualification.json").read_text())["admitted"]:
        raise ValueError("the real input must pass qualification before collection")
    task_path = prepared / "task.measured.yaml"
    original_profile = prepared / "profile.measured.yaml"
    profile = load_real2sim_tabletop_profile(original_profile)
    payload = yaml.safe_load(original_profile.read_text())
    selected_layouts = ["baseline"] if layouts is None else layouts
    selected_recipes = list(_RECIPES) if recipes is None else recipes
    by_name = {item["id"]: item for item in payload["layouts"]}
    if (
        len(set(selected_layouts)) != len(selected_layouts)
        or set(selected_layouts) - set(by_name)
        or len(set(selected_recipes)) != len(selected_recipes)
        or set(selected_recipes) - set(profile.recipes)
    ):
        raise ValueError("layouts and recipes must be unique supported selections")
    payload.update(
        layouts=[by_name[n] for n in selected_layouts],
        recipes=selected_recipes,
        source_artifacts=profile.source_artifacts,
    )
    output.mkdir(parents=True, exist_ok=False)
    selected_profile = output / "profile.yaml"
    write_yaml(selected_profile, payload)
    task = yaml.safe_load(task_path.read_text())
    for key in ("environment", "embodiment"):
        task[key]["component"] = str(
            resolve_config_path(
                task[key]["component"], base_dir=task_path.parent
            ).resolve()
        )
    for key in ("program", "integration", "execution_policy"):
        task["task_program"][key] = str(
            resolve_config_path(
                task["task_program"][key], base_dir=task_path.parent
            ).resolve()
        )
    count = len(selected_layouts) * len(selected_recipes)
    task.update(
        id="CartonStockingPanda-v1",
        real2sim={"profile": "profile.yaml"},
        collection={"target_episodes": count, "max_attempts": 1},
    )
    selected_task = output / "task.yaml"
    write_yaml(selected_task, task)
    return selected_task, selected_profile, count


def _audit_collection(
    destination: Path, profile_path: Path, expected: int
) -> dict[str, object]:
    roots = list(destination.glob("*/meta/info.json"))
    if len(roots) != 1:
        raise ValueError("collection must own exactly one recorder")
    root = roots[0].parents[1]
    rows = [
        json.loads(line)
        for line in (root / "meta/embodichain_episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    profile = load_real2sim_tabletop_profile(profile_path)
    schedule = {
        (item["id"], recipe) for item in profile.layouts for recipe in profile.recipes
    }
    actual = [(row["layout"], row["recipe"]) for row in rows]
    if len(rows) != expected or len(set(actual)) != expected or set(actual) != schedule:
        raise ValueError("confirmed episodes do not match the selected schedule")
    cases = [
        _check_case(root, row, profile, expected_origin="measured") for row in rows
    ]
    export_real2sim_replay_manifest([root], destination / "attachments.json")
    return {
        "origin": "measured",
        "dataset": str(root),
        "confirmed_episodes": len(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New artifact directory"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Pinned original DROID files; otherwise use the shared asset cache",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Reuse an example output or prepared source directory",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--layouts", nargs="+")
    parser.add_argument("--recipes", nargs="+", choices=_RECIPES)
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="All 13 x 6 real-demonstration candidates, offline replay and optional video",
    )
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.full_matrix and (
        args.prepare_only or args.layouts or args.recipes or args.viser
    ):
        parser.error("--full-matrix uses the complete fixed 78-candidate schedule")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = (args.source_dir or output / "source").resolve()
    environment = os.environ.copy()
    repository = Path(__file__).resolve().parents[4]
    paths = [
        repository,
        repository / "extensions",
        repository / "extensions/embodichain_tasks",
    ]
    environment.update(
        EMBODICHAIN_SIM_EXIT_PROCESS="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join(
            [*(str(p) for p in paths), environment.get("PYTHONPATH", "")]
        ),
    )
    stages = []

    def execute(
        name: str, module: str, arguments: list[str], *, allow_failure: bool = False
    ) -> int:
        with (output / f"{name}.log").open("w") as log:
            status = subprocess.run(
                [sys.executable, "-m", module, *arguments],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        stages.append({"stage": name, "exit_code": status, "log": f"{name}.log"})
        write_json(output / "stages.json", stages)
        if status and not allow_failure:
            raise RuntimeError(f"{name} failed; see {output/(name+'.log')}")
        return status

    prefix = "examples.expansion.trajectory.real2sim_carton."
    if args.prepared is None:
        acquire(source)
        prepared = output / "prepared"
        execute(
            "prepare",
            prefix + "prepare",
            ["prepare", "--source-dir", str(source), "--output", str(prepared)],
        )
    else:
        prepared = args.prepared.resolve()
        if (prepared / "prepared").is_dir():
            prepared /= "prepared"
        if args.full_matrix:
            acquire(source)
    if not args.full_matrix:
        task, profile, count = _deployment(
            prepared, output / "deployment", args.layouts, args.recipes
        )
        destination = output / "dataset"
        arguments = [
            "run-task",
            "--gym-config",
            str(task),
            "--dataset-dir",
            str(destination),
        ]
        if args.viser:
            arguments.append("--viser")
        print(shlex.join(["embodichain", *arguments]), flush=True)
        if args.prepare_only:
            return
        report = {
            "expected_episodes": count,
            "confirmed_episodes": 0,
            "report_complete": False,
            "sources": [],
        }
        write_json(output / "collection-report.json", report)
        execute("collect", "embodichain", arguments)
        verified = _audit_collection(destination, profile, count)
        report.update(
            confirmed_episodes=count, report_complete=True, sources=[verified]
        )
        write_json(output / "collection-report.json", report)
        print(f"Report: {output/'collection-report.json'}")
        return
    status = execute(
        "matrix",
        prefix + "matrix",
        [
            "--prepared",
            str(prepared),
            "--output",
            str(output / "matrix"),
            "--jobs",
            str(args.jobs),
        ],
        allow_failure=True,
    )
    report = output / "matrix/acceptance-report.json"
    if not report.is_file() or not json.loads(report.read_text())["report_complete"]:
        raise RuntimeError("the matrix did not report all 78 candidates")
    execute(
        "replay",
        prefix + "export",
        [
            "--prepared",
            str(prepared),
            "--matrix",
            str(report),
            "--source-dir",
            str(source),
            "--output",
            str(output / "replay"),
        ],
    )
    if not args.no_video:
        execute(
            "video",
            prefix + "video",
            [
                "--prepared",
                str(prepared),
                "--replay",
                str(output / "replay"),
                "--output",
                str(output / "video"),
            ],
        )
    print(f"Report: {report}")
    print(f"Replay: {output/'replay/index.html'}")
    if status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
