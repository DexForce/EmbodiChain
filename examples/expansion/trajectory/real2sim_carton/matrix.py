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

"""Execute every real-demonstration candidate in an independent B=1 host."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml
from embodichain.utils.config_paths import resolve_config_path

from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
    load_real2sim_tabletop_profile,
)
from examples.expansion.trajectory.real2sim_carton._audit import _check_case
from examples.expansion.trajectory.real2sim_carton.prepare import write_json, write_yaml

__all__: list[str] = []


def run(
    prepared: Path,
    output: Path,
    *,
    jobs: int = 2,
    timeout: float = 300.0,
    task: Path | None = None,
    profile_path: Path | None = None,
) -> Path:
    if not 1 <= jobs <= 3:
        raise ValueError("one to three independent B=1 workers are supported")
    if not json.loads((prepared / "qualification.json").read_text())["admitted"]:
        raise ValueError("the real input must pass qualification before the matrix")
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        "measured": (
            prepared / "task.measured.yaml" if task is None else task,
            (
                prepared / "profile.measured.yaml"
                if profile_path is None
                else profile_path
            ),
        ),
    }
    cases = []
    for origin, (task, path) in sources.items():
        profile = load_real2sim_tabletop_profile(path)
        if len(profile.layouts) != 13 or len(profile.recipes) != 6:
            raise ValueError("each origin must declare the complete 13 x 6 matrix")
        payload = yaml.safe_load(path.read_text())
        for layout in payload["layouts"]:
            for recipe in profile.recipes:
                cases.append((origin, task, payload, layout, recipe))
    # Use this checkout in subprocesses without changing the shared editable
    # installation. Each process owns its simulator, recorder and one attempt.
    repository = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
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
            [*(str(path) for path in paths), environment.get("PYTHONPATH", "")]
        ),
    )

    def one(case):
        origin, task, payload, layout, recipe = case
        directory = output / origin / layout["id"] / recipe
        directory.mkdir(parents=True)
        profile_path = directory / "profile.yaml"
        write_yaml(profile_path, {**payload, "layouts": [layout], "recipes": [recipe]})
        command = [
            sys.executable,
            "-m",
            "embodichain",
            "run-task",
            "--gym-config",
            str(task),
            "--real2sim-profile",
            str(profile_path),
            "--dataset-dir",
            str(directory / "dataset"),
            "--device",
            "cpu",
            "--headless",
            "--renderer",
            "no-render",
            "--disable-sensor",
            "--max_episodes",
            "1",
        ]
        result = dict(
            origin=origin,
            layout=layout["id"],
            recipe=recipe,
            status="failed",
            attempted=True,
            dataset=None,
            log=str(directory / "run-task.log"),
            profile=str(profile_path),
            error=None,
            audit=None,
        )
        started = time.monotonic()
        try:
            with Path(result["log"]).open("w") as log:
                process = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            result["exit_code"] = process.returncode
            roots = list((directory / "dataset").glob("*/meta/info.json"))
            if len(roots) != 1:
                raise ValueError("run-task did not create exactly one recorder")
            root = roots[0].parents[1]
            result["dataset"] = str(root)
            rows_path = root / "meta/embodichain_episodes.jsonl"
            rows = (
                [
                    json.loads(line)
                    for line in rows_path.read_text().splitlines()
                    if line.strip()
                ]
                if rows_path.exists()
                else []
            )
            failures = sorted((root / "meta/real2sim/failed-attempts").glob("*.json"))
            result["failed_attempts"] = [str(path) for path in failures]
            if process.returncode or len(rows) != 1:
                details = []
                for path in failures:
                    failure = json.loads(path.read_text())
                    if failure.get("error"):
                        details.append(str(failure["error"]))
                    for segment in (failure.get("execution") or {}).get("segments", []):
                        runtime = segment["metadata"].get("runtime") or {}
                        details.extend(
                            f"{segment['name']}: {item['message']}"
                            for item in runtime.get("failures", [])
                        )
                raise ValueError(
                    "; ".join(details)
                    or f"run-task exited {process.returncode} without a confirmed episode"
                )
            if (rows[0]["layout"], rows[0]["recipe"]) != (layout["id"], recipe):
                raise ValueError("committed row differs from the scheduled candidate")
            audit = _check_case(
                root,
                rows[0],
                load_real2sim_tabletop_profile(profile_path),
                expected_origin=origin,
            )
            audit["yaw_error_deg"] = next(
                item["metrics"]["placement_yaw_error_deg"]
                for item in rows[0]["validation_checks"]
                if item["id"] == "task"
            )
            result.update(status="passed", audit=audit)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "result.json", result)
        print(f"{origin}/{layout['id']}/{recipe}: {result['status']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, cases))
    summaries = {}
    for origin in sources:
        rows = [row for row in results if row["origin"] == origin]
        passed = sum(row["status"] == "passed" for row in rows)
        summaries[origin] = dict(
            expected=78,
            attempted=len(rows),
            passed=passed,
            failed=len(rows) - passed,
            matrix_complete=passed == 78,
        )
    report = dict(
        schema="real2sim-carton-matrix/v1",
        expected=78,
        attempted=len(results),
        report_complete=len(results) == 78,
        sources=summaries,
        all_passed=all(item["matrix_complete"] for item in summaries.values()),
        cases=results,
        scope="Public real motion in a declared cuboid proxy; sampled collision checks; no hardware/dynamics qualification",
    )
    path = output / "acceptance-report.json"
    write_json(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--task", type=str)
    parser.add_argument("--profile", type=str)
    args = parser.parse_args()
    path = run(
        args.prepared.resolve(),
        args.output.resolve(),
        jobs=args.jobs,
        timeout=args.timeout,
        task=None if args.task is None else resolve_config_path(args.task).resolve(),
        profile_path=(
            None
            if args.profile is None
            else resolve_config_path(args.profile).resolve()
        ),
    )
    print(path)
    if not json.loads(path.read_text())["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
