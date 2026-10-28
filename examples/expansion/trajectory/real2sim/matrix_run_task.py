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

"""Run and audit the fixed Real2Sim matrix in independent B=1 shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
    load_real2sim_tabletop_profile,
)

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 3:
        raise ValueError("jobs must be between one and three")
    profile = load_real2sim_tabletop_profile(args.profile)
    payload = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    if len(profile.layouts) != 9 or len(profile.recipes) != 6:
        raise ValueError(
            "the acceptance matrix requires exactly nine layouts and six recipes"
        )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("matrix output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    task = args.task.resolve()
    environment = os.environ.copy()
    environment.update(
        EMBODICHAIN_SIM_EXIT_PROCESS="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )

    def run_layout(layout):
        shard = output / str(layout["id"])
        shard.mkdir()
        local_profile = shard / "profile.yaml"
        local_data = dict(payload)
        local_data["layouts"] = [
            next(value for value in payload["layouts"] if value["id"] == layout["id"])
        ]
        local_profile.write_text(
            yaml.safe_dump(local_data, sort_keys=False), encoding="utf-8"
        )
        log_path = shard / "run-task.log"
        command = [
            sys.executable,
            "-m",
            "embodichain",
            "run-task",
            "--gym-config",
            str(task),
            "--real2sim-profile",
            str(local_profile),
            "--dataset-dir",
            str(shard / "dataset"),
            "--device",
            args.device,
            "--headless",
            "--renderer",
            "no-render",
            "--disable-sensor",
            "--max_episodes",
            str(len(profile.recipes)),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        datasets = list(
            (shard / "dataset").glob("*/meta/real2sim/collection-report.json")
        )
        row = {
            "layout": layout["id"],
            "run_task_exit_code": process.returncode,
            "log": str(log_path),
            "dataset": None,
            "audit": None,
            "error": None,
        }
        if process.returncode or len(datasets) != 1:
            row["error"] = "run-task did not complete the full layout shard"
            print(f"{layout['id']}: failed (exit {process.returncode})", flush=True)
            return row
        dataset = datasets[0].parents[2]
        row["dataset"] = str(dataset)
        audit_path = shard / "audit.json"
        audit_log = shard / "audit.log"
        with audit_log.open("w", encoding="utf-8") as log:
            audited = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("audit.py")),
                    "--dataset",
                    str(dataset),
                    "--profile",
                    str(local_profile),
                    "--output",
                    str(audit_path),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row["audit"] = str(audit_path) if audit_path.is_file() else None
        if audited.returncode:
            row["error"] = "independent shard audit failed"
        print(
            f"{layout['id']}: {'passed' if row['error'] is None else 'failed'}",
            flush=True,
        )
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        shards = list(pool.map(run_layout, profile.layouts))
    cases, failures, attempts = [], [], []
    for shard in shards:
        if shard["error"] is not None:
            failures.append(
                {
                    "layout": shard["layout"],
                    "error": shard["error"],
                    "log": shard["log"],
                }
            )
            continue
        result = json.loads(Path(shard["audit"]).read_text(encoding="utf-8"))
        cases.extend(result["cases"])
        failures.extend(result["failures"])
        attempts.extend(result["failed_attempts"])
    expected = {
        (layout["id"], recipe)
        for layout in profile.layouts
        for recipe in profile.recipes
    }
    actual = {(case["layout"], case["recipe"]) for case in cases}
    for layout, recipe in sorted(expected - actual):
        failures.append(
            {"layout": layout, "recipe": recipe, "error": "missing validated episode"}
        )
    report = {
        "origin": "synthetic_reference",
        "execution": "independent B=1 processes; six sequential episodes per layout",
        "jobs": args.jobs,
        "expected_cases": len(expected),
        "validated_cases": len(cases),
        "failure_count": len(failures),
        "failures": failures,
        "failed_attempts": attempts,
        "synthetic_matrix_complete": not failures and len(cases) == len(expected),
        "real_data_qualified": False,
        "continuous_collision_qualified": False,
        "velocity_limit_origin": profile.velocity_limit_origin,
        "acceleration_limit_origin": profile.acceleration_limit_origin,
        "motion_headroom_fraction": profile.motion_headroom_fraction,
        "metrics": {
            "maximum_fk_position_error_m": max(
                (c["fk_position_error_m"] for c in cases), default=None
            ),
            "maximum_fk_rotation_error_rad": max(
                (c["fk_rotation_error_rad"] for c in cases), default=None
            ),
            "minimum_pick_lift_m": min((c["lift_m"] for c in cases), default=None),
            "maximum_placement_xy_error_m": max(
                (c["xy_error_m"] for c in cases), default=None
            ),
            "maximum_held_translation_drift_m": max(
                (c["held_translation_drift_m"] for c in cases), default=None
            ),
            "maximum_held_rotation_drift_deg": max(
                (c["held_rotation_drift_deg"] for c in cases), default=None
            ),
            "maximum_speed_ratio": max(
                (c["maximum_speed_ratio"] for c in cases), default=None
            ),
            "maximum_acceleration_ratio": max(
                (c["maximum_acceleration_ratio"] for c in cases), default=None
            ),
        },
        "shards": shards,
        "cases": cases,
    }
    path = output / "acceptance-report.json"
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"{len(cases)}/{len(expected)} independently validated; {len(failures)} failures: {path}"
    )
    if failures or len(cases) != len(expected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
