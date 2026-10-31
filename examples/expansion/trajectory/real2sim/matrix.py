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

"""Run the fixed 9 x 6 synthetic Real2Sim matrix.

Every case runs in its own B=1 simulator process. The report records failures
without dropping cases. Sampled collision and LeRobot output are opt-in.
The current example is not a substitute for the formal run-task acceptance.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from layout_replay import _LAYOUTS

__all__: list[str] = []

_RECIPES = ("nominal", "joint_7", "joint_11", "task_space", "time_115", "time_130")


def main() -> None:
    """Write per-case evidence and an explicit complete diagnostic report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    collision_mode = parser.add_mutually_exclusive_group(required=True)
    collision_mode.add_argument("--diagnostic-no-collision-check", action="store_true")
    collision_mode.add_argument("--validate-sampled-collision", action="store_true")
    parser.add_argument("--save-lerobot", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    options = parser.parse_args()
    if options.save_lerobot and not options.validate_sampled_collision:
        raise ValueError("LeRobot output requires sampled collision validation")
    if not 1 <= options.jobs <= 3:
        raise ValueError("jobs must be between one and three")
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    def run_case(selection):
        layout, recipe = selection
        case = output / layout / recipe
        case.parent.mkdir(parents=True, exist_ok=True)
        logfile = case.parent / f"{recipe}.log"
        with logfile.open("w", encoding="utf-8") as log:
            command = [
                sys.executable,
                str(Path(__file__).with_name("layout_replay.py")),
                "--reference-dir",
                str(options.reference_dir.resolve()),
                "--output-dir",
                str(case),
                "--layout",
                layout,
                "--recipe",
                recipe,
                "--device",
                options.device,
                (
                    "--validate-sampled-collision"
                    if options.validate_sampled_collision
                    else "--diagnostic-no-collision-check"
                ),
            ]
            if options.save_lerobot:
                command.append("--save-lerobot")
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        path = case / "report.json"
        details = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        segments = (details.get("execution") or {}).get("segments", [])
        failure = next(
            (
                item.get("metadata", {}).get("runtime", {}).get("message")
                or item.get("failure_reason")
                for item in segments
                if not item.get("success")
            ),
            None,
        )
        row = {
            "layout": layout,
            "recipe": recipe,
            "diagnostic_passed": bool(
                process.returncode == 0 and details.get("task_verified")
            ),
            "collision_check": details.get("collision_check", "unverified"),
            "geometry_compatibility": details.get(
                "geometry_compatibility", "unverified"
            ),
            "dataset_committed": bool(details.get("dataset_committed")),
            "receipt_confirmed": bool(details.get("receipt", {}).get("confirmed")),
            "error": details.get("error")
            or failure
            or (
                None
                if process.returncode == 0
                else f"process exited {process.returncode}"
            ),
            "report": (path.relative_to(output).as_posix() if path.exists() else None),
            "log": logfile.relative_to(output).as_posix(),
        }
        print(
            f"{layout} / {recipe}: {'passed' if row['diagnostic_passed'] else 'failed'}",
            flush=True,
        )
        return row

    choices = [
        (layout, recipe) for layout in tuple(_LAYOUTS)[:9] for recipe in _RECIPES
    ]
    with ThreadPoolExecutor(max_workers=options.jobs) as pool:
        rows = list(pool.map(run_case, choices))
    collision_qualified = options.validate_sampled_collision and all(
        row["diagnostic_passed"]
        and row["collision_check"] == "sampled_plan_and_observed_geometry"
        and row["geometry_compatibility"] == "asset_identity_matched"
        for row in rows
    )
    persistence_qualified = options.save_lerobot and all(
        row["dataset_committed"] and row["receipt_confirmed"] for row in rows
    )
    report = {
        "origin": "synthetic_reference",
        "expected_cases": 54,
        "diagnostic_passed": sum(row["diagnostic_passed"] for row in rows),
        "collision_qualified": collision_qualified,
        "collision_scope": (
            "sampled plan and measured states"
            if options.validate_sampled_collision
            else "not checked"
        ),
        "persistence_qualified": persistence_qualified,
        "formal_acceptance_complete": False,
        "remaining_qualification": [
            "source-qualified run-task collection dispatcher",
            "authoritative gripper acceleration limits",
        ],
        "cases": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["diagnostic_passed"] != report["expected_cases"]:
        raise RuntimeError("one or more diagnostic cases failed; see report.json")


if __name__ == "__main__":
    main()
