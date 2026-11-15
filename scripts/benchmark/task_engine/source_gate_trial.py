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
"""Repeat the eight development failures or run the frozen new-instruction suite."""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from .hybrid_pilot import trial
from .grade_cohort import CASES, audit, events, quoted_obligations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("development", "holdout", "regression", "camera"),
        required=True,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old = (
        json.loads((args.dataset / "task_descriptions.json").read_text())
        if args.suite in {"development", "regression"}
        else {}
    )
    cases = (
        {
            f"task{i}": {
                "scene": f"task{i}",
                "instruction": old[f"task{i}"]["zh"],
                "expected": CASES[i],
            }
            for i in (1173, 1175, 1177, 1179, 1182, 1187, 1188, 1189)
        }
        if args.suite == "development"
        else json.loads(
            Path(__file__)
            .with_name(
                "camera_holdout.json" if args.suite == "camera" else "holdout.json"
            )
            .read_text()
        )
    )
    if args.suite == "regression":
        cases = {k: cases[k] for k in ("h01", "h05", "h02", "h07")}
        cases["task1179"] = dict(
            scene="task1179", instruction=old["task1179"]["zh"], expected=CASES[1179]
        )
    routes = (
        ("hybrid",)
        if args.suite in {"regression", "development"}
        else ("template", "goal", "hybrid")
    )

    def run(pair):
        case, route = pair
        item = cases[case]
        root = args.output / case
        root.mkdir(exist_ok=True)
        row = trial(
            route,
            case,
            args.dataset / (item["scene"] + ".png"),
            item["instruction"],
            root,
        )
        errors = []
        if row["status"] in {"covered", "uncovered"}:
            stream = events(root, route)
            errors = audit(case, stream, checklist=item["expected"])
            if case.startswith("task"):
                errors += quoted_obligations(root, route, int(case[4:]))
            if item.get("ambiguous"):
                errors.append("Pour outcome requires clarification")
            if item.get("protected") and any(
                item["protected"] in obj for _, obj, _ in stream
            ):
                errors.append("Protected object manipulated")
            if item.get("requires_fluid"):
                errors.append("No qualified liquid-transfer producer")
        row.update(missing=errors, passed=row["status"] == "covered" and not errors)
        (root / (route + "-trial.json")).write_text(
            json.dumps(row, ensure_ascii=False, indent=2)
        )
        print(case, route, row["status"], len(row["calls"]), row["passed"], flush=True)
        return row

    with ThreadPoolExecutor(5) as pool:
        rows = list(
            pool.map(run, ((case, route) for case in cases for route in routes))
        )
    (args.output / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
