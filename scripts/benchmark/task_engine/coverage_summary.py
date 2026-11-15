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

"""Summarize measured authoring coverage without counting invalid plans as successes."""

from __future__ import annotations

import json
from pathlib import Path
import argparse
from collections import Counter


def summarize(root: Path) -> dict:
    rows = {}
    total = Counter()
    for record in json.loads((root / "comparison.json").read_text()):
        case = record["case"]
        folder = root / case / "goal"
        if "goal_error" in record:
            rows[case] = {"error": record["goal_error"]}
            continue
        spec = json.loads((folder / "task_spec_v1.json").read_text())
        groups = {
            scope: {p["id"] for p in spec["objectives"][scope]}
            for scope in ("goal", "milestones")
        }
        clauses = spec["interpretation"]["clauses"]
        hits = {
            c["target"]
            for c in clauses
            if c["quote"] and c["quote"] in spec["inputs"]["instruction"]
        }
        cov = record["goal_coverage"]
        history = folder / "planning_history.json"
        history = json.loads(history.read_text()) if history.exists() else None
        first = history["attempts"][0]["coverage"] if history else cov
        counts = {
            "valid": 1,
            "passed": int(cov["status"] == "covered"),
            "conflicts": len(cov["resource_conflicts"]),
            "cases_with_conflicts": int(bool(cov["resource_conflicts"])),
            "quoted": len(hits & set.union(*groups.values())),
            "first_valid": int(first is not None),
            "first_passed": int(first is not None and first["status"] == "covered"),
            "first_conflicts": len(first["resource_conflicts"]) if first else 0,
            "replaced": int(bool(history and history["selected_round"] > 0)),
        }
        for scope, ids in groups.items():
            counts[scope] = len(ids)
            counts[scope + "_covered"] = len(ids & cov["producers"].keys())
            counts[scope + "_first"] = (
                len(ids & first["producers"].keys()) if first else 0
            )
        total.update(counts)
        rows[case] = counts
    return {"cases": len(rows), "totals": dict(total), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    print(
        json.dumps(summarize(parser.parse_args().cohort), ensure_ascii=False, indent=2)
    )
