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

"""Formal evidence gates cannot be satisfied by smoke or incompatible results."""

from __future__ import annotations
import json
import pytest
from scripts.benchmark.expansion.task_pose_coverage import _report


@pytest.mark.parametrize(
    "change",
    [
        {"suite": "smoke"},
        {"execution_status": "error", "error": "crash"},
        {"declared_budget": {"rollouts": 12}},
        {"coverage_domain": "other"},
        {"source_digest": "other"},
    ],
)
def test_formal_gate_rejects_incomplete_or_incompatible_records(tmp_path, change):
    rows = []
    for seed in (7, 17, 27, 37, 47):
        for group in ("C", "D"):
            count = 10 if group == "C" else 20
            r = {
                "suite": "formal",
                "declared_budget": {
                    "rollouts": 128,
                    "candidates": 4096,
                    "ik_calls": 32768,
                    "planning_calls": 4096,
                    "seconds": 1800,
                },
                "scene": "open",
                "installation": "nominal",
                "seed": seed,
                "group": group,
                "rows": 1,
                "operator": "object_pose",
                "nominal": False,
                "media": False,
                "execution_status": "completed",
                "error": None,
                "coverage_domain": "domain",
                "versions": {"sdk": "v1"},
                "source_digest": "same",
                "coverage": {"pick": count, "place": count},
                "unique_cells": 2 * count,
                "executed": 128,
                "physical_accepted": 128,
                "committed": 128,
                "wall_s": 100,
                "peak_rss_mib": 100,
                "path": str(tmp_path / "missing"),
            }
            if group == "D":
                r.update(change)
            rows.append(r)
    _report(tmp_path, rows)
    assert (
        json.loads((tmp_path / "acceptance.json").read_text())["open"]["status"]
        == "unqualified"
    )
