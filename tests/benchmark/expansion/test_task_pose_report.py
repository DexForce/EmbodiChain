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


def test_boundary_report_derivatives_use_recorded_control_grid(tmp_path):
    import json
    import numpy as np
    from scripts.benchmark.expansion.skill_boundaries import _episodes

    folder = tmp_path / "collection/dataset/object_pose/episodes/test"
    folder.mkdir(parents=True)
    manifest = {
        "episode_id": "test",
        "observations": ["full_joint_positions"],
        "metadata": {
            "expert_action_spec": {"joint_names": [f"joint{i}" for i in range(6)]},
            "program_segments": [{"start_step": 0}, {"start_step": 2, "name": "place"}],
            "expansion_recipe": {"ordinal": 0},
            "initial_state": [],
        },
    }
    np.savez(
        folder / "episode.npz",
        manifest=np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8),
        timestamps=np.arange(5) * 0.04,
        actions=np.repeat(np.array([0, 0.1, 0.15, 0.2])[:, None], 6, axis=1),
        observation_0=np.repeat(
            np.array([0, 0.01, 0.03, 0.06, 0.10])[:, None], 6, axis=1
        ),
    )
    row = _episodes(tmp_path)[0]
    assert row["command_jump_rad"] == pytest.approx(0.05)
    assert row["command_difference_speed_rad_s"] == pytest.approx(1.25)
    assert row["command_difference_acceleration_rad_s2"] == pytest.approx(31.25)
    assert row["measured_control_speed_rad_s"] == pytest.approx(0.5)
