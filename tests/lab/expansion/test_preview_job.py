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

"""Boundary plans cover declared domains without resampling failed endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from embodichain.lab.expansion._preview_job import (
    boundary_samples,
    plan_preview,
    collect_preview,
)


def test_endpoint_and_stratified_interior_sampling_is_deterministic():
    points, roles = boundary_samples(7, 5, 17)
    assert roles[:3] == ["center", "lower", "upper"]
    np.testing.assert_array_equal(points[1], np.zeros(7))
    np.testing.assert_array_equal(points[2], np.ones(7))
    assert np.all((points[3:] > 0) & (points[3:] < 1))
    for column in points[3:].T:
        assert sorted((column * 5).astype(int)) == list(range(5))
    np.testing.assert_array_equal(points, boundary_samples(7, 5, 17)[0])
    assert boundary_samples(0, 5, 17)[0].shape == (1, 0)


def test_plan_keeps_job_bounds_and_produces_explicit_motion_draws(tmp_path):
    from embodichain.lab.expansion.configured import load_expansion_job

    root = Path(__file__).resolve().parents[3]
    original = root / "examples/expansion/engine/job.motion.yaml"
    before = original.read_bytes()
    plan = plan_preview(original, tmp_path / "preview", interior=1)
    assert len(plan["records"]) == 4
    assert (
        plan["planned_attempts"] == 7
    )  # exact-zero center is one nominal, not two copies
    low = yaml.safe_load((tmp_path / "preview" / plan["records"][1]["job"]).read_text())
    high = yaml.safe_load(
        (tmp_path / "preview" / plan["records"][2]["job"]).read_text()
    )
    assert low["mass_scale"]["cube"] == [0.98, 0.98] and high["mass_scale"]["cube"] == [
        1.02,
        1.02,
    ]
    low_motion = yaml.safe_load(Path(low["motion_profile"]).read_text())
    assert set(low_motion["trajectory"]["spatial"]["joint_samples"].values()) == {-1.0}
    assert low["collision_profile"] == load_expansion_job(original).collision_profile
    assert original.read_bytes() == before
    with pytest.raises(FileExistsError):
        plan_preview(original, tmp_path / "preview", interior=1)
    with pytest.raises(ValueError, match="Nothing executed"):
        plan_preview(original, tmp_path / "too-large", max_episodes=1)
    assert not (tmp_path / "too-large").exists()


def test_collection_retains_failed_and_unrun_points_without_retry(
    monkeypatch, tmp_path
):
    import embodichain.lab.expansion._preview_job as module

    (tmp_path / "configs").mkdir()
    (tmp_path / "logs").mkdir()
    for i in range(2):
        (tmp_path / f"configs/{i}.yaml").write_text("max_wall_time_s: 10\n")
    plan = dict(
        output=str(tmp_path),
        collection_budget_s=10,
        recipes_per_point=1,
        planned_attempts=2,
        records=[
            dict(
                semantic_id="test",
                index=i,
                role=role,
                job=f"configs/{i}.yaml",
                result=f"jobs/{i}",
                confirmed=0,
                attempts=0,
                planned=1,
                status="unrun",
            )
            for i, role in enumerate(["lower", "upper"])
        ],
    )
    monkeypatch.setattr(module, "load_expansion_job", lambda path: None)
    monkeypatch.setattr(
        module, "preflight_expansion_job", lambda cfg: {"semantic_id": "test"}
    )
    calls = []

    class FailedProcess:
        returncode = 2

        def __init__(self, args, **kw):
            calls.append(args)

        def wait(self, timeout):
            return 2

    monkeypatch.setattr(module.subprocess, "Popen", FailedProcess)
    result = collect_preview(plan)
    assert len(calls) == 2 and all(r["status"] == "rejected" for r in result["records"])
    assert result["confirmed"] == 0 and not result["coverage_complete"]
    assert (
        json.loads((tmp_path / "preview-plan.json").read_text())["records"][0]["role"]
        == "lower"
    )


def test_source_change_stops_before_collecting(monkeypatch, tmp_path):
    import embodichain.lab.expansion._preview_job as module

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/job.yaml").write_text("max_wall_time_s: 10\n")
    plan = {
        "output": str(tmp_path),
        "collection_budget_s": 10,
        "records": [
            {
                "index": 0,
                "job": "configs/job.yaml",
                "semantic_id": "pinned",
                "status": "unrun",
            }
        ],
    }
    monkeypatch.setattr(module, "load_expansion_job", lambda path: None)
    monkeypatch.setattr(
        module, "preflight_expansion_job", lambda cfg: {"semantic_id": "changed"}
    )
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("changed source must not execute"),
    )
    with pytest.raises(ValueError, match="changed after planning"):
        collect_preview(plan)
    assert (
        json.loads((tmp_path / "preview-plan.json").read_text())["records"][0]["status"]
        == "source_changed"
    )


def test_reused_affordance_is_not_full_recipe_coverage():
    from embodichain.lab.expansion._preview_job import _exact_recipe

    sample = {"env_ids": [3, 7], "success": [True, True], "reused": [False, True]}
    metadata = {
        "expansion": [
            {
                "env_id": 7,
                "affordance_requested": 1,
                "trajectory_requested": 1,
                "trajectory_selected": 1,
            }
        ],
        "program_segments": [
            {
                "metadata": {
                    "runtime": {
                        "calls": [
                            {
                                "plan_attempts": [
                                    {
                                        "planner_diagnostics": {
                                            "metadata": {
                                                "affordance_sample": {"grasp": sample}
                                            }
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        ],
    }
    assert not _exact_recipe(metadata)
    sample["reused"][1] = False
    assert _exact_recipe(metadata)
    metadata["expansion"][0]["trajectory_selected"] = 0
    assert not _exact_recipe(metadata)
