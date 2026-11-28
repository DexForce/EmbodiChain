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


def test_one_world_schedule_and_interrupted_receipts(tmp_path, monkeypatch):
    from embodichain.lab.expansion import workflow

    source = (
        Path(__file__).resolve().parents[3]
        / "examples/expansion/engine/job.preview.yaml"
    )
    root = tmp_path / "preview"
    plan = plan_preview(source, root, interior=1)

    def interrupt(cfg, *, schedule):
        assert cfg.batch_size == 8 and cfg.target_episodes == cfg.max_attempts == 14
        assert [(pt.index, i) for pt, i in schedule.rows] == [
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (0, 0),
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
        ]
        assert list(schedule.rows[1][0].cfg.mass_scale["cube"]) == [0.98, 0.98]
        attempt = Path(cfg.output_dir) / "attempts/00000001/attempt.json"
        attempt.parent.mkdir(parents=True)
        attempt.write_text(
            json.dumps(dict(ordinal=1, status="rejected", reason="collision"))
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(workflow, "_run_expansion_job", interrupt)
    with pytest.raises(KeyboardInterrupt):
        collect_preview(plan)
    result = json.loads((root / "preview-plan.json").read_text())
    assert result["records"][1]["outcomes"] == [
        dict(ordinal=1, status="rejected", reason="collision")
    ]
    assert result["num_envs"] == 8 and not result["complete"]
    assert result["confirmed"] == 0 and not result["coverage_complete"]


def test_frozen_point_source_change_is_reported_before_collection(tmp_path):
    source = (
        Path(__file__).resolve().parents[3]
        / "examples/expansion/engine/job.motion.yaml"
    )
    plan = plan_preview(source, tmp_path / "preview", interior=1)
    job = Path(plan["output"]) / plan["records"][1]["job"]
    cfg = yaml.safe_load(job.read_text())
    cfg["friction"] = {"cube": [0.5, 0.5]}
    job.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="changed after planning"):
        collect_preview(plan)
    assert plan["records"][1]["status"] == "source_changed"
    assert not (job.parent.parent / plan["shared_result"]).exists()


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
