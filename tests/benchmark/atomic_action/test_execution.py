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

"""Drive commands must not stand in for actual physical measurements."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark.atomic_action import execution

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("incorrect_plan", [False, True])
def test_replay_scores_actual_state_and_keeps_failed_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, incorrect_plan: bool
) -> None:
    """A stuck arm fails despite a valid commanded trajectory; other rows stay local."""
    commands, steps = [], []
    positions = torch.tensor([[[0.0, 0.0], [0.3, 0.2], [0.0, 0.0]]]).repeat(2, 1, 1)
    if incorrect_plan:
        positions.zero_()
    trajectory = SimpleNamespace(
        positions=positions, dt=torch.full((2, 2), 0.01), robot_dof=2, waypoint_count=3
    )
    segment = SimpleNamespace(name="move_joints", start=0, stop=2)
    plan = SimpleNamespace(
        skill_id="move_joints",
        segments=(segment,),
        diagnostics=SimpleNamespace(messages=()),
    )
    compiled = SimpleNamespace(
        trajectory=trajectory,
        action_plans=(plan,),
        plan_success=torch.tensor([True, False]),
        segment=lambda *_: segment,
    )
    scene = SimpleNamespace(
        sim=SimpleNamespace(
            sim_config=SimpleNamespace(physics_dt=0.01),
            update=lambda *, step: steps.append(step),
        ),
        robot=SimpleNamespace(
            get_joint_ids=lambda *, name: [0, 1],
            set_qpos=lambda qpos, *, target: commands.append((qpos.clone(), target)),
        ),
        engine=SimpleNamespace(
            initial_context=lambda **_: None, compile=lambda *_: compiled
        ),
        invocations=(
            SimpleNamespace(motion_policy=SimpleNamespace(strategy="motion_gen")),
        ),
        initial_scene=None,
        metadata={
            "skill": "move_joint",
            "arm_joint_ids": [0, 1],
            "joint_targets": [[0.3, 0.2]],
        },
        observe=lambda: {"qpos": torch.zeros(2, 2)},
    )
    monkeypatch.setattr(execution, "build_scene", lambda _: scene)
    monkeypatch.setattr(execution, "create_affordance_sampling_context", lambda _: None)
    monkeypatch.setattr(
        execution, "_memory", lambda **_: {"cpu": 0.0, "gpu": 0.0, "peak": 0.0}
    )
    monkeypatch.setattr(
        execution,
        "_augment",
        lambda *_: (
            trajectory,
            [{"reference_row": row, "operator": "none"} for row in range(2)],
        ),
    )
    args = SimpleNamespace(num_envs=2, _recipe_indices=(0, 1), _config={"seed": 7})
    result = execution._run(
        args,
        {"steps_per_waypoint": 16, "settle_steps": 80, "save_trajectories": False},
        tmp_path,
    )
    assert all(target for _, target in commands)
    assert steps == [16, 16, 16, 80, 80]
    assert result["rows"][0]["failure_stage"] == "reached"
    assert result["rows"][0]["metrics"]["max_tracking_error_rad"] == pytest.approx(
        0.0 if incorrect_plan else 0.3
    )
    assert result["rows"][1]["failure_stage"] == "planning"
    assert all(not row["success"] for row in result["rows"])
    assert all(
        row["expansion"]["protected_max_joint_delta_rad"] == 0.0
        for row in result["rows"]
    )
