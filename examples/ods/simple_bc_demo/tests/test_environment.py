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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.ods.simple_bc_demo.environment import GOAL, PickPlaceEnv


def _env() -> tuple[PickPlaceEnv, torch.Tensor, torch.Tensor, torch.Tensor]:
    env = PickPlaceEnv.__new__(PickPlaceEnv)
    pose = torch.tensor([[*GOAL, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    velocity = torch.zeros(2, 6)
    qpos = torch.zeros(2, 9)
    qpos[:, 7:] = 0.04
    cube = SimpleNamespace(
        get_local_pose=lambda **kwargs: pose,
        body_data=SimpleNamespace(vel=velocity),
        clear_dynamics=Mock(),
        set_local_pose=Mock(),
    )
    tcp = torch.eye(4).repeat(2, 1, 1)
    tcp[:, :3, 3] = pose[:, :3] + torch.tensor([0.0, 0.0, 0.2])
    env.sim = SimpleNamespace(
        device=torch.device("cpu"), get_rigid_object=lambda name: cube
    )
    env.cfg = SimpleNamespace(events=None, sim_steps_per_control=8)
    env.sim_cfg = SimpleNamespace(physics_dt=1 / 240)
    env._num_envs = 2
    env.robot = SimpleNamespace(
        get_qpos=lambda: qpos,
        get_joint_ids=lambda name: list(range(7)),
        compute_fk=lambda **kwargs: tcp,
    )
    env.goal_pose = pose[:, :3].clone()
    env._lifted = torch.zeros(2, dtype=torch.bool)
    env._success = torch.zeros(2, dtype=torch.bool)
    env._stable_steps = torch.zeros(2, dtype=torch.long)
    return env, pose, velocity, qpos


def test_push_to_goal_without_lift_never_succeeds() -> None:
    env, _, _, _ = _env()
    for _ in range(12):
        env._update_sim_state()
    assert not env.is_task_success().any()


@pytest.mark.parametrize("invalid", ["closed", "moving", "height", "nonfinite"])
def test_success_requires_release_height_stability_and_finite_measurement(
    invalid: str,
) -> None:
    env, pose, vel, qpos = _env()
    env._lifted[:] = True
    if invalid == "closed":
        qpos[:, 7:] = 0
    elif invalid == "moving":
        vel[:, 0] = 0.5
    elif invalid == "height":
        pose[:, 2] += 0.02
    else:
        pose[:, 0] = float("nan")
    for _ in range(12):
        env._update_sim_state()
    assert not env.is_task_success().any()


def test_success_clock_is_step_owned_and_result_queries_are_pure() -> None:
    env, _, _, _ = _env()
    env._lifted[:] = True
    for _ in range(8):
        env._update_sim_state()
    for _ in range(10):
        assert not env.is_task_success().any()
    env._update_sim_state()
    assert env.is_task_success().all()


def test_partial_reset_preserves_other_rows(monkeypatch) -> None:
    from embodichain.lab.gym.envs import EmbodiedEnv

    monkeypatch.setattr(
        EmbodiedEnv, "_initialize_episode", lambda *args, **kwargs: None
    )
    env, _, _, _ = _env()
    env._lifted[:] = True
    env._success[:] = True
    env._stable_steps[:] = 9
    env._initialize_episode([1], save_data=False)
    assert env._success.tolist() == [True, False]
    assert env._lifted.tolist() == [True, False]
    assert env._stable_steps.tolist() == [9, 0]
