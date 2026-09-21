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
import torch
import pytest
from embodichain_tasks.classic_control.humanoid.humanoid_run import (
    HumanoidRunEnv,
    _CONFIG,
)
from embodichain_tasks.classic_control.humanoid.mdp import (
    humanoid_observation,
    humanoid_reward,
)


def test_humanoid_action_clipping_gear_and_joint_mapping():
    calls = []
    gear = torch.tensor([100.0, 300.0, 25.0])
    env = SimpleNamespace(
        _action=torch.zeros(1, 3),
        _gear=gear,
        _effort_limit=torch.tensor([100.0, 50.0, 25.0]),
        policy_joint_ids=torch.tensor([2, 0, 1]),
        robot=SimpleNamespace(set_qf=lambda qf, **kw: calls.append((qf, kw))),
    )
    action = torch.tensor([[1.0, -1.0, 0.2]])
    HumanoidRunEnv._step_action(env, action)
    torch.testing.assert_close(env._action, torch.tensor([[0.4, -0.4, 0.2]]))
    torch.testing.assert_close(calls[0][0], torch.tensor([[40.0, -50.0, 5.0]]))
    assert torch.equal(calls[0][1]["joint_ids"], env.policy_joint_ids)


def test_humanoid_observation_layout_and_motor_cost():
    count = 2
    z = torch.zeros(count)
    v = torch.zeros(count, 3)
    j = torch.zeros(count, 17)
    obs = humanoid_observation(
        torch.ones(count) * 1.34,
        v,
        v,
        z,
        z,
        z,
        torch.ones(count),
        torch.ones(count),
        j,
        j,
        j,
        angular_velocity_scale=0.25,
        joint_velocity_scale=0.1,
    )
    assert obs.shape == (2, 63)
    torch.testing.assert_close(obs[:, 0], torch.ones(count) * 1.34)
    reward = humanoid_reward(
        action=j,
        terminated=torch.tensor([False, True]),
        heading_projection=torch.ones(count),
        up_projection=torch.ones(count),
        joint_velocity=j,
        scaled_joint_position=j,
        progress=torch.ones(count),
        motor_effort_ratio=torch.ones(17),
        heading_weight=0.5,
        up_weight=0.1,
        actions_cost_scale=0.01,
        energy_cost_scale=0.05,
        joint_velocity_scale=0.1,
        death_cost=-1.0,
        alive_reward_scale=2.0,
    )
    # One unit of progress + alive 2 + upright 0.1 + heading 0.5; death overrides.
    torch.testing.assert_close(reward, torch.tensor([3.6, -1.0]))


@pytest.mark.parametrize(
    "position, expected_cost",
    [
        (-1.01, 2.0),
        (-1.0, 2.0),
        (-0.99, 2.0),
        (-0.98, 0.0),
        (0.0, 0.0),
        (0.98, 0.0),
        (0.99, 2.0),
        (1.0, 2.0),
        (1.01, 2.0),
    ],
)
def test_humanoid_reward_penalizes_both_joint_limits(
    position: float, expected_cost: float
) -> None:
    joint_position = torch.zeros(3, 17)
    # Two joints approach the same limit; the first environment stays centered.
    joint_position[1:, [0, 16]] = position
    reward = humanoid_reward(
        action=torch.zeros_like(joint_position),
        terminated=torch.tensor([False, False, True]),
        heading_projection=torch.ones(3),
        up_projection=torch.ones(3),
        joint_velocity=torch.zeros_like(joint_position),
        scaled_joint_position=joint_position,
        progress=torch.ones(3),
        motor_effort_ratio=torch.ones(17),
        heading_weight=0.5,
        up_weight=0.1,
        actions_cost_scale=0.01,
        energy_cost_scale=0.05,
        joint_velocity_scale=0.1,
        death_cost=-1.0,
        alive_reward_scale=2.0,
    )
    # Base reward is 3.6; each near-limit joint costs one, and death overrides it.
    torch.testing.assert_close(reward, torch.tensor([3.6, 3.6 - expected_cost, -1.0]))


def test_humanoid_deployment_preserves_effort_contract(monkeypatch):
    from pathlib import Path
    from embodichain.lab.sim import cfg as sim_cfg
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.utils.utility import load_config

    monkeypatch.setattr(sim_cfg, "get_data_path", lambda value: value)
    path = Path("embodichain_tasks/configs/tasks/classic_control/humanoid/env.yaml")
    cfg = config_to_cfg(load_config(str(path)), source_path=path)
    assert cfg.sim_cfg.physics_dt == pytest.approx(1 / 120)
    assert cfg.sim_steps_per_control == 2
    assert cfg.max_episode_steps == 900
    assert cfg.robot.control_parts["whole_body"] == _CONFIG["joint_names"]
    assert cfg.robot.joint_drive_props.drive_type == "force"
