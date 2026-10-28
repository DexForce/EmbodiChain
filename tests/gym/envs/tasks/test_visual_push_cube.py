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

"""Contract checks for the visual PushCube deployment and expert."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
from embodichain.utils.utility import load_config
from embodichain_tasks.manipulation.push_cube import VisualPushCubeEnv


def test_deployment_composes_franka_camera_and_policy_action_terms():
    path = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/task.franka.visual.yaml"
    )
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent).config
    cfg = config_to_cfg(physical, manager_modules=get_manager_modules())
    assert physical["id"] == "PushCube-Franka-Visual-v1"
    assert physical["physics"] == "default"
    assert cfg.robot.uid == "FrankaPanda"
    assert [(camera.uid, camera.width, camera.height) for camera in cfg.sensor] == [
        ("cam_high", 224, 224)
    ]
    assert cfg.max_episode_steps == 300
    assert cfg.target_control_frequency == 30


def test_measured_success_uses_xy_radius_and_requires_cube_on_table():
    pose = torch.tensor(
        [
            [0.099, 0.0, 0.02, 0, 0, 0, 1],
            [0.101, 0.0, 0.02, 0, 0, 0, 1],
            [0.01, 0.0, -0.1, 0, 0, 0, 1],
            [0.01, 0.0, 0.2, 0, 0, 0, 1],
        ]
    )
    cube = SimpleNamespace(get_local_pose=lambda: pose)
    env = SimpleNamespace(
        sim=SimpleNamespace(get_rigid_object=lambda uid: cube),
        goal_position=torch.zeros(4, 3),
    )
    success, fail, metrics = VisualPushCubeEnv.compute_task_state(env)
    assert success.tolist() == [True, False, False, False]
    assert fail.tolist() == [False, False, True, False]
    torch.testing.assert_close(
        metrics["goal_distance"], torch.tensor([0.099, 0.101, 0.01, 0.01])
    )


def test_expert_fails_when_ik_rejects_a_waypoint():
    class Robot:
        def __init__(self):
            self.calls = 0

        def get_joint_ids(self, name):
            return list(range(7))

        def get_qpos(self):
            return torch.zeros(1, 9)

        def compute_fk(self, qpos, **kwargs):
            return torch.eye(4).unsqueeze(0)

        def compute_ik(self, pose, **kwargs):
            self.calls += 1
            return torch.tensor([self.calls == 1]), torch.full((1, 7), 0.25)

    env = SimpleNamespace(robot=Robot(), num_envs=1, device="cpu")
    actions = iter(
        VisualPushCubeEnv._expert_waypoints(env, torch.eye(4).unsqueeze(0), 2)
    )
    torch.testing.assert_close(next(actions), torch.tensor([[0.25] * 7 + [-1.0]]))
    with pytest.raises(RuntimeError, match="IK failed"):
        next(actions)
