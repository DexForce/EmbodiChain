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
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
from embodichain.utils.utility import load_config
from embodichain_tasks.manipulation.push_cube import PushCubeEnv
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.managers import EventManager, SceneEntityCfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS


def _configuration(filename: str):
    path = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/push_cube"
        / filename
    )
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent).config
    return physical, config_to_cfg(physical, manager_modules=get_manager_modules())


def _task(monkeypatch, filename="task.franka.visual.yaml", num_envs=4):
    _, cfg = _configuration(filename)
    monkeypatch.setattr(EmbodiedEnv, "__init__", lambda *args, **kwargs: None)
    task = PushCubeEnv(cfg)
    task._num_envs = num_envs
    task.sim = SimpleNamespace(device="cpu")
    return task


def test_deployment_composes_franka_camera_and_policy_action_terms():
    physical, cfg = _configuration("task.franka.visual.yaml")
    assert physical["id"] == "PushCube-Franka-Visual-v1"
    assert physical["physics"] == "default"
    assert cfg.robot.uid == "FrankaPanda"
    assert [(camera.uid, camera.width, camera.height) for camera in cfg.sensor] == [
        ("cam_high", 224, 224)
    ]
    assert cfg.max_episode_steps == 300
    assert cfg.target_control_frequency == 30
    assert REGISTERED_ENVS["PushCubeRL"].supports_handwritten_demo is False
    assert (
        REGISTERED_ENVS["PushCube-Franka-Visual-v1"].supports_handwritten_demo is True
    )
    assert REGISTERED_ENVS["PushCubeRL"].cls is PushCubeEnv
    assert REGISTERED_ENVS["PushCube-Franka-Visual-v1"].cls is PushCubeEnv
    assert set(cfg.events.__dict__) >= {
        "randomize_cube",
        "set_goal",
        "sync_goal_marker",
    }
    assert isinstance(
        cfg.events.set_goal.params["reference_entity_cfg"], SceneEntityCfg
    )
    assert isinstance(cfg.events.sync_goal_marker.params["entity_cfg"], SceneEntityCfg)


def test_configured_reset_synchronizes_marker_for_selected_environments():
    _, cfg = _configuration("task.franka.visual.yaml")
    cfg.seed = 123
    cube_pose = torch.eye(4).repeat(2, 1, 1)
    cube_pose[:, :3, 3] = torch.tensor(cfg.rigid_object[0].init_pos)
    marker_pose = torch.eye(4).repeat(2, 1, 1)
    marker_pose[:, :3, 3] = torch.tensor(cfg.background[0].init_pos)
    cube_before, marker_before = cube_pose.clone(), marker_pose.clone()

    def set_cube(pose, env_ids):
        cube_pose[env_ids] = pose

    def set_marker(pose, env_ids):
        marker_pose[env_ids] = pose

    entities = {
        "cube": SimpleNamespace(
            cfg=cfg.rigid_object[0],
            get_local_pose=lambda **kwargs: cube_pose,
            set_local_pose=set_cube,
            clear_dynamics=Mock(),
        ),
        "goal_marker": SimpleNamespace(set_local_pose=set_marker),
    }
    goal_before = torch.eye(4).repeat(2, 1, 1)
    env = SimpleNamespace(
        cfg=cfg,
        num_envs=2,
        device=torch.device("cpu"),
        goal_pose=goal_before.clone(),
        sim=SimpleNamespace(
            asset_uids=list(entities),
            get_rigid_object=entities.get,
            get_rigid_object_uid_list=lambda: list(entities),
            update=Mock(),
        ),
    )

    EventManager(cfg.events, env).apply("reset", torch.tensor([1]))

    torch.testing.assert_close(
        env.goal_pose[1, :3, 3], cube_pose[1, :3, 3] + torch.tensor([0.2, 0.0, -0.019])
    )
    torch.testing.assert_close(marker_pose[1], env.goal_pose[1])
    torch.testing.assert_close(cube_pose[0], cube_before[0])
    torch.testing.assert_close(env.goal_pose[0], goal_before[0])
    torch.testing.assert_close(marker_pose[0], marker_before[0])
    env.sim.update.assert_not_called()


def test_measured_success_uses_configured_radius_and_table_bounds(monkeypatch):
    task = _task(monkeypatch)
    pose = torch.eye(4).repeat(4, 1, 1)
    pose[:, :3, 3] = torch.tensor(
        [[0.099, 0.0, 0.02], [0.101, 0.0, 0.02], [0.01, 0.0, -0.1], [0.01, 0.0, 0.2]]
    )
    cube = SimpleNamespace(get_local_pose=lambda **kwargs: pose)
    task.sim.get_rigid_object = lambda uid: cube
    task.goal_pose = torch.eye(4).repeat(4, 1, 1)
    success, fail, metrics = task.compute_task_state()
    assert success.tolist() == [True, False, False, False]
    assert fail.tolist() == [False, False, True, False]
    torch.testing.assert_close(
        metrics["goal_distance"], torch.tensor([0.099, 0.101, 0.01, 0.01])
    )
    assert torch.equal(task.is_task_success(), success)
    assert not task.check_truncated({}, {}).any()


def test_existing_rl_threshold_metrics_and_fall_truncation_are_preserved(monkeypatch):
    task = _task(monkeypatch, "env.json", num_envs=2)
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[:, :3, 3] = torch.tensor([[0.075, 0.0, 0.2], [0.01, 0.0, -0.11]])
    task.sim.get_rigid_object = lambda uid: SimpleNamespace(
        get_local_pose=lambda **kw: pose
    )
    success, fail, metrics = task.compute_task_state()
    assert not success.any() and not fail.any()
    assert set(metrics) == {"distance_to_goal"}
    task.goal_pose = torch.eye(4).repeat(2, 1, 1)
    assert task.compute_task_state()[0].tolist() == [True, True]
    assert task.check_truncated({}, {}).tolist() == [False, True]
    # Existing extensions.success_threshold still controls the shared task.
    _, cfg = _configuration("env.json")
    cfg.extensions["success_threshold"] = 0.05
    tighter = PushCubeEnv(cfg)
    tighter._num_envs, tighter.sim, tighter.goal_pose = 2, task.sim, task.goal_pose
    assert tighter.compute_task_state()[0].tolist() == [False, True]


def test_expert_fails_when_ik_rejects_a_waypoint(monkeypatch):
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

    env = _task(monkeypatch, num_envs=1)
    env.robot = Robot()
    actions = iter(PushCubeEnv._expert_waypoints(env, torch.eye(4).unsqueeze(0), 2))
    torch.testing.assert_close(next(actions), torch.tensor([[0.25] * 7 + [-1.0]]))
    with pytest.raises(RuntimeError, match="IK failed"):
        next(actions)


def test_expert_segments_take_geometry_and_timing_from_configuration(monkeypatch):
    _, cfg = _configuration("task.franka.visual.yaml")
    cfg.extensions["push_cube"]["expert"].update(
        approach_offset=[-0.12, 0.03],
        approach_height=0.25,
        descend_height=0.04,
        push_displacement=[0.18, 0.02, 0.0],
        steps=[2, 3, 4],
    )
    monkeypatch.setattr(EmbodiedEnv, "__init__", lambda *args, **kwargs: None)
    task = PushCubeEnv(cfg)
    task._num_envs = 1
    task.sim = SimpleNamespace(
        device="cpu",
        get_rigid_object=lambda uid: SimpleNamespace(
            get_local_pose=lambda: torch.tensor([[-0.5, 0.1, 0.02, 0.0, 0.0, 0.0, 1.0]])
        ),
    )
    planned = []

    def waypoints(target, steps):
        planned.append((target.clone(), steps))
        yield torch.zeros(1, 8)

    task._expert_waypoints = waypoints
    names = []
    for segment in task.create_demo_segments():
        names.append(segment.name)
        list(segment.actions)
    assert names == ["approach", "descend", "push"]
    assert [steps for _, steps in planned] == [2, 3, 4]
    positions = torch.stack([pose[0, :3, 3] for pose, _ in planned])
    torch.testing.assert_close(
        positions,
        torch.tensor([[-0.62, 0.13, 0.25], [-0.62, 0.13, 0.04], [-0.44, 0.15, 0.04]]),
    )
    torch.testing.assert_close(
        planned[0][0][0, :3, :3], torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    )
