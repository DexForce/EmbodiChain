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

import torch

from embodichain.lab.gym.envs.managers.cfg import EventCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.event_manager import EventManager
from embodichain.lab.gym.envs.managers.randomization.spatial import (
    sample_rigid_object_pose_from_workspace,
)
from embodichain.lab.sim.workspace import WorkspaceSample


class _MockRobot:
    def __init__(self, sample: WorkspaceSample):
        self.sample = sample
        self.loaded_part = None
        self.sampled_part = None

    def get_workspace(self, name):
        self.loaded_part = name
        return object()

    def sample_reachable_pose(self, *, name, **kwargs):
        del kwargs
        self.sampled_part = name
        return self.sample


class _MockRigidObject:
    def __init__(self, num_envs: int):
        self.pose = torch.eye(4).unsqueeze(0).repeat(num_envs, 1, 1)
        self.set_pose = None
        self.clear_count = 0

    def get_local_pose(self, to_matrix=False):
        assert to_matrix
        return self.pose

    def set_local_pose(self, pose, env_ids=None):
        self.set_pose = pose.clone()
        self.pose[env_ids] = pose

    def clear_dynamics(self):
        self.clear_count += 1


class _MockSim:
    def __init__(self, robot, rigid_object):
        self.robot = robot
        self.rigid_object = rigid_object
        self.update_count = 0
        self.asset_uids = ["robot", "cube"]

    def get_robot(self, uid):
        assert uid == "robot"
        return self.robot

    def get_rigid_object(self, uid):
        assert uid == "cube"
        return self.rigid_object

    def update(self, step):
        self.update_count += step


class _MockEnv:
    def __init__(self, robot, rigid_object):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.sim = _MockSim(robot, rigid_object)


def _workspace_sample(valid: torch.Tensor) -> WorkspaceSample:
    poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)
    poses[0, 0, :3, 3] = torch.tensor([0.4, 0.2, 0.8])
    poses[1, 0, :3, 3] = torch.tensor([0.6, -0.2, 0.9])
    return WorkspaceSample(
        eef_pose=poses,
        qpos=torch.zeros(2, 1, 2),
        indices=torch.tensor([[0], [1]]),
        valid=valid.reshape(2, 1),
    )


def test_workspace_randomizer_moves_only_valid_environments():
    """The randomizer preserves invalid envs and applies the reference height."""
    robot = _MockRobot(_workspace_sample(torch.tensor([True, False])))
    rigid_object = _MockRigidObject(num_envs=2)
    rigid_object.pose[1, :3, 3] = torch.tensor([1.0, 1.0, 1.0])
    env = _MockEnv(robot, rigid_object)
    robot_cfg = SceneEntityCfg(uid="robot", control_parts=["arm"])
    entity_cfg = SceneEntityCfg(uid="cube")
    cfg = EventCfg(
        func=sample_rigid_object_pose_from_workspace,
        params={"robot_cfg": robot_cfg, "entity_cfg": entity_cfg},
    )
    randomizer = sample_rigid_object_pose_from_workspace(cfg, env)

    randomizer(
        env,
        torch.tensor([0, 1]),
        robot_cfg=robot_cfg,
        entity_cfg=entity_cfg,
        reference_height=0.75,
    )

    assert robot.loaded_part == "arm"
    assert robot.sampled_part == "arm"
    assert torch.allclose(
        rigid_object.set_pose[0, :3, 3], torch.tensor([0.4, 0.2, 0.75])
    )
    assert torch.allclose(
        rigid_object.set_pose[1, :3, 3], torch.tensor([1.0, 1.0, 1.0])
    )
    assert rigid_object.clear_count == 1


def test_workspace_randomizer_leaves_all_poses_when_sampling_fails():
    """The randomizer does not write object state when all candidates fail."""
    robot = _MockRobot(_workspace_sample(torch.tensor([False, False])))
    rigid_object = _MockRigidObject(num_envs=2)
    env = _MockEnv(robot, rigid_object)
    robot_cfg = SceneEntityCfg(uid="robot", control_parts=["arm"])
    entity_cfg = SceneEntityCfg(uid="cube")
    cfg = EventCfg(
        func=sample_rigid_object_pose_from_workspace,
        params={"robot_cfg": robot_cfg, "entity_cfg": entity_cfg},
    )
    randomizer = sample_rigid_object_pose_from_workspace(cfg, env)

    randomizer(
        env,
        torch.tensor([0, 1]),
        robot_cfg=robot_cfg,
        entity_cfg=entity_cfg,
    )

    assert rigid_object.set_pose is None
    assert rigid_object.clear_count == 0


def test_workspace_randomizer_registers_with_event_manager():
    """EventManager resolves and invokes the class-style randomizer."""
    robot = _MockRobot(_workspace_sample(torch.tensor([True, True])))
    rigid_object = _MockRigidObject(num_envs=2)
    env = _MockEnv(robot, rigid_object)
    cfg = EventCfg(
        func=sample_rigid_object_pose_from_workspace,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg(uid="robot", control_parts=["arm"]),
            "entity_cfg": SceneEntityCfg(uid="cube"),
        },
    )

    manager = EventManager({"workspace_object": cfg}, env)
    manager.apply("reset", torch.tensor([0, 1]))

    registered_cfg = manager.get_functor_cfg("workspace_object")
    assert isinstance(registered_cfg.func, sample_rigid_object_pose_from_workspace)
    assert rigid_object.set_pose is not None
