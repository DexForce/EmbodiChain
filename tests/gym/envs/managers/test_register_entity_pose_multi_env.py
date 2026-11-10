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

"""Multi-env tests for get_pose / register_entity_pose (mock-only, no sim)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import get_pose, register_entity_pose
from embodichain.lab.sim.objects import RigidObject, Robot

NUM_JOINTS = 14
ARM_JOINTS = [0, 1, 2, 3, 4, 5]
# Static grasp pose given once, as in task configs (shape (1, 4, 4)).
GRASP = torch.tensor(
    [
        [
            [1.0, 0.0, 0.0, 0.03],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.01],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ]
)


def _make_robot(num_envs: int) -> MagicMock:
    robot = MagicMock(spec=Robot)
    robot.control_parts = {"left_arm": None, "right_arm": None}
    qpos = torch.arange(num_envs * NUM_JOINTS, dtype=torch.float32)
    robot.get_qpos.return_value = qpos.reshape(num_envs, NUM_JOINTS)
    robot.get_joint_ids.return_value = ARM_JOINTS

    def compute_fk(qpos, name=None, to_matrix=True, env_ids=None):
        # Same batch rule as Robot.compute_fk: qpos rows must match env_ids
        # (all envs when env_ids is None), since the base link poses are
        # fetched for those envs.
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        n = num_envs if env_ids is None else len(env_ids)
        if qpos.shape[0] != n:
            raise ValueError(f"batch size mismatch: expected {n}, got {qpos.shape[0]}")
        return torch.eye(4).repeat(qpos.shape[0], 1, 1)

    robot.compute_fk.side_effect = compute_fk
    return robot


def _make_env(num_envs: int, assets: dict, extra_attrs: dict | None = None):
    functor = SimpleNamespace(extra_attrs=extra_attrs or {})
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        affordance_datas={},
        sim=SimpleNamespace(get_asset=lambda uid: assets[uid]),
        event_manager=SimpleNamespace(get_functor=lambda name: functor),
    )


@pytest.mark.parametrize("num_envs", [1, 2, len(ARM_JOINTS)])
def test_get_pose_robot_uses_every_arm_joint_of_every_env(num_envs):
    robot = _make_robot(num_envs)
    env = _make_env(num_envs, {"robot": robot})
    cfg = SceneEntityCfg(uid="robot", control_parts=["left_arm"])

    name, pose = get_pose(env, torch.arange(num_envs), cfg)

    assert name == "left_arm_pose"
    assert pose.shape == (num_envs, 4, 4)
    # compute_fk also accepts a 1-D qpos for a single env, so compare per env.
    fk_qpos = robot.compute_fk.call_args.args[0].reshape(num_envs, -1)
    expected = robot.get_qpos.return_value[:, ARM_JOINTS]
    # num_envs == len(ARM_JOINTS) used to silently return the diagonal.
    assert torch.equal(fk_qpos, expected)


def test_get_pose_robot_subset_of_envs():
    robot = _make_robot(4)
    env = _make_env(4, {"robot": robot})
    cfg = SceneEntityCfg(uid="robot", control_parts=["left_arm"])
    env_ids = torch.tensor([1, 3])

    get_pose(env, env_ids, cfg)

    call = robot.compute_fk.call_args
    assert torch.equal(call.kwargs["env_ids"], env_ids)
    assert torch.equal(
        call.args[0], robot.get_qpos.return_value[env_ids][:, ARM_JOINTS]
    )


@pytest.mark.parametrize("num_envs", [1, 3])
def test_register_entity_pose_broadcasts_static_pose_object(num_envs):
    obj = MagicMock(spec=RigidObject)
    poses = torch.eye(4).repeat(num_envs, 1, 1)
    poses[:, 0, 3] = torch.arange(num_envs, dtype=torch.float32) * 0.5 + 0.1
    obj.get_local_pose.return_value = poses
    env = _make_env(
        num_envs,
        {"pen": obj},
        extra_attrs={"pen": {"grasp_pose_object": GRASP.tolist()}},
    )

    register_entity_pose(
        env,
        torch.arange(num_envs),
        SceneEntityCfg(uid="pen"),
        compute_relative=False,
        compute_pose_object_to_arena=True,
    )

    grasp_arena = env.affordance_datas["pen_grasp_pose"]
    assert grasp_arena.shape == (num_envs, 4, 4)
    for i in range(num_envs):
        assert torch.allclose(grasp_arena[i], poses[i] @ GRASP[0])
