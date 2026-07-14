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

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.success import (
    evaluate_configured_success,
)


class _FakeSim:
    def __init__(self, objects=None) -> None:
        self.objects = objects or {}

    def get_rigid_object(self, uid: str):
        return self.objects.get(uid)


class _FakeEnv:
    num_envs = 1
    device = torch.device("cpu")
    sim = _FakeSim()


def test_success_unknown_rigid_object_uid_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown rigid object uid: 'missing'"):
        evaluate_configured_success(
            _FakeEnv(),
            {
                "type": "object_xy_near",
                "object": "missing",
                "target_xy": [0.0, 0.0],
            },
        )


class _FakeObject:
    def __init__(self, position) -> None:
        self.position = torch.as_tensor(position, dtype=torch.float32).reshape(1, 3)

    def get_local_pose(self, to_matrix: bool = True):
        pose = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
        pose[:, :3, 3] = self.position
        return pose

    def get_vertices(self, env_ids=None, scale: bool = True):
        vertices = torch.tensor(
            [
                [-0.5, -0.2, -0.05],
                [-0.5, 0.2, 0.05],
                [0.5, -0.2, 0.05],
                [0.5, 0.2, -0.05],
            ],
            dtype=torch.float32,
        )
        return [vertices]


def test_object_lifted_requires_initial_height() -> None:
    env = _FakeEnv()
    env.sim = _FakeSim({"apple": _FakeObject([0.0, 0.0, 0.2])})
    env.obj_info = {}

    with pytest.raises(ValueError, match="requires an initial height"):
        evaluate_configured_success(
            env,
            {
                "type": "object_lifted",
                "object": "apple",
                "min_height": 0.1,
            },
        )


def test_object_lifted_uses_recorded_initial_height() -> None:
    env = _FakeEnv()
    env.sim = _FakeSim({"apple": _FakeObject([0.0, 0.0, 0.25])})
    env.obj_info = {"apple": {"height": torch.tensor(0.1)}}

    success = evaluate_configured_success(
        env,
        {
            "type": "object_lifted",
            "object": "apple",
            "min_height": 0.1,
        },
    )

    assert success.tolist() == [True]


def test_object_held_by_both_grippers_uses_surface_distance() -> None:
    env = _FakeEnv()
    env.sim = _FakeSim({"tray": _FakeObject([0.0, 0.0, 0.2])})
    env.close_state = torch.tensor([0.0])
    env.open_state = torch.tensor([0.05])
    env.get_current_gripper_state_agent = lambda: (
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )
    left_pose = torch.eye(4).unsqueeze(0)
    right_pose = torch.eye(4).unsqueeze(0)
    left_pose[:, :3, 3] = torch.tensor([0.55, 0.0, 0.2])
    right_pose[:, :3, 3] = torch.tensor([-0.55, 0.0, 0.2])
    env.get_current_xpos_agent = lambda: (left_pose, right_pose)

    success = evaluate_configured_success(
        env,
        {
            "type": "object_held_by_both_grippers",
            "object": "tray",
            "max_distance": 0.10,
        },
    )

    assert success.tolist() == [True]
    assert torch.linalg.norm(left_pose[0, :3, 3] - torch.tensor([0.0, 0.0, 0.2])) > 0.5


def test_both_grippers_open_and_clear_of_object() -> None:
    env = _FakeEnv()
    env.sim = _FakeSim({"tray": _FakeObject([0.0, 0.0, 0.2])})
    env.close_state = torch.tensor([0.0])
    env.open_state = torch.tensor([0.05])
    env.get_current_gripper_state_agent = lambda: (
        torch.tensor([0.05]),
        torch.tensor([0.05]),
    )
    left_pose = torch.eye(4).unsqueeze(0)
    right_pose = torch.eye(4).unsqueeze(0)
    left_pose[:, :3, 3] = torch.tensor([0.8, 0.0, 0.2])
    right_pose[:, :3, 3] = torch.tensor([-0.8, 0.0, 0.2])
    env.get_current_xpos_agent = lambda: (left_pose, right_pose)

    success = evaluate_configured_success(
        env,
        {
            "op": "all",
            "terms": [
                {"type": "both_grippers_open"},
                {
                    "type": "grippers_clear_of_object",
                    "object": "tray",
                    "min_distance": 0.05,
                },
            ],
        },
    )

    assert success.tolist() == [True]


def test_both_grippers_open_returns_false_before_gripper_cache_init() -> None:
    env = _FakeEnv()

    def get_uninitialized_gripper_state():
        return (
            env.left_arm_current_gripper_state,
            env.right_arm_current_gripper_state,
        )

    env.get_current_gripper_state_agent = get_uninitialized_gripper_state

    success = evaluate_configured_success(env, {"type": "both_grippers_open"})

    assert success.tolist() == [False]


def test_both_grippers_open_returns_false_before_open_state_init() -> None:
    env = _FakeEnv()
    env.get_current_gripper_state_agent = lambda: (
        torch.tensor([0.05]),
        torch.tensor([0.05]),
    )

    success = evaluate_configured_success(env, {"type": "both_grippers_open"})

    assert success.tolist() == [False]
