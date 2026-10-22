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

"""Tests for shared trajectory state capture and restore helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from tensordict import TensorDict

from embodichain.lab.gym.utils.trajectory_state import (
    capture_trajectory_state,
    restore_trajectory_state,
)

NUM_ENVS = 2
NUM_STEPS = 3


def _make_state_buffer() -> TensorDict:
    """Create a minimal robot/articulation/rigid-object trajectory buffer."""
    return TensorDict(
        {
            "robot": {
                "root_pose": torch.zeros(NUM_ENVS, NUM_STEPS, 7),
                "qpos": torch.zeros(NUM_ENVS, NUM_STEPS, 2),
                "qvel": torch.zeros(NUM_ENVS, NUM_STEPS, 2),
            },
            "articulations": {
                "arm": {
                    "root_pose": torch.zeros(NUM_ENVS, NUM_STEPS, 7),
                    "qpos": torch.zeros(NUM_ENVS, NUM_STEPS, 2),
                    "qvel": torch.zeros(NUM_ENVS, NUM_STEPS, 2),
                }
            },
            "rigid_objects": {
                "cube": {
                    "pose": torch.zeros(NUM_ENVS, NUM_STEPS, 7),
                    "lin_vel": torch.zeros(NUM_ENVS, NUM_STEPS, 3),
                    "ang_vel": torch.zeros(NUM_ENVS, NUM_STEPS, 3),
                }
            },
        },
        batch_size=[NUM_ENVS, NUM_STEPS],
    )


def test_capture_trajectory_state_writes_selected_environment_slot() -> None:
    """Capture writes every supported state field at the requested row and step."""
    robot_qpos = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    robot_qvel = torch.tensor([[1.1, 1.2], [1.3, 1.4]])
    articulation_qpos = robot_qpos + 2.0
    articulation_qvel = robot_qvel + 2.0
    rigid_body_state = torch.zeros(NUM_ENVS, 13)
    rigid_body_state[:, 7:10] = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    rigid_body_state[:, 10:13] = torch.tensor([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]])
    robot = SimpleNamespace(
        get_local_pose=lambda: torch.ones(NUM_ENVS, 7),
        get_qpos=lambda: robot_qpos,
        get_qvel=lambda: robot_qvel,
    )
    articulation = SimpleNamespace(
        get_local_pose=lambda: torch.full((NUM_ENVS, 7), 2.0),
        get_qpos=lambda: articulation_qpos,
        get_qvel=lambda: articulation_qvel,
    )
    rigid_object = SimpleNamespace(
        get_local_pose=lambda: torch.full((NUM_ENVS, 7), 3.0),
        body_state=rigid_body_state,
    )
    env = SimpleNamespace(
        robot=robot,
        sim=SimpleNamespace(
            _articulations={"arm": articulation},
            _rigid_objects={"cube": rigid_object},
        ),
    )
    states = _make_state_buffer()
    env_ids = torch.tensor([1])
    step_ids = torch.tensor([2])

    capture_trajectory_state(env, states, env_ids, step_ids)

    assert torch.equal(states["robot"]["qpos"][1, 2], robot_qpos[1])
    assert torch.equal(states["robot"]["qvel"][1, 2], robot_qvel[1])
    assert torch.equal(
        states["articulations"]["arm"]["qpos"][1, 2], articulation_qpos[1]
    )
    assert torch.equal(
        states["rigid_objects"]["cube"]["lin_vel"][1, 2],
        rigid_body_state[1, 7:10],
    )
    assert torch.equal(
        states["rigid_objects"]["cube"]["ang_vel"][1, 2],
        rigid_body_state[1, 10:13],
    )
    assert not states["robot"]["qpos"][0].any()


def test_restore_trajectory_state_writes_complete_robot_state() -> None:
    """Restore writes full qpos, including mimic joints, and optional velocity."""
    recorded_qpos = torch.tensor([[0.1, 0.02, 0.02]])
    recorded_qvel = torch.tensor([[0.3, 0.04, 0.04]])
    robot = SimpleNamespace(
        set_local_pose=MagicMock(),
        set_qpos=MagicMock(),
        set_qvel=MagicMock(),
    )
    env = SimpleNamespace(
        robot=robot,
        sim=SimpleNamespace(_articulations={}, _rigid_objects={}),
    )
    states = {
        "robot": {
            "root_pose": torch.zeros((1, 7)),
            "qpos": recorded_qpos,
            "qvel": recorded_qvel,
        }
    }

    restore_trajectory_state(env, states)

    robot.set_qpos.assert_called_once_with(recorded_qpos, target=False)
    robot.set_qvel.assert_called_once_with(recorded_qvel, target=False)


def test_cable_capture_is_world_observation_and_selected_rows_only() -> None:
    from embodichain.lab.gym.utils.gym_utils import build_trajectory_buffer

    cable_state = torch.arange(2 * 3 * 13, dtype=torch.float32).reshape(2, 3, 13)
    robot = SimpleNamespace(
        dof=2,
        get_local_pose=lambda: torch.zeros(2, 7),
        get_qpos=lambda: torch.zeros(2, 2),
        get_qvel=lambda: torch.zeros(2, 2),
    )
    cable = SimpleNamespace(
        data=SimpleNamespace(num_segments=3, segment_state_w=cable_state)
    )
    env = SimpleNamespace(
        robot=robot,
        sim=SimpleNamespace(
            _articulations={}, _rigid_objects={}, _cable_objects={"wire": cable}
        ),
    )
    buffer = build_trajectory_buffer(env, max_steps=3, num_envs=2, device="cpu")
    capture_trajectory_state(
        env, buffer["states"], torch.tensor([1]), torch.tensor([2])
    )
    torch.testing.assert_close(
        buffer["states", "cables", "wire", "segment_state_w"][1, 2], cable_state[1]
    )
    assert not buffer["states", "cables", "wire", "segment_state_w"][0].any()
    filtered = build_trajectory_buffer(
        env, max_steps=3, num_envs=2, device="cpu", uids=["other"]
    )
    assert "cables" not in filtered["states"].keys()


def test_cable_restore_fails_before_any_robot_or_scene_write() -> None:
    import pytest

    robot = MagicMock()
    states = {
        "robot": {"qpos": torch.ones(2, 2), "root_pose": torch.ones(2, 7)},
        "cables": {"wire": {"segment_state_w": torch.zeros(2, 3, 13)}},
    }
    with pytest.raises(NotImplementedError, match="solver-history"):
        restore_trajectory_state(SimpleNamespace(robot=robot, sim=MagicMock()), states)
    assert robot.mock_calls == []
