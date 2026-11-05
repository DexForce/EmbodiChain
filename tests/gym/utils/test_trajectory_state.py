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

import pytest
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


def test_recorded_robot_positions_keep_exact_slots_precision_and_ownership(monkeypatch):
    import pytest
    from embodichain.lab.gym.utils.trajectory_state import (
        read_recorded_robot_joint_snapshot,
    )

    states = _make_state_buffer()
    values = states["robot"]["qpos"].double()
    values[:, :, 0] = torch.tensor([-99.0, 1.0, 2.0])
    states["robot"]["qpos"] = values
    captured = read_recorded_robot_joint_snapshot(
        states,
        torch.tensor([2, 2]),
        robot_uid="native_robot",
        frame_indices=[1, 2],
        episode_ids=[4, 9],
    )
    assert captured.qpos[:, 0].tolist() == [1.0, 2.0]
    assert captured.qpos.dtype == values.dtype and captured.qpos.device == values.device
    assert captured.robot_uid == "native_robot" and captured.episode_ids == (4, 9)
    values.zero_()
    assert captured.qpos[:, 0].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="not fully recorded"):
        read_recorded_robot_joint_snapshot(
            states,
            torch.tensor([1, 2]),
            robot_uid="native_robot",
            frame_indices=[2, 2],
            episode_ids=[4, 9],
        )
    with pytest.raises(ValueError, match="terminal robot state slot"):
        read_recorded_robot_joint_snapshot(
            states,
            torch.tensor([3, 3]),
            robot_uid="native_robot",
            frame_indices=[3, 3],
            episode_ids=[4, 9],
        )


def _recorded_joint_reader(dtype=torch.float64, *, invalid=False):
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers import ObservationManager
    from embodichain.lab.gym.envs.managers.observation_types import (
        ObservationDescriptor,
        ObservationFrame,
        ObservationSample,
    )
    from embodichain.lab.gym.utils.gym_utils import (
        _init_raw_rollout,
        _write_raw_rollout,
    )
    from embodichain.lab.gym.utils.trajectory_state import RecordedTrajectoryReader

    context = EpisodeContext(2)
    context._begin_episode([0, 1])
    context._finish_initialization([0, 1])
    env = SimpleNamespace(episode_context=context, _closed=False)
    manager = ObservationManager({}, env)
    storage = TensorDict({}, batch_size=[2])
    _init_raw_rollout(storage, context, {}, torch.zeros(2, 2, 1))
    manager.bind_rollout_storage(storage)
    descriptor = ObservationDescriptor(
        "native_robot", "qpos", "joint", "joint_order", ("joint_units",), (2,), dtype
    )
    for slot in range(3):
        values = torch.tensor(
            [[slot, slot + 0.5], [slot + 10, slot + 10.5]], dtype=dtype
        )
        available = [True, not (invalid and slot == 2)]
        errors = (None, "original acquisition missing" if not available[1] else None)
        frame = ObservationFrame(
            None,
            {
                "qpos": ObservationSample(
                    descriptor, values, torch.tensor(available), errors
                )
            },
            context.episode_ids,
            slot,
            timestamp=slot * 0.04,
            clock_domain="simulation_time",
            episode_owner=context,
        )
        manager._bind_schema(frame)
        _write_raw_rollout(
            storage,
            context,
            frame,
            **(
                {"field_ids": ("qpos",), "env_ids": [0, 1]}
                if slot == 0
                else {"action": torch.zeros(2, 1)}
            ),
        )
    owner = {"token": object()}
    reader = RecordedTrajectoryReader(
        observations=manager,
        episode_context=context,
        is_closed=lambda: env._closed,
        robot_identity=lambda: ("native_robot", owner["token"]),
        step_counts=lambda: storage["raw_steps"],
    )
    return reader, env, manager, storage, owner


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_recorded_robot_reader_uses_exact_raw_slots_and_owned_original_precision(dtype):
    reader, _, _, storage, _ = _recorded_joint_reader(dtype)
    counts = reader.read_step_counts(episode_ids=[0, 0])
    snapshot = reader.read_robot_qpos([1, 2], episode_ids=[0, 0])
    assert snapshot.qpos.tolist() == [[1.0, 1.5], [12.0, 12.5]]
    assert snapshot.qpos.dtype == dtype
    assert snapshot.qpos.device == storage["raw_obs"]["qpos"].device
    assert snapshot.env_ids == (0, 1) and snapshot.frame_indices == (1, 2)
    assert snapshot.robot_uid == "native_robot" and snapshot.episode_ids == (0, 0)
    storage["raw_obs"]["qpos"].zero_()
    storage["raw_steps"].zero_()
    assert snapshot.qpos.tolist() == [[1.0, 1.5], [12.0, 12.5]]
    assert counts == (2, 2)


def test_recorded_robot_reader_preserves_unavailable_raw_acquisition():
    reader, _, _, _, _ = _recorded_joint_reader(invalid=True)
    with pytest.raises(RuntimeError, match="original acquisition missing"):
        reader.read_robot_qpos([1, 2], episode_ids=[0, 0])


@pytest.mark.parametrize("change", ["robot", "episode", "close"])
def test_recorded_robot_reader_checks_owner_and_lifetime_after_copy(
    monkeypatch, change
):
    reader, env, manager, _, owner = _recorded_joint_reader()
    original = manager.read_history

    def changed_owner(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "robot":
            owner["token"] = object()
        elif change == "episode":
            env.episode_context._begin_episode([0])
        else:
            env._closed = True
        return result

    monkeypatch.setattr(manager, "read_history", changed_owner)
    with pytest.raises(
        (ValueError, RuntimeError), match="robot or episode changed|another|after close"
    ):
        reader.read_robot_qpos([1, 2], episode_ids=[0, 0])


def test_recorded_robot_reader_requires_declared_identity_history_and_completed_slot():
    from embodichain.lab.gym.utils.trajectory_state import RecordedTrajectoryReader

    reader, env, manager, _, _ = _recorded_joint_reader()
    for frames in ([0, 2], [3, 2]):
        with pytest.raises(ValueError):
            reader.read_robot_qpos(frames, episode_ids=[0, 0])
    no_identity = RecordedTrajectoryReader(
        observations=manager,
        episode_context=env.episode_context,
        is_closed=lambda: False,
    )
    with pytest.raises(RuntimeError, match="robot identity is unavailable"):
        no_identity.read_robot_qpos([1, 2], episode_ids=[0, 0])
    disabled = RecordedTrajectoryReader(
        observations=None, episode_context=env.episode_context, is_closed=lambda: False
    )
    with pytest.raises(RuntimeError, match="history is unavailable"):
        disabled.read_robot_qpos([1, 2], episode_ids=[0, 0])
