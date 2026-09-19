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

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.envs.wrapper.replay import ReplayWrapper


class _DeltaActions:
    def process_action(self, action, mode):
        return action + 10 if mode == "pre" else action


class _ReplayEnv(gym.Env):
    """CPU boundary double retaining production controller preprocessing."""

    _preprocess_action = EmbodiedEnv._preprocess_action
    _prepare_controller_action = EmbodiedEnv._prepare_controller_action

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.robot = SimpleNamespace(
            dof=3,
            joint_names=["unused", "elbow", "wrist"],
            get_qpos=lambda: torch.zeros(num_envs, 3),
        )
        self.active_joint_ids = [1, 2]
        self.max_episode_steps = 20
        self.step_dt = 0.04
        self.action_manager = _DeltaActions()
        self._traj_buffer = None
        self._demo_no_auto_reset = False
        self.physical_objective = None
        self.step_count = 0
        self.reset_count = 0
        self.last_action = None
        self.last_command = None

    def step(self, action):
        self.last_action = action
        self.last_command = self._preprocess_action(action)
        self.step_count += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return {}, torch.zeros(self.num_envs), done, done.clone(), {}

    def reset(self, *, seed=None, options=None):
        self.reset_count += 1
        return {}, {}


def _trajectory(*, mode="position", kind="expert_controller", steps=2):
    width = 4 if mode == "position_velocity" else 2
    actions = torch.arange(steps * width, dtype=torch.float32).reshape(1, steps, width)
    return {
        "states": TensorDict({}, batch_size=[1, steps]),
        "actions": actions,
        "meta": {
            "num_steps": steps,
            "num_envs": 1,
            "lengths": [steps],
            "robot_dof": 3,
            "active_joint_ids": [1, 2],
            "action_kind": kind,
            "expert_trajectory_schema_version": 1,
            "joint_command_mode": mode,
            "joint_names": ["elbow", "wrist"],
            "qpos_slice": [0, 2],
            "qvel_slice": [2, 4] if mode == "position_velocity" else None,
            "step_dt": 0.04,
        },
    }


@pytest.mark.parametrize("kind", [None, "raw_policy"])
def test_raw_delta_actions_keep_preprocessing_despite_expert_metadata(kind):
    trajectory = _trajectory()
    if kind is None:
        del trajectory["meta"]["action_kind"]
    else:
        trajectory["meta"]["action_kind"] = kind
    env = _ReplayEnv()
    replay = ReplayWrapper(env, trajectory)
    replay.step(None)
    assert isinstance(env.last_action, torch.Tensor)
    torch.testing.assert_close(env.last_command, torch.tensor([[10.0, 11.0]]))


@pytest.mark.parametrize("mode", ["position", "position_velocity"])
def test_expert_controller_replay_decodes_targets_without_delta_preprocessing(mode):
    env = _ReplayEnv()
    replay = ReplayWrapper(env, _trajectory(mode=mode))
    replay.step(None)
    assert isinstance(env.last_action, ControllerAction)
    torch.testing.assert_close(env.last_command["qpos"], torch.tensor([[0.0, 1.0]]))
    if mode == "position_velocity":
        torch.testing.assert_close(env.last_command["qvel"], torch.tensor([[2.0, 3.0]]))
    else:
        assert "qvel" not in env.last_command


@pytest.mark.parametrize(
    "field,value",
    [
        ("step_dt", 0.02),
        ("step_dt", float("nan")),
        ("step_dt", True),
        ("joint_names", ["wrist", "elbow"]),
        ("qpos_slice", [1, 3]),
        ("qvel_slice", [2, 4]),
        ("expert_trajectory_schema_version", 2),
        ("expert_trajectory_schema_version", True),
        ("joint_command_mode", "torque"),
        ("action_kind", "guess"),
    ],
)
def test_incompatible_controller_metadata_fails_before_step(field, value):
    env = _ReplayEnv()
    trajectory = _trajectory()
    trajectory["meta"][field] = value
    with pytest.raises(ValueError):
        ReplayWrapper(env, trajectory)
    assert env.step_count == 0


@pytest.mark.parametrize(
    "field",
    [
        "step_dt",
        "joint_names",
        "qpos_slice",
        "qvel_slice",
        "expert_trajectory_schema_version",
        "joint_command_mode",
    ],
)
def test_missing_controller_metadata_fails_before_step(field):
    trajectory = _trajectory()
    del trajectory["meta"][field]
    with pytest.raises(ValueError):
        ReplayWrapper(_ReplayEnv(), trajectory)


@pytest.mark.parametrize(
    "actions",
    [
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2, 2, dtype=torch.long),
        torch.full((1, 2, 2), float("nan")),
    ],
)
def test_invalid_controller_action_payload_fails_before_step(actions):
    trajectory = _trajectory()
    trajectory["actions"] = actions
    with pytest.raises(ValueError):
        ReplayWrapper(_ReplayEnv(), trajectory)


def test_objective_replay_rejects_vector_execution_before_broadcast():
    env = _ReplayEnv(num_envs=2)
    env.physical_objective = object()
    with pytest.raises(ValueError, match="single"):
        ReplayWrapper(env, _trajectory())


def test_objective_replay_does_not_step_exhausted_rows():
    env = _ReplayEnv()
    env.physical_objective = object()
    replay = ReplayWrapper(env, _trajectory(steps=1))
    replay.step(None)
    with pytest.raises(RuntimeError, match="exhausted"):
        replay.step(None)
    assert env.step_count == 1


def test_objective_replay_rejects_partial_reset_before_resetting_scene():
    env = _ReplayEnv()
    env.physical_objective = object()
    replay = ReplayWrapper(env, _trajectory())
    with pytest.raises(ValueError, match="reset_ids"):
        replay.reset(options={"reset_ids": []})
    assert env.reset_count == 0


def test_dynamic_empty_trajectory_fails_before_indexing():
    with pytest.raises(ValueError, match="length"):
        ReplayWrapper(_ReplayEnv(), _trajectory(steps=0))


def test_kinematic_replay_ignores_controller_schema():
    trajectory = _trajectory()
    trajectory["meta"]["step_dt"] = 123.0
    ReplayWrapper(_ReplayEnv(), trajectory, mode="kinematic")


def _objective_env():
    from embodichain.lab.gym.envs.objectives import (
        MeasuredRigidObjectState,
        OrderedPlacementObjective,
    )
    from embodichain.lab.gym.envs.objectives.config import decode_objective

    class ObjectiveReplayEnv(_ReplayEnv):
        def __init__(self):
            super().__init__()
            self.physical_objective = OrderedPlacementObjective(
                decode_objective(
                    {
                        "type": "ordered_stable_regions",
                        "object_uid": "cube",
                        "regions": [
                            {"name": "goal", "lower": [-1, -1, -1], "upper": [1, 1, 1]}
                        ],
                        "hold_seconds": 0.08,
                        "max_linear_speed": 0.1,
                        "max_angular_speed": 0.1,
                    }
                ),
                1,
                "cpu",
            )
            self.sim = SimpleNamespace(enable_physics=lambda enabled: None)
            self.robot.set_local_pose = lambda value: None
            self.robot.set_qpos = lambda value, target: None

        def step(self, action):
            obs, reward, term, trunc, info = super().step(action)
            self.physical_objective.update(
                MeasuredRigidObjectState(
                    torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3)
                ),
                self.step_dt,
            )
            info["physical_objective"] = self.physical_objective.snapshot()
            return obs, reward, term, trunc, info

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed, options=options)
            self.physical_objective.reset()
            return {}, {"physical_objective": self.physical_objective.snapshot()}

        def get_obs(self):
            return {}

    return ObjectiveReplayEnv()


def test_exhaustion_cannot_complete_a_physical_dwell():
    env = _objective_env()
    replay = ReplayWrapper(env, _trajectory(steps=1))
    replay.step(None)
    assert env.physical_objective.snapshot_row(0)["success"] is False
    with pytest.raises(RuntimeError, match="exhausted"):
        replay.step(None)
    snapshot = env.physical_objective.snapshot_row(0)
    assert snapshot["progress"] == 0
    assert snapshot["metrics"]["hold_seconds"] == pytest.approx(0.04)


def test_restoring_initial_replay_state_does_not_credit_physical_dwell():
    env = _objective_env()
    trajectory = _trajectory(steps=1)
    trajectory["states"] = TensorDict(
        {"robot": {"root_pose": torch.zeros(1, 1, 7), "qpos": torch.zeros(1, 1, 3)}},
        batch_size=[1, 1],
    )
    replay = ReplayWrapper(env, trajectory)
    replay.step(None)
    _, info = replay.reset()
    assert info["physical_objective"]["progress"].tolist() == [0]
    assert info["physical_objective"]["metrics"]["hold_seconds"].tolist() == [0.0]
    replay.step(None)
    assert env.physical_objective.snapshot_row(0)["success"] is False


@pytest.mark.parametrize("mode", ["position", "position_velocity"])
def test_expert_replay_can_record_decoded_controller_targets(mode):
    from embodichain.lab.gym.envs.expert_trajectory import build_expert_action_spec

    env = _ReplayEnv()
    trajectory = _trajectory(mode=mode)
    env.expert_action_spec = build_expert_action_spec(
        joint_names=["elbow", "wrist"], joint_command_mode=mode
    )
    env._traj_buffer = TensorDict(
        {"actions": torch.zeros_like(trajectory["actions"])}, batch_size=[1, 2]
    )
    env._traj_steps = torch.zeros(1, dtype=torch.long)
    env._seed_trajectory_states = lambda ids: None
    replay = ReplayWrapper(env, trajectory)
    for _ in range(2):
        replay.step(None)
        EmbodiedEnv._write_trajectory_step(env)
    assert env._traj_steps.tolist() == [2]
    torch.testing.assert_close(env._traj_buffer["actions"], trajectory["actions"])


def test_policy_recording_preserves_raw_delta_before_controller_transform():
    env = _ReplayEnv()
    env._traj_buffer = TensorDict({"actions": torch.zeros(1, 1, 2)}, batch_size=[1, 1])
    env._traj_steps = torch.zeros(1, dtype=torch.long)
    env._seed_trajectory_states = lambda ids: None
    env.step(torch.tensor([[1.0, 2.0]]))
    EmbodiedEnv._write_trajectory_step(env)
    torch.testing.assert_close(env.last_command, torch.tensor([[11.0, 12.0]]))
    torch.testing.assert_close(
        env._traj_buffer["actions"][0, 0], torch.tensor([1.0, 2.0])
    )
