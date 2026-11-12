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
from embodichain.lab.gym.utils.gym_utils import _init_raw_rollout
from tensordict import TensorDict

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.envs.wrapper.replay import ReplayWrapper


class _DeltaActions:
    def __init__(self):
        self.action = None

    def process_action(self, action):
        self.action = action + 10

    def apply_action(self):
        return None

    def action_trace(self):
        return None


class _ReplayEnv(gym.Env):
    """CPU boundary double retaining production controller preprocessing."""

    _preprocess_action = EmbodiedEnv._preprocess_action
    _prepare_controller_action = EmbodiedEnv._prepare_controller_action
    _step_action = EmbodiedEnv._step_action
    _active_qpos_command = EmbodiedEnv._active_qpos_command
    _record_raw_action = EmbodiedEnv._record_raw_action

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.robot = SimpleNamespace(
            dof=3,
            joint_names=["unused", "elbow", "wrist"],
            get_qpos=lambda: torch.zeros(num_envs, 3),
            set_qpos=lambda **kwargs: None,
            set_qvel=lambda **kwargs: None,
            set_qf=lambda **kwargs: None,
        )
        self.active_joint_ids = [1, 2]
        self.max_episode_steps = 20
        self.step_dt = 0.04
        self.action_manager = _DeltaActions()
        self.dataset_manager = None
        self._traj_buffer = None
        self._demo_no_auto_reset = False
        self.physical_objective = None
        self.step_count = 0
        self.reset_count = 0
        self.last_action = None
        self.last_command = None

    def step(self, action):
        self.last_action = action
        prepared = self._preprocess_action(action)
        self.last_command = self._step_action(prepared)
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
    torch.testing.assert_close(env.action_manager.action, torch.tensor([[10.0, 11.0]]))


@pytest.mark.parametrize("mode", ["position", "position_velocity"])
@pytest.mark.parametrize("kind", ["expert", "expert_controller"])
def test_expert_controller_replay_decodes_targets_without_delta_preprocessing(
    mode, kind
):
    env = _ReplayEnv()
    replay = ReplayWrapper(env, _trajectory(mode=mode, kind=kind))
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
            self.robot.set_local_pose = lambda *args, **kwargs: None
            self.robot.set_qpos = lambda *args, **kwargs: None

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


def _enable_action_recording(env, actions):
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.gym.envs.managers.observation_types import ObservationFrame

    context = EpisodeContext(env.num_envs)
    context._begin_episode(list(range(env.num_envs)))
    context._finish_initialization(list(range(env.num_envs)))
    env.episode_context = context
    env.observation_manager = ObservationManager({}, env)
    env.observation_manager._env._rollout_storage = getattr(
        env.observation_manager._env, "_rollout_storage", None
    )
    if env.observation_manager._env._rollout_storage is None:
        env.observation_manager._env._rollout_storage = TensorDict(
            {},
            batch_size=[len(env.observation_manager._env.episode_context.episode_ids)],
        )
    _init_raw_rollout(
        env.observation_manager._env._rollout_storage,
        env.observation_manager._env.episode_context,
        **dict(fields={}, actions=actions),
    )
    env.observation_manager.bind_rollout_storage(
        env.observation_manager._env._rollout_storage
    )
    env._traj_steps = env._rollout_storage["raw_steps"]

    def frame():
        return ObservationFrame(
            None, {}, context.episode_ids, env.step_count, episode_owner=context
        )

    EmbodiedEnv._record_initial_observation(env, frame(), list(range(env.num_envs)))
    return frame


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
    frame = _enable_action_recording(env, env._traj_buffer["actions"])
    replay = ReplayWrapper(env, trajectory)
    for _ in range(2):
        replay.step(None)
        EmbodiedEnv._write_trajectory_step(env, frame())
    assert env._traj_steps.tolist() == [2]
    torch.testing.assert_close(env._traj_buffer["actions"], trajectory["actions"])


def test_policy_recording_preserves_raw_delta_before_controller_transform():
    env = _ReplayEnv()
    env._traj_buffer = TensorDict({"actions": torch.zeros(1, 1, 2)}, batch_size=[1, 1])
    frame = _enable_action_recording(env, env._traj_buffer["actions"])
    env._trajectory_action_kind = "policy"
    env.step(torch.tensor([[1.0, 2.0]]))
    EmbodiedEnv._write_trajectory_step(env, frame())
    torch.testing.assert_close(env.action_manager.action, torch.tensor([[11.0, 12.0]]))
    torch.testing.assert_close(
        env._traj_buffer["actions"][0, 0], torch.tensor([11.0, 12.0])
    )


class _TaskReplayEnv(_ReplayEnv):
    """Replay protocol with a virtual target independent of physical state."""

    def __init__(self, num_envs=1):
        super().__init__(num_envs)
        self.goal_pose = torch.zeros(num_envs, 3)
        self.termination_manager = object()
        self.sim = SimpleNamespace(enable_physics=lambda value: None)
        self.restored = False

    def reset(self, *, seed=None, options=None):
        self.goal_pose.fill_(99)
        return {}, {}

    def restore_task_initial_state(self, state):
        self.goal_pose.copy_(torch.tensor(state["targets"]["goal_pose"]))
        self.restored = True

    def get_obs(self):
        return {"goal": self.goal_pose.clone()}

    def get_info(self):
        return {"restored": self.restored}


def test_dynamic_replay_restores_virtual_goal_before_first_observation(monkeypatch):
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.wrapper.replay.restore_trajectory_state",
        lambda *args: None,
    )
    trajectory = _trajectory(kind="raw_policy")
    trajectory["meta"]["demo_episodes"] = [
        {
            "task_initial_state": {
                "schema_version": 1,
                "targets": {"goal_pose": [[1.0, 2.0, 3.0]]},
            }
        }
    ]
    env = _TaskReplayEnv()
    replay = ReplayWrapper(env, trajectory)
    obs, info = replay.reset()
    torch.testing.assert_close(obs["goal"], torch.tensor([[1.0, 2.0, 3.0]]))
    assert info["restored"] is True
    with pytest.raises(ValueError, match="reset_ids"):
        replay.reset(options={"reset_ids": [0]})


def test_dynamic_task_replay_requires_recorded_context():
    with pytest.raises(ValueError, match="task_initial_state"):
        ReplayWrapper(_TaskReplayEnv(), _trajectory(kind="raw_policy"))


def test_dynamic_replay_ignores_inactive_rl_restore_method():
    env = _TaskReplayEnv()
    env.termination_manager = None
    replay = ReplayWrapper(env, _trajectory(kind="raw_policy"))
    assert replay._task_outcome_replay is False


def test_kinematic_task_replay_does_not_claim_measured_goal_acceptance():
    replay = ReplayWrapper(
        _TaskReplayEnv(), _trajectory(kind="raw_policy"), mode="kinematic"
    )
    assert replay._task_outcome_replay is False


def test_replay_reset_reseeds_raw_history_after_restoring_source_state():
    from itertools import count
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.observation_types import (
        ObservationDescriptor,
        ObservationSample,
        ObservationFrame,
    )
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager

    env = _ReplayEnv()
    context = EpisodeContext(1)
    context._begin_episode([0])
    context._finish_initialization([0])
    state = torch.zeros(1, 3)
    env.robot.get_qpos = lambda: state
    env.robot.set_qpos = lambda values, target: state.copy_(values)
    env.robot.set_local_pose = lambda values: None
    env.sim = SimpleNamespace(enable_physics=lambda enabled: None)
    env.episode_context = context
    env.observation_manager = ObservationManager({}, env)
    env.observation_manager._env._rollout_storage = getattr(
        env.observation_manager._env, "_rollout_storage", None
    )
    if env.observation_manager._env._rollout_storage is None:
        env.observation_manager._env._rollout_storage = TensorDict(
            {},
            batch_size=[len(env.observation_manager._env.episode_context.episode_ids)],
        )
    _init_raw_rollout(
        env.observation_manager._env._rollout_storage,
        env.observation_manager._env.episode_context,
        **dict(fields={}, actions=torch.zeros(1, 2, 2)),
    )
    env.observation_manager.bind_rollout_storage(
        env.observation_manager._env._rollout_storage
    )
    descriptor = ObservationDescriptor(
        "robot", "qpos", "joint", "joint_order", ("joint_units",), (3,), torch.float32
    )
    revisions = count()

    def capture():
        sample = ObservationSample(
            descriptor, state.clone(), torch.ones(1, dtype=torch.bool), (None,)
        )
        return ObservationFrame(
            {"qpos": state.clone()},
            {"qpos": sample},
            context.episode_ids,
            next(revisions),
            episode_owner=context,
        )

    env._capture_observation = capture
    env._record_initial_observation = (
        lambda frame, ids: EmbodiedEnv._record_initial_observation(env, frame, ids)
    )
    EmbodiedEnv._record_initial_observation(env, capture(), [0])
    trajectory = _trajectory()
    trajectory["states"] = TensorDict(
        {
            "robot": {
                "root_pose": torch.zeros(1, 2, 7),
                "qpos": torch.tensor([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]),
            }
        },
        batch_size=[1, 2],
    )
    obs, _ = ReplayWrapper(env, trajectory).reset()
    snapshot = env.observation_manager.read_history(
        [0], fields=["qpos"], episode_ids=[0]
    )
    torch.testing.assert_close(snapshot.values["qpos"], torch.tensor([[7.0, 8.0, 9.0]]))
    torch.testing.assert_close(obs["qpos"], torch.tensor([[7.0, 8.0, 9.0]]))


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("record_policy", [False, True])
def test_expert_opt_in_uses_policy_controller_with_full_joint_target(
    completed: bool,
    record_policy: bool,
) -> None:
    """Converted expert targets keep measured holds on completed demo rows."""
    from types import MethodType
    from unittest.mock import MagicMock

    from embodichain.lab.gym.envs.expert_trajectory import ExpertTrajectoryCfg
    from embodichain.lab.gym.envs.managers import (
        ActionManager,
        ActionTermCfg,
        ParallelGripperAction,
    )

    env = _ReplayEnv(num_envs=2)
    env._explicit_policy_rollout = record_policy
    env._record_raw_actions = record_policy
    env._raw_action_history = [[], []]
    env._validate_policy_rollout_action = MethodType(
        EmbodiedEnv._validate_policy_rollout_action, env
    )
    if record_policy:
        env._traj_buffer = {}
        env._trajectory_action_kind = "policy"
    env._demo_no_auto_reset = True
    env._demo_active_mask = torch.tensor([not completed, True])
    env._mask_controller_demo_action = MethodType(
        EmbodiedEnv._mask_controller_demo_action, env
    )
    env.cfg = SimpleNamespace(
        expert_trajectory=ExpertTrajectoryCfg(use_action_manager=True)
    )
    measured = torch.tensor([[0.0, 0.015, 0.025], [0.0, 0.01, 0.02]])
    env.robot.get_qpos = lambda: measured
    env.robot.get_joint_ids = lambda part, **kwargs: [1, 2]
    env.robot.body_data = SimpleNamespace(
        qpos_limits=torch.tensor([[[-1.0, 1.0], [0.0, 0.04], [0.0, 0.04]]])
    )
    env.robot.set_qpos = MagicMock()
    env.action_manager = ActionManager(
        {
            "gripper": ActionTermCfg(
                func=ParallelGripperAction,
                params={
                    "part_name": "hand",
                    "command_mode": "binary",
                    "open_command": {"elbow": 0.04, "wrist": 0.04},
                    "close_command": {"elbow": 0.0, "wrist": 0.0},
                },
            )
        },
        env,
    )
    requested = torch.full((2, 2), 0.04)
    prepared = env._preprocess_action(ControllerAction(requested))
    applied = env._step_action(prepared)
    expected = requested.clone()
    if completed:
        expected[0] = measured[0, 1:]
    torch.testing.assert_close(env.robot.set_qpos.call_args.kwargs["qpos"], expected)
    torch.testing.assert_close(prepared.value, expected)
    assert env._step_uses_action_manager
    if record_policy:
        labels = env.action_manager.action.clone()
        assert labels.shape == (2, 1)
        torch.testing.assert_close(applied, labels)
        torch.testing.assert_close(env._last_action_trace.requested, labels)
        torch.testing.assert_close(env._traj_raw_action, labels)
        torch.testing.assert_close(env._raw_action_history[1][0], labels[1])
        if completed:
            assert env._raw_action_history[0] == []
        env.action_manager.action.zero_()
        torch.testing.assert_close(env._traj_raw_action, labels)
