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

"""Provider-free behavior tests for :class:`ReplayWrapper`."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

import embodichain.lab.gym.envs.wrapper.replay as replay_module
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.envs.wrapper import ReplayWrapper


class _FakeSim:
    def __init__(self) -> None:
        self.physics_enabled: list[bool] = []
        self.updates: list[tuple[float, int]] = []

    def enable_physics(self, enabled: bool) -> None:
        self.physics_enabled.append(enabled)

    def update(self, physics_dt: float, substeps: int) -> None:
        self.updates.append((physics_dt, substeps))


class _FakeEnv(gym.Env):
    metadata = {}

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.max_episode_steps = 10
        self.step_dt = 0.04
        self.robot = SimpleNamespace(dof=2, joint_names=["joint0", "joint1"])
        self.active_joint_ids = [0, 1]
        self.sim = _FakeSim()
        self.sim_cfg = SimpleNamespace(physics_dt=0.01)
        self.cfg = SimpleNamespace(sim_steps_per_control=4)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(num_envs, 2))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(num_envs, 1))
        self.received_actions: list[object] = []
        self.last_restored = None
        self.closed = False
        self._replay_no_auto_reset = False

    def reset(self, *, seed=None, options=None):
        del seed, options
        return self.get_obs(), {}

    def step(self, action):
        self.received_actions.append(action)
        return (
            self.get_obs(),
            torch.ones(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )

    def get_obs(self):
        return {"obs": torch.zeros(self.num_envs, 1)}

    def close(self):
        self.closed = True


def _trajectory(
    *,
    num_envs: int = 1,
    num_steps: int = 2,
    action_width: int = 2,
    meta_updates: dict[str, object] | None = None,
) -> dict:
    qpos = torch.arange(num_envs * num_steps * 2, dtype=torch.float32).reshape(
        num_envs, num_steps, 2
    )
    states = TensorDict(
        {
            "robot": TensorDict(
                {
                    "root_pose": torch.zeros(num_envs, num_steps, 7),
                    "qpos": qpos,
                },
                batch_size=[num_envs, num_steps],
            )
        },
        batch_size=[num_envs, num_steps],
    )
    meta: dict[str, object] = {
        "num_envs": num_envs,
        "num_steps": num_steps,
        "lengths": [num_steps] * num_envs,
        "robot_dof": 2,
        "active_joint_ids": [0, 1],
        "expert_trajectory_schema_version": 1,
        "joint_command_mode": "position",
        "joint_names": ["joint0", "joint1"],
        "qpos_slice": [0, 2],
        "qvel_slice": None,
        "step_dt": 0.04,
    }
    if meta_updates:
        meta.update(meta_updates)
    return {
        "states": states,
        "actions": torch.arange(
            num_envs * num_steps * action_width, dtype=torch.float32
        ).reshape(num_envs, num_steps, action_width),
        "meta": meta,
    }


@pytest.fixture
def fake_restore(monkeypatch):
    def restore(env, state):
        env.last_restored = state.clone()

    monkeypatch.setattr(replay_module, "restore_trajectory_state", restore)


@pytest.mark.parametrize("mode", ["kinematic", "control"])
def test_state_replay_modes_restore_states_and_truncate(fake_restore, mode):
    env = _FakeEnv()
    wrapper = ReplayWrapper(env, _trajectory(), mode=mode)

    wrapper.reset()
    assert env.sim.physics_enabled == [False]
    _, reward, terminated, truncated, info = wrapper.step(None)
    assert torch.equal(env.last_restored["robot"]["qpos"], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(reward, torch.zeros(1))
    assert not bool(terminated.item())
    assert not bool(truncated.item())
    assert info == {}

    _, _, _, truncated, _ = wrapper.step(None)
    assert bool(truncated.item())


def test_control_mode_clamps_scrub_to_last_recorded_state(fake_restore):
    env = _FakeEnv()
    wrapper = ReplayWrapper(env, _trajectory(), mode="control")

    wrapper.reset()
    wrapper.go_to_step(99)

    assert wrapper.control_max_step == 1
    assert torch.equal(env.last_restored["robot"]["qpos"], torch.tensor([[2.0, 3.0]]))
    assert wrapper._replay_steps.tolist() == [1]


def test_broadcast_replay_preserves_per_env_lengths():
    trajectory = _trajectory(
        num_envs=1,
        meta_updates={"lengths": torch.tensor([2])},
    )
    wrapper = ReplayWrapper(_FakeEnv(num_envs=2), trajectory, mode="kinematic")

    assert wrapper._lengths.tolist() == [2, 2]


@pytest.mark.parametrize(
    ("seal", "status", "accepted"),
    [
        (None, "not_recorded", None),
        ({"accepted": True, "observation_revision": 17}, "not_recorded", None),
        ({"accepted": False, "observation_revision": None}, "not_run", False),
    ],
)
def test_trajectory_replays_without_sealed_observations(
    fake_restore, seal, status, accepted
):
    from embodichain.lab.evaluation import (
        EvaluationSpec,
        PredicateSpec,
        builtin_predicates,
    )

    env = _FakeEnv(num_envs=2)
    env._final_acceptance_rules = (
        (
            "near",
            builtin_predicates().compile(
                EvaluationSpec(
                    {
                        "near": PredicateSpec(
                            "near_target",
                            {
                                "position": "rigid_objects/a/pose",
                                "target": "rigid_objects/b/pose",
                            },
                            {"tolerance": 0.5},
                        )
                    },
                    ("near",),
                )
            ),
        ),
    )
    trajectory = _trajectory()
    if seal is not None:
        trajectory["meta"]["demo_episodes"] = [{"final_acceptance": seal}]
    wrapper = ReplayWrapper(env, trajectory, mode="kinematic")
    assert (
        wrapper.final_acceptance
        == (
            {
                "status": status,
                "accepted": accepted,
                "recorded_accepted": seal["accepted"] if seal is not None else None,
            },
        )
        * 2
    )
    wrapper.reset()
    _, _, _, truncated, info = wrapper.step(None)
    assert not truncated.any()
    assert info == {}
    assert trajectory["meta"]["num_envs"] == 1


def test_dynamic_replay_decodes_position_velocity_expert_action(fake_restore):
    env = _FakeEnv()
    trajectory = _trajectory(
        action_width=4,
        meta_updates={
            "action_kind": "expert",
            "joint_command_mode": "position_velocity",
            "qpos_slice": [0, 2],
            "qvel_slice": [2, 4],
        },
    )
    wrapper = ReplayWrapper(env, trajectory, mode="dynamic")

    wrapper.reset()
    wrapper.step(None)

    assert env.sim.physics_enabled == [False, True]
    assert env._replay_no_auto_reset is True
    action = env.received_actions[0]
    assert isinstance(action, ControllerAction)
    assert torch.equal(action.value["qpos"], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(action.value["qvel"], torch.tensor([[2.0, 3.0]]))


def test_replay_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Invalid replay mode"):
        ReplayWrapper(_FakeEnv(), _trajectory(), mode="unsupported")


def _cable_replay_case():
    env = _FakeEnv()
    trajectory = _trajectory()
    cable_state = torch.zeros(1, 3, 13)
    cable_state[..., 6] = 1
    trajectory["states"]["cables"] = TensorDict(
        {
            "wire": {
                "segment_state_w": cable_state[:, None].expand(-1, 2, -1, -1).clone()
            }
        },
        batch_size=[1, 2],
    )
    env.sim._cable_objects = {
        "wire": SimpleNamespace(data=SimpleNamespace(segment_state_w=cable_state))
    }
    env.sim._articulations = {}
    env.sim._rigid_objects = {}
    env.robot.get_qpos = lambda: torch.tensor([[0.0, 1.0]])
    env.robot.get_local_pose = lambda: torch.zeros(1, 7)
    return env, trajectory


def test_cable_dynamic_replay_uses_reset_without_restoring_observations(fake_restore):
    env, trajectory = _cable_replay_case()
    wrapper = ReplayWrapper(env, trajectory, mode="dynamic")
    wrapper.reset()
    assert env.last_restored is None
    assert env.sim.physics_enabled == [True]
    wrapper.step(None)
    torch.testing.assert_close(env.received_actions[0], trajectory["actions"][:, 0])
    with pytest.raises(NotImplementedError, match="scrubbing"):
        wrapper.go_to_step(1)
    with pytest.raises(ValueError, match="reset_ids"):
        wrapper.reset(options={"reset_ids": [0]})


def test_cable_dynamic_replay_rejects_a_mid_episode_initial_observation(fake_restore):
    env, trajectory = _cable_replay_case()
    trajectory["states"]["cables", "wire", "segment_state_w"][0, 0, 0, 0] = 0.1
    wrapper = ReplayWrapper(env, trajectory, mode="dynamic")
    with pytest.raises(ValueError, match="differs from the full reset"):
        wrapper.reset()
    assert env.last_restored is None and not env.received_actions


@pytest.mark.parametrize("mode", ["kinematic", "control"])
def test_cable_replay_rejects_history_free_state_modes_before_mutation(mode):
    env, trajectory = _cable_replay_case()
    with pytest.raises(NotImplementedError, match="dynamic"):
        ReplayWrapper(env, trajectory, mode=mode)
    assert env.sim.physics_enabled == []


def test_cable_replay_rejects_broadcast_world_states():
    env, trajectory = _cable_replay_case()
    with pytest.raises(ValueError, match="environment counts"):
        ReplayWrapper(_FakeEnv(num_envs=2), trajectory, mode="dynamic")


def test_cable_scene_replay_rejects_recording_without_cable_initial_observations():
    env, trajectory = _cable_replay_case()
    del trajectory["states"]["cables"]
    with pytest.raises(ValueError, match="every scene cable"):
        ReplayWrapper(env, trajectory, mode="dynamic")
    assert env.sim.physics_enabled == []


def test_cable_replay_preserves_task_context_and_initial_observation_history(
    fake_restore,
):
    from unittest.mock import Mock

    env, trajectory = _cable_replay_case()
    initial = {"schema_version": 1, "targets": {"goal": [0.0, 0.0, 1.0]}}
    trajectory["meta"]["demo_episodes"] = [{"task_initial_state": initial}]
    env.termination_manager = object()
    env.restore_task_initial_state = Mock()
    env.get_info = lambda: {"task_context": "restored"}
    env.observation_manager = SimpleNamespace(has_history=True)
    frame = SimpleNamespace(policy={"recorded_observation": torch.ones(1, 1)})
    env._capture_observation = lambda: frame
    env._record_initial_observation = Mock()
    env._seed_recording_state = Mock()

    wrapper = ReplayWrapper(env, trajectory, mode="dynamic")
    obs, info = wrapper.reset()

    assert env.last_restored is None
    env.restore_task_initial_state.assert_called_once_with(initial)
    assert obs is frame.policy
    assert info == {"task_context": "restored"}
    assert env._record_initial_observation.call_args.args[0] is frame
    assert env._record_initial_observation.call_args.args[1].tolist() == [0]
    assert env._seed_recording_state.call_args.args[0] is frame.policy


@pytest.mark.parametrize("source_env_id", [0, 1])
@pytest.mark.parametrize("target_origin", [(0.0, 0.0, 0.0), (10.0, -2.0, 3.0)])
def test_auto_saved_cable_row_replays_at_target_arena_origin(
    tmp_path, source_env_id, target_origin
):
    from copy import deepcopy
    from types import MethodType

    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.utils.gym_utils import load_trajectory

    target, _ = _cable_replay_case()
    local_state = target.sim._cable_objects["wire"].data.segment_state_w.clone()
    local_state[0, :, 0] = torch.tensor([0.1, 0.2, 0.3])
    local_state[..., 7:] = 0.01
    target.sim.arena_offsets = torch.tensor([target_origin])
    target_state = local_state.clone()
    target_state[..., :3] += target.sim.arena_offsets[:, None, :]
    target.sim._cable_objects["wire"].data.segment_state_w = target_state

    source = _FakeEnv(num_envs=2)
    source.robot.uid = "robot"
    source.physics_dt = 0.01
    source.control_frequency = 25
    source.sim._articulations = {}
    source.sim._rigid_objects = {}
    source.sim.arena_offsets = torch.tensor([[0.0, 0.0, 0.0], [5.0, 2.0, 1.0]])
    trajectory = _trajectory(num_envs=2)
    trajectory["states"]["robot", "qpos"][:] = torch.tensor([0.0, 1.0])
    states = local_state[:, None].repeat(2, 2, 1, 1)
    states[..., :3] += source.sim.arena_offsets[:, None, None, :]
    trajectory["states"]["cables"] = TensorDict(
        {"wire": {"segment_state_w": states}}, batch_size=[2, 2]
    )
    source._traj_buffer = TensorDict(
        {"states": trajectory["states"], "actions": trajectory["actions"]},
        batch_size=[2, 2],
    )
    source._traj_steps = torch.tensor([2, 2])
    source.cfg.trajectory_auto_save = True
    source.cfg.trajectory_save_dir = str(tmp_path)
    source._traj_save_count = 0
    source._demo_episode_metadata = [
        {"env_id": row, "length": 2, "episode_uuid": f"episode-{row}"}
        for row in range(2)
    ]
    source.get_demo_episode_metadata = lambda row: deepcopy(
        source._demo_episode_metadata[row]
    )
    source._trajectory_action_metadata = lambda: {"action_kind": "policy"}
    source.episode_context = EpisodeContext(2)
    source.episode_context._begin_episode([0, 1])
    source.episode_context._finish_initialization([0, 1])
    source.save_trajectory = MethodType(EmbodiedEnv.save_trajectory, source)

    path = EmbodiedEnv._save_trajectory_for_env(source, source_env_id)
    saved = load_trajectory(path)
    assert saved["meta"]["num_envs"] == 1
    assert saved["meta"]["env_ids"] == [source_env_id]
    assert saved["meta"]["arena_origins_w"] == [
        source.sim.arena_offsets[source_env_id].tolist()
    ]
    recorded = saved["states"]["cables", "wire", "segment_state_w"].clone()
    wrapper = ReplayWrapper(target, saved)
    wrapper.reset()
    wrapper.step(None)
    torch.testing.assert_close(
        target.received_actions[0], trajectory["actions"][[source_env_id], 0]
    )
    torch.testing.assert_close(
        saved["states"]["cables", "wire", "segment_state_w"], recorded
    )
    torch.testing.assert_close(
        target.sim._cable_objects["wire"].data.segment_state_w, target_state
    )
    target_state[..., 0] += 0.1
    with pytest.raises(ValueError, match="differs from the full reset"):
        wrapper.reset()


@pytest.mark.parametrize("origins", [[[0.0, 0.0]], [[float("nan"), 0.0, 0.0]]])
def test_cable_replay_rejects_invalid_source_origins_before_reset(origins):
    env, trajectory = _cable_replay_case()
    trajectory["meta"]["arena_origins_w"] = origins
    with pytest.raises(ValueError, match="arena_origins_w"):
        ReplayWrapper(env, trajectory)
    assert env.sim.physics_enabled == []
