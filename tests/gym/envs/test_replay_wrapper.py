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
        self.robot = SimpleNamespace(dof=2)
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
