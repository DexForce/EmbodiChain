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

"""Environment graph/reset semantics over actual CPU functional dynamics."""

from __future__ import annotations
from types import SimpleNamespace
import torch
import warp as wp
import pytest
from embodichain.lab.gym.envs.differentiable_dynamics_env import (
    DifferentiableDynamicsEnv,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg, DefaultPhysicsCfg
from embodichain.lab.sim.sim_manager import SimulationManagerCfg
from embodichain.lab.sim.diff.dynamics import FunctionalDynamics

# Reuse a real registered-provider fixture without importing tutorial code.
import importlib.util
from pathlib import Path

_fixture_spec = importlib.util.spec_from_file_location(
    "_dynamics_fixture",
    Path(__file__).resolve().parents[2] / "sim" / "diff" / "test_dynamics.py",
)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
manager = _fixture_module.manager


class _Task(DifferentiableDynamicsEnv):
    def _action_to_control(self, action, state):
        return action.flatten()

    def _read_outputs(self, state):
        q = state.physical["joint_q"]
        return {
            "obs": q[:, None],
            "reward": q.square(),
            "terminated": q > self.threshold,
            "truncated": torch.zeros_like(q, dtype=torch.bool),
        }


def make_env(
    manager, monkeypatch: pytest.MonkeyPatch, *, threshold: float = 10.0, steps: int = 2
):
    # Bypass only the scene/render construction already covered by EmbodiedEnv;
    # production step/reset/facade and actual solver execution remain exercised.
    env = object.__new__(_Task)
    env.sim = manager
    env._num_envs = 2
    env.cfg = SimpleNamespace(sim_steps_per_control=steps, ignore_terminations=False)
    env.sim_cfg = SimpleNamespace(physics_dt=0.1, device="cpu")
    env.threshold = threshold
    env._dynamics = FunctionalDynamics(manager)
    env._functional_state = env._dynamics.create_state()
    env._resetting_functional = False
    env._elapsed_steps = torch.zeros(2, dtype=torch.int32)
    env.max_episode_steps = 100
    env.sensors = {}
    env._update_physical_objective = lambda: None
    env._hook_after_sim_step = lambda **kwargs: None

    def native_reset(self, seed=None, options=None):
        ids = options.get("reset_ids", torch.arange(2))
        selected = torch.zeros(3, dtype=torch.bool)
        selected[ids] = True
        self.sim.backend.reset_state(wp.from_torch(selected, dtype=wp.bool))
        self._elapsed_steps[ids] = 0
        return self.get_obs(), {}

    monkeypatch.setattr(EmbodiedEnv, "reset", native_reset)
    return env


def test_control_interval_and_multistep_action_gradient(manager, monkeypatch) -> None:
    env = make_env(manager, monkeypatch)
    first = torch.tensor([[1.0], [2.0]], requires_grad=True)
    second = torch.tensor([[0.5], [1.0]], requires_grad=True)
    obs, reward, *_ = env.step(first)
    obs2, reward2, *_ = env.step(second)
    torch.testing.assert_close(obs2, torch.tensor([[0.6], [1.2]]))
    assert manager.backend.step_index == 4
    (reward.sum() + reward2.sum()).backward()
    torch.testing.assert_close(first.grad, torch.tensor([[0.8], [1.6]]))
    torch.testing.assert_close(second.grad, torch.tensor([[0.48], [0.96]]))


def test_terminal_reset_keeps_reward_graph_and_other_rows(manager, monkeypatch) -> None:
    env = make_env(manager, monkeypatch, threshold=0.3, steps=1)
    action = torch.tensor([[2.0], [1.0]], requires_grad=True)
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated.tolist() == [True, False]
    torch.testing.assert_close(obs, torch.tensor([[0.0], [0.2]]))
    assert env.functional_state.valid_history_lengths.tolist() == [0, 2]
    assert manager.backend.step_index == 1
    assert info["terminal_mask"].tolist() == [True, False]
    loss = reward.sum() + obs.sum() + env.functional_state.history["memory"].sum()
    loss.backward()
    torch.testing.assert_close(action.grad, torch.tensor([[0.16], [0.38]]))


def test_detach_state_stops_prior_control_gradient_without_reset(
    manager, monkeypatch
) -> None:
    env = make_env(manager, monkeypatch, steps=1)
    first = torch.ones(2, 1, requires_grad=True)
    env.step(first)
    before = env.functional_state.physical["joint_q"].detach().clone()
    obs = env.detach_state()
    assert not obs.requires_grad
    torch.testing.assert_close(env.functional_state.physical["joint_q"], before)
    second = torch.ones(2, 1, requires_grad=True)
    _, reward, *_ = env.step(second)
    reward.sum().backward()
    assert first.grad is None
    assert second.grad is not None
    assert manager.backend.step_index == 2


def test_max_episode_steps_truncate_and_reset(manager, monkeypatch) -> None:
    env = make_env(manager, monkeypatch, steps=1)
    env.max_episode_steps = 1
    obs, reward, terminated, truncated, _ = env.step(
        torch.ones(2, 1, requires_grad=True)
    )
    assert truncated.tolist() == [True, True] and not terminated.any()
    assert not obs.any() and env._elapsed_steps.tolist() == [0, 0]
    reward.sum().backward()


def test_configuration_rejects_nonfunctional_paths_before_scene_creation() -> None:
    cfg = EmbodiedEnvCfg(sim_cfg=SimulationManagerCfg(physics_cfg=DefaultPhysicsCfg()))
    with pytest.raises(ValueError, match="NewtonPhysicsCfg"):
        DifferentiableDynamicsEnv._validate_dynamics_cfg(cfg)
    cfg.sim_cfg.physics_cfg = NewtonPhysicsCfg(requires_grad=False)
    with pytest.raises(ValueError, match="requires_grad"):
        DifferentiableDynamicsEnv._validate_dynamics_cfg(cfg)
    cfg.sim_cfg.physics_cfg.requires_grad = True
    cfg.actions = {"unsupported": object()}
    with pytest.raises(ValueError, match="functional hooks"):
        DifferentiableDynamicsEnv._validate_dynamics_cfg(cfg)


def test_sensor_sampling_follows_macro_steps_and_ignore_terminations(
    manager, monkeypatch
) -> None:
    env = make_env(manager, monkeypatch, threshold=0.0, steps=3)
    env.cfg.ignore_terminations = True
    calls = []
    env.sensors = {
        "sensor": SimpleNamespace(
            requires_substep_update=True,
            begin_control_step=lambda: calls.append("begin"),
            update_physics_step=lambda dt: calls.append(dt),
        )
    }
    obs, _, terminated, _, _ = env.step(torch.ones(2, 1))
    assert calls == ["begin", 0.1, 0.1, 0.1]
    assert not terminated.any()
    torch.testing.assert_close(obs, torch.tensor([[0.6], [0.6]]))
