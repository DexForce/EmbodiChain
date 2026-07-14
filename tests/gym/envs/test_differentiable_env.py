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
"""Tests for DifferentiableEmbodiedEnv."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from embodichain.lab.gym.envs.differentiable_env import (
    DifferentiableEmbodiedEnv,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.sim.diff import NewtonStepFunc
import embodichain.lab.sim.diff.bridge as diff_bridge
from embodichain.lab.sim.sim_manager import SimulationManagerCfg

_CONTROL_SUBSTEPS = 3


class _FakeTape:
    """Minimal Warp tape context used to exercise the PyTorch bridge."""

    def __enter__(self) -> "_FakeTape":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        return False


class _FakeWarp:
    """Subset of Warp used by ``NewtonStepFunc.forward`` in these tests."""

    float32 = object()
    Tape = _FakeTape

    @staticmethod
    def from_torch(tensor: torch.Tensor, **_: Any) -> torch.Tensor:
        return tensor


class _FakeModel:
    """Provides the alternate Newton state buffer."""

    def __init__(self) -> None:
        self.state_out = object()

    def state(self) -> object:
        return self.state_out


class _FakeNewtonManager:
    """Provides the state buffers and solver timestep consumed by the bridge."""

    def __init__(self) -> None:
        self._state_0 = object()
        self._model = _FakeModel()
        self.solver_dt = 0.01


class _FakeStepper:
    """Records native differentiable-stepper invocations."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object, float]] = []

    def create_contacts(self) -> object:
        return object()

    def step(
        self,
        state_in: object,
        state_out: object,
        *,
        contacts: object,
        dt: float,
    ) -> None:
        self.calls.append((state_in, state_out, contacts, dt))


class _FakeManager:
    """Minimal SimulationManager surface used by the differentiable bridge."""

    def __init__(self) -> None:
        self.physics = SimpleNamespace(newton_manager=_FakeNewtonManager())
        self.steppers: list[_FakeStepper] = []

    def create_differentiable_stepper(self) -> _FakeStepper:
        stepper = _FakeStepper()
        self.steppers.append(stepper)
        return stepper


def _route_env(
    manager: _FakeManager,
    *,
    mode: str | None = None,
) -> tuple[DifferentiableEmbodiedEnv, list[object]]:
    """Build an uninitialized environment with only the route dependencies."""
    env = object.__new__(DifferentiableEmbodiedEnv)
    env.sim = manager
    env.cfg = SimpleNamespace(sim_steps_per_control=_CONTROL_SUBSTEPS)
    if mode is not None:
        env.differentiable_step_mode = mode
    final_states: list[object] = []

    def _apply_action(_action_wp: torch.Tensor, tape: Any) -> None:
        del tape

    def _read_outputs(final_state: object) -> dict[str, Any]:
        final_states.append(final_state)
        return {
            "obs": torch.zeros(1, 1),
            "reward": torch.zeros(1),
            "terminated": torch.zeros(1, dtype=torch.bool),
            "truncated": torch.zeros(1, dtype=torch.bool),
            "_order": ("obs", "reward", "terminated", "truncated"),
            "_grad_track": {},
        }

    env._apply_action_kernel = _apply_action
    env._read_outputs = _read_outputs
    return env, final_states


def _diff_env_cfg(
    requires_grad: bool = True, backend: str = "newton"
) -> EmbodiedEnvCfg:
    if backend == "newton":
        physics_cfg = NewtonPhysicsCfg(
            requires_grad=requires_grad,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        )
    else:
        physics_cfg = DefaultPhysicsCfg()
    sim_cfg = SimulationManagerCfg(
        physics_cfg=physics_cfg,
        num_envs=2,
        headless=True,
    )
    return EmbodiedEnvCfg(sim_cfg=sim_cfg)


def test_default_dynamics_route_uses_bridge_stepper_without_bypass(monkeypatch):
    """Default state construction delegates every substep to the bridge."""
    manager = _FakeManager()
    env, final_states = _route_env(manager)
    monkeypatch.setattr(diff_bridge, "wp", _FakeWarp)

    sim_state = env._build_sim_state_dict(torch.zeros(1, requires_grad=True))
    outputs = NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    assert "step_fn" not in sim_state
    assert len(outputs) == 4
    assert len(manager.steppers) == 1
    assert len(manager.steppers[0].calls) == _CONTROL_SUBSTEPS
    assert final_states == [manager.physics.newton_manager._model.state_out]


def test_default_step_helper_advances_each_dynamics_substep():
    """The public default helper remains usable outside the bridge route."""
    manager = _FakeManager()
    env, _ = _route_env(manager)

    final_state = env._make_step_fn()()

    assert final_state is manager.physics.newton_manager._model.state_out
    assert len(manager.steppers) == 1
    assert len(manager.steppers[0].calls) == _CONTROL_SUBSTEPS


def test_kinematics_route_uses_only_named_kinematic_hook():
    """FK stepping is selected only through the explicit kinematics mode."""
    manager = _FakeManager()
    env, _ = _route_env(manager, mode="kinematics")
    expected_state = object()
    kinematic_calls: list[None] = []

    def _kinematic_step() -> object:
        kinematic_calls.append(None)
        return expected_state

    def _generic_step_fn() -> object:
        raise AssertionError("The generic step helper must not route kinematics.")

    env._make_kinematic_step_fn = lambda: _kinematic_step
    env._make_step_fn = _generic_step_fn

    sim_state = env._build_sim_state_dict(torch.zeros(1))

    assert sim_state["step_fn"]() is expected_state
    assert kinematic_calls == [None]
    assert manager.steppers == []


def test_kinematics_route_requires_named_hook():
    """Kinematics mode rejects environments that do not define its hook."""
    manager = _FakeManager()
    env, _ = _route_env(manager, mode="kinematics")

    with pytest.raises(
        NotImplementedError, match=r"kinematics.*_make_kinematic_step_fn"
    ):
        env._build_sim_state_dict(torch.zeros(1))


def test_invalid_differentiable_step_mode_raises_clear_error():
    """Unsupported stepping modes fail before creating a bridge callback."""
    manager = _FakeManager()
    env, _ = _route_env(manager, mode="unsupported")

    with pytest.raises(
        ValueError, match=r"differentiable_step_mode.*dynamics.*kinematics"
    ):
        env._build_sim_state_dict(torch.zeros(1))


def test_construct_without_requires_grad_raises():
    with pytest.raises(Exception, match=r"requires_grad"):
        DifferentiableEmbodiedEnv(_diff_env_cfg(requires_grad=False))


def test_construct_on_default_backend_raises():
    with pytest.raises(Exception, match=r"Newton"):
        DifferentiableEmbodiedEnv(_diff_env_cfg(backend="default"))


def _import_franka_env():
    """Import the Franka APG env, skipping if the URDF is unavailable.

    The URDF resolves through ``newton.utils.download_asset`` which
    requires network access on first run. Tests skip cleanly when the
    asset cannot be fetched.
    """
    from embodichain.lab.gym.envs.tasks.special.franka_reach_apg import (
        FrankaReachApgEnv,
    )

    return FrankaReachApgEnv


def test_franka_apg_smoke_backward():
    """Verify reward is autograd-tracked and action.grad flows back."""
    try:
        FrankaReachApgEnv = _import_franka_env()
    except FileNotFoundError as e:
        pytest.skip(f"Franka URDF not available: {e}")

    env = FrankaReachApgEnv(num_envs=2)
    env.reset(seed=0)
    action = torch.zeros(2, 7, requires_grad=True, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward.requires_grad, "Reward must be autograd-tracked."
    loss = reward.sum()
    loss.backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()


def test_franka_apg_one_iter_loss_reduces():
    """Verify a single SGD step reduces the APG loss."""
    try:
        FrankaReachApgEnv = _import_franka_env()
    except FileNotFoundError as e:
        pytest.skip(f"Franka URDF not available: {e}")

    env = FrankaReachApgEnv(num_envs=2)
    env.reset(seed=0)
    action = torch.zeros(2, 7, requires_grad=True, device=env.device)
    opt = torch.optim.SGD([action], lr=0.01)

    losses = []
    for _ in range(3):
        env.reset(seed=0)
        opt.zero_grad()
        _, reward, _, _, _ = env.step(action)
        loss = (-reward).sum()
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    assert losses[-1] < losses[0], f"APG did not reduce loss: {losses}"
