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

import numpy as np
import pytest
import torch
import warp as wp

from embodichain.lab.gym.envs.differentiable_env import (
    DifferentiableEmbodiedEnv,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.sim.diff import NewtonStepFunc, differentiable_step
import embodichain.lab.sim.diff.bridge as diff_bridge
from embodichain.lab.sim.sim_manager import SimulationManagerCfg

_CONTROL_SUBSTEPS = 3


@wp.kernel
def _write_bridge_joint_force_kernel(
    action: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
) -> None:
    """Write one tape-tracked action value into Newton joint force."""
    joint_f[0] = action[0]


@wp.kernel
def _bridge_terminal_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_id: int,
    target: wp.vec3,
    loss: wp.array(dtype=wp.float32),
) -> None:
    """Measure a terminal body-position loss inside the Warp tape."""
    delta = wp.transform_get_translation(body_q[body_id]) - target
    loss[0] = wp.dot(delta, delta)


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
    """Allocates independent Newton trajectory states."""

    def __init__(self) -> None:
        self.states: list[_FakeState] = []

    def state(self) -> "_FakeState":
        state = _FakeState(f"trajectory-{len(self.states)}")
        self.states.append(state)
        return state


class _FakeState:
    """State buffer with explicit detached-copy observability."""

    def __init__(self, name: str, value: int = 0) -> None:
        self.name = name
        self.value = value
        self.assign_sources: list[_FakeState] = []

    def assign(self, other: "_FakeState") -> None:
        """Copy a state outside the fake Warp tape."""
        self.value = other.value
        self.assign_sources.append(other)


class _FakeNewtonManager:
    """Provides the state buffers and solver timestep consumed by the bridge."""

    def __init__(self, *, num_substeps: int = 1) -> None:
        self._state_0 = _FakeState("live-state-0")
        self._state_1 = _FakeState("live-state-1")
        self._model = _FakeModel()
        self.num_substeps = num_substeps
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
        state_out.value = state_in.value + 1


class _FakeManager:
    """Minimal SimulationManager surface used by the differentiable bridge."""

    def __init__(self, *, num_substeps: int = 1) -> None:
        self.is_newton_backend = True
        self.physics = SimpleNamespace(
            newton_manager=_FakeNewtonManager(num_substeps=num_substeps)
        )
        self.steppers: list[_FakeStepper] = []

    def create_differentiable_stepper(self) -> _FakeStepper:
        stepper = _FakeStepper()
        self.steppers.append(stepper)
        return stepper


class _CountingSolver:
    """Record solver calls while delegating to the real Newton solver."""

    def __init__(self, solver: Any, call_counts: dict[str, int]) -> None:
        self._solver = solver
        self._call_counts = call_counts

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Count and delegate one solver step."""
        self._call_counts["solver"] += 1
        return self._solver.step(*args, **kwargs)


class _CountingStepper:
    """Record bridge stepper calls while retaining the real primitive."""

    def __init__(self, stepper: Any, call_counts: dict[str, int]) -> None:
        self._stepper = stepper
        self._call_counts = call_counts

    def create_contacts(self) -> Any:
        """Allocate a real contact buffer."""
        return self._stepper.create_contacts()

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Count and delegate one differentiable step."""
        self._call_counts["stepper"] += 1
        return self._stepper.step(*args, **kwargs)


class _RealBridgeManager:
    """Expose a real DexSim Newton manager through the bridge surface."""

    def __init__(self, newton_manager: Any) -> None:
        self.physics = SimpleNamespace(newton_manager=newton_manager)
        self.call_counts = {"stepper": 0, "solver": 0}

    def create_differentiable_stepper(self) -> _CountingStepper:
        """Create and instrument a real differentiable Newton stepper."""
        stepper = self.physics.newton_manager.create_differentiable_stepper()
        stepper.solver = _CountingSolver(stepper.solver, self.call_counts)
        return _CountingStepper(stepper, self.call_counts)


def _route_env(
    manager: _FakeManager,
    *,
    mode: str | None = None,
    control_substeps: int = _CONTROL_SUBSTEPS,
) -> tuple[DifferentiableEmbodiedEnv, list[object]]:
    """Build an uninitialized environment with only the route dependencies."""
    env = object.__new__(DifferentiableEmbodiedEnv)
    env.sim = manager
    env.cfg = SimpleNamespace(sim_steps_per_control=control_substeps)
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
    """Default state construction delegates every physics substep to the bridge."""
    manager = _FakeManager()
    env, final_states = _route_env(manager)
    monkeypatch.setattr(diff_bridge, "wp", _FakeWarp)

    sim_state = env._build_sim_state_dict(torch.zeros(1, requires_grad=True))
    outputs = NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    assert "step_fn" not in sim_state
    assert len(outputs) == 4
    assert len(manager.steppers) == 1
    assert len(manager.steppers[0].calls) == _CONTROL_SUBSTEPS
    assert final_states[0].value == _CONTROL_SUBSTEPS


def test_dynamics_bridge_keeps_an_odd_trajectory_across_control_steps(monkeypatch):
    """Each control step begins from the prior detached solver final state."""
    manager = _FakeManager()
    env, final_states = _route_env(manager)
    monkeypatch.setattr(diff_bridge, "wp", _FakeWarp)

    for _ in range(2):
        sim_state = env._build_sim_state_dict(torch.zeros(1, requires_grad=True))
        NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    calls = [call for stepper in manager.steppers for call in stepper.calls]
    nm = manager.physics.newton_manager

    assert [state.value for state in final_states] == [3, 6]
    assert nm._state_0.value == 6
    assert nm._state_1.value == 6
    assert [state.value for state in nm._state_0.assign_sources] == [3, 6]
    assert [state.value for state in nm._state_1.assign_sources] == [3, 6]
    assert len({id(state) for call in calls for state in call[:2]}) == 8
    assert all(
        state not in {nm._state_0, nm._state_1} for call in calls for state in call[:2]
    )
    assert len({id(call[2]) for call in calls}) == 6


def test_dynamics_bridge_multiplies_control_and_newton_substeps(monkeypatch):
    """One control step preserves both EmbodiChain and Newton time semantics."""
    manager = _FakeManager(num_substeps=3)
    env, _ = _route_env(manager, control_substeps=2)
    monkeypatch.setattr(diff_bridge, "wp", _FakeWarp)

    sim_state = env._build_sim_state_dict(torch.zeros(1, requires_grad=True))
    NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    calls = manager.steppers[0].calls
    assert len(calls) == 6
    assert {call[3] for call in calls} == {manager.physics.newton_manager.solver_dt}


def test_differentiable_step_uses_detached_trajectory_and_commits_final(monkeypatch):
    """The public helper does not reuse live state or contact buffers."""
    manager = _FakeManager()
    monkeypatch.setattr(diff_bridge, "wp", _FakeWarp)

    result = differentiable_step(
        manager,
        apply_control_fn=lambda _tape: None,
        substeps=_CONTROL_SUBSTEPS,
    )
    calls = manager.steppers[0].calls
    nm = manager.physics.newton_manager

    assert result["final_state"].value == _CONTROL_SUBSTEPS
    assert nm._state_0.value == _CONTROL_SUBSTEPS
    assert nm._state_1.value == _CONTROL_SUBSTEPS
    assert len({id(state) for call in calls for state in call[:2]}) == 4
    assert all(
        state not in {nm._state_0, nm._state_1} for call in calls for state in call[:2]
    )
    assert len({id(call[2]) for call in calls}) == _CONTROL_SUBSTEPS


@pytest.mark.parametrize("substeps", (0, -1))
def test_differentiable_step_rejects_nonpositive_substeps(substeps: int) -> None:
    """The public helper rejects an invalid empty solver trajectory."""
    manager = _FakeManager()

    with pytest.raises(ValueError, match=r"solver_steps.*positive"):
        differentiable_step(
            manager,
            apply_control_fn=lambda _tape: None,
            substeps=substeps,
        )


def test_cpu_newton_bridge_retains_trajectory_and_joint_force_gradient(tmp_path):
    """The real bridge retains a solver trajectory and action gradient on CPU."""
    newton = pytest.importorskip("newton")
    pytest.importorskip("dexsim.engine.newton_physics")
    from dexsim.engine.newton_physics import (
        NewtonCfg,
        NewtonCollisionPipelineCfg,
        NewtonManager,
        SemiImplicitSolverCfg,
    )

    previous_kernel_cache_dir = wp.config.kernel_cache_dir
    nm = None
    wp.config.kernel_cache_dir = str(tmp_path / "warp_cache")
    try:
        cfg = NewtonCfg()
        cfg.device = "cpu"
        cfg.dt = 1.0 / 60.0
        cfg.num_substeps = 2
        cfg.requires_grad = True
        cfg.use_cuda_graph = False
        cfg.solver_cfg = SemiImplicitSolverCfg()
        cfg.collision_pipeline_cfg = NewtonCollisionPipelineCfg(
            broad_phase="explicit",
            requires_grad=True,
        )
        nm = NewtonManager(cfg)
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4,
            kd=1.0e1,
            kf=0.0,
            mu=0.0,
        )
        body_id = nm._builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            mass=1.0,
            label="embodichain_bridge_gradient_ball",
        )
        nm._builder.add_shape_sphere(body=body_id, radius=0.1, cfg=shape_cfg)
        nm._builder.add_ground_plane(cfg=shape_cfg)
        nm.start_simulation()
        assert nm._model.joint_count == 1
        assert nm._control.joint_f is not None
        assert nm._control.joint_f.shape[0] == 6

        manager = _RealBridgeManager(nm)
        initial_state = nm._model.state()
        initial_state.assign(nm._state_0)
        target = wp.vec3(0.5, 0.0, 0.5)

        def _restore_initial_state() -> None:
            nm._state_0.assign(initial_state)
            nm._state_1.assign(initial_state)

        def _run(
            action_value: float, *, requires_grad: bool
        ) -> tuple[torch.Tensor, torch.Tensor]:
            loss_wp = wp.zeros(
                1,
                dtype=wp.float32,
                device=nm._state_0.body_q.device,
                requires_grad=True,
            )

            def _apply_control(action_wp: Any) -> None:
                wp.launch(
                    _write_bridge_joint_force_kernel,
                    dim=1,
                    inputs=[action_wp, nm._control.joint_f],
                    device=nm._control.joint_f.device,
                )

            def _read_reward(final_state: Any) -> dict[str, Any]:
                loss_wp.zero_()
                wp.launch(
                    _bridge_terminal_loss_kernel,
                    dim=1,
                    inputs=[final_state.body_q, body_id, target, loss_wp],
                    device=final_state.body_q.device,
                )
                return {
                    "reward": wp.to_torch(loss_wp),
                    "_order": ("reward",),
                    "_grad_track": {"reward": loss_wp},
                }

            action = torch.tensor(
                [action_value], dtype=torch.float32, requires_grad=requires_grad
            )
            sim_state = {
                "manager": manager,
                "substeps": 2,
                "action_to_control_kernel": _apply_control,
                "kernel_args": (),
                "obs_reward_fn": _read_reward,
            }
            return NewtonStepFunc.apply(action, sim_state)[0], action

        reward, action = _run(1.0, requires_grad=True)
        reward.backward()

        assert manager.call_counts == {"stepper": 4, "solver": 4}
        assert action.grad is not None
        analytic_gradient = float(action.grad[0])
        assert np.isfinite(analytic_gradient)
        assert not np.isclose(analytic_gradient, 0.0)
        assert np.allclose(
            nm._state_0.body_q.numpy(), nm._state_1.body_q.numpy(), atol=1.0e-6
        )

        def _reward_value(action_value: float) -> float:
            _restore_initial_state()
            value, _ = _run(action_value, requires_grad=False)
            return float(value.detach())

        epsilon = 1.0e-3
        finite_difference_gradient = (
            _reward_value(1.0 + epsilon) - _reward_value(1.0 - epsilon)
        ) / (2.0 * epsilon)
        assert np.isclose(
            analytic_gradient,
            finite_difference_gradient,
            rtol=2.0e-2,
            atol=1.0e-4,
        )
    finally:
        if nm is not None:
            nm.clear()
        wp.config.kernel_cache_dir = previous_kernel_cache_dir


def test_dynamics_environment_does_not_expose_generic_step_helper():
    """Only the low-level bridge may accept an arbitrary dynamics callback."""
    assert "_make_step_fn" not in DifferentiableEmbodiedEnv.__dict__


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
