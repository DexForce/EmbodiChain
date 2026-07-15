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


class _FakeModel:
    """Keep the pre-contract bridge path runnable for clean RED failures."""

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
        """Copy state and retain every publication source for assertions."""
        self.value = other.value
        self.assign_sources.append(other)


class _FakeStepper:
    """Fallback used only while proving the old private route is rejected."""

    def __init__(self, *, raise_on_step: bool = False) -> None:
        self.calls: list[tuple[object, object, object, float]] = []
        self._raise_on_step = raise_on_step

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
        if self._raise_on_step:
            raise RuntimeError("injected trajectory-step failure")
        state_out.value = state_in.value + 1


class _RecordingTape:
    """Expose construction, exit, and recording ownership of the fake tape."""

    def __init__(self, warp: "_RecordingWarp") -> None:
        self._warp = warp

    def __enter__(self) -> "_RecordingTape":
        assert not self._warp.tape_active
        self._warp.tape_active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, exc_value, traceback
        assert self._warp.tape_active
        self._warp.tape_active = False
        self._warp.events.append("tape.exit")
        return False

    def reset(self) -> None:
        """Model the tape cleanup required before releasing a trajectory."""
        assert not self._warp.tape_active
        self._warp.events.append("tape.reset")

    def backward(self, *_args: Any, **_kwargs: Any) -> None:
        """Provide the minimal action gradient required by bridge tests."""
        self._warp.events.append("tape.backward")
        if self._warp.raise_on_tape_backward:
            raise RuntimeError("injected tape backward failure")
        if self._warp.last_action is not None:
            self._warp.last_action.grad = torch.ones_like(self._warp.last_action)

    def zero(self) -> None:
        """Keep the current bridge executable until it migrates to reset()."""
        self._warp.events.append("tape.zero")


class _RecordingWarp:
    """Tiny Warp fake that makes tape ownership observable to a manager."""

    float32 = object()

    def __init__(self) -> None:
        self.tape_active = False
        self.events: list[str] = []
        self.last_action: torch.Tensor | None = None
        self.raise_on_tape_backward = False

    def Tape(self) -> _RecordingTape:
        """Record construction before returning a tape context manager."""
        self.events.append("tape.construct")
        return _RecordingTape(self)

    def from_torch(
        self, tensor: torch.Tensor, *, requires_grad: bool = False, **_: Any
    ) -> torch.Tensor:
        """Preserve the test tensor as the fake Warp action array."""
        action = tensor.detach().clone().requires_grad_(requires_grad)
        self.last_action = action
        return action

    def to_torch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Expose a fake Warp array to the PyTorch bridge."""
        if self.last_action is not None and tensor is self.last_action.grad:
            self.events.append("action-gradient.capture")
        return tensor


class _ManagerOwnedTrajectory:
    """Fake public trajectory whose stepping must occur inside the tape."""

    def __init__(
        self,
        manager: "_TrajectoryNewtonManager",
        *,
        physics_steps: int,
        physics_dt: float,
    ) -> None:
        self._manager = manager
        self.control = object()
        self.physics_steps = physics_steps
        self.physics_dt = physics_dt
        self.total_solver_steps = physics_steps * manager.num_substeps
        self.states = [
            _FakeState(f"trajectory-state-{index}")
            for index in range(self.total_solver_steps + 1)
        ]
        self.states[0].assign(manager._state_0)
        self.contacts = [object() for _ in range(self.total_solver_steps)]
        self.step_calls = 0
        self._released = False

    @property
    def final_state(self) -> _FakeState:
        """Return the terminal state owned by this one taped trajectory."""
        return self.states[-1]

    def step(self) -> _FakeState:
        """Advance the owned trajectory and expose tape placement."""
        self._manager.events.append("trajectory.step")
        assert self._manager.warp.tape_active
        self.step_calls += 1
        if self._manager.raise_on_trajectory_step:
            raise RuntimeError("injected trajectory-step failure")
        for state_in, state_out in zip(self.states, self.states[1:]):
            state_out.value = state_in.value + 1
        return self.final_state

    def release(self) -> None:
        """Release this trajectory's model lease after its tape is reset."""
        if self._released:
            return
        self._manager._release_differentiable_trajectory(self)
        self._released = True


class _TrajectoryNewtonManager:
    """Fake Newton manager for the manager-owned trajectory bridge contract."""

    def __init__(self, warp: _RecordingWarp, *, num_substeps: int = 1) -> None:
        self.warp = warp
        self.events = warp.events
        self._state_0 = _FakeState("live-state-0")
        self._state_1 = _FakeState("live-state-1")
        # Keep the old private path runnable so each regression fails on the
        # missing public trajectory contract rather than a fake-only error.
        self._model = _FakeModel()
        self._control = object()
        self.num_substeps = num_substeps
        self.solver_dt = 0.01
        self._dt = self.solver_dt * self.num_substeps
        self.physics_dt = self._dt
        self.trajectory_requests: list[dict[str, Any]] = []
        self.trajectories: list[_ManagerOwnedTrajectory] = []
        self.commits: list[_ManagerOwnedTrajectory] = []
        self.commit_assignment_counts: list[tuple[int, int]] = []
        self._active_trajectory: _ManagerOwnedTrajectory | None = None
        self.raise_on_trajectory_step = False

    def create_differentiable_trajectory(
        self, *, physics_steps: int, physics_dt: float
    ) -> _ManagerOwnedTrajectory:
        """Create the public trajectory before the bridge opens its tape."""
        if physics_steps < 1:
            raise ValueError("physics_steps must be positive")
        if self._active_trajectory is not None:
            raise RuntimeError(
                "A differentiable trajectory is still active; release it after "
                "backward before creating another trajectory."
            )
        self.events.append("create")
        trajectory = _ManagerOwnedTrajectory(
            self,
            physics_steps=physics_steps,
            physics_dt=physics_dt,
        )
        self.trajectories.append(trajectory)
        self._active_trajectory = trajectory
        self.trajectory_requests.append(
            {
                "physics_steps": physics_steps,
                "physics_dt": physics_dt,
                "tape_active": self.warp.tape_active,
            }
        )
        return trajectory

    def commit_differentiable_trajectory(
        self, trajectory: _ManagerOwnedTrajectory
    ) -> None:
        """Record a detached post-tape publication through the manager API."""
        assert not self.warp.tape_active
        assert trajectory in self.trajectories
        before = (len(self._state_0.assign_sources), len(self._state_1.assign_sources))
        self._state_0.assign(trajectory.final_state)
        self._state_1.assign(trajectory.final_state)
        self.commits.append(trajectory)
        self.commit_assignment_counts.append(
            (
                len(self._state_0.assign_sources) - before[0],
                len(self._state_1.assign_sources) - before[1],
            )
        )
        self.events.append("commit")

    def _release_differentiable_trajectory(
        self, trajectory: _ManagerOwnedTrajectory
    ) -> None:
        """Release the one active trajectory once tape ownership has ended."""
        assert self._active_trajectory is trajectory
        self._active_trajectory = None
        self.events.append("trajectory.release")


class _TrajectorySimulationManager:
    """Bridge-facing manager exposing public and legacy test doubles."""

    def __init__(self, warp: _RecordingWarp, *, num_substeps: int = 1) -> None:
        self.is_newton_backend = True
        self.physics = SimpleNamespace(
            newton_manager=_TrajectoryNewtonManager(warp, num_substeps=num_substeps)
        )
        self.steppers: list[_FakeStepper] = []

    def create_differentiable_stepper(self) -> _FakeStepper:
        """Keep the pre-contract bridge executable for a clean RED failure."""
        stepper = _FakeStepper(
            raise_on_step=self.physics.newton_manager.raise_on_trajectory_step
        )
        self.steppers.append(stepper)
        return stepper


class _RealBridgeManager:
    """Expose only the public Newton-trajectory surface to the bridge."""

    def __init__(self, newton_manager: Any) -> None:
        self.is_newton_backend = True
        self.physics = SimpleNamespace(newton_manager=newton_manager)

    def create_differentiable_stepper(self) -> None:
        """Fail if the bridge retains the removed SimulationManager route."""
        raise AssertionError(
            "NewtonStepFunc must use NewtonManager.create_differentiable_trajectory(), "
            "not SimulationManager.create_differentiable_stepper()."
        )


def _route_env(
    manager: Any,
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

    def _apply_dynamics_action(
        _action_wp: torch.Tensor, _control: Any, tape: Any
    ) -> None:
        del tape

    def _apply_kinematic_action(_action_wp: torch.Tensor, tape: Any) -> None:
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

    env._apply_dynamics_action_kernel = _apply_dynamics_action
    env._apply_action_kernel = _apply_kinematic_action
    env._read_outputs = _read_outputs
    return env, final_states


def _manager_owned_trajectory_sim_state(
    manager: _TrajectorySimulationManager,
    *,
    action_to_control_kernel: Any,
    step_mode: str | None = None,
    step_fn: Any | None = None,
) -> dict[str, Any]:
    """Build the narrow bridge input used by manager-owned trajectory tests."""
    nm = manager.physics.newton_manager

    def _read_outputs(final_state: _FakeState) -> dict[str, Any]:
        del final_state
        assert nm.warp.tape_active
        nm.events.append("outputs")
        return {
            "obs": torch.zeros(1, 1),
            "reward": torch.zeros(1),
            "terminated": torch.zeros(1, dtype=torch.bool),
            "truncated": torch.zeros(1, dtype=torch.bool),
            "_order": ("obs", "reward", "terminated", "truncated"),
            "_grad_track": {},
        }

    sim_state: dict[str, Any] = {
        "manager": manager,
        "substeps": _CONTROL_SUBSTEPS,
        "physics_dt": nm.physics_dt,
        "action_to_control_kernel": action_to_control_kernel,
        "kernel_args": ("kernel-argument",),
        "obs_reward_fn": _read_outputs,
    }
    if step_mode is not None:
        sim_state["step_mode"] = step_mode
    if step_fn is not None:
        sim_state["step_fn"] = step_fn
    return sim_state


def _assert_tape_reset_then_trajectory_release(events: list[str]) -> None:
    """Require one terminal tape reset followed immediately by release."""
    assert events.count("tape.reset") == 1
    assert events.count("trajectory.release") == 1
    reset_index = events.index("tape.reset")
    release_index = events.index("trajectory.release")
    assert events.index("tape.exit") < reset_index < release_index
    assert events[-2:] == ["tape.reset", "trajectory.release"]


def _assert_backward_captures_gradient_then_releases(events: list[str]) -> None:
    """Require gradient capture before terminal tape and trajectory cleanup."""
    tracked_events = {
        "tape.backward",
        "action-gradient.capture",
        "tape.reset",
        "trajectory.release",
    }
    assert [event for event in events if event in tracked_events] == [
        "tape.backward",
        "action-gradient.capture",
        "tape.reset",
        "trajectory.release",
    ]


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


def test_default_dynamics_manager_trajectory_lifecycle_is_fully_ordered(
    monkeypatch,
) -> None:
    """Allocate, tape, action, step, output, and commit stay in one order."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager

    def _apply_action(_action: torch.Tensor, *_args: Any) -> None:
        assert warp.tape_active
        nm.events.append("action")

    NewtonStepFunc.apply(
        torch.zeros(1, requires_grad=True),
        _manager_owned_trajectory_sim_state(
            manager,
            action_to_control_kernel=_apply_action,
            step_mode="dynamics",
        ),
    )

    assert nm.trajectory_requests == [
        {
            "physics_steps": _CONTROL_SUBSTEPS,
            "physics_dt": nm.physics_dt,
            "tape_active": False,
        }
    ]
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert trajectory.step_calls == 1
    assert nm.events == [
        "create",
        "tape.construct",
        "action",
        "trajectory.step",
        "outputs",
        "tape.exit",
        "commit",
    ]
    assert nm.commits == [trajectory]
    assert nm.commit_assignment_counts == [(1, 1)]
    assert [len(state.assign_sources) for state in (nm._state_0, nm._state_1)] == [
        1,
        1,
    ]


def test_default_dynamics_action_hook_receives_trajectory_local_control(
    monkeypatch,
) -> None:
    """The taped action write never targets the manager's shared control."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    received: list[tuple[tuple[Any, ...], bool]] = []

    def _apply_action(_action: torch.Tensor, *args: Any) -> None:
        received.append((args, warp.tape_active))

    NewtonStepFunc.apply(
        torch.zeros(1, requires_grad=True),
        _manager_owned_trajectory_sim_state(
            manager,
            action_to_control_kernel=_apply_action,
            step_mode="dynamics",
        ),
    )

    nm = manager.physics.newton_manager
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert received == [((trajectory.control, "kernel-argument"), True)]
    assert received[0][0][0] is not nm._control


def test_dynamics_legacy_action_type_error_is_not_retried_after_creation(
    monkeypatch,
) -> None:
    """Dynamics propagates a legacy callback error instead of falling back."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager
    legacy_calls: list[tuple[torch.Tensor, Any]] = []

    def _legacy_action(action_wp: torch.Tensor, tape: Any) -> None:
        legacy_calls.append((action_wp, tape))
        raise TypeError("original legacy action TypeError")

    sim_state = _manager_owned_trajectory_sim_state(
        manager,
        action_to_control_kernel=_legacy_action,
        step_mode="dynamics",
    )
    # With no extra kernel arguments, the legacy two-argument callback is
    # entered once with local control in its obsolete ``tape`` position.
    # Retrying after its body raises TypeError would invoke it a second time.
    sim_state["kernel_args"] = ()

    with pytest.raises(TypeError) as exc_info:
        NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    assert str(exc_info.value) == "original legacy action TypeError"
    assert len(legacy_calls) == 1
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert legacy_calls[0][1] is trajectory.control
    assert trajectory.step_calls == 0
    assert nm.commits == []
    assert nm.commit_assignment_counts == []
    assert [state.assign_sources for state in (nm._state_0, nm._state_1)] == [
        [],
        [],
    ]
    assert nm._active_trajectory is None
    assert trajectory._released
    _assert_tape_reset_then_trajectory_release(nm.events)


def test_default_dynamics_commits_manager_trajectory_once_after_tape_closes(
    monkeypatch,
) -> None:
    """A public commit is the sole detached publication of live state."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)

    NewtonStepFunc.apply(
        torch.zeros(1, requires_grad=True),
        _manager_owned_trajectory_sim_state(
            manager,
            action_to_control_kernel=lambda _action, *_args: None,
            step_mode="dynamics",
        ),
    )

    nm = manager.physics.newton_manager
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert nm.commits == [trajectory]
    assert nm.events[-1] == "commit"
    assert nm.commit_assignment_counts == [(1, 1)]
    assert [state.assign_sources for state in (nm._state_0, nm._state_1)] == [
        [trajectory.final_state],
        [trajectory.final_state],
    ]


@pytest.mark.parametrize("failure_site", ("action", "trajectory_step"))
def test_failed_manager_trajectory_forward_resets_and_releases_without_commit(
    monkeypatch, failure_site: str
) -> None:
    """A failed taped forward releases its manager lease without publishing it."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager

    if failure_site == "action":

        def _apply_action(_action: torch.Tensor, *_args: Any) -> None:
            assert warp.tape_active
            nm.events.append("action.error")
            raise RuntimeError("injected action failure")

        error_match = "injected action failure"
    else:
        nm.raise_on_trajectory_step = True

        def _apply_action(_action: torch.Tensor, *_args: Any) -> None:
            assert warp.tape_active
            nm.events.append("action")

        error_match = "injected trajectory-step failure"

    with pytest.raises(RuntimeError, match=error_match):
        NewtonStepFunc.apply(
            torch.zeros(1, requires_grad=True),
            _manager_owned_trajectory_sim_state(
                manager,
                action_to_control_kernel=_apply_action,
                step_mode="dynamics",
            ),
        )

    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert nm.commits == []
    assert nm.commit_assignment_counts == []
    assert [state.assign_sources for state in (nm._state_0, nm._state_1)] == [
        [],
        [],
    ]
    assert nm._active_trajectory is None
    assert trajectory._released
    _assert_tape_reset_then_trajectory_release(nm.events)


def test_backward_resets_tape_then_releases_manager_trajectory(monkeypatch) -> None:
    """A grad-tracked trajectory resets its tape before releasing after backward."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager
    action = torch.zeros(1, requires_grad=True)

    outputs = NewtonStepFunc.apply(
        action,
        _manager_owned_trajectory_sim_state(
            manager,
            action_to_control_kernel=lambda _action, *_args: None,
            step_mode="dynamics",
        ),
    )
    outputs[0].sum().backward()

    assert action.grad is not None
    _assert_backward_captures_gradient_then_releases(nm.events)
    _assert_tape_reset_then_trajectory_release(nm.events)
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert nm.commits == [trajectory]
    assert nm._active_trajectory is None
    assert trajectory._released


def test_backward_exception_resets_tape_then_releases_manager_trajectory(
    monkeypatch,
) -> None:
    """A tape-backward failure cannot leave a manager trajectory leased."""
    warp = _RecordingWarp()
    warp.raise_on_tape_backward = True
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager
    action = torch.zeros(1, requires_grad=True)

    outputs = NewtonStepFunc.apply(
        action,
        _manager_owned_trajectory_sim_state(
            manager,
            action_to_control_kernel=lambda _action, *_args: None,
            step_mode="dynamics",
        ),
    )

    with pytest.raises(RuntimeError, match="injected tape backward failure"):
        outputs[0].sum().backward()

    assert nm.events.count("tape.backward") == 1
    assert "action-gradient.capture" not in nm.events
    _assert_tape_reset_then_trajectory_release(nm.events)
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert nm.commits == [trajectory]
    assert nm._active_trajectory is None
    assert trajectory._released


def test_obs_reward_failure_releases_manager_trajectory_before_fresh_forward(
    monkeypatch,
) -> None:
    """An output-read error rolls back its lease so the next trajectory starts."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager
    failing_state = _manager_owned_trajectory_sim_state(
        manager,
        action_to_control_kernel=lambda _action, *_args: None,
        step_mode="dynamics",
    )

    def _raise_from_outputs(_final_state: _FakeState) -> dict[str, Any]:
        assert warp.tape_active
        nm.events.append("outputs.error")
        raise RuntimeError("injected output-read failure")

    failing_state["obs_reward_fn"] = _raise_from_outputs
    with pytest.raises(RuntimeError, match="injected output-read failure"):
        NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), failing_state)

    failure_events = list(nm.events)
    assert "trajectory.step" in failure_events
    assert "outputs.error" in failure_events
    assert failure_events.index("trajectory.step") < failure_events.index(
        "outputs.error"
    )
    _assert_tape_reset_then_trajectory_release(failure_events)
    assert len(nm.trajectories) == 1
    failed_trajectory = nm.trajectories[0]
    assert failed_trajectory.step_calls == 1
    assert nm.commits == []
    assert nm.commit_assignment_counts == []
    assert [state.assign_sources for state in (nm._state_0, nm._state_1)] == [
        [],
        [],
    ]
    assert nm._active_trajectory is None
    assert failed_trajectory._released

    with torch.no_grad():
        outputs = NewtonStepFunc.apply(
            torch.zeros(1, requires_grad=True),
            _manager_owned_trajectory_sim_state(
                manager,
                action_to_control_kernel=lambda _action, *_args: None,
                step_mode="dynamics",
            ),
        )

    assert len(outputs) == 4
    assert len(nm.trajectories) == 2
    fresh_trajectory = nm.trajectories[1]
    assert fresh_trajectory is not failed_trajectory
    assert nm.commits == [fresh_trajectory]
    assert nm._active_trajectory is None
    assert fresh_trajectory._released


def test_no_grad_forward_resets_tape_then_releases_manager_trajectory(
    monkeypatch,
) -> None:
    """A non-grad forward cannot retain a trajectory lease for backward."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager

    with torch.no_grad():
        outputs = NewtonStepFunc.apply(
            torch.zeros(1, requires_grad=True),
            _manager_owned_trajectory_sim_state(
                manager,
                action_to_control_kernel=lambda _action, *_args: None,
                step_mode="dynamics",
            ),
        )

    assert len(outputs) == 4
    assert not outputs[0].requires_grad
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert nm.commits == [trajectory]
    assert nm._active_trajectory is None
    assert trajectory._released
    assert "tape.backward" not in nm.events
    _assert_tape_reset_then_trajectory_release(nm.events)


def test_legacy_dynamics_step_fn_is_rejected_before_opening_tape(monkeypatch) -> None:
    """An untrusted callback cannot silently bypass default solver dynamics."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    legacy_calls: list[None] = []

    def _legacy_step() -> _FakeState:
        legacy_calls.append(None)
        return _FakeState("legacy-dynamics-final")

    with pytest.raises(ValueError, match=r"step_fn.*kinematics"):
        NewtonStepFunc.apply(
            torch.zeros(1, requires_grad=True),
            _manager_owned_trajectory_sim_state(
                manager,
                action_to_control_kernel=lambda _action, *_args: None,
                step_mode="dynamics",
                step_fn=_legacy_step,
            ),
        )

    assert legacy_calls == []
    assert warp.events == []


def test_missing_step_mode_with_step_fn_is_rejected_before_opening_tape(
    monkeypatch,
) -> None:
    """Historical implicit-FK dictionaries cannot bypass solver dynamics."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)

    with pytest.raises(ValueError, match=r"step_mode.*kinematics"):
        NewtonStepFunc.apply(
            torch.zeros(1, requires_grad=True),
            _manager_owned_trajectory_sim_state(
                manager,
                action_to_control_kernel=lambda _action, *_args: None,
                step_fn=lambda: _FakeState("implicit-legacy-final"),
            ),
        )

    assert warp.events == []


def test_bridge_rejects_invalid_step_mode_before_opening_tape(monkeypatch) -> None:
    """Direct bridge callers cannot open a tape for an unsupported mode."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)

    with pytest.raises(ValueError, match=r"step_mode.*dynamics.*kinematics"):
        NewtonStepFunc.apply(
            torch.zeros(1, requires_grad=True),
            _manager_owned_trajectory_sim_state(
                manager,
                action_to_control_kernel=lambda _action, *_args: None,
                step_mode="unsupported",
            ),
        )

    assert warp.events == []
    assert manager.physics.newton_manager.trajectory_requests == []
    assert manager.steppers == []


def test_explicit_kinematics_step_fn_remains_a_supported_bridge_route(
    monkeypatch,
) -> None:
    """The deliberate kinematics escape hatch does not request a trajectory."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    kinematic_calls: list[None] = []
    final_state = _FakeState("kinematic-final")

    def _kinematic_step() -> _FakeState:
        kinematic_calls.append(None)
        return final_state

    outputs = NewtonStepFunc.apply(
        torch.zeros(1, requires_grad=True),
        _manager_owned_trajectory_sim_state(
            manager,
            action_to_control_kernel=lambda _action, *_args: None,
            step_mode="kinematics",
            step_fn=_kinematic_step,
        ),
    )

    assert len(outputs) == 4
    assert kinematic_calls == [None]
    assert manager.physics.newton_manager.trajectory_requests == []


def test_environment_sim_state_marks_default_and_explicit_kinematics_routes() -> None:
    """The bridge can distinguish an explicit FK request from legacy bypasses."""
    dynamics_env, _ = _route_env(SimpleNamespace())
    kinematics_env, _ = _route_env(SimpleNamespace(), mode="kinematics")
    kinematics_env._make_kinematic_step_fn = lambda: (lambda: _FakeState("fk"))

    dynamics_state = dynamics_env._build_sim_state_dict(torch.zeros(1))
    kinematics_state = kinematics_env._build_sim_state_dict(torch.zeros(1))

    assert dynamics_state["step_mode"] == "dynamics"
    assert kinematics_state["step_mode"] == "kinematics"


def test_environment_dynamics_hook_receives_local_control_with_migration_api() -> None:
    """The default environment wrapper calls only the v1 dynamics hook."""
    env, _ = _route_env(SimpleNamespace())
    dynamics_calls: list[tuple[object, object, object]] = []
    legacy_calls: list[tuple[object, object]] = []
    action = object()
    control = object()

    def _dynamics_action(
        action_wp: object, trajectory_control: object, tape: object
    ) -> None:
        dynamics_calls.append((action_wp, trajectory_control, tape))

    def _legacy_action(action_wp: object, tape: object) -> None:
        legacy_calls.append((action_wp, tape))

    env._apply_dynamics_action_kernel = _dynamics_action
    env._apply_action_kernel = _legacy_action
    sim_state = env._build_sim_state_dict(torch.zeros(1))
    sim_state["action_to_control_kernel"](action, control, "kernel-argument")

    assert dynamics_calls == [(action, control, None)]
    assert legacy_calls == []


def test_environment_dynamics_hook_observes_only_its_active_tape(
    monkeypatch,
) -> None:
    """The bridge binds the tape through a per-step wrapper closure."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    env, _ = _route_env(manager)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    observed_tapes: list[object | None] = []

    def _dynamics_action(
        _action_wp: object,
        _trajectory_control: object,
        tape: object | None,
    ) -> None:
        observed_tapes.append(tape)

    env._apply_dynamics_action_kernel = _dynamics_action
    sim_state = env._build_sim_state_dict(torch.zeros(1))

    with torch.no_grad():
        NewtonStepFunc.apply(torch.zeros(1), sim_state)

    assert len(observed_tapes) == 1
    assert isinstance(observed_tapes[0], _RecordingTape)

    sim_state["action_to_control_kernel"](object(), object())
    assert observed_tapes[-1] is None


def test_environment_rejects_legacy_dynamics_action_hook_with_migration_error(
    monkeypatch,
) -> None:
    """Default dynamics cannot silently keep the pre-local-control hook."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    env = object.__new__(DifferentiableEmbodiedEnv)
    env.sim = manager
    env.cfg = SimpleNamespace(sim_steps_per_control=_CONTROL_SUBSTEPS)
    env._apply_dynamics_action_kernel = None
    env._apply_action_kernel = lambda _action, tape: None
    env._read_outputs = lambda _state: {
        "obs": torch.zeros(1, 1),
        "reward": torch.zeros(1),
        "terminated": torch.zeros(1, dtype=torch.bool),
        "truncated": torch.zeros(1, dtype=torch.bool),
        "_order": ("obs", "reward", "terminated", "truncated"),
        "_grad_track": {},
    }

    with pytest.raises(
        NotImplementedError, match=r"legacy.*_apply_dynamics_action_kernel"
    ):
        sim_state = env._build_sim_state_dict(torch.zeros(1, requires_grad=True))
        NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    assert warp.events == []


def test_environment_kinematics_hook_keeps_its_strict_legacy_signature(
    monkeypatch,
) -> None:
    """FK-only bridge execution receives action and tape, never local control."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    env, _ = _route_env(manager, mode="kinematics")
    monkeypatch.setattr(diff_bridge, "wp", warp)
    calls: list[tuple[torch.Tensor, object]] = []
    final_state = _FakeState("kinematic-final")

    def _kinematic_action(action_wp: torch.Tensor, tape: object) -> None:
        calls.append((action_wp, tape))

    env._apply_action_kernel = _kinematic_action
    env._make_kinematic_step_fn = lambda: (lambda: final_state)
    action = torch.zeros(1, requires_grad=True)
    sim_state = env._build_sim_state_dict(action)
    outputs = NewtonStepFunc.apply(action, sim_state)

    assert len(outputs) == 4
    assert len(calls) == 1
    assert torch.equal(calls[0][0], action)
    assert isinstance(calls[0][1], _RecordingTape)
    assert manager.physics.newton_manager.trajectory_requests == []


def test_grad_terminal_step_defers_reset_until_after_backward(monkeypatch) -> None:
    """A terminal grad step must return before touching fenced live state."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    env, _ = _route_env(manager)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager
    reset_calls: list[torch.Tensor] = []

    def _terminal_outputs(_final_state: object) -> dict[str, Any]:
        return {
            "obs": torch.full((1, 1), 7.0),
            "reward": torch.full((1,), 3.0),
            "terminated": torch.ones(1, dtype=torch.bool),
            "truncated": torch.zeros(1, dtype=torch.bool),
            "_order": ("obs", "reward", "terminated", "truncated"),
            "_grad_track": {},
        }

    def _reset(*, options: dict[str, Any]):
        if nm._active_trajectory is not None:
            raise RuntimeError("reset crossed an active Newton trajectory fence")
        reset_ids = torch.as_tensor(options["reset_ids"]).clone()
        reset_calls.append(reset_ids)
        return torch.full((1, 1), -1.0), {}

    env._read_outputs = _terminal_outputs
    env.reset = _reset
    action = torch.zeros(1, requires_grad=True)

    obs, reward, terminated, truncated, info = env.step(action)

    assert torch.equal(obs.detach(), torch.full((1, 1), 7.0))
    assert terminated.tolist() == [True]
    assert truncated.tolist() == [False]
    assert reset_calls == []
    assert info["requires_reset_after_backward"] is True
    assert torch.equal(info["deferred_reset_ids"], torch.tensor([0]))
    assert nm._active_trajectory is not None

    reward.sum().backward()

    assert action.grad is not None
    assert nm._active_trajectory is None
    env.reset(options={"reset_ids": info["deferred_reset_ids"]})
    assert len(reset_calls) == 1
    assert torch.equal(reset_calls[0], torch.tensor([0]))


def test_no_grad_terminal_step_keeps_synchronous_auto_reset(monkeypatch) -> None:
    """A terminal no-grad step may reset after its tape is released."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    env, _ = _route_env(manager)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager
    reset_calls: list[torch.Tensor] = []

    env._read_outputs = lambda _state: {
        "obs": torch.full((1, 1), 7.0),
        "reward": torch.full((1,), 3.0),
        "terminated": torch.ones(1, dtype=torch.bool),
        "truncated": torch.zeros(1, dtype=torch.bool),
        "_order": ("obs", "reward", "terminated", "truncated"),
        "_grad_track": {},
    }

    def _reset(*, options: dict[str, Any]):
        assert nm._active_trajectory is None
        reset_ids = torch.as_tensor(options["reset_ids"]).clone()
        reset_calls.append(reset_ids)
        return torch.full((1, 1), -1.0), {}

    env.reset = _reset
    with torch.no_grad():
        obs, reward, terminated, truncated, info = env.step(
            torch.zeros(1, requires_grad=True)
        )

    assert torch.equal(obs, torch.full((1, 1), -1.0))
    assert not reward.requires_grad
    assert terminated.tolist() == [True]
    assert truncated.tolist() == [False]
    assert len(reset_calls) == 1
    assert torch.equal(reset_calls[0], torch.tensor([0]))
    assert "deferred_reset_ids" not in info
    assert "requires_reset_after_backward" not in info


def test_default_dynamics_route_uses_manager_trajectory_without_bypass(monkeypatch):
    """Default state construction delegates one control step to Newton."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    env, final_states = _route_env(manager)
    monkeypatch.setattr(diff_bridge, "wp", warp)

    sim_state = env._build_sim_state_dict(torch.zeros(1, requires_grad=True))
    outputs = NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    nm = manager.physics.newton_manager
    assert sim_state["step_mode"] == "dynamics"
    assert "step_fn" not in sim_state
    assert len(outputs) == 4
    assert len(nm.trajectories) == 1
    assert nm.trajectories[0].total_solver_steps == _CONTROL_SUBSTEPS
    assert final_states[0].value == _CONTROL_SUBSTEPS


def test_dynamics_bridge_keeps_an_odd_continuous_horizon_in_one_trajectory(
    monkeypatch,
):
    """A continuous odd horizon is one lease-owning manager trajectory."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    env, final_states = _route_env(manager, control_substeps=5)
    monkeypatch.setattr(diff_bridge, "wp", warp)

    sim_state = env._build_sim_state_dict(torch.zeros(1, requires_grad=True))
    NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    nm = manager.physics.newton_manager
    assert [state.value for state in final_states] == [5]
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert trajectory.total_solver_steps == 5
    assert nm._state_0.value == 5
    assert nm._state_1.value == 5
    assert nm._state_0.assign_sources == [trajectory.final_state]
    assert nm._state_1.assign_sources == [trajectory.final_state]
    assert len({id(state) for state in trajectory.states}) == 6
    assert all(state not in {nm._state_0, nm._state_1} for state in trajectory.states)
    assert len({id(contact) for contact in trajectory.contacts}) == 5


def test_dynamics_bridge_rejects_a_second_outstanding_manager_trajectory(
    monkeypatch,
) -> None:
    """A second grad forward requires release of the first trajectory lease."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    sim_state = _manager_owned_trajectory_sim_state(
        manager,
        action_to_control_kernel=lambda _action, *_args: None,
        step_mode="dynamics",
    )

    NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    with pytest.raises(RuntimeError, match=r"trajectory.*active.*release"):
        NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)


def test_dynamics_bridge_multiplies_control_and_newton_substeps(monkeypatch):
    """One control step preserves EmbodiChain and Newton time semantics."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp, num_substeps=3)
    env, _ = _route_env(manager, control_substeps=2)
    monkeypatch.setattr(diff_bridge, "wp", warp)

    sim_state = env._build_sim_state_dict(torch.zeros(1, requires_grad=True))
    NewtonStepFunc.apply(torch.zeros(1, requires_grad=True), sim_state)

    nm = manager.physics.newton_manager
    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert trajectory.physics_steps == 2
    assert trajectory.physics_dt == nm.physics_dt
    assert trajectory.total_solver_steps == 6


def test_differentiable_step_uses_manager_owned_trajectory_and_local_control(
    monkeypatch,
):
    """The low-level helper also delegates state publication to Newton."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    received: list[tuple[Any, ...]] = []

    result = differentiable_step(
        manager,
        apply_control_fn=lambda *args: received.append(args),
        substeps=_CONTROL_SUBSTEPS,
    )
    nm = manager.physics.newton_manager

    assert len(nm.trajectories) == 1
    trajectory = nm.trajectories[0]
    assert any(trajectory.control in args for args in received)
    assert result["trajectory"] is trajectory
    assert nm.commits == [trajectory]
    assert nm.commit_assignment_counts == [(1, 1)]


def test_differentiable_step_rejects_substeps_not_divisible_by_newton_substeps(
    monkeypatch,
) -> None:
    """A low-level solver horizon must map to whole Newton physics steps."""
    warp = _RecordingWarp()
    manager = _TrajectorySimulationManager(warp, num_substeps=2)
    monkeypatch.setattr(diff_bridge, "wp", warp)
    nm = manager.physics.newton_manager
    control_calls: list[tuple[Any, ...]] = []

    with pytest.raises(ValueError, match=r"substeps.*divisible.*num_substeps"):
        differentiable_step(
            manager,
            apply_control_fn=lambda *args: control_calls.append(args),
            substeps=3,
        )

    assert control_calls == []
    assert nm.trajectory_requests == []
    assert manager.steppers == []
    assert warp.events == []


@pytest.mark.parametrize("substeps", (0, -1))
def test_differentiable_step_rejects_nonpositive_substeps(substeps: int) -> None:
    """The public helper rejects an invalid empty solver trajectory."""
    manager = _TrajectorySimulationManager(_RecordingWarp())

    with pytest.raises(ValueError, match=r"positive"):
        differentiable_step(
            manager,
            apply_control_fn=lambda *_args: None,
            substeps=substeps,
        )


def test_cpu_newton_manager_trajectory_retains_local_control_gradient_and_fd(tmp_path):
    """The real bridge keeps a local control trajectory across two steps."""
    newton = pytest.importorskip("newton")
    pytest.importorskip("dexsim.engine.newton_physics")
    from dexsim.engine.newton_physics import (
        NewtonCfg,
        NewtonCollisionPipelineCfg,
        NewtonManager,
        SemiImplicitSolverCfg,
    )

    assert hasattr(
        NewtonManager, "create_differentiable_trajectory"
    ), "NewtonManager must publish create_differentiable_trajectory() first."

    previous_kernel_cache_dir = wp.config.kernel_cache_dir
    previous_verify_access = wp.config.verify_autograd_array_access
    nm = None
    wp.config.kernel_cache_dir = str(tmp_path / "warp_cache")
    wp.config.verify_autograd_array_access = True
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
            label="embodichain_manager_trajectory_gradient_ball",
        )
        nm._builder.add_shape_sphere(body=body_id, radius=0.1, cfg=shape_cfg)
        nm._builder.add_ground_plane(cfg=shape_cfg)
        nm.start_simulation()
        assert nm._model.joint_count == 1

        manager = _RealBridgeManager(nm)
        initial_state = nm._model.state()
        initial_state.assign(nm._state_0)
        target = wp.vec3(0.5, 0.0, 0.5)

        def _restore_initial_state() -> None:
            nm._state_0.assign(initial_state)
            nm._state_1.assign(initial_state)

        def _run(
            action_value: float, *, requires_grad: bool
        ) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
            loss_wp = wp.zeros(
                1,
                dtype=wp.float32,
                device=nm._state_0.body_q.device,
                requires_grad=True,
            )
            local_controls: list[Any] = []

            def _apply_control(action_wp: Any, *args: Any) -> None:
                assert len(args) == 1, "Bridge must pass exactly one local control."
                control = args[0]
                assert control.joint_f is not None
                local_controls.append(control)
                wp.launch(
                    _write_bridge_joint_force_kernel,
                    dim=1,
                    inputs=[action_wp, control.joint_f],
                    device=control.joint_f.device,
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
                "step_mode": "dynamics",
                "substeps": 2,
                "physics_dt": cfg.dt,
                "action_to_control_kernel": _apply_control,
                "kernel_args": (),
                "obs_reward_fn": _read_reward,
            }
            return NewtonStepFunc.apply(action, sim_state)[0], action, local_controls

        reward, action, local_controls = _run(1.0, requires_grad=True)
        reward.backward()

        assert len(local_controls) == 1
        assert action.grad is not None
        analytic_gradient = float(action.grad[0])
        assert np.isfinite(analytic_gradient)
        assert not np.isclose(analytic_gradient, 0.0)
        first_final_state = nm._state_0.body_q.numpy().copy()
        assert np.allclose(
            nm._state_0.body_q.numpy(), nm._state_1.body_q.numpy(), atol=1.0e-6
        )

        continuation_reward, continuation_action, continuation_controls = _run(
            1.0, requires_grad=True
        )
        continuation_reward.backward()
        assert len(continuation_controls) == 1
        assert continuation_action.grad is not None
        assert np.isfinite(continuation_action.grad).all()
        assert not np.allclose(first_final_state, nm._state_0.body_q.numpy())

        def _reward_value(action_value: float) -> float:
            _restore_initial_state()
            value, _action, controls = _run(action_value, requires_grad=False)
            assert len(controls) == 1
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
        wp.config.verify_autograd_array_access = previous_verify_access
        wp.config.kernel_cache_dir = previous_kernel_cache_dir


def test_dynamics_environment_does_not_expose_generic_step_helper():
    """Only the low-level bridge may accept an arbitrary dynamics callback."""
    assert "_make_step_fn" not in DifferentiableEmbodiedEnv.__dict__


def test_kinematics_route_uses_only_named_kinematic_hook():
    """FK stepping is selected only through the explicit kinematics mode."""
    manager = SimpleNamespace()
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

    assert sim_state["step_mode"] == "kinematics"
    assert sim_state["step_fn"]() is expected_state
    assert kinematic_calls == [None]


def test_kinematics_route_requires_named_hook():
    """Kinematics mode rejects environments that do not define its hook."""
    manager = SimpleNamespace()
    env, _ = _route_env(manager, mode="kinematics")

    with pytest.raises(
        NotImplementedError, match=r"kinematics.*_make_kinematic_step_fn"
    ):
        env._build_sim_state_dict(torch.zeros(1))


def test_invalid_differentiable_step_mode_raises_clear_error():
    """Unsupported stepping modes fail before creating a bridge callback."""
    manager = SimpleNamespace()
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


def test_franka_kinematics_build_snapshots_live_primal_before_bridge(
    monkeypatch,
) -> None:
    """Franka must detach taped FK inputs before the parent opens a tape."""
    from embodichain.lab.gym.envs.tasks.special import franka_reach_apg

    env = object.__new__(franka_reach_apg.FrankaReachApgEnv)
    live_joint_q = object()
    snapshot_joint_q = object()
    fresh_fk_state = object()
    events: list[str] = []
    env.sim = SimpleNamespace(
        physics=SimpleNamespace(
            newton_manager=SimpleNamespace(
                _state_0=SimpleNamespace(joint_q=live_joint_q),
                _model=SimpleNamespace(
                    state=lambda: (events.append("state"), fresh_fk_state)[1]
                ),
            )
        )
    )

    def _clone(array: object) -> object:
        assert array is live_joint_q
        events.append("clone")
        return snapshot_joint_q

    def _parent_build(_self: object, _action: torch.Tensor) -> dict[str, Any]:
        events.append("parent")
        assert env._current_joint_q_snapshot is snapshot_joint_q
        assert env._fk_state is fresh_fk_state
        return {"prepared": True}

    monkeypatch.setattr(franka_reach_apg.wp, "clone", _clone)
    monkeypatch.setattr(
        DifferentiableEmbodiedEnv,
        "_build_sim_state_dict",
        _parent_build,
    )

    result = env._build_sim_state_dict(torch.zeros(1, 7))

    assert result == {"prepared": True}
    assert events == ["clone", "state", "parent"]


def test_franka_action_kernel_reads_snapshot_instead_of_live_state(monkeypatch) -> None:
    """The recorded action kernel must not capture mutable manager state."""
    from embodichain.lab.gym.envs.tasks.special import franka_reach_apg

    env = object.__new__(franka_reach_apg.FrankaReachApgEnv)
    live_joint_q = object()
    snapshot_joint_q = object()
    target_joint_q = object()
    action_wp = object()
    launch_inputs: list[object] = []
    env.sim = SimpleNamespace(
        num_envs=1,
        physics=SimpleNamespace(
            newton_manager=SimpleNamespace(
                _state_0=SimpleNamespace(joint_q=live_joint_q)
            )
        ),
    )
    env._current_joint_q_snapshot = snapshot_joint_q
    env._n_joints_per_env = 9
    env._wp_device = "cpu"
    env._limit_lo_wp = object()
    env._limit_hi_wp = object()
    env._action_scale = 0.2

    monkeypatch.setattr(
        franka_reach_apg.wp,
        "zeros",
        lambda *_args, **_kwargs: target_joint_q,
    )

    def _launch(*_args: Any, inputs: list[object], **_kwargs: Any) -> None:
        launch_inputs.extend(inputs)

    monkeypatch.setattr(franka_reach_apg.wp, "launch", _launch)

    env._apply_action_kernel(action_wp, tape=object())

    assert launch_inputs[0] is action_wp
    assert launch_inputs[1] is snapshot_joint_q
    assert launch_inputs[1] is not live_joint_q
    assert launch_inputs[2] is target_joint_q


def test_franka_snapshot_keeps_gradient_after_live_state_mutation_and_matches_fd(
    monkeypatch,
    tmp_path,
) -> None:
    """Detached FK input survives live writes before backward under strict mode."""
    from embodichain.lab.gym.envs.tasks.special import franka_reach_apg

    env = object.__new__(franka_reach_apg.FrankaReachApgEnv)
    device = "cpu"
    live_joint_q = wp.zeros(7, dtype=wp.float32, device=device)
    env.sim = SimpleNamespace(
        num_envs=1,
        physics=SimpleNamespace(
            newton_manager=SimpleNamespace(
                _state_0=SimpleNamespace(joint_q=live_joint_q),
                _model=SimpleNamespace(state=lambda: object()),
            )
        ),
    )
    env._wp_device = device
    env._n_joints_per_env = 7
    env._limit_lo_wp = wp.array(
        np.full(7, -10.0, dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )
    env._limit_hi_wp = wp.array(
        np.full(7, 10.0, dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )
    env._action_scale = 0.2
    monkeypatch.setattr(
        DifferentiableEmbodiedEnv,
        "_build_sim_state_dict",
        lambda _self, _action: {},
    )
    env._build_sim_state_dict(torch.zeros(1, 7))

    previous_verify_access = wp.config.verify_autograd_array_access
    previous_kernel_cache_dir = wp.config.kernel_cache_dir
    wp.config.verify_autograd_array_access = True
    wp.config.kernel_cache_dir = str(tmp_path / "warp_cache")
    tape = wp.Tape()
    try:
        action_wp = wp.array(
            np.zeros(7, dtype=np.float32),
            dtype=wp.float32,
            device=device,
            requires_grad=True,
        )
        with tape:
            env._apply_action_kernel(action_wp, tape=tape)
        analytic_output = env._new_joint_q

        wp.copy(
            live_joint_q,
            wp.array(
                np.full(7, 5.0, dtype=np.float32),
                dtype=wp.float32,
                device=device,
            ),
        )
        tape.backward(grads={analytic_output: wp.ones_like(analytic_output)})
        analytic_gradient = action_wp.grad.numpy().copy()

        assert np.isfinite(analytic_gradient).all()
        assert np.all(np.abs(analytic_gradient) > 0.0)

        def _loss(action_value: float) -> float:
            values = np.zeros(7, dtype=np.float32)
            values[0] = action_value
            finite_difference_action = wp.array(
                values,
                dtype=wp.float32,
                device=device,
            )
            env._apply_action_kernel(finite_difference_action, tape=object())
            return float(env._new_joint_q.numpy().sum())

        epsilon = 1.0e-3
        finite_difference_gradient = (_loss(epsilon) - _loss(-epsilon)) / (
            2.0 * epsilon
        )
        assert np.isclose(
            analytic_gradient[0],
            finite_difference_gradient,
            rtol=1.0e-4,
            atol=1.0e-5,
        )
    finally:
        tape.reset()
        wp.config.verify_autograd_array_access = previous_verify_access
        wp.config.kernel_cache_dir = previous_kernel_cache_dir


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
