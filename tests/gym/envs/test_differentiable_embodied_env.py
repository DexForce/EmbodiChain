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
"""Tests for the Newton-only kinematic :class:`DifferentiableEnv`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import warp as wp

import embodichain.lab.gym.envs.differentiable_env as differentiable_env_module
from embodichain.lab.gym.envs.differentiable_env import DifferentiableEnv
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.sim.diff import NewtonStepFunc, tape_context
from embodichain.lab.sim.diff.runtime import NewtonDifferentiableRuntime
from embodichain.lab.sim.sim_manager import SimulationManagerCfg


@wp.kernel
def _scale_action_kernel(
    action: wp.array(dtype=wp.float32),
    state: wp.array(dtype=wp.float32),
) -> None:
    """Map one action to a task-owned kinematic state."""
    state[0] = 2.0 * action[0]


@wp.kernel
def _square_reward_kernel(
    state: wp.array(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
) -> None:
    """Compute a differentiable scalar reward from kinematic state."""
    reward[0] = state[0] * state[0]


def _bridge_state(*, is_newton_backend: bool = True) -> dict[str, Any]:
    """Build a one-dimensional kinematics bridge input on CPU."""
    state_wp = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=True)
    reward_wp = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=True)
    manager = SimpleNamespace(is_newton_backend=is_newton_backend)

    def _apply_action(action_wp: Any, tape: Any) -> None:
        del tape
        wp.launch(
            _scale_action_kernel,
            dim=1,
            inputs=[action_wp, state_wp],
            device="cpu",
        )

    def _read_outputs(final_state: Any) -> dict[str, Any]:
        assert final_state is state_wp
        wp.launch(
            _square_reward_kernel,
            dim=1,
            inputs=[state_wp, reward_wp],
            device="cpu",
        )
        return {
            "obs": wp.to_torch(state_wp),
            "reward": wp.to_torch(reward_wp),
            "terminated": torch.zeros(1, dtype=torch.bool),
            "truncated": torch.zeros(1, dtype=torch.bool),
            "_order": ("obs", "reward", "terminated", "truncated"),
            "_grad_track": {
                "obs": None,
                "reward": reward_wp,
                "terminated": None,
                "truncated": None,
            },
        }

    return {
        "manager": manager,
        "action_kernel": _apply_action,
        "kernel_args": (),
        "step_fn": lambda: state_wp,
        "obs_reward_fn": _read_outputs,
        "last_info": {},
    }


def _bare_env() -> DifferentiableEnv:
    """Build an uninitialized environment with only its hook dependencies."""
    env = object.__new__(DifferentiableEnv)
    env.sim = SimpleNamespace(is_newton_backend=True)
    env._apply_action_kernel = lambda _action, tape: None
    env._make_kinematic_step_fn = lambda: (lambda: object())
    env._read_outputs = lambda _state: {
        "obs": torch.zeros(1, 1),
        "reward": torch.zeros(1),
        "terminated": torch.zeros(1, dtype=torch.bool),
        "truncated": torch.zeros(1, dtype=torch.bool),
        "_order": ("obs", "reward", "terminated", "truncated"),
        "_grad_track": {},
    }
    return env


def _diff_env_cfg(
    requires_grad: bool = True,
    backend: str = "newton",
) -> EmbodiedEnvCfg:
    """Build the minimum config required for constructor validation."""
    physics_cfg = (
        NewtonPhysicsCfg(
            requires_grad=requires_grad,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        )
        if backend == "newton"
        else DefaultPhysicsCfg()
    )
    return EmbodiedEnvCfg(
        sim_cfg=SimulationManagerCfg(
            physics_cfg=physics_cfg,
            num_envs=2,
            headless=True,
        )
    )


def test_public_class_is_renamed_without_legacy_alias() -> None:
    """The public module exports only the concise environment name."""
    assert differentiable_env_module.__all__ == ["DifferentiableEnv"]
    assert not hasattr(differentiable_env_module, "DifferentiableEmbodiedEnv")


def test_environment_builds_only_a_kinematic_bridge_state() -> None:
    """The base environment exposes no solver mode, substeps, or control hook."""
    env = _bare_env()
    expected_state = object()
    env._make_kinematic_step_fn = lambda: (lambda: expected_state)

    sim_state = env._build_sim_state_dict(torch.zeros(1))

    assert sim_state["step_fn"]() is expected_state
    assert "step_mode" not in sim_state
    assert "substeps" not in sim_state
    assert "action_to_control_kernel" not in sim_state
    assert "_apply_dynamics_action_kernel" not in DifferentiableEnv.__dict__
    assert "differentiable_step_mode" not in DifferentiableEnv.__dict__


def test_environment_action_hook_receives_only_action_and_tape() -> None:
    """The bridge adapter never supplies a Newton control buffer."""
    env = _bare_env()
    action_wp = object()
    tape = object()
    calls: list[tuple[object, object]] = []

    def _apply_action(action: object, tape: object) -> None:
        calls.append((action, tape))

    env._apply_action_kernel = _apply_action

    sim_state = env._build_sim_state_dict(torch.zeros(1))
    sim_state["action_kernel"](action_wp, tape)

    assert calls == [(action_wp, tape)]


def test_kinematic_bridge_propagates_reward_gradient_to_action() -> None:
    """Warp reverse mode is bridged back to the original torch action."""
    action = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)

    _, reward, _, _ = NewtonStepFunc.apply(action, _bridge_state())
    reward.sum().backward()

    assert action.grad is not None
    assert torch.allclose(action.grad, torch.tensor([4.0]))


def test_kinematic_bridge_no_grad_call_releases_tape_synchronously() -> None:
    """Inference output does not retain a custom autograd node."""
    action = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)

    with torch.no_grad():
        _, reward, _, _ = NewtonStepFunc.apply(action, _bridge_state())

    assert not reward.requires_grad


def test_kinematic_bridge_rejects_default_backend_before_opening_tape() -> None:
    """Direct bridge callers receive the same Newton-only contract."""
    action = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)

    with pytest.raises(RuntimeError, match="Newton backend"):
        NewtonStepFunc.apply(
            action,
            _bridge_state(is_newton_backend=False),
        )


def test_kinematic_bridge_requires_a_named_step_callback() -> None:
    """An arbitrary dynamics fallback cannot replace the kinematics hook."""
    sim_state = _bridge_state()
    sim_state["step_fn"] = None

    with pytest.raises(TypeError, match="callable step_fn"):
        NewtonStepFunc.apply(torch.zeros(1), sim_state)


def test_tape_context_rejects_default_backend() -> None:
    """Expert tape composition remains Newton-only."""
    manager = SimpleNamespace(is_newton_backend=False)

    with pytest.raises(RuntimeError, match="Newton backend"):
        with tape_context(manager):
            pass


def test_kinematic_runtime_does_not_validate_or_expose_a_solver() -> None:
    """FK access needs a grad model and live state, not dynamics resources."""
    model = object()
    current_state = object()
    backend = SimpleNamespace(
        model=model,
        cfg=SimpleNamespace(
            requires_grad=True,
            solver_cfg=SimpleNamespace(solver_type="mujoco_warp"),
        ),
        _runtime=SimpleNamespace(current_state=current_state),
        state_0=object(),
        state_1=object(),
    )
    runtime = NewtonDifferentiableRuntime(lambda: backend)

    assert runtime.model is model
    assert runtime.current_state is current_state
    assert runtime.live_states == (backend.state_0, backend.state_1)
    assert not hasattr(runtime, "control")
    assert not hasattr(runtime, "create_differentiable_trajectory")


def test_construct_without_requires_grad_raises() -> None:
    """Newton kinematic models must opt into Warp gradients."""
    with pytest.raises(RuntimeError, match="requires_grad"):
        DifferentiableEnv(_diff_env_cfg(requires_grad=False))


def test_construct_on_default_backend_raises() -> None:
    """The Default backend is rejected before environment initialization."""
    with pytest.raises(
        RuntimeError,
        match="DifferentiableEnv requires NewtonPhysicsCfg",
    ):
        DifferentiableEnv(_diff_env_cfg(backend="default"))


@pytest.mark.parametrize("grad_enabled", (True, False))
def test_terminal_reset_respects_tape_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    grad_enabled: bool,
) -> None:
    """Tracked terminal steps defer reset; inference resets synchronously."""
    env = _bare_env()
    reset_calls: list[torch.Tensor] = []
    sim_state = {"last_info": {}}
    env._build_sim_state_dict = lambda _action: sim_state

    outputs = (
        torch.full((1, 1), 7.0),
        torch.full((1,), 3.0),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    monkeypatch.setattr(
        NewtonStepFunc,
        "apply",
        staticmethod(lambda _action, _state: outputs),
    )

    def _reset(*, options: dict[str, Any]):
        reset_ids = torch.as_tensor(options["reset_ids"]).clone()
        reset_calls.append(reset_ids)
        return torch.full((1, 1), -1.0), {}

    env.reset = _reset
    action = torch.zeros(1, requires_grad=True)
    context = torch.enable_grad() if grad_enabled else torch.no_grad()
    with context:
        obs, _, _, _, info = env.step(action)

    if grad_enabled:
        assert torch.equal(obs, torch.full((1, 1), 7.0))
        assert reset_calls == []
        assert info["requires_reset_after_backward"] is True
        assert torch.equal(info["deferred_reset_ids"], torch.tensor([0]))
    else:
        assert torch.equal(obs, torch.full((1, 1), -1.0))
        assert len(reset_calls) == 1
        assert "requires_reset_after_backward" not in info


def _import_franka_env():
    """Import the Franka APG environment after resolving task packages."""
    from embodichain_tasks.special.franka_reach_apg import FrankaReachApgEnv

    return FrankaReachApgEnv


def test_franka_kinematics_build_snapshots_live_primal_before_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Franka detaches taped FK inputs before the parent opens a tape."""
    from embodichain_tasks.special import franka_reach_apg

    env = object.__new__(franka_reach_apg.FrankaReachApgEnv)
    live_joint_q = object()
    snapshot_joint_q = object()
    fresh_fk_state = object()
    events: list[str] = []
    env.sim = SimpleNamespace(
        differentiable_runtime=SimpleNamespace(
            current_state=SimpleNamespace(joint_q=live_joint_q),
            model=SimpleNamespace(
                state=lambda: (events.append("state"), fresh_fk_state)[1]
            ),
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
    monkeypatch.setattr(DifferentiableEnv, "_build_sim_state_dict", _parent_build)

    result = env._build_sim_state_dict(torch.zeros(1, 7))

    assert result == {"prepared": True}
    assert events == ["clone", "state", "parent"]


def test_franka_action_kernel_reads_snapshot_instead_of_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded action kernel never captures mutable manager state."""
    from embodichain_tasks.special import franka_reach_apg

    env = object.__new__(franka_reach_apg.FrankaReachApgEnv)
    snapshot_joint_q = object()
    target_joint_q = object()
    action_wp = object()
    launch_inputs: list[object] = []
    env.sim = SimpleNamespace(num_envs=1)
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
    assert launch_inputs[2] is target_joint_q


def test_franka_snapshot_keeps_gradient_after_live_state_mutation_and_matches_fd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Detached FK input survives live writes before backward under strict mode."""
    from embodichain_tasks.special import franka_reach_apg

    env = object.__new__(franka_reach_apg.FrankaReachApgEnv)
    device = "cpu"
    live_joint_q = wp.zeros(7, dtype=wp.float32, device=device)
    env.sim = SimpleNamespace(
        num_envs=1,
        differentiable_runtime=SimpleNamespace(
            current_state=SimpleNamespace(joint_q=live_joint_q),
            model=SimpleNamespace(state=lambda: object()),
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
        DifferentiableEnv,
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


@pytest.mark.requires_sim
@pytest.mark.gpu
def test_franka_apg_smoke_backward() -> None:
    """Reward remains tracked and produces a finite action gradient."""
    try:
        FrankaReachApgEnv = _import_franka_env()
    except FileNotFoundError as exc:
        pytest.skip(f"Franka URDF not available: {exc}")

    env = FrankaReachApgEnv(num_envs=2)
    try:
        env.reset(seed=0)
        action = torch.zeros(2, 7, requires_grad=True, device=env.device)
        _, reward, _, _, _ = env.step(action)
        assert reward.requires_grad
        reward.sum().backward()
        assert action.grad is not None
        assert torch.isfinite(action.grad).all()
    finally:
        env.close()


@pytest.mark.requires_sim
@pytest.mark.gpu
def test_franka_apg_one_iter_loss_reduces() -> None:
    """A short action optimization reduces the kinematic reach loss."""
    try:
        FrankaReachApgEnv = _import_franka_env()
    except FileNotFoundError as exc:
        pytest.skip(f"Franka URDF not available: {exc}")

    env = FrankaReachApgEnv(num_envs=2)
    try:
        env.reset(seed=0)
        action = torch.zeros(2, 7, requires_grad=True, device=env.device)
        optimizer = torch.optim.SGD([action], lr=0.01)
        losses: list[float] = []
        for _ in range(3):
            env.reset(seed=0)
            optimizer.zero_grad()
            _, reward, _, _, _ = env.step(action)
            loss = (-reward).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        assert losses[-1] < losses[0], f"APG did not reduce loss: {losses}"
    finally:
        env.close()
