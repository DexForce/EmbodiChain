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
"""Warp-tape <-> PyTorch-autograd bridge for Newton physics."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import TYPE_CHECKING, Any, Callable, Iterator

import torch
import warp as wp

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["NewtonStepFunc", "differentiable_step", "tape_context"]


def _physics_dt(nm: Any, sim_state: dict[str, Any]) -> float:
    """Resolve the outer Newton step duration represented by one control step."""
    physics_dt = sim_state.get("physics_dt")
    if physics_dt is None:
        physics_dt = float(nm.solver_dt) * int(nm.num_substeps)
    try:
        physics_dt = float(physics_dt)
    except (TypeError, ValueError) as exc:
        raise TypeError("physics_dt must be a positive finite float.") from exc
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("physics_dt must be a positive finite float.")
    return physics_dt


def _resolve_step_mode(sim_state: dict[str, Any]) -> tuple[str, Callable | None]:
    """Validate the explicit dynamics-versus-kinematics bridge contract."""
    step_mode = sim_state.get("step_mode", "dynamics")
    if step_mode not in {"dynamics", "kinematics"}:
        raise ValueError(
            "step_mode must be 'dynamics' or 'kinematics', " f"got {step_mode!r}."
        )

    step_fn = sim_state.get("step_fn")
    if step_mode == "dynamics" and step_fn is not None:
        raise ValueError(
            "step_fn is only supported when step_mode='kinematics'; "
            "the dynamics route always uses Newton solver dynamics."
        )
    if step_mode == "kinematics" and step_fn is None:
        raise ValueError("step_mode='kinematics' requires a named step_fn.")
    return step_mode, step_fn


def _reset_tape_then_release(tape: wp.Tape | None, trajectory: Any | None) -> None:
    """End tape ownership before releasing the trajectory's model lease."""
    try:
        if tape is not None:
            tape.reset()
    finally:
        if trajectory is not None:
            trajectory.release()


def _abort_forward(
    tape: wp.Tape | None,
    trajectory: Any | None,
) -> None:
    """Best-effort cleanup which never masks the original forward failure."""
    try:
        _reset_tape_then_release(tape, trajectory)
    except BaseException:
        # The active trajectory must not mask the action/solver/output failure
        # which caused the abort. DexSim release is idempotent and this path is
        # only entered to preserve the original exception.
        pass


@contextmanager
def tape_context(manager: "SimulationManager") -> Iterator[wp.Tape]:
    """Open a Warp tape bound to the manager's Newton state.

    Advanced users compose their own Warp kernels inside this context, then
    call ``tape.backward()`` outside the with-block.
    """
    if not manager.is_newton_backend:
        raise RuntimeError(
            "tape_context requires the Newton backend with requires_grad=True."
        )
    tape = wp.Tape()
    with tape:
        yield tape


def differentiable_step(
    manager: "SimulationManager",
    *,
    apply_control_fn: Callable[[wp.Tape, Any], None],
    substeps: int,
    dt: float | None = None,
) -> dict[str, Any]:
    """Run a low-level manager-owned Newton trajectory inside a Warp tape.

    ``substeps`` remains a legacy solver-step count. It must therefore divide
    evenly into whole Newton physics steps; the public trajectory transaction
    owns every detached state, contact, control, and generation lease.

    The returned tape and trajectory remain active for the caller to use in a
    custom backward pass. After ``tape.backward()`` (or when abandoning the
    result), callers must invoke ``tape.reset()`` and then
    ``trajectory.release()`` in a ``finally`` block. The helper releases both
    automatically only when forward construction itself fails.

    Args:
        manager: The owning :class:`SimulationManager` (must be Newton).
        apply_control_fn: Callable that writes the trajectory-local joint/body
            control targets inside the tape. It receives ``(tape, control)``
            and must launch Warp kernels targeting ``control``, never the
            manager's shared control buffer.
        substeps: Number of solver substeps to run.
        dt: Solver dt; defaults to the manager's configured solver dt.

    Returns:
        A dict carrying the tape, trajectory, and detached final state for the
        caller to retain through backward before resetting/releasing it.
    """
    if not manager.is_newton_backend:
        raise RuntimeError("differentiable_step requires the Newton backend.")
    nm = manager.physics.newton_manager
    if isinstance(substeps, bool) or int(substeps) != substeps or substeps <= 0:
        raise ValueError("substeps must be a positive integer.")
    substeps = int(substeps)
    num_substeps = int(nm.num_substeps)
    if num_substeps <= 0:
        raise ValueError("Newton num_substeps must be positive.")
    if substeps % num_substeps != 0:
        raise ValueError(
            "substeps must be divisible by Newton num_substeps so the "
            "trajectory represents whole physics steps."
        )
    dt_val = float(nm.solver_dt if dt is None else dt)
    if not math.isfinite(dt_val) or dt_val <= 0.0:
        raise ValueError("dt must be a positive finite solver time step.")

    trajectory = None
    tape = None
    try:
        trajectory = nm.create_differentiable_trajectory(
            physics_steps=substeps // num_substeps,
            physics_dt=dt_val * num_substeps,
        )
        tape = wp.Tape()
        with tape:
            apply_control_fn(tape, trajectory.control)
            final_state = trajectory.step()
        nm.commit_differentiable_trajectory(trajectory)
    except BaseException:
        _abort_forward(tape, trajectory)
        raise

    return {
        "tape": tape,
        "trajectory": trajectory,
        "final_state": final_state,
        "states": trajectory.states,
        "contacts": trajectory.contacts,
        "control": trajectory.control,
    }


class NewtonStepFunc(torch.autograd.Function):
    """torch.autograd.Function bridging Warp tape autodiff to PyTorch.

    Forward: validates an explicit step mode before creating a tape. The
    default ``dynamics`` route allocates a manager-owned detached trajectory,
    launches the action-to-local-control Warp kernel, records its solver
    horizon, and commits it only after tape exit. The explicitly selected
    ``kinematics`` route retains its named FK ``step_fn`` escape hatch.
    Observation/reward kernels run inside the tape so their outputs carry
    gradient back to ``action_wp``.

    Backward: copies upstream PyTorch grads into the corresponding Warp
    ``.grad`` buffers, calls ``tape.backward()``, and returns
    ``wp.to_torch(action_wp.grad)`` reshaped to the action's tensor shape.

    Callers must supply a ``sim_state`` dict with the following keys:
        manager: SimulationManager (Newton, requires_grad=True)
        substeps: int control-level physics updates (used by the default
            solver-based step route)
        step_mode: ``"dynamics"`` (default) or explicit ``"kinematics"``
        action_to_control_kernel: dynamics callable
            ``(action_wp, trajectory_control, *kernel_args)``; kinematics
            retains ``(action_wp, tape, *kernel_args)``
        kernel_args: tuple consumed by action_to_control_kernel
        obs_reward_fn: callable(final_state) -> dict with torch outputs
        physics_dt: optional outer Newton step duration (defaults to
            ``solver_dt * num_substeps``)
        step_fn: required only when ``step_mode == "kinematics"``

    The ``obs_reward_fn`` must return a dict containing:
        _order: tuple of output names (returned in this order)
        _grad_track: dict mapping name -> Warp array (or None) whose
            ``.grad`` should be seeded from the upstream PyTorch grad
        <name>: torch tensor for each name in ``_order``
    """

    @classmethod
    def apply(cls, action_torch: torch.Tensor, sim_state: dict[str, Any]) -> Any:
        """Capture the caller's grad mode before PyTorch enters ``forward``.

        ``torch.autograd.Function.forward`` always executes with grad mode
        disabled, and ``ctx.needs_input_grad`` alone remains true when a
        requires-grad action is passed through an outer ``torch.no_grad()``
        block. Passing the ambient mode as a non-differentiable argument lets
        the bridge synchronously reset/release no-grad trajectories instead of
        retaining an unreachable manager lease.
        """
        return super().apply(action_torch, sim_state, torch.is_grad_enabled())

    @staticmethod
    def forward(
        ctx: Any,
        action_torch: torch.Tensor,
        sim_state: dict[str, Any],
        outer_grad_enabled: bool,
    ) -> tuple[torch.Tensor, ...]:
        manager = sim_state["manager"]
        substeps = int(sim_state["substeps"])
        kernel = sim_state["action_to_control_kernel"]
        kernel_args = sim_state["kernel_args"]
        obs_reward_fn = sim_state["obs_reward_fn"]
        step_mode, step_fn = _resolve_step_mode(sim_state)
        tape_binder = (
            sim_state.get("_bind_dynamics_tape") if step_mode == "dynamics" else None
        )

        # Save the original action shape so backward can reshape the gradient.
        ctx.saved_action_shape = action_torch.shape

        nm = manager.physics.newton_manager

        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        needs_action_grad = bool(outer_grad_enabled and ctx.needs_input_grad[0])
        action_wp = wp.from_torch(
            action_flat,
            dtype=wp.float32,
            requires_grad=needs_action_grad,
        )

        trajectory = None
        tape = None
        try:
            if step_mode == "dynamics":
                if substeps <= 0:
                    raise ValueError("substeps must be a positive integer.")
                trajectory = nm.create_differentiable_trajectory(
                    physics_steps=substeps,
                    physics_dt=_physics_dt(nm, sim_state),
                )

            tape = wp.Tape()
            try:
                with tape:
                    if tape_binder is not None:
                        tape_binder(tape)
                    if step_mode == "dynamics":
                        kernel(action_wp, trajectory.control, *kernel_args)
                        final_state = trajectory.step()
                    else:
                        # The explicit FK route keeps the historical callback
                        # shape and receives the open tape, but never detached
                        # solver control.
                        kernel(action_wp, tape, *kernel_args)
                        final_state = step_fn()

                    # Validate and materialize outputs inside the tape. A malformed
                    # output dictionary is a forward failure and must not publish a
                    # detached dynamics trajectory.
                    outputs = obs_reward_fn(final_state)
                    outputs_order = tuple(outputs["_order"])
                    output_values = tuple(outputs[name] for name in outputs_order)
                    outputs_grad_track = outputs.get("_grad_track", {})
            finally:
                if tape_binder is not None:
                    tape_binder(None)

            if trajectory is not None:
                nm.commit_differentiable_trajectory(trajectory)
        except BaseException:
            _abort_forward(tape, trajectory)
            raise

        if not needs_action_grad:
            _reset_tape_then_release(tape, trajectory)
            return output_values

        ctx.tape = tape
        ctx.trajectory = trajectory
        ctx.action_wp = action_wp
        ctx.outputs_order = outputs_order
        ctx.outputs_grad_track = outputs_grad_track
        ctx._bridge_released = False
        return output_values

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None]:
        if getattr(ctx, "_bridge_released", False):
            raise RuntimeError(
                "NewtonStepFunc backward was already consumed; create a new "
                "differentiable trajectory for another backward pass."
            )

        action_grad = None
        try:
            # Copy each upstream grad back into the corresponding Warp .grad.
            for name, grad_t in zip(ctx.outputs_order, grad_outputs):
                wp_arr = ctx.outputs_grad_track.get(name)
                if grad_t is None or wp_arr is None:
                    continue
                # Warp allocates .grad lazily for arrays with requires_grad=True
                # that participate in the tape; allocate defensively in case
                # the array was created but never written inside the tape.
                if wp_arr.grad is None:
                    wp_arr.grad = wp.zeros_like(wp_arr)
                wp.copy(
                    wp_arr.grad,
                    wp.from_torch(
                        grad_t.detach().clone().contiguous(),
                        dtype=wp.float32,
                    ),
                )
            ctx.tape.backward()
            action_wp_grad = getattr(ctx.action_wp, "grad", None)
            if action_wp_grad is not None:
                # Capture the action gradient before reset invalidates tape
                # storage, then terminate tape ownership before releasing the
                # trajectory's active manager token.
                action_grad = wp.to_torch(action_wp_grad).clone()
        finally:
            try:
                _reset_tape_then_release(ctx.tape, ctx.trajectory)
            finally:
                ctx._bridge_released = True

        if action_grad is None:
            return None, None, None
        # Reshape to the original action layout; metadata inputs have no
        # gradient.
        return action_grad.reshape(ctx.saved_action_shape), None, None
