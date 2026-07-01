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
from typing import TYPE_CHECKING, Any, Callable, Iterator

import torch
import warp as wp

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["NewtonStepFunc", "differentiable_step", "tape_context"]


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
    apply_control_fn: Callable[[wp.Tape], None],
    substeps: int,
    dt: float | None = None,
) -> dict:
    """Run one EmbodiChain-level physics step inside a Warp tape.

    Args:
        manager: The owning :class:`SimulationManager` (must be Newton).
        apply_control_fn: Callable that writes the joint/body control
            targets inside the tape. Invoked once at the start of the
            step. Receives the open tape; must launch Warp kernels (or
            call dexsim setters that are tape-aware) to populate
            ``manager.physics.newton_manager._control``.
        substeps: Number of solver substeps to run (typically
            ``sim_cfg.sim_steps_per_control``).
        dt: Solver dt; defaults to the manager's configured dt.

    Returns:
        A dict carrying the tape and the state buffers for the caller to
        save in autograd context.
    """
    if not manager.is_newton_backend:
        raise RuntimeError("differentiable_step requires the Newton backend.")
    nm = manager.physics.newton_manager
    stepper = manager.create_differentiable_stepper()
    state_in = nm._state_0
    state_out = nm._model.state()
    contacts = stepper.create_contacts()
    dt_val = nm.solver_dt if dt is None else float(dt)

    tape = wp.Tape()
    with tape:
        apply_control_fn(tape)
        for _ in range(substeps):
            stepper.step(state_in, state_out, contacts=contacts, dt=dt_val)
            state_in, state_out = state_out, state_in

    # The final state lives in state_in after the swap.
    return {
        "tape": tape,
        "final_state": state_in,
        "stepper": stepper,
    }


class NewtonStepFunc(torch.autograd.Function):
    """torch.autograd.Function bridging Warp tape autodiff to PyTorch.

    Forward: launches the action-to-control Warp kernel, runs the
    caller-provided ``step_fn`` (differentiable solver loop or FK bypass),
    and reads observation / reward as torch tensors via ``wp.to_torch``
    (zero-copy where possible). The obs/reward kernels launched by
    ``obs_reward_fn`` run INSIDE the open Warp tape so that their outputs
    carry gradient back to ``action_wp``.

    Backward: copies upstream PyTorch grads into the corresponding Warp
    ``.grad`` buffers, calls ``tape.backward()``, and returns
    ``wp.to_torch(action_wp.grad)`` reshaped to the action's tensor shape.

    Callers must supply a ``sim_state`` dict with the following keys:
        manager: SimulationManager (Newton, requires_grad=True)
        substeps: int (used by the default solver-based step_fn)
        action_to_control_kernel: callable(action_wp, *kernel_args)
        kernel_args: tuple consumed by action_to_control_kernel
        obs_reward_fn: callable(final_state) -> dict with torch outputs
        step_fn: optional callable() -> final Newton state; when omitted
            the bridge runs the differentiable stepper for ``substeps``
            iterations (the original solver-based path)

    The ``obs_reward_fn`` must return a dict containing:
        _order: tuple of output names (returned in this order)
        _grad_track: dict mapping name -> Warp array (or None) whose
            ``.grad`` should be seeded from the upstream PyTorch grad
        <name>: torch tensor for each name in ``_order``
    """

    @staticmethod
    def forward(ctx, action_torch: torch.Tensor, sim_state: dict):
        manager = sim_state["manager"]
        substeps = int(sim_state["substeps"])
        kernel = sim_state["action_to_control_kernel"]
        kernel_args = sim_state["kernel_args"]
        obs_reward_fn = sim_state["obs_reward_fn"]
        step_fn = sim_state.get("step_fn")

        # Save the original action shape so backward can reshape the gradient.
        ctx.saved_action_shape = action_torch.shape

        nm = manager.physics.newton_manager

        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        action_wp = wp.from_torch(action_flat, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            kernel(action_wp, *kernel_args)  # writes inputs for stepping
            if step_fn is not None:
                final_state = step_fn()
            else:
                stepper = manager.create_differentiable_stepper()
                state_in = nm._state_0
                state_out = nm._model.state()
                contacts = stepper.create_contacts()
                dt_val = nm.solver_dt
                for _ in range(substeps):
                    stepper.step(state_in, state_out, contacts=contacts, dt=dt_val)
                    state_in, state_out = state_out, state_in
                final_state = state_in
            # Compute obs/reward INSIDE the tape so the reward/obs kernels
            # participate in the Warp autodiff graph. The torch tensors
            # returned by obs_reward_fn are built via wp.to_torch of
            # tape-tracked Warp arrays, so they carry gradient back to
            # action_wp when tape.backward() is called.
            outputs = obs_reward_fn(final_state)

        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.outputs_order = outputs["_order"]
        ctx.outputs_grad_track = outputs.get("_grad_track", {})
        return tuple(outputs[k] for k in outputs["_order"])

    @staticmethod
    def backward(ctx, *grad_outputs):
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
                wp.from_torch(grad_t.detach().clone().contiguous(), dtype=wp.float32),
            )
        ctx.tape.backward()
        action_grad = wp.to_torch(ctx.action_wp.grad).clone()
        ctx.tape.zero()
        # Reshape to the original action layout; second input (sim_state)
        # has no gradient.
        return action_grad.reshape(ctx.saved_action_shape), None
