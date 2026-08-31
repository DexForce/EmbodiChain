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
"""Warp-tape ↔ PyTorch-autograd bridge for Newton kinematics."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

import torch
import warp as wp

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["NewtonStepFunc", "tape_context"]


def _validate_manager(manager: Any) -> None:
    """Validate the Newton gradient boundary without touching its solver."""
    if not bool(getattr(manager, "is_newton_backend", False)):
        raise RuntimeError(
            "Differentiable kinematics require the Newton backend with "
            "requires_grad=True."
        )
    runtime = getattr(manager, "differentiable_runtime", None)
    if runtime is not None:
        # Model access validates finalization and requires_grad while remaining
        # independent of the configured Newton solver.
        _ = runtime.model


def _reset_tape(tape: wp.Tape | None) -> None:
    """Release all arrays retained by a completed Warp tape."""
    if tape is not None:
        tape.reset()


def _abort_forward(tape: wp.Tape | None) -> None:
    """Best-effort cleanup that never masks the original forward failure."""
    try:
        _reset_tape(tape)
    except BaseException:
        pass


@contextmanager
def tape_context(manager: "SimulationManager") -> Iterator[wp.Tape]:
    """Open a Warp tape for expert Newton kinematics kernels.

    Args:
        manager: Prepared Newton-backed simulation manager in gradient mode.

    Yields:
        The active Warp tape. Call ``backward`` and then ``reset`` after the
        context when retaining it manually.

    Raises:
        RuntimeError: If the manager does not use a finalized Newton gradient
            model.
    """
    _validate_manager(manager)
    tape = wp.Tape()
    with tape:
        yield tape


class NewtonStepFunc(torch.autograd.Function):
    """Bridge one task-defined Newton kinematics step into PyTorch autograd.

    Forward records the action kernel, named kinematics callback, and output
    kernels inside one Warp tape. It does not create contacts, call a Newton
    solver, or advance simulation time. Backward seeds the tracked Warp output
    arrays from PyTorch gradients and returns the resulting action gradient.

    ``sim_state`` must contain:

    - ``manager``: a prepared Newton-backed :class:`SimulationManager`;
    - ``action_kernel``: callable ``(action_wp, tape, *kernel_args)``;
    - ``kernel_args``: tuple forwarded to the action kernel;
    - ``step_fn``: zero-argument task kinematics callback returning a state;
    - ``obs_reward_fn``: callable that maps that state to an output dictionary.

    The output dictionary contains ``_order``, ``_grad_track``, and one torch
    tensor for every name in ``_order``. ``_grad_track`` maps a name to the
    backing Warp array whose gradient should be seeded, or to ``None`` for a
    non-differentiable output.
    """

    @classmethod
    def apply(cls, action_torch: torch.Tensor, sim_state: dict[str, Any]) -> Any:
        """Capture ambient grad mode before PyTorch enters ``forward``."""
        return super().apply(action_torch, sim_state, torch.is_grad_enabled())

    @staticmethod
    def forward(
        ctx: Any,
        action_torch: torch.Tensor,
        sim_state: dict[str, Any],
        outer_grad_enabled: bool,
    ) -> tuple[torch.Tensor, ...]:
        """Record one kinematics step and materialize its torch outputs."""
        _validate_manager(sim_state["manager"])
        action_kernel = sim_state["action_kernel"]
        kernel_args = sim_state["kernel_args"]
        step_fn = sim_state["step_fn"]
        obs_reward_fn = sim_state["obs_reward_fn"]
        if not callable(step_fn):
            raise TypeError("Differentiable kinematics require a callable step_fn.")

        ctx.saved_action_shape = action_torch.shape
        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        needs_action_grad = bool(outer_grad_enabled and ctx.needs_input_grad[0])
        action_wp = wp.from_torch(
            action_flat,
            dtype=wp.float32,
            requires_grad=needs_action_grad,
        )

        tape = None
        try:
            tape = wp.Tape()
            with tape:
                action_kernel(action_wp, tape, *kernel_args)
                final_state = step_fn()
                outputs = obs_reward_fn(final_state)
                outputs_order = tuple(outputs["_order"])
                output_values = tuple(outputs[name] for name in outputs_order)
                outputs_grad_track = outputs.get("_grad_track", {})
        except BaseException:
            _abort_forward(tape)
            raise

        if not needs_action_grad:
            _reset_tape(tape)
            return output_values

        ctx.tape = tape
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
        """Run Warp reverse mode and return the bridged action gradient."""
        if getattr(ctx, "_bridge_released", False):
            raise RuntimeError(
                "NewtonStepFunc backward was already consumed; run a fresh "
                "kinematics step before another backward pass."
            )

        action_grad = None
        try:
            for name, grad_t in zip(ctx.outputs_order, grad_outputs):
                wp_arr = ctx.outputs_grad_track.get(name)
                if grad_t is None or wp_arr is None:
                    continue
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
                action_grad = wp.to_torch(action_wp_grad).clone()
        finally:
            try:
                _reset_tape(ctx.tape)
            finally:
                ctx._bridge_released = True

        if action_grad is None:
            return None, None, None
        return action_grad.reshape(ctx.saved_action_shape), None, None
