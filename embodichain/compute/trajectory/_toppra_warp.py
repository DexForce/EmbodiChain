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

"""Internal tensor adapter for the planner's Warp TOPPRA backend.

Waypoint data and numerical work stay on the input device. Scalar reads
validate and size variable-length output; no path is solved through NumPy.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import warp as wp
from torch.autograd.function import once_differentiable

from ._toppra import _limits, _validate_grid_size, _validate_sampling
from ._warp import toppra as kernels

__all__ = []


class _ToppraAutograd(torch.autograd.Function):
    """Bridge Warp's first-order adjoints into the caller's Torch graph."""

    @staticmethod
    def forward(
        ctx: Any,
        waypoints: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        sample_count: int | None,
        sample_dt: float | None,
        grid_size: int,
    ) -> tuple[torch.Tensor, ...]:
        state: dict[str, Any] = {}
        result = _launch_toppra(
            waypoints,
            velocity,
            acceleration,
            sample_count=sample_count,
            sample_dt=sample_dt,
            grid_size=grid_size,
            state=state,
        )
        ctx.state = state
        ctx.save_for_backward(waypoints, velocity, acceleration)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(result["success"])
        return tuple(
            result[key]
            for key in ("success", "positions", "velocities", "accelerations", "dt")
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, success_grad: None, *output_grads: torch.Tensor | None
    ) -> tuple:
        saved = ctx.saved_tensors
        waypoints = saved[0]
        state = ctx.state
        stream = (
            wp.ScopedStream(
                wp.stream_from_torch(torch.cuda.current_stream(waypoints.device))
            )
            if waypoints.is_cuda
            else nullcontext()
        )
        with stream:
            # A retained Torch graph may be differentiated repeatedly. Copy the
            # returned input gradient before the next backward clears the tape.
            state["tape"].zero()
            seeds = {
                array: wp.from_torch(
                    grad.to(torch.float64).contiguous(), dtype=wp.float64
                )
                for array, grad in zip(state["outputs"], output_grads)
                if grad is not None
            }
            if seeds:
                state["tape"].backward(grads=seeds)
            gradients = tuple(
                wp.to_torch(array.grad).to(tensor.dtype).clone() if needed else None
                for array, tensor, needed in zip(
                    state["inputs"], saved, ctx.needs_input_grad
                )
            )
        return *gradients, None, None, None


def _retime_toppra_warp(
    waypoints: torch.Tensor,
    velocity: object,
    acceleration: object,
    *,
    sample_count: int | None = None,
    sample_dt: float | None = None,
    grid_size: int = 100,
) -> dict[str, torch.Tensor]:
    """Retime (B, N, DOF) paths, isolating invalid/infeasible rows.

    Failed rows contain zeros. Stationary rows have two zero-time samples.
    Shorter successful rows pad with their endpoint and zero derivatives/dt.
    The floating outputs use the input dtype and device. Computation uses
    float64 on both Warp CPU and CUDA, and honors the current Torch stream.

    Waypoints and tensor motion limits support first-order Warp adjoints on
    CPU and CUDA with either sampling mode. Derivatives hold discrete filtering,
    grid topology, sample counts and active constraints fixed, and are not
    guaranteed at their switches. In time mode, sample_dt sets an upper bound
    on spacing; samples remain uniform over the full solved duration.
    Failed rows contribute zero gradients; stationary rows differentiate only
    the retained position. Higher-order derivatives are unsupported.
    """
    _validate_sampling(sample_count, sample_dt)
    _validate_grid_size(grid_size)
    if waypoints.ndim != 3 or min(waypoints.shape) < 1 or waypoints.shape[1] < 2:
        raise ValueError("waypoints must have shape (B >= 1, N >= 2, DOF >= 1)")
    if waypoints.dtype not in (torch.float32, torch.float64):
        raise ValueError("waypoints must have float32 or float64 dtype")
    velocity = _tensor_limits(velocity, waypoints.shape[2], waypoints.device)
    acceleration = _tensor_limits(acceleration, waypoints.shape[2], waypoints.device)
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (waypoints, velocity, acceleration)
    ):
        values = _ToppraAutograd.apply(
            waypoints, velocity, acceleration, sample_count, sample_dt, grid_size
        )
        return dict(
            zip(("success", "positions", "velocities", "accelerations", "dt"), values)
        )
    return _launch_toppra(
        waypoints,
        velocity,
        acceleration,
        sample_count=sample_count,
        sample_dt=sample_dt,
        grid_size=grid_size,
    )


def _tensor_limits(value: object, dofs: int, device: torch.device) -> torch.Tensor:
    """Normalize tensor limits without detaching their gradient or visiting CPU."""
    if not isinstance(value, torch.Tensor):
        # Keep Python/NumPy validation on the host; no extra CUDA round trips
        # are needed for the existing constant-limit planning interface.
        return torch.as_tensor(_limits(value, dofs), dtype=torch.float64, device=device)
    bounds = value.to(device=device, dtype=torch.float64)
    if bounds.ndim == 0:
        bounds = bounds.expand(dofs)
    if bounds.shape == (dofs,):
        bounds = torch.stack((-bounds, bounds), dim=-1)
    if bounds.shape != (dofs, 2):
        raise ValueError("limits must be finite scalar, (DOF,), or (DOF, 2)")
    if not bool(
        (
            torch.isfinite(bounds).all()
            & (bounds[:, 0] <= 0).all()
            & (bounds[:, 1] >= 0).all()
        ).item()
    ):
        raise ValueError("limits must be finite and contain zero")
    return bounds.contiguous()


def _launch_toppra(
    waypoints: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    *,
    sample_count: int | None = None,
    sample_dt: float | None = None,
    grid_size: int = 100,
    state: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Run the shared forward kernels, recording a tape only for autograd."""
    batch, count, dofs = waypoints.shape
    # NumPy 1.x promotes a float32 scalar comparison to float64; NumPy 2.x
    # keeps float32. Match the installed CPU backend at the dedup boundary.
    scalar_type = np.float32 if waypoints.dtype == torch.float32 else np.float64
    duplicate_tolerance = np.asarray(
        1e-6, dtype=np.result_type(scalar_type(0), 1e-6)
    ).item()
    wp.init()
    device = wp.device_from_torch(waypoints.device)
    stream = (
        wp.ScopedStream(
            wp.stream_from_torch(torch.cuda.current_stream(waypoints.device))
        )
        if waypoints.is_cuda
        else nullcontext()
    )
    differentiable = state is not None
    tape = wp.Tape() if differentiable else nullcontext()
    with stream:
        tensor = waypoints.detach().to(dtype=torch.float64).contiguous()
        points = wp.from_torch(tensor, dtype=wp.float64, requires_grad=differentiable)
        clean = wp.zeros((batch, count, dofs), dtype=wp.float64, device=device)
        kept = wp.zeros((batch, count), dtype=int, device=device)
        counts = wp.zeros(batch, dtype=int, device=device)
        intervals = wp.zeros(batch, dtype=int, device=device)
        success = wp.zeros(batch, dtype=int, device=device)
        base = max(grid_size - 1, 10 * count - 1)
        capacity = base + count - 1
        wp.launch(
            kernels._prepare,
            dim=batch,
            inputs=[
                points,
                clean,
                kept,
                counts,
                intervals,
                success,
                grid_size,
                int(waypoints.dtype == torch.float32),
                wp.float64(duplicate_tolerance),
            ],
            device=device,
        )
        with tape:
            if differentiable:
                clean = wp.zeros(
                    (batch, count, dofs),
                    dtype=wp.float64,
                    device=device,
                    requires_grad=True,
                )
                wp.launch(
                    kernels._gather,
                    dim=clean.shape,
                    inputs=[points, kept, counts, success, clean],
                    device=device,
                )
            diagonal = wp.zeros_like(clean, requires_grad=False)
            eliminated = wp.zeros_like(clean)
            second = wp.zeros_like(clean)
            coefficients = wp.zeros(
                (batch, count - 1, dofs, 4),
                dtype=wp.float64,
                device=device,
                requires_grad=differentiable,
            )
            wp.launch(
                kernels._spline,
                dim=(batch, dofs),
                inputs=[
                    clean,
                    counts,
                    success,
                    diagonal,
                    eliminated,
                    second,
                    coefficients,
                ],
                device=device,
            )
            velocity_wp = wp.from_torch(
                velocity.detach(), dtype=wp.float64, requires_grad=differentiable
            )
            acceleration_wp = wp.from_torch(
                acceleration.detach(), dtype=wp.float64, requires_grad=differentiable
            )
            velocity_caps = wp.zeros(
                (batch, capacity, dofs),
                dtype=wp.float64,
                device=device,
                requires_grad=differentiable,
            )
            a = wp.zeros(
                (batch, capacity, 6 * dofs + 2),
                dtype=wp.float64,
                device=device,
                requires_grad=differentiable,
            )
            b, c = wp.zeros_like(a), wp.zeros_like(a)
            wp.launch(
                kernels._constraints,
                dim=(batch, capacity),
                inputs=[
                    coefficients,
                    counts,
                    intervals,
                    success,
                    velocity_wp,
                    acceleration_wp,
                    velocity_caps,
                    a,
                    b,
                    c,
                ],
                device=device,
            )
            wp.launch(
                kernels._velocity_caps,
                dim=(batch, capacity),
                inputs=[velocity_caps, intervals, success, a, b, c],
                device=device,
            )
            caps = wp.zeros(
                (batch, capacity),
                dtype=wp.float64,
                device=device,
                requires_grad=differentiable,
            )
            wp.launch(
                kernels._project_caps,
                dim=(batch, capacity),
                inputs=[a, b, c, intervals, caps],
                device=device,
            )
            controllable = wp.zeros(
                (batch, capacity + 1),
                dtype=wp.float64,
                device=device,
                requires_grad=differentiable,
            )
            x, times = wp.zeros_like(controllable), wp.zeros_like(controllable)
            durations = wp.zeros(
                batch, dtype=wp.float64, device=device, requires_grad=differentiable
            )
            wp.launch(
                kernels._solve,
                dim=batch,
                inputs=[
                    a,
                    b,
                    c,
                    intervals,
                    caps,
                    controllable,
                    x,
                    times,
                    durations,
                    success,
                ],
                device=device,
            )
            duration_tensor = wp.to_torch(durations)
            ok = wp.to_torch(success).bool()
            moving = ok & (wp.to_torch(counts) > 1)
            if sample_count is not None:
                lengths = torch.full(
                    (batch,), sample_count, device=waypoints.device, dtype=torch.int32
                )
            else:
                # Integer output lengths are frozen for this forward/backward
                # pair; timing and sampling still differentiate through duration.
                desired = torch.ceil(duration_tensor.detach() / sample_dt) + 1
                if bool((desired > torch.iinfo(torch.int32).max).any().item()):
                    raise ValueError("sample_dt requests too many samples")
                lengths = desired.to(torch.int32).clamp_min(2)
            lengths = torch.where(moving, lengths, 2).contiguous()
            samples = int(lengths.max().item())
            positions = wp.zeros(
                (batch, samples, dofs),
                dtype=wp.float64,
                device=device,
                requires_grad=differentiable,
            )
            velocities, accelerations = wp.zeros_like(positions), wp.zeros_like(
                positions
            )
            dt = wp.zeros(
                (batch, samples),
                dtype=wp.float64,
                device=device,
                requires_grad=differentiable,
            )
            wp.launch(
                kernels._sample,
                dim=(batch, samples, dofs),
                inputs=[
                    clean,
                    coefficients,
                    counts,
                    intervals,
                    success,
                    x,
                    times,
                    durations,
                    wp.from_torch(lengths),
                    positions,
                    velocities,
                    accelerations,
                    dt,
                ],
                device=device,
            )
        if state is not None:
            state.update(
                tape=tape,
                inputs=(points, velocity_wp, acceleration_wp),
                outputs=(positions, velocities, accelerations, dt),
            )
        return {
            "success": ok,
            "positions": wp.to_torch(positions).to(waypoints.dtype),
            "velocities": wp.to_torch(velocities).to(waypoints.dtype),
            "accelerations": wp.to_torch(accelerations).to(waypoints.dtype),
            "dt": wp.to_torch(dt).to(waypoints.dtype),
        }
