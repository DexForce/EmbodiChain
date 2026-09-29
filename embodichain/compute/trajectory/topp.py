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

"""Differentiable time-optimal path parameterization under joint box limits.

The problem solved is the one TOPP-RA solves for joint velocity and
acceleration limits: given a geometric joint path ``q(s)``, find the fastest
``s(t)`` whose joint velocities and accelerations stay inside per-joint bounds,
starting and ending at rest. With the squared path speed ``x = sdot^2`` and the
path acceleration ``u = sddot`` as variables, every constraint is linear in
``(x, u)`` at each grid point, so each stage is a two-variable linear program.
Its feasible set projects onto ``x`` in closed form, which turns the
reachability sweeps into ``min``/``max``/division and makes the whole
computation differentiable almost everywhere with respect to the path.

Given the same path and grid, the result matches the ``toppra`` library's
``TOPPRA`` with ``ParametrizeConstAccel`` to numerical precision, for both its
collocation and its default interpolation discretization.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = [
    "TimeOptimalParameterization",
    "TimeOptimalTrajectory",
    "parameterize_time_optimal",
    "retime_time_optimal",
]

_BIG = 1.0e12
_DISCRETIZATION_SETS = {"collocation": 1, "interpolation": 2}

# The parameterization only enforces its limits at grid points; between them
# the reconstructed trajectory can exceed them, less as the grid is refined.
# Roughly 100 grid points per waypoint, floor 1000, keeps the overshoot in
# joint acceleration below about one percent on seven-joint arm paths.
_GRIDPOINTS_PER_WAYPOINT = 100
_GRIDPOINTS_MIN = 1000


@dataclass(frozen=True)
class TimeOptimalParameterization:
    """Time-optimal speed profile along a path grid.

    Attributes:
        sdot_sq: Squared path speed ``x`` at every grid point, ``(B, N+1)``.
            The first and last entries are zero: the motion is rest-to-rest.
        sddot: Constant path acceleration ``u`` on each grid segment, ``(B, N)``.
        segment_time: Time spent on each grid segment, ``(B, N)``.
        duration: Total duration, ``(B,)``.
    """

    sdot_sq: torch.Tensor
    sddot: torch.Tensor
    segment_time: torch.Tensor
    duration: torch.Tensor


@dataclass(frozen=True)
class TimeOptimalTrajectory:
    """Joint trajectory sampled on a time grid after time-optimal retiming.

    Rows shorter than the longest are padded by holding their final pose with
    zero velocity, zero acceleration and zero interval, matching the batched
    ``PlanResult`` convention.

    Attributes:
        positions: Joint positions, ``(B, K, J)``.
        velocities: Joint velocities, ``(B, K, J)``.
        accelerations: Joint accelerations, ``(B, K, J)``.
        dt: Interval used to reach each sample; the first entry is zero,
            ``(B, K)``.
        duration: Total duration per environment, ``(B,)``.
        success: Whether every returned value is finite, ``(B,)``.
    """

    positions: torch.Tensor
    velocities: torch.Tensor
    accelerations: torch.Tensor
    dt: torch.Tensor
    duration: torch.Tensor
    success: torch.Tensor


def _eps(dtype: torch.dtype) -> tuple[float, float]:
    """Return the degeneracy threshold and square-root floor for a dtype."""
    return (1.0e-12, 1.0e-12) if dtype == torch.float64 else (1.0e-7, 1.0e-10)


def _constraint_rows(
    path_velocity: torch.Tensor,
    path_acceleration: torch.Tensor,
    ds: torch.Tensor,
    num_sets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the joint-acceleration rows ``a u + b x`` for each stage.

    Collocation applies each joint's constraint at the stage's own grid point.
    Interpolation additionally applies it at the next grid point, reached with
    ``x + 2 ds u``, which contributes the row ``(a' + 2 ds b') u + b' x``. The
    last stage has no next point and repeats its own rows, as ``toppra`` does.
    """
    a_sets = [path_velocity]
    b_sets = [path_acceleration]
    if num_sets == 2:
        shifted = (
            path_velocity[:, 1:] + 2.0 * ds.view(1, -1, 1) * path_acceleration[:, 1:]
        )
        a_sets.append(torch.cat([shifted, path_velocity[:, -1:]], dim=1))
        b_sets.append(
            torch.cat([path_acceleration[:, 1:], path_acceleration[:, -1:]], dim=1)
        )
    return torch.stack(a_sets, dim=2), torch.stack(b_sets, dim=2)


def _parameterize_torch(
    path_velocity: torch.Tensor,
    path_acceleration: torch.Tensor,
    ds: torch.Tensor,
    velocity_limits: torch.Tensor,
    acceleration_limits: torch.Tensor,
    num_sets: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation; the Warp kernels compute the same values."""
    eps, sqrt_floor = _eps(path_velocity.dtype)
    batch, count, _ = path_velocity.shape
    n = count - 1
    a, b = _constraint_rows(path_velocity, path_acceleration, ds, num_sets)
    active = a.abs() > eps
    one = torch.ones_like(a)
    big = torch.full_like(a, _BIG)
    safe_a = torch.where(active, a, one)
    # Row r bounds u to [-c_r - k_r x, c_r - k_r x].
    k = torch.where(active, b / safe_a, torch.zeros_like(a))
    c = torch.where(active, acceleration_limits / safe_a.abs(), big)
    # A joint that does not move along the path still bounds x through |b x| <= A.
    still = (~active) & (b.abs() > eps)
    still_cap = torch.where(
        still, acceleration_limits / torch.where(still, b.abs(), one), big
    )
    moving = path_velocity.abs() > eps
    velocity_cap = torch.where(
        moving,
        velocity_limits**2
        / torch.where(moving, path_velocity, torch.ones_like(path_velocity)) ** 2,
        torch.full_like(path_velocity, _BIG),
    )
    k = k.flatten(2)
    c = c.flatten(2)
    # Every lower acceleration bound must sit below every upper one.
    dk = k.unsqueeze(-2) - k.unsqueeze(-1)
    crossing = dk > eps
    pair_cap = torch.where(
        crossing,
        (c.unsqueeze(-2) + c.unsqueeze(-1))
        / torch.where(crossing, dk, torch.ones_like(dk)),
        torch.full_like(dk, _BIG),
    )
    mvc = torch.minimum(
        torch.minimum(velocity_cap.amin(-1), still_cap.flatten(2).amin(-1)),
        pair_cap.flatten(-2).amin(-1),
    )

    xmax: list[torch.Tensor] = [None] * count  # type: ignore[list-item]
    xmax[n] = torch.zeros(batch, dtype=a.dtype, device=a.device)
    ones = torch.ones_like(k[:, 0])
    for i in range(n - 1, -1, -1):
        two = 2.0 * ds[i]
        coef = 1.0 - two * k[:, i]
        lower = coef > eps
        upper = coef < -eps
        # Maximal deceleration must still reach the next stage's set ...
        land = torch.where(
            lower,
            (xmax[i + 1].unsqueeze(-1) + two * c[:, i])
            / torch.where(lower, coef, ones),
            torch.full_like(coef, _BIG),
        )
        # ... and maximal acceleration must not be forced below zero speed.
        stay = torch.where(
            upper,
            two * c[:, i] / torch.where(upper, -coef, ones),
            torch.full_like(coef, _BIG),
        )
        cap = torch.minimum(mvc[:, i], torch.minimum(land, stay).amin(-1))
        xmax[i] = cap.clamp_min(0.0)

    x = [torch.zeros(batch, dtype=a.dtype, device=a.device)]
    u = []
    for i in range(n):
        two = 2.0 * ds[i]
        u_hi = (c[:, i] - k[:, i] * x[i].unsqueeze(-1)).amin(-1)
        u_i = torch.minimum(u_hi, (xmax[i + 1] - x[i]) / two)
        u.append(u_i)
        x.append(torch.minimum((x[i] + two * u_i).clamp_min(0.0), xmax[i + 1]))
    x_t = torch.stack(x, dim=1)
    u_t = torch.stack(u, dim=1)
    root = x_t.clamp_min(sqrt_floor).sqrt()
    segment_time = 2.0 * ds / (root[:, :-1] + root[:, 1:])
    return x_t, u_t, segment_time


class _WarpParameterization(torch.autograd.Function):
    """Bridge the Warp kernels into torch autograd through a ``wp.Tape``."""

    @classmethod
    def apply(cls, *args: Any) -> Any:  # type: ignore[override]
        """Capture the caller's grad mode, which ``forward`` cannot observe."""
        return super().apply(torch.is_grad_enabled(), *args)

    @staticmethod
    def forward(
        ctx: Any,
        grad_enabled: bool,
        path_velocity: torch.Tensor,
        path_acceleration: torch.Tensor,
        ds: torch.Tensor,
        velocity_limits: torch.Tensor,
        acceleration_limits: torch.Tensor,
        num_sets: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import warp as wp

        from ._warp.topp import build_topp_kernels

        wp.init()
        double = path_velocity.dtype == torch.float64
        scalar = wp.float64 if double else wp.float32
        batch, count, joints = path_velocity.shape
        n = count - 1
        rows, cap, sweep = build_topp_kernels(joints, num_sets, double)
        needs_grad = bool(
            grad_enabled and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2])
        )
        device = wp.device_from_torch(path_velocity.device)

        def bridged(tensor: torch.Tensor, requires_grad: bool = False) -> Any:
            return wp.from_torch(
                tensor.detach().contiguous().clone(),
                dtype=scalar,
                requires_grad=requires_grad,
            )

        def zeros(*shape: int) -> Any:
            return wp.zeros(
                shape, dtype=scalar, device=device, requires_grad=needs_grad
            )

        qs = bridged(path_velocity, needs_grad)
        qss = bridged(path_acceleration, needs_grad)
        wds = bridged(ds)
        vmax = bridged(velocity_limits)
        amax = bridged(acceleration_limits)
        k = zeros(batch, count, num_sets, joints)
        c = zeros(batch, count, num_sets, joints)
        xcap = zeros(batch, count)
        mvc = zeros(batch, count)
        xmax = zeros(batch, count)
        x = zeros(batch, count)
        u = zeros(batch, n)
        seg = zeros(batch, n)

        stream = (
            wp.ScopedStream(wp.stream_from_torch(path_velocity.device))
            if path_velocity.is_cuda
            else nullcontext()
        )
        tape = wp.Tape() if needs_grad else None
        with stream, tape if tape is not None else nullcontext():
            wp.launch(
                rows,
                dim=(batch, count),
                inputs=[qs, qss, wds, vmax, amax, n],
                outputs=[k, c, xcap],
                device=device,
            )
            wp.launch(
                cap,
                dim=(batch, count),
                inputs=[k, c, xcap],
                outputs=[mvc],
                device=device,
            )
            wp.launch(
                sweep,
                dim=batch,
                inputs=[k, c, mvc, wds, n],
                outputs=[xmax, x, u, seg],
                device=device,
            )
        outputs = tuple(wp.to_torch(array).clone() for array in (x, u, seg))
        if needs_grad:
            ctx.tape = tape
            ctx.arrays = (qs, qss, x, u, seg)
            ctx.dtype = path_velocity.dtype
            ctx.torch_device = path_velocity.device
        else:
            ctx.tape = None
        return outputs

    @staticmethod
    def backward(
        ctx: Any, *grads: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        import warp as wp

        tape = ctx.tape
        if tape is None:
            return (None,) * 7
        qs, qss, x, u, seg = ctx.arrays
        for array, grad in zip((x, u, seg), grads):
            if grad is not None:
                wp.copy(
                    array.grad,
                    wp.from_torch(grad.detach().contiguous().to(ctx.dtype)),
                )
        stream = (
            wp.ScopedStream(wp.stream_from_torch(ctx.torch_device))
            if ctx.torch_device.type == "cuda"
            else nullcontext()
        )
        with stream:
            tape.backward()
        grad_velocity = wp.to_torch(qs.grad).clone()
        grad_acceleration = wp.to_torch(qss.grad).clone()
        tape.reset()
        ctx.tape = None
        ctx.arrays = None
        return None, grad_velocity, grad_acceleration, None, None, None, None


def _resolve_backend(backend: str, joints: int) -> str:
    """Pick the Warp kernels unless the joint count exceeds their unroll limit."""
    from ._warp.topp import MAX_JOINTS

    if backend == "auto":
        return "warp" if joints <= MAX_JOINTS else "torch"
    if backend not in ("torch", "warp"):
        raise ValueError(f"backend must be 'auto', 'torch' or 'warp', got {backend!r}.")
    if backend == "warp" and joints > MAX_JOINTS:
        raise ValueError(
            f"The Warp backend supports at most {MAX_JOINTS} joints, got {joints}; "
            "use backend='torch'."
        )
    return backend


def _limit_vector(
    limit: float | torch.Tensor, joints: int, like: torch.Tensor, name: str
) -> torch.Tensor:
    """Expand a scalar or per-joint limit and check it is positive and finite."""
    vector = torch.as_tensor(limit, dtype=like.dtype, device=like.device)
    if vector.dim() == 0:
        vector = vector.expand(joints)
    if vector.shape != (joints,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({joints},), got {tuple(vector.shape)}."
        )
    if not bool(torch.isfinite(vector).all()) or bool((vector <= 0).any()):
        raise ValueError(f"{name} must be finite and positive.")
    return vector.contiguous()


def parameterize_time_optimal(
    path_velocity: torch.Tensor,
    path_acceleration: torch.Tensor,
    gridpoints: torch.Tensor,
    velocity_limits: float | torch.Tensor,
    acceleration_limits: float | torch.Tensor,
    *,
    discretization: Literal["interpolation", "collocation"] = "interpolation",
    backend: Literal["auto", "torch", "warp"] = "auto",
) -> TimeOptimalParameterization:
    """Solve the fastest rest-to-rest speed profile along a sampled path.

    Differentiable with respect to ``path_velocity`` and ``path_acceleration``
    almost everywhere: the result is piecewise smooth, and a gradient changes
    discontinuously where the binding constraint switches from one joint to
    another. At a given grid point only the binding joints receive gradient.

    Args:
        path_velocity: ``dq/ds`` at every grid point, ``(B, N+1, J)``.
        path_acceleration: ``d^2q/ds^2`` at every grid point, ``(B, N+1, J)``.
        gridpoints: Strictly increasing path coordinates, ``(N+1,)``, shared by
            the batch.
        velocity_limits: Symmetric joint velocity limit, scalar or ``(J,)``.
        acceleration_limits: Symmetric joint acceleration limit, scalar or
            ``(J,)``.
        discretization: ``interpolation`` also enforces each stage's
            acceleration rows at the next grid point, which is ``toppra``'s
            default and is conservative; ``collocation`` enforces them only at
            the stage's own point.
        backend: ``warp`` runs the two sequential sweeps as one thread per
            environment and supports up to 16 joints; ``torch`` is the
            reference implementation with no joint limit but a Python loop over
            the grid. ``auto`` selects Warp when the joint count allows it.

    Returns:
        The speed profile, segment times and duration.

    Raises:
        ValueError: For mismatched shapes, a non-increasing grid, non-positive
            limits, an unknown discretization or an unsupported backend.
    """
    if path_velocity.dim() != 3 or path_velocity.shape != path_acceleration.shape:
        raise ValueError(
            "path_velocity and path_acceleration must share shape (B, N+1, J)."
        )
    if path_velocity.dtype not in (torch.float32, torch.float64):
        raise ValueError("Path derivatives must be float32 or float64.")
    if discretization not in _DISCRETIZATION_SETS:
        raise ValueError(
            f"discretization must be 'interpolation' or 'collocation', got {discretization!r}."
        )
    batch, count, joints = path_velocity.shape
    if count < 2:
        raise ValueError("A path needs at least two grid points.")
    grid = torch.as_tensor(
        gridpoints, dtype=path_velocity.dtype, device=path_velocity.device
    )
    if grid.shape != (count,):
        raise ValueError(
            f"gridpoints must have shape ({count},), got {tuple(grid.shape)}."
        )
    ds = grid.diff()
    if bool((ds <= 0).any()):
        raise ValueError("gridpoints must be strictly increasing.")
    velocity = _limit_vector(velocity_limits, joints, path_velocity, "velocity_limits")
    acceleration = _limit_vector(
        acceleration_limits, joints, path_velocity, "acceleration_limits"
    )
    num_sets = _DISCRETIZATION_SETS[discretization]
    if _resolve_backend(backend, joints) == "warp":
        x, u, seg = _WarpParameterization.apply(
            path_velocity, path_acceleration, ds, velocity, acceleration, num_sets
        )
    else:
        x, u, seg = _parameterize_torch(
            path_velocity, path_acceleration, ds, velocity, acceleration, num_sets
        )
    return TimeOptimalParameterization(
        sdot_sq=x, sddot=u, segment_time=seg, duration=seg.sum(dim=-1)
    )


@torch.no_grad()
def _not_a_knot_matrix(
    count: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Moment system of a uniform not-a-knot cubic spline with four or more knots."""
    matrix = torch.zeros(count, count, dtype=dtype, device=device)
    matrix[0, :3] = torch.tensor([1.0, -2.0, 1.0], dtype=dtype)
    matrix[-1, -3:] = torch.tensor([1.0, -2.0, 1.0], dtype=dtype)
    rows = torch.arange(1, count - 1, device=device)
    matrix[rows, rows - 1] = 1.0
    matrix[rows, rows] = 4.0
    matrix[rows, rows + 1] = 1.0
    return matrix


def _spline_moments(knots: torch.Tensor) -> torch.Tensor:
    """Second derivatives at the knots of ``scipy``'s not-a-knot cubic spline.

    Knots sit at uniform path coordinates on ``[0, 1]``. Two knots give a line
    and three a parabola, which is where ``scipy`` special-cases not-a-knot.
    """
    count = knots.shape[1]
    h = 1.0 / (count - 1)
    if count == 2:
        return torch.zeros_like(knots)
    curvature = (knots[:, 2:] - 2.0 * knots[:, 1:-1] + knots[:, :-2]) / h**2
    if count == 3:
        return curvature.expand(-1, 3, -1).contiguous()
    rhs = torch.zeros_like(knots)
    rhs[:, 1:-1] = 6.0 * curvature
    matrix = _not_a_knot_matrix(count, knots.dtype, knots.device)
    return torch.linalg.solve(matrix, rhs)


def _evaluate_spline(
    knots: torch.Tensor, moments: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a uniform cubic spline and its first two derivatives at ``s``.

    Args:
        knots: Knot values, ``(B, M, J)``.
        moments: Second derivatives at the knots, ``(B, M, J)``.
        s: Path coordinates in ``[0, 1]``, ``(B, K)``.

    Returns:
        ``q``, ``dq/ds`` and ``d^2q/ds^2``, each ``(B, K, J)``.
    """
    count, joints = knots.shape[1], knots.shape[2]
    h = 1.0 / (count - 1)
    s = s.clamp(0.0, 1.0)
    index = torch.clamp((s.detach() / h).floor().long(), 0, count - 2)
    gather = lambda values, offset: values.gather(
        1, (index + offset).unsqueeze(-1).expand(-1, -1, joints)
    )
    y0, y1 = gather(knots, 0), gather(knots, 1)
    m0, m1 = gather(moments, 0), gather(moments, 1)
    left = ((index + 1).to(s.dtype) * h - s).unsqueeze(-1)
    right = (s - index.to(s.dtype) * h).unsqueeze(-1)
    slope0 = y0 / h - m0 * h / 6.0
    slope1 = y1 / h - m1 * h / 6.0
    q = (
        m0 * left**3 / (6.0 * h)
        + m1 * right**3 / (6.0 * h)
        + slope0 * left
        + slope1 * right
    )
    dq = -m0 * left**2 / (2.0 * h) + m1 * right**2 / (2.0 * h) - slope0 + slope1
    ddq = m0 * left / h + m1 * right / h
    return q, dq, ddq


def _distinct_waypoints(waypoints: torch.Tensor, tolerance: float) -> list[list[int]]:
    """Per row, indices that drop consecutive repeats while keeping both endpoints.

    A run of near-identical samples, such as a converged environment holding
    its pose inside a batch, is geometrically a single point. Keeping it would
    put zero-length segments into the path, where the cubic fit bends back on
    itself to pass through the repeat. Each row is compared against its last
    kept point rather than its previous sample, so slow drift is not merged away.

    Args:
        waypoints: ``(B, M, J)`` tensor; read once on the host.
        tolerance: Largest per-joint difference counted as the same point.

    Returns:
        One list of kept indices per row; a single index means a stationary row.
    """
    points = waypoints.detach().cpu().numpy()
    rows = []
    for row in points:
        kept = [0]
        for index in range(1, row.shape[0]):
            if abs(row[index] - row[kept[-1]]).max() >= tolerance:
                kept.append(index)
        if len(kept) > 1 and kept[-1] != row.shape[0] - 1:
            kept[-1] = row.shape[0] - 1
        rows.append(kept)
    return rows


def retime_time_optimal(
    waypoints: torch.Tensor,
    velocity_limits: float | torch.Tensor,
    acceleration_limits: float | torch.Tensor,
    *,
    sample_interval: float,
    gridpoints: int | None = None,
    discretization: Literal["interpolation", "collocation"] = "interpolation",
    backend: Literal["auto", "torch", "warp"] = "auto",
    duplicate_tolerance: float = 1.0e-6,
) -> TimeOptimalTrajectory:
    """Fit a path through joint waypoints and sample its fastest timing.

    Consecutive waypoints closer than ``duplicate_tolerance`` are merged, a
    not-a-knot cubic spline is fit through the rest at uniform path
    coordinates as ``toppra.SplineInterpolator`` does, the path is
    parameterized with :func:`parameterize_time_optimal`, and the result is
    sampled at ``ceil(duration / sample_interval) + 1`` evenly spaced times.

    Every output is differentiable with respect to the retained waypoints,
    including the sample times, which scale with the duration.

    Args:
        waypoints: Joint waypoints, ``(B, M, J)``.
        velocity_limits: Symmetric joint velocity limit, scalar or ``(J,)``.
        acceleration_limits: Symmetric joint acceleration limit, scalar or
            ``(J,)``.
        sample_interval: Target output period in seconds. The realized period
            is ``duration / (samples - 1)``, never longer than this.
        gridpoints: Parameterization grid points. ``None`` uses 100 per
            retained waypoint with a floor of 1000; limits are only enforced
            at grid points, so a coarser grid overshoots them more between
            points.
        discretization: See :func:`parameterize_time_optimal`.
        backend: See :func:`parameterize_time_optimal`.
        duplicate_tolerance: Largest per-joint difference at which consecutive
            waypoints count as the same point.

    Returns:
        The sampled trajectory. A row whose waypoints are all one point has
        zero duration and two identical samples.

    Raises:
        ValueError: For a malformed waypoint tensor, a non-positive sample
            interval, or any error raised by :func:`parameterize_time_optimal`.
    """
    if waypoints.dim() != 3 or waypoints.shape[1] < 1:
        raise ValueError("waypoints must have shape (B, M, J) with M >= 1.")
    if not math.isfinite(sample_interval) or sample_interval <= 0.0:
        raise ValueError("sample_interval must be finite and positive.")
    batch, _, joints = waypoints.shape
    dtype, device = waypoints.dtype, waypoints.device

    kept = _distinct_waypoints(waypoints, duplicate_tolerance)
    counts = [len(indices) for indices in kept]
    moving = [b for b in range(batch) if counts[b] >= 2]
    grid_count = gridpoints or max(
        _GRIDPOINTS_MIN, _GRIDPOINTS_PER_WAYPOINT * max(counts)
    )
    grid = torch.linspace(0.0, 1.0, int(grid_count), dtype=dtype, device=device)

    positions = waypoints[:, :1].expand(batch, 2, joints).contiguous()
    velocities = torch.zeros_like(positions)
    accelerations = torch.zeros_like(positions)
    dt = torch.zeros(batch, 2, dtype=dtype, device=device)
    duration = torch.zeros(batch, dtype=dtype, device=device)
    if moving:
        # Rows that share a waypoint count share one spline solve.
        groups: dict[int, list[int]] = {}
        for row, b in enumerate(moving):
            groups.setdefault(counts[b], []).append(row)
        fitted = {}
        path_velocity = grid.new_zeros(len(moving), grid.shape[0], joints)
        path_acceleration = torch.zeros_like(path_velocity)
        for count, rows in groups.items():
            index = torch.tensor(rows, device=device)
            sources = torch.tensor([moving[row] for row in rows], device=device)
            picks = torch.tensor([kept[moving[row]] for row in rows], device=device)
            knots = waypoints[sources.unsqueeze(-1), picks]
            moments = _spline_moments(knots)
            _, dq, ddq = _evaluate_spline(knots, moments, grid.expand(len(rows), -1))
            path_velocity = path_velocity.index_copy(0, index, dq)
            path_acceleration = path_acceleration.index_copy(0, index, ddq)
            fitted[count] = (index, knots, moments)

        profile = parameterize_time_optimal(
            path_velocity,
            path_acceleration,
            grid,
            velocity_limits,
            acceleration_limits,
            discretization=discretization,
            backend=backend,
        )
        total = profile.duration
        steps = [
            (
                max(2, math.ceil(value / sample_interval) + 1)
                if math.isfinite(value)
                else 2
            )
            for value in total.detach().tolist()
        ]
        length = max(steps)
        steps_t = torch.tensor(steps, device=device).unsqueeze(-1)
        sample = torch.arange(length, device=device).unsqueeze(0)
        real = sample < steps_t
        fraction = (sample.to(dtype) / (steps_t - 1).to(dtype)).clamp(max=1.0)
        times = fraction * total.unsqueeze(-1)
        elapsed = torch.cat(
            [total.new_zeros(len(moving), 1), profile.segment_time.cumsum(-1)], dim=-1
        )
        segment = torch.clamp(
            torch.searchsorted(
                elapsed.detach().contiguous(), times.detach().contiguous(), right=True
            )
            - 1,
            0,
            profile.segment_time.shape[1] - 1,
        )
        tau = times - elapsed.gather(1, segment)
        speed0 = profile.sdot_sq.gather(1, segment).clamp_min(_eps(dtype)[1]).sqrt()
        accel = profile.sddot.gather(1, segment)
        s = grid[segment] + speed0 * tau + 0.5 * accel * tau**2
        speed = (speed0 + accel * tau).clamp_min(0.0)

        q_all = grid.new_zeros(len(moving), length, joints)
        dq_all = torch.zeros_like(q_all)
        ddq_all = torch.zeros_like(q_all)
        for index, knots, moments in fitted.values():
            q, dq, ddq = _evaluate_spline(knots, moments, s[index])
            q_all = q_all.index_copy(0, index, q)
            dq_all = dq_all.index_copy(0, index, dq)
            ddq_all = ddq_all.index_copy(0, index, ddq)
        # Padding samples hold the final pose, which the clamped time grid
        # already evaluates to; only their derivatives need zeroing.
        mask = real.unsqueeze(-1)
        vel_m = torch.where(
            mask, dq_all * speed.unsqueeze(-1), torch.zeros_like(dq_all)
        )
        acc_m = torch.where(
            mask,
            dq_all * accel.unsqueeze(-1) + ddq_all * speed.unsqueeze(-1) ** 2,
            torch.zeros_like(dq_all),
        )
        interval = total.unsqueeze(-1) / (steps_t - 1).to(dtype)
        dt_m = torch.where(
            real & (sample > 0), interval.expand(-1, length), torch.zeros_like(times)
        )

        index = torch.tensor(moving, device=device)
        positions = waypoints[:, :1].expand(batch, length, joints).contiguous()
        positions = positions.index_copy(0, index, q_all)
        velocities = torch.zeros_like(positions).index_copy(0, index, vel_m)
        accelerations = torch.zeros_like(positions).index_copy(0, index, acc_m)
        dt = positions.new_zeros(batch, length).index_copy(0, index, dt_m)
        duration = duration.index_copy(0, index, total)

    success = (
        torch.isfinite(duration)
        & torch.isfinite(positions).flatten(1).all(-1)
        & torch.isfinite(velocities).flatten(1).all(-1)
        & torch.isfinite(accelerations).flatten(1).all(-1)
    )
    return TimeOptimalTrajectory(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        dt=dt,
        duration=duration,
        success=success,
    )
