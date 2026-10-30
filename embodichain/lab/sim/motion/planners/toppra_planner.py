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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import field

import torch

from embodichain.compute.trajectory._toppra_common import _validate_grid_size
from embodichain.compute.trajectory._toppra_warp import _retime_toppra_warp
from embodichain.compute.trajectory._toppra_stopped import _retime_toppra_stopped
from embodichain.utils import logger, configclass
from .base_planner import (
    validate_plan_options,
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    _infer_batch_size,
)
from ._joint_limits import _resolve_joint_constraints
from .utils import MoveType, TrajectorySampleMethod, PlanState, PlanResult

__all__ = [
    "ToppraPlanner",
    "ToppraPlannerCfg",
    "ToppraPlanOptions",
    "retime_joint_paths",
]


def _failed_result(
    batch: int, dofs: int, device: torch.device, dtype: torch.dtype
) -> PlanResult:
    """Return independent zero-filled rows in the caller's floating precision."""
    positions = torch.zeros((batch, 2, dofs), dtype=dtype, device=device)
    return PlanResult(
        success=torch.zeros(batch, dtype=torch.bool, device=device),
        positions=positions,
        velocities=torch.zeros_like(positions),
        accelerations=torch.zeros_like(positions),
        dt=torch.zeros((batch, 2), dtype=dtype, device=device),
    )


def retime_joint_paths(
    positions: torch.Tensor,
    *,
    constraints: Mapping[str, float | Sequence[float] | torch.Tensor],
    path_lengths: torch.Tensor | None = None,
    sample_method: TrajectorySampleMethod = TrajectorySampleMethod.QUANTITY,
    sample_interval: float | int | None = None,
    grid_size: int = 100,
    device: torch.device | None = None,
) -> PlanResult:
    """Re-parameterize already-planned joint paths under dynamic limits.

    This is the time parameterization of :class:`ToppraPlanner`, exposed for
    planners that produce geometry without executable timing. A closed-loop
    policy rollout, for example, emits one joint sample per control step on a
    nominal ``dt`` that no dynamic limit informed; feeding those samples here
    replaces that timing with a solved one that respects the limits.

    Warp handles both CPU and CUDA inputs. Sufficiently large CPU inference
    batches share a bounded thread pool; small requests and gradient recording
    run inline. Interval constraints enforce limits between grid points.
    Floating outputs use float64 so finite differences of dense position samples
    do not amplify float32 quantization into spurious dynamic-limit violations.
    Input filtering retains float32/float64 input precision; other input dtypes
    are promoted to float32 before solving. Consumers that narrow or resample
    these positions must recheck the resulting trajectory's dynamic limits.

    The path geometry is resampled along the fitted spline, so poses derived
    from the input samples must be recomputed from the returned positions.

    Args:
        positions: Planned joint paths of shape ``(B, N, DOF)``.
        constraints: ``velocity`` and ``acceleration`` limits, each a scalar,
            one value per joint, or signed ``(DOF, 2)`` bounds, matching
            :class:`ToppraPlanOptions`.
        path_lengths: Optional valid prefix length for each batch row. This
            excludes rectangular tail padding from spline fitting while
            preserving explicit repeated endpoints inside each prefix.
        sample_method: Fixed output quantity or approximately fixed time step.
        sample_interval: Output count for ``QUANTITY`` or seconds for ``TIME``.
            ``None`` keeps the input sample count and is only meaningful for
            ``QUANTITY``, which leaves array shapes unchanged while the timing
            behind them changes.
        grid_size: Minimum parameterization grid size, as for
            :class:`ToppraPlannerCfg`.
        device: Device for the assembled tensors; defaults to the input's.

    Returns:
        PlanResult whose ``success`` is per-env and false where the path could
        not be parameterized.

    Raises:
        ValueError: If ``positions`` is not a ``(B, N, DOF)`` tensor,
            ``path_lengths`` is malformed, the constraints omit ``velocity``
            or ``acceleration``, or ``TIME`` sampling is requested without an
            explicit interval in seconds.
    """
    if positions.dim() != 3:
        raise ValueError(
            f"positions must have shape (B, N, DOF), got {tuple(positions.shape)}."
        )
    missing = sorted({"velocity", "acceleration"}.difference(constraints))
    if missing:
        raise ValueError(f"constraints is missing required keys: {missing}.")
    if sample_interval is None and sample_method is TrajectorySampleMethod.TIME:
        # The default is a sample count; reading it as seconds would silently
        # collapse a dense path to two samples whenever its duration is shorter.
        raise ValueError(
            "sample_interval is required for TIME sampling; it is a duration in "
            "seconds, not the sample count the default supplies."
        )
    resolved_device = positions.device if device is None else device
    batch, count, dofs = positions.shape
    if path_lengths is not None:
        if not isinstance(path_lengths, torch.Tensor) or path_lengths.shape != (batch,):
            raise ValueError(f"path_lengths must have shape ({batch},)")
        if path_lengths.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("path_lengths must contain integers")
        path_lengths = path_lengths.to(device=positions.device, dtype=torch.int32)
        if bool(((path_lengths < 2) | (path_lengths > count)).any().item()):
            raise ValueError(f"path_lengths must be between 2 and {count}")
        path_lengths = path_lengths.contiguous()
    resolved_interval = count if sample_interval is None else sample_interval
    velocity, acceleration = constraints["velocity"], constraints["acceleration"]

    quantity = sample_method == TrajectorySampleMethod.QUANTITY
    waypoints = (
        positions
        if positions.dtype in (torch.float32, torch.float64)
        else positions.float()
    )
    try:
        result = _retime_toppra_warp(
            waypoints,
            velocity,
            acceleration,
            path_lengths=path_lengths,
            sample_count=int(resolved_interval) if quantity else None,
            sample_dt=None if quantity else float(resolved_interval),
            grid_size=grid_size,
            output_dtype=torch.float64,
        )
    except (ValueError, FloatingPointError, OverflowError):
        return _failed_result(batch, dofs, resolved_device, torch.float64)
    return PlanResult(
        **{key: value.to(resolved_device) for key, value in result.items()}
    )


@configclass
class ToppraPlannerCfg(BasePlannerCfg):
    """Configure the shared CPU/CUDA Warp time parameterizer."""

    planner_type: str = "toppra"
    grid_size: int = field(default=100, kw_only=True)
    """Minimum timing grid size; each spline segment is subdivided uniformly."""


class _ToppraConstraints(dict):
    """Copy config containers while preserving runtime tensor references."""

    def __deepcopy__(self, memo: dict) -> _ToppraConstraints:
        result = _ToppraConstraints()
        memo[id(self)] = result
        for key, value in self.items():
            result[deepcopy(key, memo)] = (
                value if isinstance(value, torch.Tensor) else deepcopy(value, memo)
            )
        return result


@configclass
class ToppraPlanOptions(PlanOptions):

    control_part: str | None = field(default=None, kw_only=True)
    """Control group whose joint order matches the trajectory columns."""

    constraints: dict = {
        "velocity": None,
        "acceleration": None,
    }
    """Constraints for the planner, including velocity and acceleration limits.

    Values may be scalars, per-joint magnitudes, or signed bounds. ``None``
    velocity resolves robot speed caps at planning time. Acceleration is a
    planning budget; ``None`` uses its compatibility default. Explicit speed
    requests are capped by available robot limits.
    Tensor bounds retain their references through option construction/copying
    so gradients reach the caller, including through non-leaf expressions.
    """

    sample_method: TrajectorySampleMethod = TrajectorySampleMethod.TIME
    """Method for sampling the trajectory.

    Options are 'time' for uniform time intervals or 'quantity' for a fixed number of samples.
    """

    sample_interval: float | int = 0.01
    """Interval for sampling the trajectory.

    If sample_method is 'time', this is the time interval in seconds.
    If sample_method is 'quantity', this is the total number of samples.
    Both modes support first-order gradients of waypoints and tensor motion limits.
    """

    stop_at_waypoints: bool = False
    """Require rest at every supplied waypoint on the TIME control grid.

    Available only with TIME sampling. Each adjacent pair is solved separately,
    and its duration is conservatively rounded up to whole sample intervals.
    Shared endpoints are retained exactly, with zero velocity, so a caller may
    insert a stationary hand command there. The default preserves the existing
    continuous spline and upper-bound time spacing.
    """

    def __post_init__(self) -> None:
        """Preserve tensor graphs and validate the optional stopped path mode."""
        if type(self.stop_at_waypoints) is not bool:
            raise TypeError("stop_at_waypoints must be boolean.")
        if self.stop_at_waypoints and self.sample_method != TrajectorySampleMethod.TIME:
            raise ValueError("stop_at_waypoints requires TIME sampling.")
        self.constraints = _ToppraConstraints(self.constraints)


class ToppraPlanner(BasePlanner):
    """Rest-to-rest joint-path timing with differentiable CPU/CUDA Warp kernels.

    Uniform not-a-knot cubic geometry is timed under conservative interval
    bounds. Results preserve the planner's float32 output contract.
    """

    supported_move_types = frozenset({MoveType.JOINT_MOVE})
    preserve_plan_samples = True
    """Keep the solved native time grid through MotionGenerator normalization.

    A facade-level linear resampling can invalidate the solved derivatives.
    Backend-neutral counts still configure the planner when no explicit native
    options are supplied; explicit native sampling options remain authoritative.
    """

    def __init__(self, cfg: ToppraPlannerCfg) -> None:
        """Validate the timing grid and resolve the configured robot."""
        _validate_grid_size(cfg.grid_size)
        super().__init__(cfg)

    def default_plan_options(self) -> ToppraPlanOptions:
        """Return a valid time-sampled default when no count is requested."""
        return ToppraPlanOptions()

    def close(self) -> None:
        """Complete the planner lifecycle; CPU workers belong to the shared runtime."""

    @validate_plan_options(options_cls=ToppraPlanOptions)
    def plan(
        self,
        target_states: list[PlanState],
        options: ToppraPlanOptions = ToppraPlanOptions(),
    ) -> PlanResult:
        r"""Execute trajectory planning.

        Args:
            target_states: list of :class:`PlanState` waypoints. Tensor fields
                carry a leading batch dim ``B``: ``qpos`` is ``(B, DOF)``.
            options: :class:`ToppraPlanOptions` with constraints and sampling.

        Returns:
            PlanResult containing the planned trajectory details. All tensor
            fields are env-batched with leading dim ``B``: ``success`` ``(B,)``,
            ``positions``/``velocities``/``accelerations`` ``(B, N, DOF)``,
            ``dt`` ``(B, N)``, ``duration`` ``(B,)``.
        """
        for i, t in enumerate(target_states):
            if t.qpos is None:
                logger.log_error(f"Target state at index {i} missing qpos", ValueError)

        b = _infer_batch_size(target_states) or 1
        dofs = target_states[0].qpos.shape[-1]

        constraints = _resolve_joint_constraints(
            self.robot,
            options.constraints,
            control_part=options.control_part,
            dof=dofs,
            batch_size=b,
        )

        waypoints_tensor = torch.stack([s.qpos for s in target_states], dim=1).to(
            self.device
        )
        if waypoints_tensor.dtype not in (torch.float32, torch.float64):
            waypoints_tensor = waypoints_tensor.to(torch.float64)
        try:
            if options.stop_at_waypoints:
                result = _retime_toppra_stopped(
                    waypoints_tensor,
                    constraints["velocity"],
                    constraints["acceleration"],
                    sample_dt=float(options.sample_interval),
                    grid_size=self.cfg.grid_size,
                )
            else:
                result = _retime_toppra_warp(
                    waypoints_tensor,
                    constraints["velocity"],
                    constraints["acceleration"],
                    sample_count=(
                        int(options.sample_interval)
                        if options.sample_method == TrajectorySampleMethod.QUANTITY
                        else None
                    ),
                    sample_dt=(
                        options.sample_interval
                        if options.sample_method == TrajectorySampleMethod.TIME
                        else None
                    ),
                    grid_size=self.cfg.grid_size,
                    output_dtype=torch.float64,
                )
        except (ValueError, FloatingPointError, OverflowError):
            return _failed_result(b, dofs, self.device, torch.float64)
        return PlanResult(**result)
