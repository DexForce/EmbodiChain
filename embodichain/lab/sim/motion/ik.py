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

"""Stateful adapters for explicit Cartesian samples and continuous IK paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import math

import torch

from embodichain.compute.kinematics.ik_path import resolve_seeded_ik_path
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.utils import logger

from .limits import authored_joint_velocity_limits
from .motion_generator import MotionGenerator, MotionGenOptions
from .planners.utils import MoveType, PlanResult, PlanState, interpolate_xpos_batched
from .search import record_search

__all__ = ["resolve_continuous_ik_path", "sample_cartesian_targets"]


def resolve_continuous_ik_path(
    generator: MotionGenerator,
    targets: list[PlanState],
    options: MotionGenOptions,
    *,
    candidate_predicate: Callable[[torch.Tensor], bool] | None = None,
    operation: str = "Cartesian path",
) -> PlanResult | None:
    """Resolve one unchanged-time Cartesian path on a continuous joint branch.

    The previous accepted joint posture seeds the next IK query. Bounded seed
    perturbations handle joint-limit margins, while authored/runtime velocity
    checks retain the original time grid. The caller's random stream is restored.

    Args:
        generator: Stateful owner of the robot and solver integration.
        targets: Required end-effector poses; their order is preserved.
        options: Explicit start posture, control part and positive interval.
        candidate_predicate: Optional serial-candidate qualification callback.
        operation: Diagnostic name owned by the calling skill or service.

    Returns:
        A timed ``(1, T + 1, D)`` plan, or ``None`` when bounded search fails.
        Failure does not establish that the targets are globally unreachable.

    Raises:
        ValueError: If the request is not a single path with explicit timing,
            start posture, controlled part and end-effector pose targets.
    """
    if (
        options.start_qpos is None
        or options.start_qpos.ndim != 2
        or options.start_qpos.shape[0] != 1
        or options.control_part is None
        or options.interpolation_dt is None
        or not math.isfinite(options.interpolation_dt)
        or options.interpolation_dt <= 0
        or not targets
        or any(t.move_type is not MoveType.EEF_MOVE or t.xpos is None for t in targets)
    ):
        raise ValueError("Continuous IK requires one explicit timed Cartesian path.")
    robot = generator.robot
    seed = options.start_qpos.clone()
    bounds = robot.get_qpos_limits(name=options.control_part).to(seed)
    lower, upper = bounds[..., 0], bounds[..., 1]
    limits = authored_joint_velocity_limits(robot, options.control_part).to(seed)
    step_dt = options.interpolation_dt

    def solve(
        target: torch.Tensor, trial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = MotionGenerator.generate(
            generator,
            [PlanState(move_type=MoveType.EEF_MOVE, xpos=target)],
            options=replace(
                options,
                start_qpos=trial,
                sample_count=2,
                preserve_cartesian_samples=True,
            ),
        )
        return result.success, result.positions[:, -1]

    # Failed searches must not perturb the later grasp-candidate random stream.
    devices = [seed.device.index] if seed.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        poses = torch.stack([target.xpos.reshape(4, 4) for target in targets])
        points = resolve_seeded_ik_path(
            poses,
            seed,
            solve=solve,
            lower=lower,
            upper=upper,
            limits=limits,
            dt=step_dt,
            candidate_predicate=candidate_predicate,
            record_operation=lambda name, count: record_search(name, count=count),
        )
    if points is None:
        return None
    path = points.unsqueeze(0)
    dt = path.new_full(path.shape[:2], step_dt)
    dt[:, 0] = 0.0
    success = joint_velocity_validity(path, dt, limits)
    if not success.all():
        return None
    logger.log_info(
        f"{operation} continuous IK recovery accepted: "
        f"{path.shape[1]} unchanged-time samples."
    )
    return PlanResult(success=success, positions=path, dt=dt)


def sample_cartesian_targets(
    start: torch.Tensor,
    targets: list[PlanState],
    count: int,
) -> list[PlanState]:
    """Allocate a fixed sample budget across all Cartesian target intervals.

    Args:
        start: Initial poses, shape ``(B, 4, 4)``.
        targets: Required target poses, with one sample reserved for each.
        count: Total budget including the start sample.

    Returns:
        ``count - 1`` target states, excluding the start and including every
        requested waypoint in order. No IK or feasibility check is performed.

    Raises:
        ValueError: If the budget cannot include all targets or shapes disagree.
    """
    if start.ndim != 3 or start.shape[1:] != (4, 4):
        raise ValueError("Cartesian sampling start must have shape (B, 4, 4).")
    if any(
        target.move_type is not MoveType.EEF_MOVE or target.xpos is None
        for target in targets
    ):
        raise ValueError("Cartesian sampling requires end-effector pose targets.")
    if count - 1 < len(targets):
        raise ValueError("Cartesian approach sampling needs one sample per target.")
    result: list[PlanState] = []
    previous = start
    remaining = count - 1
    for index, target in enumerate(targets):
        assert target.xpos is not None
        pose = target.xpos.to(device=start.device, dtype=start.dtype)
        if pose.ndim == 2:
            pose = pose.unsqueeze(0).expand(start.shape[0], -1, -1)
        if pose.shape != start.shape:
            raise ValueError("Cartesian approach targets must match the start batch.")
        intervals = remaining // (len(targets) - index)
        interpolated = interpolate_xpos_batched(previous, pose, intervals + 1)
        result.extend(
            PlanState(move_type=MoveType.EEF_MOVE, xpos=interpolated[:, point])
            for point in range(1, intervals + 1)
        )
        previous = pose
        remaining -= intervals
    return result
