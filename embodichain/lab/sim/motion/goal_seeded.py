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

"""Goal-conditioned common-seed IK for an explicit ordered waypoint set."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING
from collections.abc import Callable
import torch
from embodichain.compute.geometry.pose_checks import sampled_pose_path_matches
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix
from .solvers.tolerances import pose_validation_tolerances
from .search import record_search
from .limits import authored_joint_velocity_limits
from embodichain.compute.kinematics.ik_path import resolve_seeded_ik_path

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "qualify_ik_waypoints",
    "solve_goal_seeded_waypoints",
    "solve_seeded_pose_path",
    "solve_bounded_endpoint",
]


def qualify_ik_waypoints(
    robot: Robot,
    part: str,
    q: torch.Tensor,
    desired: torch.Tensor,
    ok: torch.Tensor,
    bounds: torch.Tensor,
) -> dict[str, Any]:
    """Compare per-waypoint IK, limits and FK with the source residual law.

    Args:
        robot: Stateful FK/solver owner.
        part: Explicit control group.
        q: Candidate joint values, one pose or a B1 waypoint sequence.
        desired: Corresponding rigid end-effector target transforms.
        ok: Original solver convergence flags, with per-waypoint cardinality.
        bounds: Matching joint limits with trailing lower/upper values.

    Returns:
        Owned numeric qualification diagnostics. This does not establish path
        collision freedom or physical task effects.
    """
    if q.shape != (*desired.shape[:-2], bounds.shape[1]):
        raise ValueError("IK joint shape differs from required waypoints.")
    finite = bool(torch.isfinite(q).all())
    selected = bounds if q.ndim == 2 else bounds[:, None]
    bounded = finite and bool(((q >= selected[..., 0]) & (q <= selected[..., 1])).all())
    record = {
        "ik_ok": bool(
            torch.as_tensor(ok).numel() == math.prod(desired.shape[:-2])
            and torch.as_tensor(ok).all()
        ),
        "finite": finite,
        "in_limits": bounded,
        "fk_valid": False,
        "desired_eef": desired.detach().cpu().tolist(),
    }
    if finite:
        fk = (
            robot.compute_fk(qpos=q, name=part, to_matrix=True)
            if q.ndim == 2
            else robot.compute_batch_fk(qpos=q, name=part, to_matrix=True)
        )
        record["fk_valid"] = sampled_pose_path_matches(
            fk, desired, *pose_validation_tolerances(robot.get_solver(part))
        )
        if torch.isfinite(fk).all():
            record["position_error_m"] = (
                torch.linalg.vector_norm(fk[..., :3, 3] - desired[..., :3, 3], dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
            record["rotation_error_rad"] = (
                quat_error_magnitude(
                    quat_from_matrix(fk[..., :3, :3]),
                    quat_from_matrix(desired[..., :3, :3]),
                )
                .detach()
                .cpu()
                .tolist()
            )
    record["valid"] = all(
        record[k] for k in ("ik_ok", "finite", "in_limits", "fk_valid")
    )
    return record


def solve_goal_seeded_waypoints(
    robot: Robot,
    control_part: str,
    poses: torch.Tensor,
    start_qpos: torch.Tensor,
    *,
    goal_index: int,
    qualified_goal_seed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Solve one distinguished goal, then all mandatory poses from that same seed.

    Args:
        robot: Stateful solver/FK owner. No action or Task Program enters.
        control_part: Explicit controlled group.
        poses: Unchanged ordered B1 rigid targets ``(1,T,4,4)``.
        start_qpos: Actual initial arm state ``(1,D)``.
        goal_index: The caller-owned distinguished target in the supplied sequence.
        qualified_goal_seed: Optional existing endpoint-search branch. It is
            requalified against the exact goal, never substituted without checks.

    Returns:
        B1 validity, owned joint targets and source-compatible diagnostics.
        A failed solve returns a repeated start hold with false validity. The
        caller owns its complete recovery/RNG scope; no retry occurs here.
    """
    if (
        start_qpos.ndim != 2
        or start_qpos.shape[0] != 1
        or not start_qpos.shape[1]
        or poses.ndim != 4
        or poses.shape[0] != 1
        or poses.shape[-2:] != (4, 4)
        or type(goal_index) is not int
        or not 0 <= goal_index < poses.shape[1]
        or not torch.isfinite(start_qpos).all()
        or not torch.isfinite(poses).all()
    ):
        raise ValueError(
            "Goal-seeded IK requires an explicit B1 start and finite required poses."
        )
    bounds = robot.get_qpos_limits(name=control_part).to(start_qpos)
    if (
        bounds.shape != (*start_qpos.shape, 2)
        or not torch.isfinite(bounds).all()
        or (bounds[..., 0] > bounds[..., 1]).any()
    ):
        raise ValueError("Goal-seeded IK requires original finite joint bounds.")
    goal = poses[:, goal_index].to(start_qpos)
    if qualified_goal_seed is None:
        record_search("goal_seed_ik", budget_checkpoint=False)
        ok, qgoal = robot.compute_ik(
            pose=goal, name=control_part, joint_seed=start_qpos
        )
    else:
        qgoal = qualified_goal_seed
        ok = torch.ones(qgoal.shape[:-1], dtype=torch.bool, device=qgoal.device)
    qgoal = torch.as_tensor(qgoal).to(start_qpos)
    receipt = {
        "release_seed": qualify_ik_waypoints(
            robot, control_part, qgoal, goal, ok, bounds
        )
    }
    if receipt["release_seed"]["valid"]:
        record_search("goal_seeded_waypoint_batches", budget_checkpoint=False)
        record_search(
            "goal_seeded_waypoints", count=poses.shape[1], budget_checkpoint=False
        )
        ok, qposes = robot.compute_batch_ik(
            pose=poses,
            name=control_part,
            joint_seed=qgoal[:, None].expand(-1, poses.shape[1], -1).clone(),
        )
        qposes = torch.as_tensor(qposes).to(start_qpos)
        receipt["waypoints"] = qualify_ik_waypoints(
            robot, control_part, qposes, poses, ok, bounds
        )
        if receipt["waypoints"]["valid"]:
            return (
                torch.ones(1, dtype=torch.bool, device=start_qpos.device),
                qposes.clone(),
                receipt,
            )
    receipt["reason"] = "required_waypoint_ik_fk_limits_rejected"
    return (
        torch.zeros(1, dtype=torch.bool, device=start_qpos.device),
        start_qpos[:, None].expand(-1, poses.shape[1], -1).clone(),
        receipt,
    )


def solve_seeded_pose_path(
    robot: Robot,
    control_part: str,
    poses: torch.Tensor,
    start_qpos: torch.Tensor,
    *,
    control_dt: float,
) -> torch.Tensor | None:
    """Follow fixed B1 poses with direct Robot IK and bounded joint-limit search.

    Args:
        robot: Stateful solver and original joint-limit owner.
        control_part: Explicit active group.
        poses: Nonempty ordered world poses ``(T,4,4)``.
        start_qpos: Exact initial controlled joints ``(1,D)``.
        control_dt: Positive fixed sample interval, never retimed here.

    Returns:
        Joint path including the start with shape ``(T+1,D)``, or ``None``
        after bounded failure. The caller owns RNG scope and final FK/contact
        qualification; this adapter deliberately adds no random-state rewind.
    """
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or poses.shape[0] == 0
        or start_qpos.ndim != 2
        or start_qpos.shape[0] != 1
        or not torch.isfinite(poses).all()
        or not torch.isfinite(start_qpos).all()
    ):
        raise ValueError(
            "Seeded pose paths require finite nonempty poses and a B1 start."
        )
    bounds = robot.get_qpos_limits(name=control_part).to(start_qpos)
    return resolve_seeded_ik_path(
        poses,
        start_qpos,
        solve=lambda target, seed: robot.compute_ik(
            pose=target, name=control_part, joint_seed=seed
        ),
        lower=bounds[..., 0],
        upper=bounds[..., 1],
        limits=authored_joint_velocity_limits(robot, control_part).to(start_qpos),
        dt=control_dt,
        record_operation=lambda name, count: record_search(name, count=count),
    )


def solve_bounded_endpoint(
    robot: Robot,
    control_part: str,
    target: torch.Tensor,
    seed: torch.Tensor,
    *,
    env_ids: torch.Tensor,
    record_operation: Callable[[str, int], None] | None = None,
) -> tuple[bool, torch.Tensor, dict[str, Any]]:
    """Solve one B1 endpoint against original runtime/solver limit intersection.

    This preserves source convergence, finite and bound checks; it is neither a
    trajectory nor physical certificate. The optional measurement callback fires
    only immediately before the IK call, after validating the original limits.
    Caller-owned random state and retries are unchanged.

    Args:
        robot: Borrowed Robot exposing the selected runtime and solver limits.
        control_part: Explicit motion group to query.
        target: Required end-effector pose with shape ``(1,4,4)``.
        seed: Original joint seed with shape ``(1,D)``.
        env_ids: The supported explicit environment row ``[0]``.
        record_operation: Optional caller measurement hook before the IK query.

    Returns:
        Acceptance, an independently owned solver candidate and diagnostic
        flags. A rejected candidate remains rejected; it is not a safe-hold
        trajectory or a physical-effect result.
    """
    if (
        seed.ndim != 2
        or seed.shape[0] != 1
        or target.shape != (1, 4, 4)
        or env_ids.tolist() != [0]
    ):
        raise ValueError("Bounded endpoint IK supports only explicit B1/env0 targets.")
    runtime = robot.get_qpos_limits(name=control_part, env_ids=env_ids).to(seed)
    solver = robot.get_solver(control_part)
    lower = solver.lower_qpos_limits.to(seed)
    upper = solver.upper_qpos_limits.to(seed)
    if (
        runtime.shape != (*seed.shape, 2)
        or lower.shape != seed.shape[1:]
        or upper.shape != seed.shape[1:]
        or not torch.isfinite(runtime).all()
        or not torch.isfinite(lower).all()
        or not torch.isfinite(upper).all()
    ):
        raise ValueError("handover_exchange_joint_limits_unavailable")
    lower = torch.maximum(runtime[..., 0], lower)
    upper = torch.minimum(runtime[..., 1], upper)
    if (lower > upper).any():
        raise ValueError("handover_exchange_joint_limits_inconsistent")
    if record_operation is not None:
        record_operation("ik_queries", 1)
    record_search("bounded_endpoint_ik", budget_checkpoint=False)
    success, qpos = robot.compute_ik(
        pose=target,
        joint_seed=seed.detach().clone(),
        name=control_part,
        env_ids=env_ids,
    )
    if (
        not isinstance(success, torch.Tensor)
        or success.dtype is not torch.bool
        or success.shape != (1,)
        or not isinstance(qpos, torch.Tensor)
        or qpos.shape != seed.shape
    ):
        raise ValueError("handover_exchange_endpoint_ik_shape_unavailable")
    finite = bool(torch.isfinite(qpos).all())
    within = finite and bool(((qpos >= lower) & (qpos <= upper)).all())
    accepted = bool(success.all()) and within
    return (
        accepted,
        qpos.detach().clone(),
        {
            "control_part": control_part,
            "solver_success": bool(success.all()),
            "finite": finite,
            "within_runtime_and_solver_limits": within,
            "accepted": accepted,
        },
    )
