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

"""Internal sampled Cartesian geometry, scalar timing and robot IK adaptation.

This adapter reports IK and path derivatives, not collision freedom or joint
velocity/acceleration compliance. The caller owns geometry and timing limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from embodichain.compute.kinematics._path_derivatives import (
    joint_derivatives_from_path_time_law,
    nearest_equivalent_joint_solution,
)
from .bezier import BezierPath
from .trapezoidal_planner import TrapezoidalPlanOptions, _plan_linear_profiles
from .utils import PlanResult, TrajectorySampleMethod

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__: list[str] = []


def plan_cartesian_path(
    robot: Robot,
    control_part: str,
    start_qpos: torch.Tensor,
    start_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    *,
    options: TrapezoidalPlanOptions,
    path: BezierPath | None = None,
) -> tuple[PlanResult, torch.Tensor, PlanResult]:
    """Time a fixed-orientation line or Bezier translation before batched IK.

    Args:
        robot: Robot providing IK, root-frame Jacobians and joint limits.
        control_part: Selected arm whose joints match ``start_qpos``.
        start_qpos: Initial joint seeds shaped ``(B, D)``.
        start_pose: Initial arena-frame transforms shaped ``(B, 4, 4)``.
        goal_pose: Final transforms with the same orientation as ``start_pose``.
        options: Scalar distance timing with at least three quantity samples.
        path: Optional translation path with matching endpoint positions.

    Returns:
        Joint trajectory, desired pose samples and scalar distance trajectory.
    """
    if (
        options.sample_method is not TrajectorySampleMethod.QUANTITY
        or options.sample_interval < 3
    ):
        raise ValueError("Cartesian planning requires QUANTITY sample_count >= 3.")
    count = int(options.sample_interval)
    batch = start_qpos.shape[0]
    if start_pose.shape != (batch, 4, 4) or goal_pose.shape != start_pose.shape:
        raise ValueError("Cartesian endpoints must have shape (B, 4, 4).")
    if not torch.allclose(start_pose[:, :3, :3], goal_pose[:, :3, :3]):
        raise ValueError("Cartesian profiles require fixed endpoint orientation.")
    if path is not None:
        endpoints = path.evaluate(start_qpos.new_tensor([0.0, 1.0]).expand(batch, -1))
        expected = torch.stack((start_pose[:, :3, 3], goal_pose[:, :3, 3]), dim=1)
        if endpoints.shape != expected.shape or not torch.allclose(endpoints, expected):
            raise ValueError("Bezier path endpoints must match the Cartesian poses.")
    delta = goal_pose[:, :3, 3] - start_pose[:, :3, 3]
    distance = torch.linalg.vector_norm(delta, dim=-1) if path is None else path.length
    if not bool(torch.isfinite(distance).all()) or bool((distance <= 0).any()):
        raise ValueError("Cartesian path length must be finite and greater than zero.")
    scalar_waypoints = start_qpos.new_zeros((batch, 2, 1))
    scalar_waypoints[:, 1, 0] = distance
    scalar = _plan_linear_profiles(scalar_waypoints, options)
    desired = start_pose[:, None].expand(-1, count, -1, -1).clone()
    if path is None:
        progress = scalar.positions[..., 0] / distance[:, None]
        desired[:, :, :3, 3] = torch.lerp(
            start_pose[:, None, :3, 3], goal_pose[:, None, :3, 3], progress[..., None]
        )
        world_tangent = (delta / distance[:, None])[:, None].expand(-1, count, -1)
        world_curvature = None
    else:
        parameter = path.parameter_at_arc_length(scalar.positions[..., 0])
        desired[:, :, :3, 3] = path.evaluate(parameter)
        world_tangent = path.arc_tangent(parameter)
        world_curvature = path.arc_curvature(parameter)
    ik = robot.compute_batch_ik(desired, start_qpos, control_part, continuous=True)
    if ik is None:
        raise RuntimeError("Cartesian path IK solver is unavailable.")
    sample_success, positions = ik
    if not bool(sample_success.all()):
        failure = torch.nonzero(~sample_success, as_tuple=False)[0].tolist()
        raise RuntimeError(
            f"Cartesian path IK failed at env {failure[0]}, sample {failure[1]}."
        )
    if path is None:
        joint_ids = robot.get_joint_ids(name=control_part)
        limits = robot.get_qpos_limits(joint_ids=joint_ids).to(start_qpos)
        previous = torch.cat((start_qpos[:, None], positions[:, :-1]), dim=1)
        positions = nearest_equivalent_joint_solution(
            positions, previous, limits[:, None].expand(-1, count, -1, -1)
        )
    solver = robot.get_solver(control_part)
    if solver is None:
        raise RuntimeError(f"Kinematic solver is unavailable for {control_part!r}.")
    jacobians = solver.get_jacobian(
        positions.reshape(batch * count, -1), jac_type="full"
    ).reshape(batch, count, 6, -1)
    base_pose = robot.get_control_part_base_pose(control_part, to_matrix=True)
    world_to_root = base_pose[:, None, :3, :3].transpose(-1, -2)
    root_tangent = torch.matmul(world_to_root, world_tangent[..., None]).squeeze(-1)
    tangent = torch.cat((root_tangent, torch.zeros_like(root_tangent)), dim=-1).to(
        jacobians
    )
    curvature = None
    if world_curvature is not None:
        root_curvature = torch.matmul(
            world_to_root, world_curvature[..., None]
        ).squeeze(-1)
        curvature = torch.cat(
            (root_curvature, torch.zeros_like(root_curvature)), dim=-1
        ).to(jacobians)
    velocities, accelerations = joint_derivatives_from_path_time_law(
        jacobians,
        scalar.positions[..., 0],
        scalar.velocities[..., 0],
        scalar.accelerations[..., 0],
        tangent,
        curvature,
    )
    return (
        PlanResult(
            success=sample_success.all(dim=1),
            xpos_list=desired,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dt=scalar.dt,
        ),
        desired,
        scalar,
    )


def compute_eef_trajectory(
    robot: Robot,
    control_part: str,
    joint_positions: torch.Tensor,
    env_index: int,
) -> torch.Tensor:
    """Evaluate FK for one row of a trajectory shaped ``(B, N, DOF)``.

    Returns:
        End-effector position and quaternion as ``(N, 7)`` in the local arena
        frame, ordered as ``x, y, z, qx, qy, qz, qw`` (``xyz + xyzw``).
    """
    if joint_positions.ndim != 3:
        raise ValueError("joint_positions must have shape (B, N, DOF).")
    if not 0 <= env_index < joint_positions.shape[0]:
        raise ValueError(
            f"env_index must be in [0, {joint_positions.shape[0]}), got {env_index}."
        )
    sample_count = joint_positions.shape[1]
    poses = robot.compute_fk(
        qpos=joint_positions[env_index],
        name=control_part,
        env_ids=[env_index] * sample_count,
        to_matrix=False,
    )
    if poses is None:
        raise RuntimeError(f"Forward kinematics is unavailable for {control_part!r}.")
    return poses
