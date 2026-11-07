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

"""Shared grasp and target geometry for coordinated placement hosts."""

from __future__ import annotations

import torch

from embodichain.utils import logger
from .._atomic_fixtures import (
    compute_world_bounds,
    normalize_vector,
    rotate_pose_about_world_z,
    transform_points,
)

__all__ = [
    "get_pan_basin_vertices",
    "build_top_down_tcp_pose",
    "build_flat_object_grasp_pose",
    "build_pan_handle_grasp_pose",
    "build_support_object_target_pose",
    "build_placing_object_target_pose",
]


TABLE_TOP_Z = 0.65
PAN_TARGET_CENTER_XY = (-0.06, 0.0)
PAN_TARGET_Z_LIFT = 0.06
BREAD_ON_PAN_CLEARANCE = 0.006
PAN_HANDLE_LOCAL_Z_MIN = 0.04
PAN_BASIN_LOCAL_Z_MAX = 0.04
PAN_HANDLE_ROOT_OFFSET = 0.035
BREAD_TARGET_WORLD_YAW_DEG = 0.0


def get_pan_basin_vertices(pan_vertices: torch.Tensor) -> torch.Tensor:
    """Select basin vertices, falling back to the full mesh for an empty mask.

    Args:
        pan_vertices: Local pan mesh vertices with shape ``(N, 3)``.

    Returns:
        Vertices whose local z is within the configured basin height.
    """
    basin_vertices = pan_vertices[pan_vertices[:, 2] <= PAN_BASIN_LOCAL_Z_MAX]
    if basin_vertices.numel() == 0:
        logger.log_warning("Pan basin vertex mask is empty; falling back to full mesh.")
        return pan_vertices
    return basin_vertices


def build_top_down_tcp_pose(
    position: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Build a top-down TCP pose using the calibrated grasp orientation.

    Args:
        position: World TCP position with shape ``(3,)``.
        device: Device for the output transform.

    Returns:
        A float32 TCP transform with shape ``(4, 4)``.
    """
    pose = torch.eye(4, dtype=torch.float32, device=device)
    # The calibration was recorded to four decimal places. Reconstruct a
    # right-handed orthonormal frame before passing it to strict IK solvers;
    # the rounded matrix itself is not an element of SO(3).
    z_axis = torch.tensor([-0.0022, -0.0401, -0.9992], device=device)
    z_axis = z_axis / torch.linalg.vector_norm(z_axis)
    y_hint = torch.tensor([-0.9985, 0.0540, 0.0], device=device)
    x_axis = torch.cross(y_hint, z_axis, dim=0)
    x_axis = x_axis / torch.linalg.vector_norm(x_axis)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    pose[:3, :3] = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    pose[:3, 3] = position
    return pose


def build_flat_object_grasp_pose(
    object_pose: torch.Tensor,
    local_vertices: torch.Tensor,
    local_min: torch.Tensor,
    local_max: torch.Tensor,
    device: torch.device,
    *,
    world_xy_offset: tuple[float, float] = (0.0, 0.0),
    z_clearance: float = 0.02,
) -> torch.Tensor:
    """Build a top-down grasp centered over the object's world-space top.

    Args:
        object_pose: World object transform with shape ``(4, 4)``.
        local_vertices: Object mesh vertices with shape ``(N, 3)``.
        local_min: Local axis-aligned lower bounds with shape ``(3,)``.
        local_max: Local axis-aligned upper bounds with shape ``(3,)``.
        device: Device for the output transform.
        world_xy_offset: Additional world XY grasp offset in metres.
        z_clearance: TCP clearance above the object in metres.

    Returns:
        A world TCP transform with shape ``(4, 4)``.
    """
    local_center = 0.5 * (local_min + local_max)
    local_center = local_center.to(device=device, dtype=torch.float32)
    grasp_position = object_pose[:3, 3] + object_pose[:3, :3] @ local_center
    _, world_max = compute_world_bounds(object_pose, local_vertices)
    grasp_position[0] += world_xy_offset[0]
    grasp_position[1] += world_xy_offset[1]
    grasp_position[2] = world_max[2] + z_clearance
    return build_top_down_tcp_pose(grasp_position, device)


def build_pan_handle_grasp_pose(
    pan_pose: torch.Tensor,
    pan_vertices: torch.Tensor,
    device: torch.device,
    *,
    z_clearance: float = 0.006,
) -> torch.Tensor:
    """Build a downward-facing grasp displaced toward the pan handle root.

    Args:
        pan_pose: World pan transform with shape ``(4, 4)``.
        pan_vertices: Local pan mesh vertices with shape ``(N, 3)``.
        device: Device for the output transform.
        z_clearance: TCP clearance above the handle in metres.

    Returns:
        A world TCP transform with shape ``(4, 4)``.
    """
    handle_vertices = pan_vertices[pan_vertices[:, 2] > PAN_HANDLE_LOCAL_Z_MIN]
    if handle_vertices.numel() == 0:
        logger.log_warning(
            "Pan handle vertex mask is empty; falling back to full mesh."
        )
        handle_vertices = pan_vertices

    handle_world = transform_points(pan_pose, handle_vertices)
    handle_min = handle_world.min(dim=0).values
    handle_max = handle_world.max(dim=0).values
    grasp_position = 0.5 * (handle_min + handle_max)

    pan_world = transform_points(pan_pose, pan_vertices)
    pan_center_xy = pan_world[:, :2].mean(dim=0)
    handle_dir_xy = grasp_position[:2] - pan_center_xy
    handle_axis = normalize_vector(
        torch.tensor(
            [handle_dir_xy[0], handle_dir_xy[1], 0.0],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor([1.0, -0.2, 0.0], dtype=torch.float32, device=device),
    )
    grasp_position[:2] -= handle_axis[:2] * PAN_HANDLE_ROOT_OFFSET
    grasp_position[2] = handle_max[2] + z_clearance

    y_axis = handle_axis
    z_axis = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    x_axis = normalize_vector(
        torch.cross(y_axis, z_axis, dim=0),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device),
    )
    y_axis = normalize_vector(
        torch.cross(z_axis, x_axis, dim=0),
        y_axis,
    )

    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = grasp_position
    return pose


def build_support_object_target_pose(
    pan_pose: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Place the support pan at the shared scene's raised target.

    Args:
        pan_pose: Initial pan transform with shape ``(4, 4)``.
        device: Device for the output transform.

    Returns:
        A copied world transform preserving the initial orientation.
    """
    pose = pan_pose.clone().to(device=device, dtype=torch.float32)
    pose[0, 3] = PAN_TARGET_CENTER_XY[0]
    pose[1, 3] = PAN_TARGET_CENTER_XY[1]
    pose[2, 3] = TABLE_TOP_Z + 0.001 + PAN_TARGET_Z_LIFT
    return pose


def build_placing_object_target_pose(
    bread_pose: torch.Tensor,
    bread_vertices: torch.Tensor,
    pan_vertices: torch.Tensor,
    support_target_pose: torch.Tensor,
    device: torch.device,
    *,
    pan_triangles: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align the placing object over the pan basin with bottom clearance.

    Args:
        bread_pose: Initial placing-object transform with shape ``(4, 4)``.
        bread_vertices: Local placing-object mesh vertices with shape ``(N, 3)``.
        pan_vertices: Local support mesh vertices with shape ``(M, 3)``.
        support_target_pose: World support target with shape ``(4, 4)``.
        device: Device for the output transform.
        pan_triangles: Optional mesh topology. When provided, intersect the
            basin-center vertical ray with the actual inner support surface;
            a bounds-only support uses its upper bound.

    Returns:
        A copied placing-object target with the shared basin-clearance policy.
    """
    pose = rotate_pose_about_world_z(
        bread_pose.clone().to(device=device, dtype=torch.float32),
        BREAD_TARGET_WORLD_YAW_DEG,
    )
    pan_basin_world = transform_points(
        support_target_pose, get_pan_basin_vertices(pan_vertices)
    )
    basin_center_xy = 0.5 * (
        pan_basin_world[:, :2].min(dim=0).values
        + pan_basin_world[:, :2].max(dim=0).values
    )
    # Align mesh footprints, not object origins or historical world offsets.
    # Asset origins need not coincide with the placing object's geometric center.
    bread_min, bread_max = compute_world_bounds(pose, bread_vertices)
    bread_center_xy = 0.5 * (bread_min[:2] + bread_max[:2])
    pose[:2, 3] += basin_center_xy - bread_center_xy
    if pan_triangles is None:
        pan_top_z = pan_basin_world[:, 2].max()
    else:
        # A tray's rim is not its resting surface. The uppermost triangle under
        # the basin center identifies the interior floor, including floor thickness.
        world_vertices = transform_points(support_target_pose, pan_vertices)
        triangles = world_vertices[pan_triangles.to(device=device, dtype=torch.long)]
        a, b, c = triangles.unbind(dim=1)
        edge_b, edge_c = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        delta = basin_center_xy - a[:, :2]
        determinant = edge_b[:, 0] * edge_c[:, 1] - edge_b[:, 1] * edge_c[:, 0]
        nondegenerate = determinant.abs() > 1e-10
        denominator = torch.where(
            nondegenerate, determinant, torch.ones_like(determinant)
        )
        u = (delta[:, 0] * edge_c[:, 1] - delta[:, 1] * edge_c[:, 0]) / denominator
        v = (edge_b[:, 0] * delta[:, 1] - edge_b[:, 1] * delta[:, 0]) / denominator
        inside = nondegenerate & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
        if not bool(inside.any()):
            raise ValueError("Pan basin center has no supporting mesh surface.")
        heights = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
        pan_top_z = heights[inside].max()
    bread_bottom_z = compute_world_bounds(pose, bread_vertices)[0][2]
    pose[2, 3] += pan_top_z + BREAD_ON_PAN_CLEARANCE - bread_bottom_z
    return pose
