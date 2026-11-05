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

"""Pure Cartesian entry and release paths for drawer payloads."""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "drawer_entry_poses",
    "drawer_unoccupied_target",
    "drawer_exposed_floor_targets",
]


def drawer_entry_poses(
    target: torch.Tensor,
    link: torch.Tensor,
    vertices: np.ndarray | torch.Tensor,
    lower: tuple[float, float, float] | np.ndarray | torch.Tensor,
    upper: tuple[float, float, float] | np.ndarray | torch.Tensor,
    axis: tuple[float, float, float],
    axis_sign: float,
    *,
    preserve_vertical: bool = False,
    release_lift: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross the rim with a full payload reserve and form a bounded release.

    Inputs are existing batched object/link poses and object-local vertices.
    This numeric proposal neither performs IK nor certifies physical clearance.
    Upright release preserves the supported target; ordinary release stays above
    the rim with the original additional 12 cm hand reserve.
    """
    outward = -link.new_tensor(axis) * axis_sign
    vertices = link.new_tensor(vertices)
    local_rotation = link[:, :3, :3].transpose(-1, -2) @ target[:, :3, :3]
    points = vertices @ local_rotation.transpose(-1, -2)
    low, high = link.new_tensor(lower), link.new_tensor(upper)
    local_target = (
        link[:, :3, :3].transpose(-1, -2)
        @ (target[:, :3, 3] - link[:, :3, 3]).unsqueeze(-1)
    ).squeeze(-1)
    rim = target.clone()
    rise = (high[2] + 0.025 - (local_target[:, 2] + points[:, :, 2].amin(1))).clamp_min(
        0
    )
    rim[:, :3, 3] += link[:, :3, 2] * rise[:, None]
    if not preserve_vertical:
        rim[:, :3, 3] += link[:, :3, 2] * 0.12
    front = torch.maximum(low * outward, high * outward).sum()
    back_extent = (points @ outward).amin(1)
    distance = (front + 0.04 - back_extent - (local_target @ outward)).clamp_min(0)
    outside = rim.clone()
    outside[:, :3, 3] += (link[:, :3, :3] @ outward) * distance[:, None]
    release = target.clone() if preserve_vertical else rim.clone()
    release[:, :3, 3] += link[:, :3, 2] * release_lift
    return outside, rim, release


def drawer_unoccupied_target(
    target: torch.Tensor,
    link: torch.Tensor,
    vertices: np.ndarray,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    axis: tuple[float, float, float],
    occupied_payloads: tuple[tuple[np.ndarray, torch.Tensor], ...],
    *,
    side: int = 0,
) -> torch.Tensor:
    """Select a floor position around already-qualified occupied payloads.

    Each occupied item supplies object-local vertices and an observed world pose.
    The caller determines which payloads are physically inside this drawer.
    """
    from embodichain.compute.geometry.landing import select_container_landing
    import trimesh

    occupied = []
    for other_vertices, other_pose in occupied_payloads:
        relative = torch.linalg.inv(link) @ other_pose.to(link)
        points = link.new_tensor(other_vertices) @ relative[:3, :3].T + relative[:3, 3]
        center = (points[:, :2].amin(0) + points[:, :2].amax(0)) / 2
        radius = torch.linalg.vector_norm(points[:, :2] - center, dim=-1).amax()
        occupied.append((float(center[0]), float(center[1]), float(radius)))
    if not occupied:
        return target
    low, high = np.asarray(lower), np.asarray(upper)
    floor = trimesh.creation.box(extents=[high[0] - low[0], high[1] - low[1], 0.002])
    floor.apply_translation([*(low[:2] + high[:2]) / 2, low[2] - 0.001])
    local = torch.linalg.inv(link) @ target
    vertices = np.asarray(vertices) @ local[:3, :3].cpu().numpy().T
    preference = local[:3, 3].cpu().numpy().copy()
    if side:
        rail = np.asarray(axis)
        lateral = np.cross([0.0, 0.0, 1.0], rail)
        preference += side * 0.25 * float((high - low) @ np.abs(lateral)) * lateral
    landing = select_container_landing(floor, vertices, preference, occupied=occupied)
    if landing is None:
        raise ValueError("Drawer occupied-floor landing is unavailable.")
    result = target.clone()
    position = local[:3, 3].clone()
    position[:2] = position.new_tensor(landing[:2])
    result[:3, 3] = link[:3, :3] @ position + link[:3, 3]
    return result


def drawer_exposed_floor_targets(
    vertices: np.ndarray,
    object_pose: torch.Tensor,
    link: torch.Tensor,
    opening: torch.Tensor,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    axis: tuple[float, float, float],
    axis_sign: float,
    *,
    preserve_vertical: bool = False,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Bias bounded floor proposals toward the exposed portion of an open drawer."""
    from .drawer import drawer_placement_pose_candidates

    outward = -link.new_tensor(axis) * axis_sign
    depth = ((link.new_tensor(upper) - link.new_tensor(lower)) * outward.abs()).sum()
    shift = (depth - opening).clamp_min(0) / 2
    direction = link[:, :3, :3] @ outward
    rows = []
    for row in range(len(object_pose)):
        poses = drawer_placement_pose_candidates(
            vertices,
            object_pose[row, :3, :3].detach().cpu().numpy(),
            link[row].detach().cpu().numpy(),
            np.asarray(lower),
            np.asarray(upper),
            preserve_vertical=preserve_vertical,
        )
        result = []
        for pose in poses:
            target = torch.as_tensor(
                pose, dtype=object_pose.dtype, device=object_pose.device
            )
            target[:3, 3] += direction[row] * shift[row]
            result.append(target)
        rows.append(tuple(result))
    return tuple(rows)
