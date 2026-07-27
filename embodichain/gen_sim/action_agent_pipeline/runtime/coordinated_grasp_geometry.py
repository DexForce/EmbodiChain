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

"""Generate and score geometric dual-arm grasp candidates.

This module is deliberately IK-free: it proposes geometric candidates, while
``coordinated_grasp_ik`` decides which candidates are reachable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.defaults import (
    generation_defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_runtime_types import (
    CoordinatedGraspPair,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.pose_utils import (
    _current_arm_positions,
    _normalize_vector,
    _object_world_vertices,
    _orthogonalized_axis,
    _pose_from_axes,
    _transform_local_point,
    _world_bounds_from_local_vertices,
    _world_pose_to_object_pose,
    _world_vertices_from_local_vertices,
)
from embodichain.gen_sim.action_agent_pipeline.semantics import (
    CONTAINER_LIKE_KEYWORDS as _COORDINATED_CONTAINER_LIKE_KEYWORDS,
    ROD_LIKE_KEYWORDS as _COORDINATED_ROD_LIKE_KEYWORDS,
)

__all__ = [
    "_filter_coordinated_payload_collision_candidates",
    "_coordinated_grasp_pair_candidates",
    "_coordinated_top_down_axis_indices",
    "_coordinated_top_down_grasp_candidates",
    "_coordinated_projected_top_down_grasp_candidates",
    "_coordinated_world_lateral_top_down_grasp_candidates",
    "_coordinated_world_y_constrained_top_down_grasp_candidates",
    "_coordinated_axis_inset",
    "_coordinated_grasp_style",
    "_coordinated_inset_fractions_for_style",
    "_coordinated_world_lateral_priority",
    "_coordinated_label_has_keyword",
    "_coordinated_principal_extents",
    "_coordinated_geometry_is_rod_like",
    "_coordinated_geometry_is_container_like",
    "_coordinated_novel_principal_axis_pairs",
    "_coordinated_axis_pair_is_novel",
    "_normalize_horizontal_axis",
    "_make_coordinated_top_down_world_grasp_pair",
    "_coordinated_side_axis_index",
    "_make_coordinated_side_grasp_pose",
    "_make_coordinated_grasp_pair",
    "_coordinated_grasp_pair_world_y_angle_degrees",
    "_coordinated_grasp_pair_score",
    "_assign_coordinated_world_grasp_pair_to_arms",
    "_assign_coordinated_local_grasp_pair_to_arms",
]

_GRASP_DEFAULTS = generation_defaults_section("grasp")
_COORDINATED_GRASP_STYLE_CONTAINER = "container_like"
_COORDINATED_GRASP_STYLE_ROD = "rod_like"
_COORDINATED_GRASP_STYLE_GENERIC = "generic"
_COORDINATED_ROD_LIKE_INSET_FRACTIONS = tuple(
    float(value) for value in _GRASP_DEFAULTS["coordinated_rod_like_inset_fractions"]
)
_COORDINATED_CONTAINER_LIKE_INSET_FRACTIONS = tuple(
    float(value)
    for value in _GRASP_DEFAULTS["coordinated_container_like_inset_fractions"]
)
_COORDINATED_GENERIC_INSET_FRACTIONS = tuple(
    float(value) for value in _GRASP_DEFAULTS["coordinated_generic_inset_fractions"]
)


def _filter_coordinated_payload_collision_candidates(
    candidates: Sequence[CoordinatedGraspPair],
    *,
    payload_uids: Sequence[str],
    object_initial_pose: torch.Tensor,
    env,
    device,
    env_id: int = 0,
) -> list[CoordinatedGraspPair]:
    if not payload_uids or env is None:
        return list(candidates)
    payload_bounds = []
    for payload_uid in payload_uids:
        payload = env.sim.get_rigid_object(str(payload_uid))
        if payload is None:
            raise ValueError(f"Unknown coordinated payload uid: {payload_uid!r}.")
        vertices = _object_world_vertices(payload, device, env_id=env_id)
        payload_bounds.append(
            (
                vertices.min(dim=0).values - 0.02,
                vertices.max(dim=0).values + 0.02,
            )
        )

    def _candidate_is_clear(candidate: CoordinatedGraspPair) -> bool:
        eef_positions = (
            (object_initial_pose @ candidate.left_object_to_eef)[:3, 3],
            (object_initial_pose @ candidate.right_object_to_eef)[:3, 3],
        )
        return all(
            not bool(((position >= mins) & (position <= maxs)).all())
            for position in eef_positions
            for mins, maxs in payload_bounds
        )

    return [candidate for candidate in candidates if _candidate_is_clear(candidate)]


def _coordinated_grasp_pair_candidates(
    *,
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    object_label: str | None = None,
    max_grasp_separation_angle_to_world_y_degrees: float | None = None,
    env,
    device,
    env_id: int = 0,
) -> list[CoordinatedGraspPair]:
    mins = vertices.min(dim=0).values
    maxs = vertices.max(dim=0).values
    center = (mins + maxs) * 0.5
    extents = maxs - mins
    candidates: list[CoordinatedGraspPair] = []
    grasp_style = _coordinated_grasp_style(
        object_label=object_label,
        vertices=vertices,
        object_initial_pose=object_initial_pose,
        device=device,
    )
    inset_fractions = _coordinated_inset_fractions_for_style(grasp_style)
    use_edge_closing = grasp_style == _COORDINATED_GRASP_STYLE_CONTAINER
    top_down_axis_indices = _coordinated_top_down_axis_indices(extents)
    world_lateral_priority = _coordinated_world_lateral_priority(grasp_style)
    if max_grasp_separation_angle_to_world_y_degrees is None:
        candidates.extend(
            _coordinated_world_lateral_top_down_grasp_candidates(
                vertices=vertices,
                center=center,
                object_initial_pose=object_initial_pose,
                inset_fractions=_coordinated_inset_fractions_for_style(
                    _COORDINATED_GRASP_STYLE_CONTAINER
                ),
                priority=world_lateral_priority,
                env=env,
                device=device,
                env_id=env_id,
            )
        )
    else:
        candidates.extend(
            _coordinated_world_y_constrained_top_down_grasp_candidates(
                vertices=vertices,
                object_initial_pose=object_initial_pose,
                inset_fractions=inset_fractions,
                use_edge_closing=use_edge_closing,
                max_angle_degrees=(max_grasp_separation_angle_to_world_y_degrees),
                env=env,
                device=device,
                env_id=env_id,
            )
        )
    principal_axis_pairs = _coordinated_novel_principal_axis_pairs(
        vertices,
        object_initial_pose,
        device,
    )
    principal_priority_offset = 0
    for axis_rank, (separation_axis, closing_axis) in enumerate(principal_axis_pairs):
        gripper_closing_axis = separation_axis if use_edge_closing else closing_axis
        candidates.extend(
            _coordinated_projected_top_down_grasp_candidates(
                vertices=vertices,
                separation_axis=separation_axis,
                lateral_axis=closing_axis,
                closing_axis=gripper_closing_axis,
                object_initial_pose=object_initial_pose,
                inset_fractions=inset_fractions,
                priority=principal_priority_offset + axis_rank * 20,
                axis_kind="long_axis" if axis_rank == 0 else "short_axis",
                env=env,
                device=device,
                env_id=env_id,
            )
        )
    local_priority_offset = len(principal_axis_pairs) * 20 + principal_priority_offset
    for axis_rank, axis_index in enumerate(top_down_axis_indices):
        candidates.extend(
            _coordinated_top_down_grasp_candidates(
                vertices=vertices,
                axis_index=axis_index,
                inset_fractions=inset_fractions,
                use_edge_closing=use_edge_closing,
                priority=local_priority_offset + axis_rank * 20,
                axis_kind="long_axis" if axis_rank == 0 else "short_axis",
                object_initial_pose=object_initial_pose,
                env=env,
                device=device,
                env_id=env_id,
            )
        )
    side_axis_index = _coordinated_side_axis_index(extents)
    side_axis = torch.eye(3, dtype=torch.float32, device=device)[:, side_axis_index]
    lateral_offset = max(float(extents[side_axis_index]) * 0.5, 0.04)
    vertical_offset = max(float(extents[2]) * 0.25, 0.03)

    positive_side = _make_coordinated_side_grasp_pose(
        center=center,
        side_axis=side_axis,
        side_sign=1.0,
        lateral_offset=lateral_offset,
        vertical_offset=vertical_offset,
    )
    negative_side = _make_coordinated_side_grasp_pose(
        center=center,
        side_axis=side_axis,
        side_sign=-1.0,
        lateral_offset=lateral_offset,
        vertical_offset=vertical_offset,
    )
    left_side, right_side = _assign_coordinated_local_grasp_pair_to_arms(
        positive_side,
        negative_side,
        object_initial_pose=object_initial_pose,
        env=env,
        device=device,
        env_id=env_id,
    )
    candidates.append(
        _make_coordinated_grasp_pair(
            left_side,
            right_side,
            object_initial_pose=object_initial_pose,
            env=env,
            device=device,
            env_id=env_id,
            priority=len(top_down_axis_indices) * 20 + 10,
            score_bias=10.0,
            axis_kind="side",
        )
    )
    if max_grasp_separation_angle_to_world_y_degrees is not None:
        angle_limit = float(max_grasp_separation_angle_to_world_y_degrees)
        candidates = [
            candidate
            for candidate in candidates
            if _coordinated_grasp_pair_world_y_angle_degrees(
                candidate,
                object_initial_pose=object_initial_pose,
            )
            <= angle_limit + 1e-4
        ]
        return sorted(
            candidates,
            key=lambda pair: (
                _coordinated_grasp_pair_world_y_angle_degrees(
                    pair,
                    object_initial_pose=object_initial_pose,
                ),
                pair.priority,
                pair.score,
            ),
        )
    return sorted(candidates, key=lambda pair: (pair.priority, pair.score))


def _coordinated_top_down_axis_indices(extents: torch.Tensor) -> list[int]:
    horizontal = [0, 1]
    return sorted(horizontal, key=lambda index: float(extents[index]), reverse=True)


def _coordinated_top_down_grasp_candidates(
    *,
    vertices: torch.Tensor,
    axis_index: int,
    inset_fractions: tuple[float, ...],
    use_edge_closing: bool,
    priority: int,
    axis_kind: str,
    object_initial_pose: torch.Tensor,
    env,
    device,
    env_id: int = 0,
) -> list[CoordinatedGraspPair]:
    axis_local = torch.zeros(3, dtype=torch.float32, device=device)
    axis_local[axis_index] = 1.0
    separation_axis = _normalize_vector(object_initial_pose[:3, :3] @ axis_local)
    closing_axis_index = 1 - axis_index
    closing_axis_local = torch.zeros(3, dtype=torch.float32, device=device)
    closing_axis_local[closing_axis_index] = 1.0
    lateral_axis = _normalize_vector(object_initial_pose[:3, :3] @ closing_axis_local)
    closing_axis = separation_axis if use_edge_closing else lateral_axis
    return _coordinated_projected_top_down_grasp_candidates(
        vertices=vertices,
        separation_axis=separation_axis,
        lateral_axis=lateral_axis,
        closing_axis=closing_axis,
        object_initial_pose=object_initial_pose,
        inset_fractions=inset_fractions,
        priority=priority,
        axis_kind=axis_kind,
        env=env,
        device=device,
        env_id=env_id,
    )


def _coordinated_projected_top_down_grasp_candidates(
    *,
    vertices: torch.Tensor,
    separation_axis: torch.Tensor,
    lateral_axis: torch.Tensor,
    closing_axis: torch.Tensor,
    object_initial_pose: torch.Tensor,
    inset_fractions: tuple[float, ...],
    priority: int,
    axis_kind: str,
    env,
    device,
    env_id: int = 0,
) -> list[CoordinatedGraspPair]:
    world_vertices = _world_vertices_from_local_vertices(object_initial_pose, vertices)
    world_bounds_min = world_vertices.min(dim=0).values
    world_bounds_max = world_vertices.max(dim=0).values
    grasp_z = (
        world_bounds_min[2] + (world_bounds_max[2] - world_bounds_min[2]) * 0.55 + 0.01
    )
    separation_axis = _normalize_horizontal_axis(separation_axis, device)
    lateral_axis = _normalize_horizontal_axis(lateral_axis, device)
    closing_axis = _normalize_horizontal_axis(closing_axis, device)
    lateral_axis = _orthogonalized_axis(lateral_axis, separation_axis)
    separation_projections = world_vertices @ separation_axis
    lateral_projections = world_vertices @ lateral_axis
    separation_min = separation_projections.min()
    separation_max = separation_projections.max()
    lateral_center = (lateral_projections.min() + lateral_projections.max()) * 0.5
    separation_extent = separation_max - separation_min

    candidates: list[CoordinatedGraspPair] = []
    for inset_rank, inset_fraction in enumerate(inset_fractions):
        margin = _coordinated_axis_inset(separation_extent, inset_fraction)
        first_projection = separation_min + margin
        second_projection = separation_max - margin
        first_world_pos = (
            separation_axis * first_projection + lateral_axis * lateral_center
        )
        second_world_pos = (
            separation_axis * second_projection + lateral_axis * lateral_center
        )
        first_world_pos[2] = grasp_z
        second_world_pos[2] = grasp_z

        candidates.append(
            _make_coordinated_top_down_world_grasp_pair(
                first_world_pos=first_world_pos,
                second_world_pos=second_world_pos,
                separation_axis=separation_axis,
                closing_axis=closing_axis,
                object_initial_pose=object_initial_pose,
                env=env,
                device=device,
                env_id=env_id,
                priority=priority + inset_rank,
                score_bias=0.0,
                axis_kind=axis_kind,
            )
        )
    return candidates


def _coordinated_world_lateral_top_down_grasp_candidates(
    *,
    vertices: torch.Tensor,
    center: torch.Tensor,
    object_initial_pose: torch.Tensor,
    inset_fractions: tuple[float, ...],
    priority: int,
    env,
    device,
    env_id: int = 0,
) -> list[CoordinatedGraspPair]:
    world_bounds_min, world_bounds_max = _world_bounds_from_local_vertices(
        object_initial_pose,
        vertices,
    )
    world_y_extent = world_bounds_max[1] - world_bounds_min[1]
    if float(world_y_extent) < 0.12:
        return []

    world_center = _transform_local_point(object_initial_pose, center)
    grasp_z = (
        world_bounds_min[2] + (world_bounds_max[2] - world_bounds_min[2]) * 0.55 + 0.01
    )
    separation_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    closing_axis = separation_axis
    candidates: list[CoordinatedGraspPair] = []
    for inset_rank, inset_fraction in enumerate(inset_fractions):
        margin = _coordinated_axis_inset(world_y_extent, inset_fraction)
        first_world_pos = world_center.clone()
        second_world_pos = world_center.clone()
        first_world_pos[1] = world_bounds_min[1] + margin
        second_world_pos[1] = world_bounds_max[1] - margin
        first_world_pos[2] = grasp_z
        second_world_pos[2] = grasp_z
        candidates.append(
            _make_coordinated_top_down_world_grasp_pair(
                first_world_pos=first_world_pos,
                second_world_pos=second_world_pos,
                separation_axis=separation_axis,
                closing_axis=closing_axis,
                object_initial_pose=object_initial_pose,
                env=env,
                device=device,
                env_id=env_id,
                priority=priority + inset_rank,
                score_bias=0.0,
                axis_kind="world_lateral",
            )
        )
    return candidates


def _coordinated_world_y_constrained_top_down_grasp_candidates(
    *,
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    inset_fractions: tuple[float, ...],
    use_edge_closing: bool,
    max_angle_degrees: float,
    env,
    device,
    env_id: int = 0,
) -> list[CoordinatedGraspPair]:
    angle_magnitudes = [0.0]
    if max_angle_degrees > 1e-6:
        half_angle = float(max_angle_degrees) * 0.5
        if half_angle > 1e-6:
            angle_magnitudes.append(half_angle)
        if float(max_angle_degrees) - half_angle > 1e-6:
            angle_magnitudes.append(float(max_angle_degrees))

    candidates: list[CoordinatedGraspPair] = []
    for angle_rank, angle_magnitude in enumerate(angle_magnitudes):
        signed_angles = (
            (0.0,) if angle_magnitude == 0.0 else (-angle_magnitude, angle_magnitude)
        )
        for signed_angle in signed_angles:
            angle_radians = float(np.deg2rad(signed_angle))
            separation_axis = torch.tensor(
                [np.sin(angle_radians), np.cos(angle_radians), 0.0],
                dtype=torch.float32,
                device=device,
            )
            lateral_axis = torch.tensor(
                [np.cos(angle_radians), -np.sin(angle_radians), 0.0],
                dtype=torch.float32,
                device=device,
            )
            closing_axis = separation_axis if use_edge_closing else lateral_axis
            candidates.extend(
                _coordinated_projected_top_down_grasp_candidates(
                    vertices=vertices,
                    separation_axis=separation_axis,
                    lateral_axis=lateral_axis,
                    closing_axis=closing_axis,
                    object_initial_pose=object_initial_pose,
                    inset_fractions=inset_fractions,
                    priority=angle_rank * 20,
                    axis_kind=(
                        "world_y" if angle_magnitude == 0.0 else "world_y_constrained"
                    ),
                    env=env,
                    device=device,
                    env_id=env_id,
                )
            )
    return candidates


def _coordinated_axis_inset(extent: torch.Tensor, fraction: float) -> float:
    axis_extent = float(extent)
    if axis_extent <= 1e-6:
        return 0.0
    return min(axis_extent * float(fraction), axis_extent * 0.45)


def _coordinated_grasp_style(
    *,
    object_label: str | None,
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    device,
) -> str:
    if _coordinated_label_has_keyword(
        object_label,
        _COORDINATED_CONTAINER_LIKE_KEYWORDS,
    ):
        return _COORDINATED_GRASP_STYLE_CONTAINER
    if _coordinated_label_has_keyword(
        object_label,
        _COORDINATED_ROD_LIKE_KEYWORDS,
    ):
        return _COORDINATED_GRASP_STYLE_ROD

    long_xy, short_xy, z_extent = _coordinated_principal_extents(
        vertices,
        object_initial_pose,
        device,
    )
    if _coordinated_geometry_is_rod_like(long_xy, short_xy):
        return _COORDINATED_GRASP_STYLE_ROD
    if _coordinated_geometry_is_container_like(long_xy, short_xy, z_extent):
        return _COORDINATED_GRASP_STYLE_CONTAINER
    return _COORDINATED_GRASP_STYLE_GENERIC


def _coordinated_inset_fractions_for_style(grasp_style: str) -> tuple[float, ...]:
    if grasp_style == _COORDINATED_GRASP_STYLE_ROD:
        return _COORDINATED_ROD_LIKE_INSET_FRACTIONS
    if grasp_style == _COORDINATED_GRASP_STYLE_CONTAINER:
        return _COORDINATED_CONTAINER_LIKE_INSET_FRACTIONS
    return _COORDINATED_GENERIC_INSET_FRACTIONS


def _coordinated_world_lateral_priority(grasp_style: str) -> int:
    if grasp_style == _COORDINATED_GRASP_STYLE_CONTAINER:
        return 60
    if grasp_style == _COORDINATED_GRASP_STYLE_ROD:
        return 80
    return 8


def _coordinated_label_has_keyword(
    object_label: str | None,
    keywords: tuple[str, ...],
) -> bool:
    if not object_label:
        return False
    text = str(object_label).lower()
    normalized = (
        text.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    )
    tokens = set(normalized.split())
    for keyword in keywords:
        keyword = keyword.lower()
        if keyword.isascii():
            if " " in keyword:
                if keyword in normalized:
                    return True
            elif keyword in tokens:
                return True
        elif keyword in text:
            return True
    return False


def _coordinated_principal_extents(
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    device,
) -> tuple[float, float, float]:
    world_vertices = _world_vertices_from_local_vertices(object_initial_pose, vertices)
    world_bounds_min = world_vertices.min(dim=0).values
    world_bounds_max = world_vertices.max(dim=0).values
    fallback_extents = world_bounds_max - world_bounds_min
    xy = world_vertices[:, :2]
    if xy.shape[0] < 3:
        long_xy = max(float(fallback_extents[0]), float(fallback_extents[1]))
        short_xy = min(float(fallback_extents[0]), float(fallback_extents[1]))
        return long_xy, short_xy, float(fallback_extents[2])

    centered_xy = xy - xy.mean(dim=0, keepdim=True)
    covariance = (
        centered_xy.transpose(0, 1)
        @ centered_xy
        / max(
            int(centered_xy.shape[0]),
            1,
        )
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= 1e-8:
        long_xy = max(float(fallback_extents[0]), float(fallback_extents[1]))
        short_xy = min(float(fallback_extents[0]), float(fallback_extents[1]))
        return long_xy, short_xy, float(fallback_extents[2])

    zero = torch.zeros((), dtype=torch.float32, device=device)
    axes = (
        torch.stack([eigenvectors[0, -1], eigenvectors[1, -1], zero]),
        torch.stack([eigenvectors[0, -2], eigenvectors[1, -2], zero]),
    )
    ranges = []
    for axis in axes:
        axis = _normalize_horizontal_axis(axis, device)
        projections = world_vertices @ axis
        ranges.append(float(projections.max() - projections.min()))
    long_xy = max(ranges)
    short_xy = min(ranges)
    return long_xy, short_xy, float(fallback_extents[2])


def _coordinated_geometry_is_rod_like(long_xy: float, short_xy: float) -> bool:
    short_xy = max(float(short_xy), 1e-6)
    return float(long_xy) / short_xy >= 2.4


def _coordinated_geometry_is_container_like(
    long_xy: float,
    short_xy: float,
    z_extent: float,
) -> bool:
    short_xy = max(float(short_xy), 1e-6)
    return (
        float(short_xy) >= 0.12
        and float(long_xy) / short_xy <= 2.4
        and float(z_extent) <= short_xy * 0.65
    )


def _coordinated_novel_principal_axis_pairs(
    vertices: torch.Tensor,
    object_initial_pose: torch.Tensor,
    device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    world_vertices = _world_vertices_from_local_vertices(object_initial_pose, vertices)
    if world_vertices.shape[0] < 3:
        return []
    xy = world_vertices[:, :2]
    centered_xy = xy - xy.mean(dim=0, keepdim=True)
    covariance = (
        centered_xy.transpose(0, 1)
        @ centered_xy
        / max(
            int(centered_xy.shape[0]),
            1,
        )
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= 1e-8:
        return []
    if float(eigenvalues[-1] / torch.clamp(eigenvalues[-2], min=1e-8)) < 1.2:
        return []

    zero = torch.zeros((), dtype=torch.float32, device=device)
    long_axis = torch.stack([eigenvectors[0, -1], eigenvectors[1, -1], zero])
    short_axis = torch.stack([eigenvectors[0, -2], eigenvectors[1, -2], zero])
    long_axis = _normalize_horizontal_axis(long_axis, device)
    short_axis = _normalize_horizontal_axis(short_axis, device)
    if not _coordinated_axis_pair_is_novel(long_axis, object_initial_pose, device):
        return []
    return [(long_axis, short_axis), (short_axis, long_axis)]


def _coordinated_axis_pair_is_novel(
    long_axis: torch.Tensor,
    object_initial_pose: torch.Tensor,
    device,
) -> bool:
    local_axes = []
    for axis_index in (0, 1):
        axis_local = torch.zeros(3, dtype=torch.float32, device=device)
        axis_local[axis_index] = 1.0
        try:
            local_axes.append(
                _normalize_horizontal_axis(
                    object_initial_pose[:3, :3] @ axis_local,
                    device,
                )
            )
        except ValueError:
            continue
    if not local_axes:
        return True
    max_alignment = max(abs(float(torch.dot(long_axis, axis))) for axis in local_axes)
    return max_alignment < 0.98


def _normalize_horizontal_axis(axis: torch.Tensor, device) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float32, device=device).clone()
    axis[2] = 0.0
    return _normalize_vector(axis)


def _make_coordinated_top_down_world_grasp_pair(
    *,
    first_world_pos: torch.Tensor,
    second_world_pos: torch.Tensor,
    separation_axis: torch.Tensor,
    closing_axis: torch.Tensor,
    object_initial_pose: torch.Tensor,
    env,
    device,
    priority: int,
    score_bias: float,
    axis_kind: str,
    env_id: int = 0,
) -> CoordinatedGraspPair:
    z_axis = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)
    x_axis = _orthogonalized_axis(closing_axis, z_axis)
    y_axis = _normalize_vector(torch.linalg.cross(z_axis, x_axis))
    separation_axis = _normalize_horizontal_axis(separation_axis, device)
    if float(torch.dot(y_axis, separation_axis)) < 0.0:
        x_axis = -x_axis
        y_axis = _normalize_vector(torch.linalg.cross(z_axis, x_axis))
    x_axis = _normalize_vector(x_axis)

    first_world = _pose_from_axes(
        position=first_world_pos,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
    )
    second_world = _pose_from_axes(
        position=second_world_pos,
        x_axis=-x_axis,
        y_axis=-y_axis,
        z_axis=z_axis,
    )
    left_world, right_world = _assign_coordinated_world_grasp_pair_to_arms(
        first_world,
        second_world,
        env=env,
        device=device,
        env_id=env_id,
    )
    return _make_coordinated_grasp_pair(
        _world_pose_to_object_pose(object_initial_pose, left_world),
        _world_pose_to_object_pose(object_initial_pose, right_world),
        object_initial_pose=object_initial_pose,
        env=env,
        device=device,
        env_id=env_id,
        priority=priority,
        score_bias=score_bias,
        axis_kind=axis_kind,
    )


def _coordinated_side_axis_index(extents: torch.Tensor) -> int:
    x_extent = float(extents[0])
    y_extent = float(extents[1])
    if abs(x_extent - y_extent) < 1e-6:
        return 1
    return 0 if x_extent < y_extent else 1


def _make_coordinated_side_grasp_pose(
    *,
    center: torch.Tensor,
    side_axis: torch.Tensor,
    side_sign: float,
    lateral_offset: float,
    vertical_offset: float,
) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32, device=center.device)
    pose[:3, 3] = center + side_axis * float(side_sign) * lateral_offset
    pose[2, 3] = center[2] + vertical_offset

    z_axis = _normalize_vector(-side_axis * float(side_sign))
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=center.device)
    x_axis = torch.linalg.cross(world_up, z_axis)
    if float(torch.linalg.norm(x_axis)) < 1e-6:
        x_axis = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float32, device=center.device
        )
    x_axis = _normalize_vector(x_axis)
    y_axis = _normalize_vector(torch.linalg.cross(z_axis, x_axis))
    pose[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=1)
    return pose


def _make_coordinated_grasp_pair(
    left_object_to_eef: torch.Tensor,
    right_object_to_eef: torch.Tensor,
    *,
    object_initial_pose: torch.Tensor,
    env,
    device,
    priority: int,
    score_bias: float,
    axis_kind: str,
    env_id: int = 0,
) -> CoordinatedGraspPair:
    left_world = object_initial_pose @ left_object_to_eef
    right_world = object_initial_pose @ right_object_to_eef
    score = _coordinated_grasp_pair_score(
        left_world,
        right_world,
        env=env,
        device=device,
        env_id=env_id,
    )
    return CoordinatedGraspPair(
        left_object_to_eef=left_object_to_eef,
        right_object_to_eef=right_object_to_eef,
        priority=int(priority),
        score=score + float(score_bias),
        axis_kind=axis_kind,
    )


def _coordinated_grasp_pair_world_y_angle_degrees(
    candidate: CoordinatedGraspPair,
    *,
    object_initial_pose: torch.Tensor,
) -> float:
    left_world = object_initial_pose @ candidate.left_object_to_eef
    right_world = object_initial_pose @ candidate.right_object_to_eef
    separation = left_world[:2, 3] - right_world[:2, 3]
    separation_norm = torch.linalg.norm(separation)
    if float(separation_norm) <= 1e-8:
        return 90.0
    world_y_alignment = torch.clamp(
        torch.abs(separation[1] / separation_norm),
        min=0.0,
        max=1.0,
    )
    return float(torch.rad2deg(torch.acos(world_y_alignment)))


def _coordinated_grasp_pair_score(
    left_world: torch.Tensor,
    right_world: torch.Tensor,
    *,
    env,
    device,
    env_id: int = 0,
) -> float:
    left_pos = left_world[:3, 3]
    right_pos = right_world[:3, 3]
    score = -0.2 * abs(float(left_pos[1] - right_pos[1]))
    score += 0.2 * abs(float(left_pos[0] - right_pos[0]))
    arm_positions = _current_arm_positions(env, device, env_id=env_id)
    if arm_positions is not None:
        left_arm_pos, right_arm_pos = arm_positions
        score += float(torch.linalg.norm(left_arm_pos - left_pos))
        score += float(torch.linalg.norm(right_arm_pos - right_pos))
    return score


def _assign_coordinated_world_grasp_pair_to_arms(
    first_pose: torch.Tensor,
    second_pose: torch.Tensor,
    *,
    env,
    device,
    env_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_world = first_pose[:3, 3]
    second_world = second_pose[:3, 3]
    arm_positions = _current_arm_positions(env, device, env_id=env_id)
    if arm_positions is not None:
        left_pos, right_pos = arm_positions
        direct_cost = torch.linalg.norm(left_pos - first_world) + torch.linalg.norm(
            right_pos - second_world
        )
        swapped_cost = torch.linalg.norm(left_pos - second_world) + torch.linalg.norm(
            right_pos - first_world
        )
        if float(swapped_cost) + 1e-6 < float(direct_cost):
            return second_pose, first_pose
        if float(direct_cost) + 1e-6 < float(swapped_cost):
            return first_pose, second_pose

    if float(first_world[1]) >= float(second_world[1]):
        return first_pose, second_pose
    return second_pose, first_pose


def _assign_coordinated_local_grasp_pair_to_arms(
    first_pose: torch.Tensor,
    second_pose: torch.Tensor,
    *,
    object_initial_pose: torch.Tensor,
    env,
    device,
    env_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_world = _transform_local_point(object_initial_pose, first_pose[:3, 3])
    second_world = _transform_local_point(object_initial_pose, second_pose[:3, 3])
    arm_positions = _current_arm_positions(env, device, env_id=env_id)
    if arm_positions is not None:
        left_pos, right_pos = arm_positions
        direct_cost = torch.linalg.norm(left_pos - first_world) + torch.linalg.norm(
            right_pos - second_world
        )
        swapped_cost = torch.linalg.norm(left_pos - second_world) + torch.linalg.norm(
            right_pos - first_world
        )
        if float(swapped_cost) + 1e-6 < float(direct_cost):
            return second_pose, first_pose
        if float(direct_cost) + 1e-6 < float(swapped_cost):
            return first_pose, second_pose

    if float(first_world[1]) >= float(second_world[1]):
        return first_pose, second_pose
    return second_pose, first_pose
