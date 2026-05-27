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

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions.core import ObjectSemantics
from embodichain.utils.logger import log_info

__all__ = [
    "SemanticGraspCandidatePlan",
    "SemanticGraspSelection",
    "apply_grasp_roll_offset",
    "format_semantic_candidate_message",
    "rank_semantic_grasp_candidates",
    "semantic_candidate_record_label",
    "select_ranked_semantic_grasp",
]


@dataclass
class SemanticGraspCandidatePlan:
    """Planned semantic grasp candidate and ranking metrics."""

    label: str
    direction: torch.Tensor
    candidate_idx: int
    roll_offset_rad: float
    open_length: float
    grasp_pose: torch.Tensor
    trajectory: torch.Tensor
    joint_ids: list[int]
    qpos_score: float
    grasp_height: float
    radial_ratio: float
    height_ratio: float
    geometry_score: float
    roll_score: float
    reference_pos_error: float | None
    reference_rot_error_rad: float | None
    reference_score: float | None

    @property
    def legacy_pos_error(self) -> float | None:
        """Compatibility alias for older logs and tests."""
        return self.reference_pos_error

    @property
    def legacy_rot_error_rad(self) -> float | None:
        """Compatibility alias for older logs and tests."""
        return self.reference_rot_error_rad

    @property
    def legacy_score(self) -> float | None:
        """Compatibility alias for older logs and tests."""
        return self.reference_score


@dataclass
class SemanticGraspSelection:
    """Result of ranked semantic grasp planning."""

    candidate: SemanticGraspCandidatePlan | None
    failures: list[str]
    trajectory: torch.Tensor
    joint_ids: list[int]

    @property
    def is_success(self) -> bool:
        return self.candidate is not None


def select_ranked_semantic_grasp(
    *,
    action: Any,
    target: ObjectSemantics,
    start_qpos: torch.Tensor,
    options: dict[str, Any],
    record_attempt: Callable[..., None] | None = None,
) -> SemanticGraspSelection:
    """Plan and select a semantic grasp candidate through a PickUpAction instance."""
    planned_candidates, failures = _plan_semantic_grasp_candidates(
        action=action,
        target=target,
        start_qpos=start_qpos,
        options=options,
        record_attempt=record_attempt,
    )
    ranked_candidates = rank_semantic_grasp_candidates(planned_candidates, options)
    if not ranked_candidates:
        return SemanticGraspSelection(
            candidate=None,
            failures=failures,
            trajectory=torch.empty(0, device=start_qpos.device),
            joint_ids=list(getattr(action, "joint_ids", [])),
        )

    selected = ranked_candidates[0]
    object_label = options.get("obj_name") or target.label
    object_text = f" for '{object_label}'" if object_label else ""
    log_info(
        f"Selected ranked public semantic grasp {selected.label}:"
        f"{selected.candidate_idx}{object_text} "
        f"({format_semantic_candidate_message(selected)}).",
        color="green",
    )
    if record_attempt is not None:
        record_attempt(
            label=semantic_candidate_record_label(selected),
            direction=selected.direction,
            status="selected",
            message=format_semantic_candidate_message(selected),
        )
    return SemanticGraspSelection(
        candidate=selected,
        failures=failures,
        trajectory=selected.trajectory,
        joint_ids=list(selected.joint_ids),
    )


def _plan_semantic_grasp_candidates(
    *,
    action: Any,
    target: ObjectSemantics,
    start_qpos: torch.Tensor,
    options: dict[str, Any],
    record_attempt: Callable[..., None] | None,
) -> tuple[list[SemanticGraspCandidatePlan], list[str]]:
    planned_candidates: list[SemanticGraspCandidatePlan] = []
    failures: list[str] = []
    use_reference_orientation = bool(
        options.get("public_grasp_use_legacy_orientation", False)
        or options.get("use_reference_orientation", False)
    )
    reference_pose = options.get("reference_grasp_pose")
    max_reference_pos_error = _optional_float(
        options,
        "public_grasp_legacy_pose_max_position_error",
        "reference_pose_max_position_error",
    )
    max_reference_rot_error = _optional_float(
        options,
        "public_grasp_legacy_pose_max_rotation_error",
        "reference_pose_max_rotation_error",
    )
    min_open_length = _optional_float(
        options,
        "public_grasp_candidate_min_open_length",
        "candidate_min_open_length",
    )
    max_open_length = _optional_float(
        options,
        "public_grasp_candidate_max_open_length",
        "candidate_max_open_length",
    )
    geometry_bounds = _resolve_geometry_bounds(target, options)
    attempt_context = str(options.get("attempt_context", ""))
    roll_offsets = _semantic_roll_offsets(options, use_reference_orientation)
    directions = _semantic_approach_directions(action, options)

    original_direction = action.approach_direction.detach().clone()
    original_offset_world = action.grasp_pose_offset_world.detach().clone()
    original_offset_along = float(action.cfg.grasp_pose_offset_along_approach)

    try:
        for label, direction in directions:
            direction = _as_direction_tensor(direction, device=action.device)
            _set_action_direction_and_offsets(
                action,
                direction=direction,
                options=options,
                original_offset_world=original_offset_world,
                original_offset_along=original_offset_along,
            )
            try:
                is_success, grasp_xpos, open_length = (
                    action._resolve_grasp_pose_candidates(target)
                )
            except Exception as exc:
                message = (
                    f"{label}: resolve_exception={type(exc).__name__}: {exc}; "
                    f"{attempt_context}"
                )
                failures.append(message)
                _record_attempt(
                    record_attempt,
                    label=label,
                    direction=direction,
                    status="resolve_failed",
                    message=message,
                )
                continue

            if (
                not bool(torch.as_tensor(is_success).all().item())
                or grasp_xpos.numel() == 0
            ):
                message = (
                    f"{label}: no semantic grasp candidates survived affordance "
                    f"filtering; {attempt_context}"
                )
                failures.append(message)
                _record_attempt(
                    record_attempt,
                    label=label,
                    direction=direction,
                    status="no_candidate",
                    message=message,
                )
                continue

            for candidate_idx in range(grasp_xpos.shape[1]):
                base_candidate_label = f"{label}:{candidate_idx}"
                candidate_open_length = float(
                    open_length[:, candidate_idx].mean().item()
                )
                if not torch.all(open_length[:, candidate_idx] > 0):
                    _record_attempt(
                        record_attempt,
                        label=base_candidate_label,
                        direction=direction,
                        status="filtered",
                        message=(
                            f"open_length={candidate_open_length:.5f} <= 0; "
                            f"{attempt_context}"
                        ),
                    )
                    continue
                if (
                    min_open_length is not None
                    and candidate_open_length < min_open_length
                ):
                    _record_attempt(
                        record_attempt,
                        label=base_candidate_label,
                        direction=direction,
                        status="filtered",
                        message=(
                            f"open_length={candidate_open_length:.5f}"
                            f"<{min_open_length:.5f}; {attempt_context}"
                        ),
                    )
                    continue
                if (
                    max_open_length is not None
                    and candidate_open_length > max_open_length
                ):
                    _record_attempt(
                        record_attempt,
                        label=base_candidate_label,
                        direction=direction,
                        status="filtered",
                        message=(
                            f"open_length={candidate_open_length:.5f}"
                            f">{max_open_length:.5f}; {attempt_context}"
                        ),
                    )
                    continue

                base_candidate_pose = grasp_xpos[:, candidate_idx]
                for roll_offset_rad in roll_offsets:
                    candidate_pose = base_candidate_pose
                    if use_reference_orientation and reference_pose is not None:
                        candidate_pose = _apply_reference_orientation(
                            candidate_pose,
                            reference_pose,
                        )
                    else:
                        candidate_pose = apply_grasp_roll_offset(
                            candidate_pose,
                            roll_offset_rad,
                        )
                    score_pose = action._apply_grasp_pose_offset(candidate_pose)[0]
                    (
                        grasp_height,
                        radial_ratio,
                        height_ratio,
                        geometry_score,
                    ) = _semantic_grasp_geometry_metrics(
                        score_pose,
                        direction,
                        geometry_bounds,
                        options,
                    )
                    (
                        reference_pos_error,
                        reference_rot_error,
                        reference_score,
                    ) = _reference_pose_errors(score_pose, reference_pose, options)

                    candidate_label = base_candidate_label
                    if abs(roll_offset_rad) > 1e-6:
                        candidate_label = (
                            f"{candidate_label}:roll"
                            f"{math.degrees(roll_offset_rad):+.0f}"
                        )

                    filter_reasons = []
                    if (
                        max_reference_pos_error is not None
                        and reference_pos_error is not None
                        and reference_pos_error > max_reference_pos_error
                    ):
                        filter_reasons.append(
                            f"legacy_pos_error>{max_reference_pos_error:.5f}"
                        )
                    if (
                        max_reference_rot_error is not None
                        and reference_rot_error is not None
                        and reference_rot_error > max_reference_rot_error
                    ):
                        filter_reasons.append(
                            f"legacy_rot_error>{max_reference_rot_error:.5f}"
                        )
                    if filter_reasons:
                        _record_attempt(
                            record_attempt,
                            label=candidate_label,
                            direction=direction,
                            status="filtered",
                            message=(
                                ";".join(filter_reasons)
                                + f"; open_length={candidate_open_length:.5f}, "
                                f"legacy_pos_error={_format_optional_metric(reference_pos_error)}, "
                                f"legacy_rot_error={_format_optional_metric(reference_rot_error)}; "
                                f"{attempt_context}"
                            ),
                        )
                        continue

                    is_plan_success, trajectory = action._plan_candidate_pickup(
                        candidate_pose,
                        start_qpos,
                        candidate_idx=candidate_idx,
                    )
                    if not is_plan_success:
                        message = f"{candidate_label}: plan_failed; {attempt_context}"
                        failures.append(message)
                        _record_attempt(
                            record_attempt,
                            label=candidate_label,
                            direction=direction,
                            status="plan_failed",
                            message=message,
                        )
                        continue

                    final_arm_qpos = trajectory[:, -1, : start_qpos.shape[-1]]
                    qpos_distance = torch.linalg.norm(
                        final_arm_qpos - start_qpos,
                        dim=-1,
                    )
                    qpos_score = (
                        float(qpos_distance.mean().item())
                        + 0.01 * candidate_idx
                        + 0.001 * abs(float(roll_offset_rad))
                    )
                    planned_candidate = SemanticGraspCandidatePlan(
                        label=label,
                        direction=direction.detach().clone(),
                        candidate_idx=candidate_idx,
                        roll_offset_rad=float(roll_offset_rad),
                        open_length=candidate_open_length,
                        grasp_pose=score_pose.detach().clone(),
                        trajectory=trajectory.detach().clone(),
                        joint_ids=list(action.joint_ids),
                        qpos_score=qpos_score,
                        grasp_height=grasp_height,
                        radial_ratio=radial_ratio,
                        height_ratio=height_ratio,
                        geometry_score=geometry_score,
                        roll_score=_semantic_roll_score(
                            float(roll_offset_rad),
                            options,
                        ),
                        reference_pos_error=reference_pos_error,
                        reference_rot_error_rad=reference_rot_error,
                        reference_score=reference_score,
                    )
                    planned_candidates.append(planned_candidate)
                    _record_attempt(
                        record_attempt,
                        label=candidate_label,
                        direction=direction,
                        status="planned",
                        message=(
                            f"{format_semantic_candidate_message(planned_candidate)}; "
                            f"{attempt_context}"
                        ),
                    )
    finally:
        action.approach_direction = original_direction
        action.grasp_pose_offset_world = original_offset_world
        action.cfg.grasp_pose_offset_along_approach = original_offset_along

    return planned_candidates, failures


def rank_semantic_grasp_candidates(
    candidates: list[SemanticGraspCandidatePlan],
    options: dict[str, Any],
) -> list[SemanticGraspCandidatePlan]:
    """Rank planned semantic grasp candidates."""
    candidates = _filter_auto_general_lateral_height_candidates(candidates, options)
    candidates = _filter_auto_general_geometry_candidates(candidates, options)
    rank_mode = str(options.get("public_grasp_rank_mode", "")).strip().lower()
    if rank_mode == "planned_order":
        return candidates

    rank_by_reference = bool(
        options.get("public_grasp_rank_by_legacy_pose", False)
        or options.get("rank_by_reference_pose", False)
    )
    strategy = _semantic_strategy(options)
    direction_order: dict[str, int] = {}
    if strategy == "auto_general":
        for candidate in candidates:
            direction_order.setdefault(candidate.label, len(direction_order))
    prefer_thin_top_down = bool(
        options.get("public_grasp_is_thin_object", False)
        or options.get("thin_object", False)
    ) and bool(options.get("public_grasp_thin_object_prefer_top_down", True))

    def sort_key(candidate: SemanticGraspCandidatePlan):
        primary = candidate.qpos_score
        reference_score = getattr(
            candidate,
            "reference_score",
            getattr(candidate, "legacy_score", None),
        )
        if rank_by_reference and reference_score is not None:
            primary = reference_score
        elif strategy == "auto_general":
            primary = candidate.geometry_score + candidate.roll_score
        direction_priority = direction_order.get(candidate.label, 0)
        if prefer_thin_top_down:
            direction = candidate.direction.detach().to(dtype=torch.float32)
            direction = direction / torch.linalg.norm(direction).clamp_min(1e-6)
            direction_priority = 0 if float(direction[2].item()) <= -0.75 else 1
        return (
            direction_priority,
            primary,
            candidate.geometry_score,
            candidate.roll_score,
            candidate.qpos_score,
            candidate.label,
            candidate.candidate_idx,
        )

    return sorted(candidates, key=sort_key)


def format_semantic_candidate_message(candidate: SemanticGraspCandidatePlan) -> str:
    """Format candidate metrics for logs and attempt records."""
    return (
        f"candidate_idx={candidate.candidate_idx}, "
        f"roll_offset_rad={candidate.roll_offset_rad:.5f}, "
        f"open_length={candidate.open_length:.5f}, "
        f"qpos_score={candidate.qpos_score:.5f}, "
        f"geometry_score={candidate.geometry_score:.5f}, "
        f"grasp_height={candidate.grasp_height:.5f}, "
        f"radial_ratio={candidate.radial_ratio:.5f}, "
        f"height_ratio={candidate.height_ratio:.5f}, "
        f"roll_score={candidate.roll_score:.5f}, "
        f"legacy_pos_error={_format_optional_metric(candidate.reference_pos_error)}, "
        f"legacy_rot_error={_format_optional_metric(candidate.reference_rot_error_rad)}, "
        f"legacy_score={_format_optional_metric(candidate.reference_score)}"
    )


def semantic_candidate_record_label(candidate: SemanticGraspCandidatePlan) -> str:
    """Return a stable label for a selected semantic grasp candidate."""
    label = f"{candidate.label}:{candidate.candidate_idx}"
    if abs(candidate.roll_offset_rad) <= 1e-6:
        return label
    roll_deg = math.degrees(candidate.roll_offset_rad)
    return f"{label}:roll{roll_deg:+.0f}"


def apply_grasp_roll_offset(
    pose: torch.Tensor,
    roll_offset_rad: float,
) -> torch.Tensor:
    """Apply a roll offset around a grasp pose local z axis."""
    if abs(float(roll_offset_rad)) <= 1e-6:
        return pose
    cos_roll = math.cos(float(roll_offset_rad))
    sin_roll = math.sin(float(roll_offset_rad))
    roll = torch.tensor(
        [
            [cos_roll, -sin_roll, 0.0],
            [sin_roll, cos_roll, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=pose.dtype,
        device=pose.device,
    )
    shifted_pose = pose.clone()
    shifted_pose[..., :3, :3] = shifted_pose[..., :3, :3] @ roll
    return shifted_pose


def _semantic_approach_directions(
    action: Any,
    options: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    directions = options.get("approach_directions")
    if not directions:
        directions = options.get("public_grasp_approach_directions")
    if directions:
        return [
            (str(label), _as_direction_tensor(direction, device=action.device))
            for label, direction in directions
        ]
    return [("configured", action.approach_direction.detach().clone())]


def _semantic_roll_offsets(
    options: dict[str, Any],
    use_reference_orientation: bool,
) -> list[float]:
    if use_reference_orientation:
        return [0.0]
    configured = options.get("roll_offsets")
    if configured is None:
        configured = options.get("public_grasp_roll_offsets")
    if configured is None:
        return [0.0]
    if isinstance(configured, str):
        configured = [
            value.strip()
            for value in configured.replace(";", ",").split(",")
            if value.strip()
        ]
    values = [float(value) for value in list(configured)]
    return values or [0.0]


def _set_action_direction_and_offsets(
    action: Any,
    *,
    direction: torch.Tensor,
    options: dict[str, Any],
    original_offset_world: torch.Tensor,
    original_offset_along: float,
) -> None:
    action.approach_direction = direction
    offset_world, offset_along = _semantic_pose_offsets(
        options,
        direction=direction,
        default_offset_world=original_offset_world,
        default_offset_along=original_offset_along,
    )
    action.grasp_pose_offset_world = torch.as_tensor(
        offset_world,
        dtype=torch.float32,
        device=action.device,
    )
    action.cfg.grasp_pose_offset_along_approach = float(offset_along)


def _semantic_pose_offsets(
    options: dict[str, Any],
    *,
    direction: torch.Tensor,
    default_offset_world: torch.Tensor,
    default_offset_along: float,
) -> tuple[Any, float]:
    offset_world = options.get("public_grasp_pose_offset_world")
    if offset_world is None:
        offset_world = options.get("pose_offset_world", default_offset_world)
    offset_along = options.get("public_grasp_pose_offset_along_approach")
    if offset_along is None:
        offset_along = options.get("pose_offset_along_approach", default_offset_along)
    offset_along = float(offset_along or 0.0)
    if _semantic_strategy(options) != "auto_general":
        return offset_world, offset_along

    direction = direction.detach().to(dtype=torch.float32)
    direction = direction / torch.linalg.norm(direction).clamp_min(1e-6)
    is_lateral = abs(float(direction[2].item())) < 0.75
    if _is_zero_vector_like(offset_world):
        offset_world = [0.0, 0.0, 0.011 if is_lateral else 0.006]
    if offset_along == 0.0:
        offset_along = 0.025 if is_lateral else 0.015
    return offset_world, offset_along


def _resolve_geometry_bounds(
    target: ObjectSemantics,
    options: dict[str, Any],
) -> dict[str, torch.Tensor] | None:
    geometry_bounds = options.get("geometry_bounds")
    if geometry_bounds is not None:
        return geometry_bounds
    geometry_bounds = target.properties.get("geometry_bounds")
    if geometry_bounds is not None:
        return geometry_bounds
    return _geometry_bounds_from_semantics(target)


def _geometry_bounds_from_semantics(
    target: ObjectSemantics,
) -> dict[str, torch.Tensor] | None:
    vertices = target.geometry.get("mesh_vertices")
    if vertices is None:
        return None
    vertices = torch.as_tensor(vertices, dtype=torch.float32)
    if vertices.numel() == 0:
        return None
    if target.entity is not None:
        pose = target.entity.get_local_pose(to_matrix=True).squeeze(0)
        pose = torch.as_tensor(pose, dtype=torch.float32, device=vertices.device)
        world_vertices = vertices @ pose[:3, :3].transpose(0, 1)
        world_vertices = world_vertices + pose[:3, 3]
    else:
        world_vertices = vertices
    mins = world_vertices.min(dim=0).values
    maxs = world_vertices.max(dim=0).values
    extents = maxs - mins
    return {
        "center": ((mins + maxs) * 0.5).detach(),
        "mins": mins.detach(),
        "maxs": maxs.detach(),
        "extents": extents.detach(),
        "xy_radius": torch.clamp(extents[:2].max() * 0.5, min=1e-6).detach(),
        "height": torch.clamp(extents[2], min=1e-6).detach(),
    }


def _semantic_grasp_geometry_metrics(
    grasp_pose: torch.Tensor,
    direction: torch.Tensor,
    geometry_bounds: dict[str, torch.Tensor] | None,
    options: dict[str, Any],
) -> tuple[float, float, float, float]:
    if geometry_bounds is None:
        return float(grasp_pose[2, 3].item()), 0.0, 0.0, 0.0

    pose_device = grasp_pose.device
    pose_dtype = grasp_pose.dtype
    center = geometry_bounds["center"].to(device=pose_device, dtype=pose_dtype)
    mins = geometry_bounds["mins"].to(device=pose_device, dtype=pose_dtype)
    xy_radius = geometry_bounds["xy_radius"].to(
        device=pose_device,
        dtype=pose_dtype,
    )
    height = geometry_bounds["height"].to(device=pose_device, dtype=pose_dtype)
    direction = direction.to(device=pose_device, dtype=pose_dtype)
    direction = direction / torch.linalg.norm(direction).clamp_min(1e-6)

    position = grasp_pose[:3, 3]
    xy_distance = torch.linalg.norm(position[:2] - center[:2])
    radial_ratio = float((xy_distance / xy_radius).item())
    height_ratio = float(((position[2] - mins[2]) / height).clamp(0.0, 1.5).item())
    grasp_height = float(position[2].item())

    is_top_down = abs(float(direction[2].item())) >= 0.75
    if is_top_down:
        target_radial = float(
            options.get("public_grasp_top_down_target_radial_ratio", 0.65)
        )
        min_radial = float(options.get("public_grasp_top_down_min_radial_ratio", 0.18))
        min_height = float(options.get("public_grasp_top_down_min_height_ratio", 0.30))
        target_height = options.get("public_grasp_top_down_target_height_ratio")
        height_weight = float(
            options.get("public_grasp_top_down_target_height_weight", 0.0)
        )
    else:
        target_radial = float(
            options.get("public_grasp_lateral_target_radial_ratio", 0.80)
        )
        min_radial = float(options.get("public_grasp_lateral_min_radial_ratio", 0.35))
        min_height = float(options.get("public_grasp_lateral_min_height_ratio", 0.20))
        target_height = options.get("public_grasp_lateral_target_height_ratio")
        height_weight = float(
            options.get("public_grasp_lateral_target_height_weight", 0.0)
        )

    max_radial = float(options.get("public_grasp_max_radial_ratio", 1.80))
    radial_penalty = abs(radial_ratio - target_radial)
    center_penalty = max(0.0, min_radial - radial_ratio) * 4.0
    low_height_penalty = max(0.0, min_height - height_ratio) * 3.0
    overreach_penalty = max(0.0, radial_ratio - max_radial)
    target_height_penalty = 0.0
    if target_height is not None:
        target_height_penalty = height_weight * abs(height_ratio - float(target_height))
    geometry_score = (
        radial_penalty
        + center_penalty
        + low_height_penalty
        + overreach_penalty
        + target_height_penalty
    )
    return grasp_height, radial_ratio, height_ratio, geometry_score


def _reference_pose_errors(
    candidate_pose: torch.Tensor,
    reference_pose: torch.Tensor | None,
    options: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    if reference_pose is None:
        return None, None, None
    reference_pose = reference_pose.to(
        dtype=candidate_pose.dtype,
        device=candidate_pose.device,
    )
    pos_error = float(
        torch.linalg.norm(candidate_pose[:3, 3] - reference_pose[:3, 3]).item()
    )
    rot_error = _rotation_error_rad(candidate_pose, reference_pose)
    score = (
        float(options.get("public_grasp_legacy_pose_position_weight", 1.0)) * pos_error
        + float(options.get("public_grasp_legacy_pose_rotation_weight", 0.05))
        * rot_error
    )
    return pos_error, rot_error, score


def _apply_reference_orientation(
    pose: torch.Tensor,
    reference_pose: torch.Tensor,
) -> torch.Tensor:
    reference_rot = reference_pose[:3, :3].to(dtype=pose.dtype, device=pose.device)
    oriented_pose = pose.clone()
    oriented_pose[:, :3, :3] = reference_rot
    return oriented_pose


def _rotation_error_rad(
    candidate_pose: torch.Tensor,
    reference_pose: torch.Tensor,
) -> float:
    candidate_rot = candidate_pose[:3, :3]
    reference_rot = reference_pose[:3, :3].to(
        dtype=candidate_rot.dtype,
        device=candidate_rot.device,
    )
    delta_rot = reference_rot.transpose(0, 1) @ candidate_rot
    cos_angle = ((torch.trace(delta_rot) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.acos(cos_angle).item())


def _semantic_roll_score(roll_offset_rad: float, options: dict[str, Any]) -> float:
    preferred = options.get("public_grasp_preferred_roll_offset")
    if preferred is None and _semantic_strategy(options) == "auto_general":
        preferred = options.get("auto_general_preferred_roll_offset")
    if preferred is None:
        return 0.0
    weight = float(options.get("public_grasp_roll_preference_weight", 0.04))
    return _angular_distance_rad(float(roll_offset_rad), float(preferred)) * weight


def _filter_auto_general_lateral_height_candidates(
    candidates: list[SemanticGraspCandidatePlan],
    options: dict[str, Any],
) -> list[SemanticGraspCandidatePlan]:
    if _semantic_strategy(options) != "auto_general":
        return candidates
    if not bool(options.get("public_grasp_filter_lateral_low_height", True)):
        return candidates

    filtered: list[SemanticGraspCandidatePlan] = []
    lateral_groups: dict[str, list[SemanticGraspCandidatePlan]] = {}
    for candidate in candidates:
        direction = candidate.direction.detach().to(dtype=torch.float32)
        direction = direction / torch.linalg.norm(direction).clamp_min(1e-6)
        if abs(float(direction[2].item())) < 0.75:
            lateral_groups.setdefault(candidate.label, []).append(candidate)
        else:
            filtered.append(candidate)

    quantile = float(options.get("public_grasp_lateral_min_height_quantile", 0.25))
    quantile = min(max(quantile, 0.0), 1.0)
    for group in lateral_groups.values():
        if len(group) < 4:
            filtered.extend(group)
            continue
        z_values = [float(candidate.grasp_pose[2, 3].item()) for candidate in group]
        min_z = float(np.quantile(np.asarray(z_values, dtype=np.float32), quantile))
        kept = [
            candidate
            for candidate in group
            if float(candidate.grasp_pose[2, 3].item()) >= min_z - 1e-6
        ]
        filtered.extend(kept or group)

    return filtered or candidates


def _filter_auto_general_geometry_candidates(
    candidates: list[SemanticGraspCandidatePlan],
    options: dict[str, Any],
) -> list[SemanticGraspCandidatePlan]:
    if _semantic_strategy(options) != "auto_general":
        return candidates
    if not bool(options.get("public_grasp_filter_geometry_outliers", True)):
        return candidates

    grouped: dict[str, list[SemanticGraspCandidatePlan]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.label, []).append(candidate)

    filtered: list[SemanticGraspCandidatePlan] = []
    min_group_size = int(options.get("public_grasp_geometry_filter_min_group_size", 4))
    for group in grouped.values():
        if len(group) < min_group_size:
            filtered.extend(group)
            continue

        direction = group[0].direction.detach().to(dtype=torch.float32)
        direction = direction / torch.linalg.norm(direction).clamp_min(1e-6)
        is_top_down = abs(float(direction[2].item())) >= 0.75
        if is_top_down:
            min_radial = float(
                options.get("public_grasp_top_down_min_radial_ratio", 0.18)
            )
            min_height = float(
                options.get("public_grasp_top_down_min_height_ratio", 0.30)
            )
        else:
            min_radial = float(
                options.get("public_grasp_lateral_min_radial_ratio", 0.35)
            )
            min_height = float(
                options.get("public_grasp_lateral_min_height_ratio", 0.20)
            )

        kept = [
            candidate
            for candidate in group
            if candidate.radial_ratio >= min_radial
            and candidate.height_ratio >= min_height
        ]
        filtered.extend(kept or group)

    return filtered or candidates


def _optional_float(options: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = options.get(name)
        if value is not None:
            return float(value)
    return None


def _semantic_strategy(options: dict[str, Any]) -> str | None:
    strategy = options.get("public_grasp_strategy", options.get("strategy"))
    if strategy is None:
        return None
    strategy = str(strategy).strip().lower().replace("-", "_")
    if not strategy or strategy in {"none", "default"}:
        return None
    return strategy


def _as_direction_tensor(direction, *, device) -> torch.Tensor:
    tensor = torch.as_tensor(direction, dtype=torch.float32, device=device)
    return tensor / torch.linalg.norm(tensor).clamp_min(1e-6)


def _is_zero_vector_like(value) -> bool:
    if value is None:
        return True
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    except Exception:
        return False
    return bool(tensor.numel() == 3 and torch.linalg.norm(tensor).item() <= 1e-8)


def _angular_distance_rad(angle_a: float, angle_b: float) -> float:
    return abs((angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi)


def _format_optional_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.5f}"


def _record_attempt(
    record_attempt: Callable[..., None] | None,
    **kwargs: Any,
) -> None:
    if record_attempt is not None:
        record_attempt(**kwargs)
