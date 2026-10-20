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

"""Measured physical stage evaluators and observation-only evidence helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import torch

from embodichain.utils.math import pose_inv

from ..contracts import ExecutionObservation, PhysicalEvaluation, ReplayObservationTrace
from ..evaluation import TOLERANCES, StageEvidence, evaluate_skill_stage_batch
from ..models import BenchmarkCase, CaseOutcome
from .atomic_case_utils import case_pose

__all__ = [
    "signed_rotation_about_axis",
    "motion_valid_mask",
    "executed_final_pose_errors",
    "stage_evidence_batch",
    "trace_validity",
    "merge_stage_evidence_batches",
    "replay_trace",
    "minimum_height_above_initial_support",
    "gripper_released",
    "gripper_holding",
    "pose_rotation_error",
    "PickUpEvaluator",
    "MoveHeldObjectEvaluator",
    "PlaceEvaluator",
    "PressEvaluator",
    "SlideEvaluator",
    "TwistEvaluator",
    "MoveEndEffectorEvaluator",
    "MoveJointsEvaluator",
]


def signed_rotation_about_axis(
    reference_rotation: torch.Tensor,
    current_rotation: torch.Tensor,
    world_axis: torch.Tensor,
) -> torch.Tensor:
    """Return the signed angle of ``current @ reference^T`` about ``world_axis``.

    The sine comes from the relative rotation's skew part projected on the
    axis and the cosine from its trace, so a rotation purely about the axis is
    recovered exactly over the full ``(-pi, pi]`` range.
    """
    relative = current_rotation @ reference_rotation.transpose(-1, -2)
    skew = 0.5 * torch.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        dim=-1,
    )
    sine = (skew * world_axis).sum(dim=-1)
    cosine = 0.5 * (torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0)
    return torch.atan2(sine, cosine)


def motion_valid_mask(
    motion_outcomes: tuple[CaseOutcome, ...], *, device: torch.device
) -> torch.Tensor:
    """Return the common motion-valid result as one device-local mask."""
    return torch.tensor(
        [item.motion_valid for item in motion_outcomes],
        dtype=torch.bool,
        device=device,
    )


def executed_final_pose_errors(
    case: BenchmarkCase,
    final_tcp_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final TCP errors under the case's declared rotation symmetry."""
    target = case.target_waypoints[:, -1]
    translation = torch.linalg.vector_norm(
        final_tcp_pose[:, :3, 3] - target[:, :3, 3], dim=-1
    )
    rotation = pose_rotation_error(final_tcp_pose, target)
    rotation_symmetry = case.case_parameters.get("waypoint_rotation_symmetry")
    if rotation_symmetry == "half_turn_about_z":
        symmetric_target = target.clone()
        symmetric_target[:, :3, :2] = -symmetric_target[:, :3, :2]
        symmetric_rotation = pose_rotation_error(final_tcp_pose, symmetric_target)
        rotation = torch.minimum(rotation, symmetric_rotation)
    elif rotation_symmetry is not None:
        raise ValueError(
            "waypoint_rotation_symmetry must be None or "
            f"'half_turn_about_z', got {rotation_symmetry!r}."
        )
    return translation, rotation


def stage_evidence_batch(
    *,
    stage_id: str,
    window: str,
    measurements: Mapping[str, torch.Tensor],
    method: str,
    valid_mask: torch.Tensor | None = None,
    source: Literal["simulation_state", "contact_sensor", "measured_proxy"] = (
        "simulation_state"
    ),
) -> tuple[Mapping[str, StageEvidence], ...]:
    """Convert measured batch scalars into row-local standard evidence."""
    if not measurements:
        raise ValueError("Stage evidence requires at least one measurement.")
    values = tuple(measurements.values())
    if any(value.dim() != 1 for value in values):
        raise ValueError("Stage measurement values must be one-dimensional batches.")
    batch_size = values[0].shape[0]
    if any(value.shape[0] != batch_size for value in values):
        raise ValueError("Stage measurement batches must have the same size.")
    if valid_mask is not None and (
        valid_mask.dtype != torch.bool or valid_mask.shape != (batch_size,)
    ):
        raise ValueError("Evidence validity must be a boolean environment vector.")

    def scalar(value: torch.Tensor, index: int) -> float | bool:
        item = value[index].item()
        return bool(item) if value.dtype == torch.bool else float(item)

    rows: list[Mapping[str, StageEvidence]] = []
    for env_index in range(batch_size):
        if valid_mask is not None and not bool(valid_mask[env_index].item()):
            rows.append({})
            continue
        row_measurements = {
            name: scalar(value, env_index) for name, value in measurements.items()
        }
        if any(
            type(value) is not bool and not math.isfinite(value)
            for value in row_measurements.values()
        ):
            rows.append({})
            continue
        rows.append(
            {
                stage_id: StageEvidence(
                    window=window,
                    source=source,
                    measurements=row_measurements,
                    method=method,
                )
            }
        )
    return tuple(rows)


def trace_validity(
    *traces: ReplayObservationTrace,
    observation: ExecutionObservation,
) -> torch.Tensor:
    """Preserve row-local raw-data validity before deriving boolean proxies."""
    valid = None
    for trace in traces:
        values = [
            value
            for value in (
                trace.tcp_pose,
                trace.object_pose,
                trace.articulation_joint,
                *trace.gripper_qpos.values(),
                *trace.scene_object_poses.values(),
            )
            if value is not None
        ]
        for value in values:
            rows = torch.isfinite(value).movedim(1, 0).flatten(start_dim=1).all(dim=1)
            valid = rows if valid is None else valid & rows
        for part in trace.gripper_qpos:
            for commands in (
                observation.gripper_open_qpos,
                observation.gripper_grasp_qpos,
            ):
                command = commands.get(part)
                if command is not None:
                    finite = torch.isfinite(command)
                    rows = finite.all() if command.dim() == 1 else finite.all(dim=-1)
                    valid = valid & rows.to(valid.device)
    if valid is None:
        raise ValueError("Evidence validity requires a measured trace.")
    return valid


def merge_stage_evidence_batches(
    *batches: Sequence[Mapping[str, StageEvidence]],
) -> tuple[Mapping[str, StageEvidence], ...]:
    """Merge stage batches while preserving stable environment order."""
    if not batches:
        return ()
    batch_size = len(batches[0])
    if any(len(batch) != batch_size for batch in batches):
        raise ValueError("Stage evidence batches must have the same environment size.")
    merged: list[dict[str, StageEvidence]] = []
    for env_index in range(batch_size):
        row: dict[str, StageEvidence] = {}
        for batch in batches:
            overlap = set(row) & set(batch[env_index])
            if overlap:
                raise ValueError(f"Duplicate stage evidence: {sorted(overlap)}.")
            row.update(batch[env_index])
        merged.append(row)
    return tuple(merged)


def replay_trace(
    observation: ExecutionObservation, observation_id: str
) -> ReplayObservationTrace | None:
    """Return one measured replay trace without fabricating missing evidence."""
    return observation.replay_traces.get(observation_id)


def minimum_height_above_initial_support(
    trace: ReplayObservationTrace,
    case: BenchmarkCase,
) -> torch.Tensor | None:
    """Return the minimum object-reference height above its frozen start plane."""
    if trace.object_pose is None:
        return None
    initial = torch.as_tensor(
        case.case_parameters["object_initial_pose"],
        dtype=trace.object_pose.dtype,
        device=trace.object_pose.device,
    )
    return trace.object_pose[:, :, 2, 3].min(dim=0).values - initial[:, 2, 3]


def gripper_released(
    trace: ReplayObservationTrace,
    observation: ExecutionObservation,
    part: str,
) -> torch.Tensor | None:
    """Return whether the final window sample is nearer open than grasp."""
    qpos = trace.gripper_qpos.get(part)
    open_qpos = observation.gripper_open_qpos.get(part)
    grasp_qpos = observation.gripper_grasp_qpos.get(part)
    if qpos is None or open_qpos is None or grasp_qpos is None:
        return None
    final = qpos[-1]
    open_qpos = open_qpos.to(device=final.device, dtype=final.dtype)
    grasp_qpos = grasp_qpos.to(device=final.device, dtype=final.dtype)
    return torch.linalg.vector_norm(
        final - open_qpos, dim=-1
    ) <= torch.linalg.vector_norm(final - grasp_qpos, dim=-1)


def gripper_holding(
    trace: ReplayObservationTrace,
    observation: ExecutionObservation,
    part: str,
    *,
    gap: float = 1.0e-3,
) -> torch.Tensor | None:
    """Return whether the final sample stopped between open and grasp commands."""
    qpos = trace.gripper_qpos.get(part)
    open_qpos = observation.gripper_open_qpos.get(part)
    grasp_qpos = observation.gripper_grasp_qpos.get(part)
    if qpos is None or open_qpos is None or grasp_qpos is None:
        return None
    final = qpos[-1]
    open_qpos = open_qpos.to(device=final.device, dtype=final.dtype)
    grasp_qpos = grasp_qpos.to(device=final.device, dtype=final.dtype)
    direction = torch.sign(grasp_qpos - open_qpos)
    closed = (final - open_qpos) * direction
    remaining = (grasp_qpos - final) * direction
    return ((closed >= gap) & (remaining >= gap)).all(dim=-1)


def pose_rotation_error(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the geodesic rotation error for matching pose batches."""
    relative = target[..., :3, :3].transpose(-1, -2) @ actual[..., :3, :3]
    trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1)
    return torch.arccos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))


class PickUpEvaluator:
    """Evaluate grasp, lift, and terminal retention from replayed state."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        grasp = replay_trace(observation, "pickup_grasp")
        lift = replay_trace(observation, "pickup_lift")
        settle = replay_trace(observation, "pickup_settle")
        part = next(iter(observation.gripper_open_qpos), "")
        batches: list[Sequence[Mapping[str, StageEvidence]]] = []
        if grasp is not None:
            established = gripper_holding(grasp, observation, part)
            if established is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(grasp, observation=observation),
                        stage_id="grasped",
                        window="grasp_end",
                        measurements={"grasp_established": established},
                        method="gripper_joint_state",
                        source="measured_proxy",
                    )
                )
        if lift is not None and lift.object_pose is not None:
            initial = torch.as_tensor(
                case.case_parameters["object_initial_pose"],
                dtype=lift.object_pose.dtype,
                device=lift.object_pose.device,
            )
            min_height = minimum_height_above_initial_support(lift, case)
            if min_height is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(lift, observation=observation),
                        stage_id="lifted",
                        window="lift_end",
                        measurements={
                            "lift_delta_m": lift.object_pose[-1, :, 2, 3]
                            - initial[:, 2, 3],
                            "min_height_above_support_m": min_height,
                        },
                        method="object_lift_trace",
                    )
                )
        if settle is not None:
            retained = gripper_holding(settle, observation, part)
            min_height = minimum_height_above_initial_support(settle, case)
            if (
                retained is not None
                and min_height is not None
                and lift is not None
                and lift.tcp_pose is not None
                and lift.object_pose is not None
                and settle.tcp_pose is not None
                and settle.object_pose is not None
            ):
                reference = pose_inv(lift.tcp_pose[-1]) @ lift.object_pose[-1]
                relative = pose_inv(settle.tcp_pose) @ settle.object_pose
                position_drift = torch.linalg.vector_norm(
                    relative[..., :3, 3] - reference[None, :, :3, 3], dim=-1
                ).amax(dim=0)
                rotation_drift = pose_rotation_error(
                    relative, reference[None].expand_as(relative)
                ).amax(dim=0)
                initial = case_pose(case, "object_initial_pose", device=relative.device)
                lift_remaining = (
                    settle.object_pose[:, :, 2, 3].amin(dim=0) - initial[:, 2, 3]
                )
                retained = (
                    retained
                    & (lift_remaining >= TOLERANCES["pick_min_lift_m"])
                    & (position_drift <= TOLERANCES["task_position_m"])
                    & (rotation_drift <= TOLERANCES["task_rotation_rad"])
                )
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(
                            settle, lift, observation=observation
                        ),
                        stage_id="held",
                        window="terminal_settle",
                        measurements={
                            "object_retained": retained,
                            "min_height_above_support_m": min_height,
                            "relative_position_drift_m": position_drift,
                            "relative_rotation_drift_rad": rotation_drift,
                            "remaining_lift_m": lift_remaining,
                        },
                        method="gripper_and_object_settle_trace",
                        source="measured_proxy",
                    )
                )
        evidence = (
            merge_stage_evidence_batches(*batches)
            if batches
            else tuple({} for _ in range(case.batch_size))
        )
        return evaluate_skill_stage_batch("pick_up", evidence)


class MoveHeldObjectEvaluator:
    """Evaluate transport plus the held object's settled pose and retention."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        transport = replay_trace(observation, "held_object_transport")
        settle = replay_trace(observation, "held_object_settle")
        batches: list[Sequence[Mapping[str, StageEvidence]]] = []
        for stage_id, window, trace in (
            ("transported", "actuation_end", transport),
            ("settled", "terminal_settle", settle),
        ):
            if trace is None or trace.object_pose is None:
                continue
            final = trace.object_pose[-1]
            target = case_pose(
                case, "target_object_pose", device=trace.object_pose.device
            ).to(dtype=trace.object_pose.dtype)
            min_height = minimum_height_above_initial_support(trace, case)
            if min_height is None:
                continue
            measurements: dict[str, torch.Tensor] = {
                "position_error_m": torch.linalg.vector_norm(
                    final[:, :3, 3] - target[:, :3, 3], dim=-1
                ),
                "rotation_error_rad": pose_rotation_error(final, target),
                "min_height_above_support_m": min_height,
            }
            batches.append(
                stage_evidence_batch(
                    valid_mask=trace_validity(trace, observation=observation),
                    stage_id=stage_id,
                    window=window,
                    measurements=measurements,
                    method=(
                        "object_transport_trace"
                        if stage_id == "transported"
                        else "terminal_object_pose_trace"
                    ),
                )
            )
        if settle is not None:
            part = next(iter(observation.gripper_open_qpos), "")
            retained = gripper_holding(settle, observation, part)
            min_height = minimum_height_above_initial_support(settle, case)
            if retained is not None and min_height is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(settle, observation=observation),
                        stage_id="held",
                        window="terminal_settle",
                        measurements={
                            "object_retained": retained,
                            "min_height_above_support_m": min_height,
                        },
                        method="gripper_and_object_settle_trace",
                        source="measured_proxy",
                    )
                )
        evidence = (
            merge_stage_evidence_batches(*batches)
            if batches
            else tuple({} for _ in range(case.batch_size))
        )
        return evaluate_skill_stage_batch("move_held_object", evidence)


class PlaceEvaluator:
    """Evaluate release, terminal placement, and post-release stability."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        release = replay_trace(observation, "place_release")
        settle = replay_trace(observation, "place_settle")
        part = next(iter(observation.gripper_open_qpos), "")
        batches: list[Sequence[Mapping[str, StageEvidence]]] = []
        if release is not None:
            released = gripper_released(release, observation, part)
            if released is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(release, observation=observation),
                        stage_id="released",
                        window="release_end",
                        measurements={"release_established": released},
                        method="gripper_joint_state",
                        source="measured_proxy",
                    )
                )
        if settle is not None and settle.object_pose is not None:
            poses = settle.object_pose
            target = case_pose(case, "target_object_pose", device=poses.device).to(
                dtype=poses.dtype
            )
            min_height = minimum_height_above_initial_support(settle, case)
            if min_height is not None:
                final = poses[-1]
                if "object_rotation_threshold_rad" in case.case_parameters:
                    orientation_error = pose_rotation_error(final, target)
                else:
                    orientation_error = torch.arccos(
                        torch.clamp(
                            (final[:, :3, 2] * target[:, :3, 2]).sum(dim=-1),
                            -1.0,
                            1.0,
                        )
                    )
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(settle, observation=observation),
                        stage_id="placed",
                        window="terminal_settle",
                        measurements={
                            "position_error_m": torch.linalg.vector_norm(
                                final[:, :3, 3] - target[:, :3, 3], dim=-1
                            ),
                            "orientation_error_rad": orientation_error,
                            "min_height_above_support_m": min_height,
                        },
                        method="terminal_object_pose_trace",
                    )
                )
                if poses.shape[0] >= 2:
                    position_drift = torch.linalg.vector_norm(
                        poses[:, :, :3, 3] - poses[:1, :, :3, 3], dim=-1
                    ).amax(dim=0)
                    rotation_drift = pose_rotation_error(
                        poses, poses[:1].expand_as(poses)
                    ).amax(dim=0)
                    batches.append(
                        stage_evidence_batch(
                            valid_mask=trace_validity(settle, observation=observation),
                            stage_id="stable",
                            window="settle_interval",
                            measurements={
                                "position_drift_m": position_drift,
                                "rotation_drift_rad": rotation_drift,
                                "min_height_above_support_m": min_height,
                            },
                            method="object_settle_drift_trace",
                        )
                    )
        evidence = (
            merge_stage_evidence_batches(*batches)
            if batches
            else tuple({} for _ in range(case.batch_size))
        )
        return evaluate_skill_stage_batch("place", evidence)


class PressEvaluator:
    """Evaluate contact and signed button stroke inside the press segment."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        trace = replay_trace(observation, "press_actuation")
        if trace is None or trace.articulation_joint is None:
            return evaluate_skill_stage_batch(
                "press", tuple({} for _ in range(case.batch_size))
            )
        values = trace.articulation_joint
        sign = float(case.case_parameters.get("effect_joint_sign", 1.0))
        peak = ((values - values[:1]) * sign).amax(dim=0)
        stroke = float(case.case_parameters["articulation_joint_stroke"])
        contacted = peak > float(
            case.case_parameters.get("contact_minimum_joint_delta", 1.0e-5)
        )
        evidence = merge_stage_evidence_batches(
            stage_evidence_batch(
                valid_mask=trace_validity(trace, observation=observation),
                stage_id="contacted",
                window="actuation_interval",
                measurements={"contact_established": contacted},
                method="target_joint_motion_proxy",
                source="measured_proxy",
            ),
            stage_evidence_batch(
                valid_mask=trace_validity(trace, observation=observation),
                stage_id="pressed",
                window="actuation_interval",
                measurements={"peak_signed_stroke_ratio": peak / stroke},
                method="target_joint_trace",
            ),
        )
        return evaluate_skill_stage_batch("press", evidence)


class SlideEvaluator:
    """Evaluate grasp proxy, drawer travel, and release at their phase windows."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        actuation = replay_trace(observation, "slide_actuation")
        release = replay_trace(observation, "slide_release")
        if actuation is None or actuation.articulation_joint is None:
            return evaluate_skill_stage_batch(
                "slide", tuple({} for _ in range(case.batch_size))
            )
        joint = actuation.articulation_joint
        displacement = joint[-1] - joint[0]
        effect_sign = float(case.case_parameters.get("effect_joint_sign", 1.0))
        requested_displacement = effect_sign * float(
            case.case_parameters["translation_distance_m"]
        )
        grasped = displacement * effect_sign > float(
            case.case_parameters.get("contact_minimum_joint_delta", 1.0e-5)
        )
        travel_error = (displacement - requested_displacement).abs()
        batches: list[Sequence[Mapping[str, StageEvidence]]] = [
            stage_evidence_batch(
                valid_mask=trace_validity(actuation, observation=observation),
                stage_id="grasped",
                window="grasp_end",
                measurements={"grasp_established": grasped},
                method="target_joint_motion_proxy",
                source="measured_proxy",
            ),
            stage_evidence_batch(
                valid_mask=trace_validity(actuation, observation=observation),
                stage_id="slid",
                window="actuation_end",
                measurements={"travel_error_m": travel_error},
                method="target_joint_segment_end",
            ),
        ]
        if release is not None:
            part = next(iter(observation.gripper_open_qpos), "")
            released = gripper_released(release, observation, part)
            if released is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(release, observation=observation),
                        stage_id="released",
                        window="release_end",
                        measurements={"release_established": released},
                        method="gripper_joint_state",
                        source="measured_proxy",
                    )
                )
        return evaluate_skill_stage_batch(
            "slide", merge_stage_evidence_batches(*batches)
        )


class TwistEvaluator:
    """Evaluate executed TCP twist rather than substituting knob displacement."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        actuation = replay_trace(observation, "twist_actuation")
        release = replay_trace(observation, "twist_release")
        if (
            actuation is None
            or actuation.articulation_joint is None
            or actuation.tcp_pose is None
        ):
            return evaluate_skill_stage_batch(
                "twist", tuple({} for _ in range(case.batch_size))
            )
        joint = actuation.articulation_joint
        grasped = (joint[-1] - joint[0]).abs() > float(
            case.case_parameters.get("contact_minimum_joint_delta", 1.0e-5)
        )
        tcp = actuation.tcp_pose
        target_pose = case_pose(case, "twist_target_pose", device=tcp.device).to(
            dtype=tcp.dtype
        )
        axis_local = torch.as_tensor(
            case.case_parameters["twist_axis"], dtype=tcp.dtype, device=tcp.device
        )
        axis_local = axis_local / torch.linalg.vector_norm(axis_local)
        world_axis = target_pose[:, :3, :3] @ axis_local
        executed = signed_rotation_about_axis(
            tcp[0, :, :3, :3], tcp[-1, :, :3, :3], world_axis
        )
        error = (executed - float(case.case_parameters["twist_angle_rad"])).abs()
        batches: list[Sequence[Mapping[str, StageEvidence]]] = [
            stage_evidence_batch(
                valid_mask=trace_validity(actuation, observation=observation),
                stage_id="grasped",
                window="grasp_end",
                measurements={"grasp_established": grasped},
                method="target_joint_motion_proxy",
                source="measured_proxy",
            ),
            stage_evidence_batch(
                valid_mask=trace_validity(actuation, observation=observation),
                stage_id="twisted",
                window="actuation_end",
                measurements={"tcp_axis_angle_error_rad": error},
                method="tcp_segment_rotation",
            ),
        ]
        if release is not None:
            part = next(iter(observation.gripper_open_qpos), "")
            released = gripper_released(release, observation, part)
            if released is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(release, observation=observation),
                        stage_id="released",
                        window="release_end",
                        measurements={"release_established": released},
                        method="gripper_joint_state",
                        source="measured_proxy",
                    )
                )
        return evaluate_skill_stage_batch(
            "twist", merge_stage_evidence_batches(*batches)
        )


class MoveEndEffectorEvaluator:
    """Score the physically settled TCP position without planner state."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        final_tcp_pose = observation.final_tcp_pose
        target = case.target_waypoints[:, -1].to(
            device=final_tcp_pose.device,
            dtype=final_tcp_pose.dtype,
        )
        if final_tcp_pose.shape != target.shape:
            raise ValueError(
                "Final TCP observations must match the batched terminal target pose."
            )
        position_error, rotation_error = executed_final_pose_errors(
            case, final_tcp_pose
        )
        return evaluate_skill_stage_batch(
            "move_end_effector",
            stage_evidence_batch(
                stage_id="reached",
                window="terminal_settle",
                measurements={
                    "tcp_position_error_m": position_error,
                    "tcp_rotation_error_rad": rotation_error,
                },
                method="terminal_tcp_pose",
            ),
        )


class MoveJointsEvaluator:
    """Score the physically settled arm joints without planner state."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        final_arm_qpos = observation.final_arm_qpos
        if final_arm_qpos is None:
            return evaluate_skill_stage_batch(
                "move_joints", tuple({} for _ in range(case.batch_size))
            )
        target = torch.as_tensor(
            case.case_parameters["target_qpos"],
            dtype=final_arm_qpos.dtype,
            device=final_arm_qpos.device,
        )
        if target.dim() != 3 or target.shape[0] != case.batch_size:
            raise ValueError("MoveJoints target_qpos must have shape (B, W, D).")
        target = target[:, -1]
        if final_arm_qpos.shape != target.shape:
            raise ValueError(
                "Final arm observations must match the batched terminal joint target."
            )
        error = torch.linalg.vector_norm(final_arm_qpos - target, dim=-1)
        return evaluate_skill_stage_batch(
            "move_joints",
            stage_evidence_batch(
                stage_id="reached",
                window="terminal_settle",
                measurements={"joint_error_l2_rad": error},
                method="terminal_arm_qpos",
            ),
        )
