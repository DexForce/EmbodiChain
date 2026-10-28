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

"""Fixed-geometry prototype motion sources for atomic-skill planning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from embodichain.data_pipeline.real2sim import (
    PhaseAnnotation,
    SkillMotionPrototype,
    retarget_motion_prototype,
)
from embodichain.compute.trajectory import retime_to_control_grid

from .invocation import ResolvedActionRequest
from .bindings import JointPositionTarget
from .control import GRASP_COMMAND, OPEN_COMMAND
from .nominal import InstantiatedSkillMotion
from .plans import TimedTrajectory
from .state import PlanningContext
from .trajectory_ops import build_pose_plan_states

__all__ = ["PrototypeMotionProvider"]


def _project_mimics(positions: torch.Tensor, robot: object) -> None:
    """Apply explicit driver relationships after reference commands change."""
    mimic_ids = tuple(robot.mimic_ids)
    if not mimic_ids:
        return
    values = tuple(
        zip(
            mimic_ids, robot.mimic_parents, robot.mimic_multipliers, robot.mimic_offsets
        )
    )
    if len(values) != len(mimic_ids):
        raise ValueError("robot mimic metadata is incomplete")
    pending = list(values)
    resolved = set(range(positions.shape[-1])) - set(mimic_ids)
    while pending:
        previous = len(pending)
        for item in tuple(pending):
            child, parent, multiplier, offset = item
            if parent not in resolved:
                continue
            if type(child) is not int or not 0 <= child < positions.shape[-1]:
                raise ValueError("robot mimic metadata indexes invalid joints")
            positions[:, :, child] = positions[:, :, parent] * multiplier + offset
            resolved.add(child)
            pending.remove(item)
        if len(pending) == previous:
            raise ValueError("robot mimic metadata contains an unresolved chain")


def _smooth_ik_phase_interiors(
    path: torch.Tensor,
    phases: tuple[PhaseAnnotation, ...],
    *,
    allowed_phases: frozenset[str] | None = None,
    allow_hold_for_feasibility: bool = False,
) -> torch.Tensor:
    """Reduce numerical IK jitter while retaining event derivative stencils."""
    result = path.clone()
    for phase in phases:
        if (
            (
                phase.kind != "free"
                and not (allow_hold_for_feasibility and phase.kind == "hold")
            )
            or phase.authority == "qpos"
            or phase.stop - phase.start < 5
            or (allowed_phases is not None and phase.name not in allowed_phases)
        ):
            continue
        start, stop = phase.start + 2, phase.stop - 2
        result[:, start:stop] = (
            path[:, start - 1 : stop - 1]
            + 2 * path[:, start:stop]
            + path[:, start + 1 : stop + 1]
        ) * 0.25
    return result


def _align_release_anchor(
    poses: np.ndarray, index: int, goal: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Align a Place path to its semantic release pose while preserving world-up."""
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not 0 <= index < len(poses):
        raise ValueError("release alignment requires a valid TCP path and anchor")
    goal = np.asarray(goal, dtype=np.float64)
    if (
        goal.shape != (4, 4)
        or not np.isfinite(goal).all()
        or not np.allclose(goal[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(goal[:3, :3].T @ goal[:3, :3], np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(goal[:3, :3]), 1.0, atol=1e-5, rtol=0)
    ):
        raise ValueError("release alignment requires a proper rigid goal pose")
    anchor = poses[index]
    delta = goal @ np.linalg.inv(anchor)
    yaw = np.arctan2(delta[1, 0], delta[0, 0])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    result = delta @ poses
    result[:, :3, 3] = goal[:3, 3] + (poses[:, :3, 3] - anchor[:3, 3]) @ yaw_rotation.T
    position = float(np.linalg.norm(goal[:3, 3] - anchor[:3, 3]))
    cosine_angle = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return result, position, float(np.arccos(cosine_angle))


@dataclass(frozen=True)
class PrototypeMotionProvider:
    """Instantiate a prototype with an explicit motion authority.

    ``qpos`` requires the source object and target poses. ``eef_pose``
    retargets tabletop XY/yaw changes through IK, and ``hybrid`` uses the
    authority declared for every phase. The host must independently establish
    that robot and object geometry are compatible: entity poses and joint names
    alone do not prove asset identity. Environment lifecycle and persistence
    remain host-owned.

    Args:
        prototype: Prepared, normalized source prototype.
        object_entity_id: Scene entity carrying the source object pose.
        target_entity_id: Scene entity carrying the source target pose when
            one is declared by the prototype.
        target_entity_to_reference: Optional rigid transform from that entity
            to the prototype's target reference frame.
        pose_tolerance: Absolute matrix tolerance for fixed scene references.
        gripper_mode: Use recorded values or reconstruct semantic commands.
        authority: Authoritative joint or TCP path, globally or by phase.
        motion_generator: Robot motion services for IK and collision checking.
        require_collision_validation: Require the backend's sampled checker.
        prototype_sha256: Optional content-addressed source identity.
        object_support_validator: Optional trusted host check for regrasping
            an object currently resting on the target support.
        ik_smoothing_trigger_fraction: Optional fraction of the motion policy's
            acceleration limit at which to seek a lower-acceleration nominal
            path without changing protected samples or the TCP error tube.

    A required acceleration-feasibility pass can adjust Cartesian-authoritative
    hold interiors before the nominal geometry baseline is frozen. Its FK error
    is checked and its iterations are recorded. Optional headroom smoothing
    never changes contact or hold samples.
    """

    prototype: SkillMotionPrototype
    object_entity_id: str
    target_entity_id: str | None = None
    target_entity_to_reference: np.ndarray | None = None
    pose_tolerance: float = 1e-5
    gripper_mode: Literal["recorded", "semantic"] = "recorded"
    authority: Literal["qpos", "eef_pose", "hybrid"] = "qpos"
    motion_generator: object | None = None
    require_collision_validation: bool = True
    prototype_sha256: str | None = None
    object_support_validator: Callable[[PlanningContext], bool] | None = None
    ik_smoothing_trigger_fraction: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prototype, SkillMotionPrototype):
            raise TypeError("prototype must be a SkillMotionPrototype")
        if (
            not isinstance(self.object_entity_id, str)
            or not self.object_entity_id.strip()
        ):
            raise ValueError("object_entity_id must be a nonempty string")
        if self.target_entity_id is not None and (
            not isinstance(self.target_entity_id, str)
            or not self.target_entity_id.strip()
        ):
            raise ValueError("target_entity_id must be a nonempty string or None")
        if self.target_entity_to_reference is not None:
            transform = np.array(
                self.target_entity_to_reference, dtype=np.float64, copy=True
            )
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
                or not np.allclose(
                    transform[:3, :3].T @ transform[:3, :3],
                    np.eye(3),
                    atol=1e-6,
                    rtol=0,
                )
                or not np.isclose(
                    np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6, rtol=0
                )
            ):
                raise ValueError(
                    "target_entity_to_reference must be a proper rigid transform"
                )
            transform.setflags(write=False)
            object.__setattr__(self, "target_entity_to_reference", transform)
        if not np.isfinite(self.pose_tolerance) or self.pose_tolerance < 0:
            raise ValueError("pose_tolerance must be finite and nonnegative")
        if (
            self.prototype.unit.source.target_pose is not None
            and self.target_entity_id is None
        ):
            raise ValueError("prototype target pose requires an explicit target entity")
        if self.gripper_mode not in {"recorded", "semantic"}:
            raise ValueError("gripper_mode must be recorded or semantic")
        if self.authority not in {"qpos", "eef_pose", "hybrid"}:
            raise ValueError("authority must be qpos, eef_pose, or hybrid")
        if (
            self.authority in {"eef_pose", "hybrid"}
            or self.require_collision_validation
        ) and self.motion_generator is None:
            raise ValueError(
                "Cartesian transfer or collision validation requires a motion generator"
            )
        if (
            self.authority in {"eef_pose", "hybrid"}
            and self.prototype.unit.world_tcp is None
        ):
            raise ValueError(
                "EEF and hybrid authority require explicit source TCP poses"
            )
        if self.authority == "hybrid" and any(
            phase.authority is None for phase in self.prototype.unit.phases
        ):
            raise ValueError(
                "hybrid authority requires an explicit authority for every phase"
            )
        if type(self.require_collision_validation) is not bool:
            raise TypeError("require_collision_validation must be a boolean")
        if self.object_support_validator is not None and not callable(
            self.object_support_validator
        ):
            raise TypeError("object_support_validator must be callable or None")
        if self.ik_smoothing_trigger_fraction is not None and (
            isinstance(self.ik_smoothing_trigger_fraction, bool)
            or not isinstance(self.ik_smoothing_trigger_fraction, (int, float))
            or not np.isfinite(self.ik_smoothing_trigger_fraction)
            or not 0 < self.ik_smoothing_trigger_fraction < 1
        ):
            raise ValueError(
                "ik_smoothing_trigger_fraction must be between zero and one"
            )
        if self.prototype_sha256 is not None and (
            not isinstance(self.prototype_sha256, str)
            or len(self.prototype_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.prototype_sha256)
        ):
            raise ValueError("prototype_sha256 must be a lowercase SHA-256 digest")

    def instantiate(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
    ) -> InstantiatedSkillMotion:
        """Build a fixed-scene nominal motion for the active invocation."""
        if not isinstance(request, ResolvedActionRequest) or not isinstance(
            context, PlanningContext
        ):
            raise TypeError("request and context must be resolved atomic values")
        unit = self.prototype.unit
        if request.skill_id != unit.source.skill_id:
            raise ValueError("prototype skill does not match the invocation")
        if context.batch_size != 1:
            raise ValueError("prototype motion currently requires B=1")

        def check_pose(
            entity_id: str, reference: np.ndarray, relative: np.ndarray | None = None
        ) -> torch.Tensor:
            if entity_id not in context.scene.entities:
                raise ValueError(f"missing prototype scene entity {entity_id!r}")
            pose = context.scene.entities[entity_id].pose.to(context.robot.qpos)
            if pose.shape == (4, 4):
                pose = pose.unsqueeze(0)
            if pose.shape != (1, 4, 4):
                raise ValueError("prototype entity must have one 4x4 pose")
            if relative is not None:
                pose = pose @ torch.as_tensor(
                    np.array(relative, copy=True), dtype=pose.dtype, device=pose.device
                )
            expected = torch.as_tensor(
                np.array(reference, copy=True), dtype=pose.dtype, device=pose.device
            ).unsqueeze(0)
            if self.authority == "qpos" and not torch.allclose(
                pose, expected, atol=self.pose_tolerance, rtol=0
            ):
                raise ValueError(
                    f"prototype entity {entity_id!r} changed; Cartesian replanning is required"
                )
            return pose

        if unit.source.object_pose is None:
            raise ValueError("prototype requires a fixed source object pose")
        object_pose = check_pose(self.object_entity_id, unit.source.object_pose)
        target_pose = None
        if unit.source.target_pose is not None:
            assert self.target_entity_id is not None
            target_pose = check_pose(
                self.target_entity_id,
                unit.source.target_pose,
                self.target_entity_to_reference,
            )
        if not np.allclose(
            np.diff(unit.timestamps), context.require_control_dt(), atol=1e-9, rtol=0
        ):
            raise ValueError("prototype control grid does not match the environment")
        positions = torch.as_tensor(
            np.array(unit.qpos, copy=True),
            dtype=context.robot.qpos.dtype,
            device=context.robot.qpos.device,
        ).unsqueeze(0)
        initial_qpos = (
            torch.as_tensor(
                np.array(unit.initial_qpos, copy=True),
                dtype=positions.dtype,
                device=positions.device,
            ).unsqueeze(0)
            if self.authority == "qpos"
            else context.robot.qpos.clone()
        )
        if self.gripper_mode == "semantic":
            hand = request.binding.endpoint("primary", "grasp")
            hand_target = hand.require_target(JointPositionTarget)
            hand_ids = list(hand_target.joint_ids)
            opened = hand.joint_positions(
                OPEN_COMMAND, num_envs=1, device=positions.device, dtype=positions.dtype
            )
            closed = hand.joint_positions(
                GRASP_COMMAND,
                num_envs=1,
                device=positions.device,
                dtype=positions.dtype,
            )
            for phase in unit.phases:
                count = phase.stop - phase.start
                if phase.name == "close":
                    fraction = torch.linspace(
                        0, 1, count, dtype=positions.dtype, device=positions.device
                    )
                    values = torch.lerp(
                        opened[:, None], closed[:, None], fraction[None, :, None]
                    )
                elif phase.name == "release":
                    fraction = torch.linspace(
                        0, 1, count, dtype=positions.dtype, device=positions.device
                    )
                    values = torch.lerp(
                        closed[:, None], opened[:, None], fraction[None, :, None]
                    )
                else:
                    target = (
                        opened
                        if (request.skill_id == "pick_up" and phase.name == "approach")
                        or (request.skill_id == "place" and phase.name == "retract")
                        else closed
                    )
                    values = target[:, None].expand(-1, count, -1)
                positions[:, phase.start : phase.stop, hand_ids] = values
        segment_lengths = {
            phase.name: phase.stop - phase.start for phase in unit.phases
        }
        smoothing_iterations = 0
        hold_smoothing_iterations = 0
        supported_regrasp = False
        anchor_position_correction = 0.0
        anchor_rotation_correction = 0.0
        fk_position_error_m = None
        fk_rotation_error_rad = None
        if self.authority in {"eef_pose", "hybrid"}:
            generator = self.motion_generator
            if generator is None:
                raise ValueError("EEF transfer requires a motion generator")
            arm = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            arm_ids = list(arm.joint_ids)
            hand_ids = list(
                request.binding.endpoint("primary", "grasp")
                .require_target(JointPositionTarget)
                .joint_ids
            )
            unowned = [
                index
                for index in range(positions.shape[-1])
                if index
                not in set(arm_ids) | set(hand_ids) | set(generator.robot.mimic_ids)
            ]
            if unowned:
                positions[:, :, unowned] = context.robot.qpos[:, None, unowned]
            held = None
            if (
                request.skill_id == "pick_up"
                and self.object_support_validator is not None
            ):
                supported_regrasp = self.object_support_validator(context)
                if type(supported_regrasp) is not bool:
                    raise TypeError("object support validation must return a boolean")
            if request.skill_id != "pick_up":
                endpoint = request.binding.endpoint("primary", "motion")
                held_state = context.get_held_object(endpoint.task_state_key)
                if held_state is None:
                    raise ValueError(
                        "EEF transport or placement requires a held object"
                    )
                held = held_state.object_to_eef.detach().cpu().numpy()
                if held.shape == (1, 4, 4):
                    held = held[0]
            target_tcp = retarget_motion_prototype(
                self.prototype,
                object_pose=(
                    object_pose[0].detach().cpu().numpy()
                    if request.skill_id == "pick_up"
                    else None
                ),
                target_pose=(
                    None
                    if target_pose is None
                    else target_pose[0].detach().cpu().numpy()
                ),
                held_object_to_tcp=held,
                object_supported_regrasp=supported_regrasp,
            )
            if request.skill_id == "place":
                from .goals import resolve_pose_goal
                from .primitives.place import PlaceGoal
                from .trajectory_ops import resolve_pose_target

                if not isinstance(request.goal, PlaceGoal):
                    raise ValueError("prototype Place requires a direct PlaceGoal")
                goal = resolve_pose_target(
                    resolve_pose_goal(request.goal.xpos, context, name="xpos"),
                    num_envs=1,
                    device=positions.device,
                )
                if goal.ndim == 4:
                    goal = goal[:, -1]
                target_tcp, anchor_position_correction, anchor_rotation_correction = (
                    _align_release_anchor(
                        target_tcp,
                        unit.phases[1].start,
                        goal[0].detach().cpu().numpy(),
                    )
                )
            target_tcp = torch.as_tensor(
                np.array(target_tcp, copy=True),
                dtype=positions.dtype,
                device=positions.device,
            ).unsqueeze(0)
            robot = generator.robot
            start_qpos = context.robot.qpos[:, arm_ids]
            start_pose = robot.compute_fk(
                qpos=start_qpos, name=arm.control_part, to_matrix=True
            )
            if not isinstance(start_pose, torch.Tensor) or start_pose.shape != (
                1,
                4,
                4,
            ):
                raise ValueError("current TCP pose is unavailable")
            prefix = None
            if not torch.allclose(start_pose, target_tcp[:, 0], atol=1e-5, rtol=0):
                if unit.phases[0].kind != "free":
                    raise ValueError(
                        "prototype entry connection requires an explicitly free first phase"
                    )
                options = request.motion_policy.to_motion_gen_options(
                    start_qpos=start_qpos,
                    control_part=arm.control_part,
                    interpolation_dt=context.require_control_dt(),
                )
                connection = generator.generate(
                    build_pose_plan_states(target_tcp[:, :1]), options=options
                )
                connection_success = connection.success
                if isinstance(connection_success, torch.Tensor):
                    connection_success = bool(connection_success.all())
                if (
                    connection.positions is None
                    or connection.dt is None
                    or connection_success is not True
                ):
                    raise ValueError("prototype entry connection could not be planned")
                connection_qpos, connection_dt = connection.positions, connection.dt
                if bool((connection_dt[:, 0] > 0).any()):
                    connection_qpos = torch.cat(
                        (start_qpos[:, None], connection_qpos), dim=1
                    )
                    connection_dt = torch.cat(
                        (torch.zeros_like(connection_dt[:, :1]), connection_dt), dim=1
                    )
                elif not torch.allclose(
                    connection_qpos[:, 0], start_qpos, atol=1e-4, rtol=0
                ):
                    raise ValueError(
                        "prototype connection does not begin at the observed state"
                    )
                prefix, _, _, count = retime_to_control_grid(
                    connection_qpos, connection_dt, context.require_control_dt()
                )
                prefix = prefix[:, : int(count[0])]
                start_qpos = prefix[:, -1]
            valid, path = generator._solve_cartesian_waypoints(
                target_tcp,
                start_qpos,
                arm.control_part,
                known_start=torch.ones(1, dtype=torch.bool, device=positions.device),
            )
            if (
                not bool(valid.all())
                or path.shape != (1, target_tcp.shape[1], len(arm_ids))
                or not bool(torch.isfinite(path).all())
            ):
                raise ValueError("prototype IK failed on a required sample")
            if self.authority == "hybrid":
                path = path.clone()
                for phase in unit.phases:
                    if phase.authority == "qpos":
                        path[:, phase.start : phase.stop] = positions[
                            :, phase.start : phase.stop, arm_ids
                        ]
            observed = robot.compute_batch_fk(
                qpos=path, name=arm.control_part, to_matrix=True
            )
            if (
                not isinstance(observed, torch.Tensor)
                or observed.shape != target_tcp.shape
                or not bool(torch.isfinite(observed).all())
            ):
                raise ValueError("prototype FK validation is unavailable")
            position_error = torch.linalg.vector_norm(
                observed[:, :, :3, 3] - target_tcp[:, :, :3, 3], dim=-1
            )
            relative = (
                observed[:, :, :3, :3].transpose(-1, -2) @ target_tcp[:, :, :3, :3]
            )
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(
                -1, 1
            )
            fk_position_error_m = float(position_error.max())
            fk_rotation_error_rad = float(torch.acos(cosine).max())
            if bool(
                ((position_error > 0.002) | (torch.acos(cosine) > np.deg2rad(2))).any()
            ):
                raise ValueError("prototype IK path does not satisfy the TCP targets")
            limits = robot.get_qpos_limits(name=arm.control_part)
            if (
                not isinstance(limits, torch.Tensor)
                or limits.shape != (1, len(arm_ids), 2)
                or not bool(torch.isfinite(limits).all())
            ):
                raise ValueError("prototype joint limits are unavailable")
            if bool(
                ((path < limits[:, None, :, 0]) | (path > limits[:, None, :, 1])).any()
            ):
                raise ValueError("prototype IK violates a joint limit")
            velocity_limits = robot.get_qvel_limits(name=arm.control_part)
            if (
                not isinstance(velocity_limits, torch.Tensor)
                or velocity_limits.shape != (1, len(arm_ids))
                or not bool(torch.isfinite(velocity_limits).all())
                or not bool((velocity_limits > 0).all())
            ):
                raise ValueError("prototype velocity limits are unavailable")
            if bool(
                (
                    (path[:, 1:] - path[:, :-1]).abs()
                    > velocity_limits[:, None] * context.require_control_dt() + 1e-5
                ).any()
            ):
                raise ValueError("prototype IK exceeds a joint velocity limit")
            positions[:, :, arm_ids] = path
            if prefix is not None and prefix.shape[1] > 1:
                prefix_full = (
                    context.robot.qpos[:, None, :]
                    .expand(-1, prefix.shape[1] - 1, -1)
                    .clone()
                )
                prefix_full[:, :, arm_ids] = prefix[:, :-1]
                prefix_full[:, :, hand_ids] = positions[:, :1, hand_ids]
                positions = torch.cat((prefix_full, positions), dim=1)
                segment_lengths[unit.phases[0].name] += prefix_full.shape[1]
            offset = positions.shape[1] - path.shape[1]
            # Discrete independent IK solves can add small joint-acceleration
            # spikes. Only accept a filtered interior if the complete path now
            # satisfies the explicit acceleration budget AND its FK remains in
            # the source TCP error tube. Contact and qpos-authority samples
            # stay fixed. Hold interiors may be adjusted only when necessary
            # for nominal IK feasibility, never for voluntary headroom.
            acceleration_limit = request.motion_policy.acceleration_limit
            if acceleration_limit is not None and positions.shape[1] > 2:
                dt = context.require_control_dt()

                def peak_acceleration(value: torch.Tensor) -> float:
                    velocity = (value[:, 1:] - value[:, :-1]) / dt
                    return float(
                        ((velocity[:, 1:] - velocity[:, :-1]) / dt).abs().max()
                    )

                original_arm = positions[:, :, arm_ids]
                original_peak = peak_acceleration(original_arm)
                requires_feasibility = original_peak > acceleration_limit + 1e-5
                trigger = (
                    acceleration_limit + 1e-5
                    if self.ik_smoothing_trigger_fraction is None
                    else acceleration_limit * self.ik_smoothing_trigger_fraction
                )
                if original_peak > trigger:
                    candidate_path = path
                    best = None
                    best_peak = original_peak
                    smoothing_phases = None
                    if (
                        self.ik_smoothing_trigger_fraction is not None
                        and not requires_feasibility
                    ):
                        from ._expansion_phases import FREE_PHASES

                        smoothing_phases = frozenset(
                            phase.name
                            for phase in unit.phases
                            if phase.name in FREE_PHASES.get(request.skill_id, ())
                            and "task_space_residual" in phase.allowed_operators
                        )
                    for iteration in range(1, 9):
                        candidate_path = _smooth_ik_phase_interiors(
                            candidate_path,
                            unit.phases,
                            allowed_phases=smoothing_phases,
                            allow_hold_for_feasibility=requires_feasibility,
                        )
                        candidate_arm = torch.cat(
                            (original_arm[:, :offset], candidate_path), dim=1
                        )
                        candidate_peak = peak_acceleration(candidate_arm)
                        if (
                            candidate_peak > acceleration_limit + 1e-5
                            or candidate_peak >= best_peak - 1e-7
                            or bool(
                                (
                                    (candidate_arm[:, 1:] - candidate_arm[:, :-1]).abs()
                                    > velocity_limits[:, None] * dt + 1e-5
                                ).any()
                            )
                        ):
                            continue
                        filtered_fk = robot.compute_batch_fk(
                            qpos=candidate_path, name=arm.control_part, to_matrix=True
                        )
                        if (
                            not isinstance(filtered_fk, torch.Tensor)
                            or filtered_fk.shape != target_tcp.shape
                            or not bool(torch.isfinite(filtered_fk).all())
                        ):
                            continue
                        filtered_position_error = torch.linalg.vector_norm(
                            filtered_fk[:, :, :3, 3] - target_tcp[:, :, :3, 3], dim=-1
                        )
                        filtered_relative = (
                            filtered_fk[:, :, :3, :3].transpose(-1, -2)
                            @ target_tcp[:, :, :3, :3]
                        )
                        filtered_cosine = (
                            (filtered_relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
                            / 2
                        ).clamp(-1, 1)
                        if bool(
                            (
                                (filtered_position_error > 0.002)
                                | (torch.acos(filtered_cosine) > np.deg2rad(2))
                            ).any()
                        ):
                            continue
                        if bool(
                            (
                                (candidate_path < limits[:, None, :, 0])
                                | (candidate_path > limits[:, None, :, 1])
                            ).any()
                        ):
                            continue
                        best = (
                            candidate_path,
                            iteration,
                            float(filtered_position_error.max()),
                            float(torch.acos(filtered_cosine).max()),
                        )
                        best_peak = candidate_peak
                        if (
                            self.ik_smoothing_trigger_fraction is None
                            or requires_feasibility
                        ):
                            break
                    if best is not None:
                        (
                            candidate_path,
                            smoothing_iterations,
                            fk_position_error_m,
                            fk_rotation_error_rad,
                        ) = best
                        positions[:, offset:, arm_ids] = candidate_path
                        if requires_feasibility and any(
                            phase.kind == "hold"
                            and not torch.equal(
                                candidate_path[:, phase.start : phase.stop],
                                path[:, phase.start : phase.stop],
                            )
                            for phase in unit.phases
                        ):
                            hold_smoothing_iterations = smoothing_iterations
            if bool(
                (
                    (positions[:, 1:, arm_ids] - positions[:, :-1, arm_ids]).abs()
                    > velocity_limits[:, None] * context.require_control_dt() + 1e-5
                ).any()
            ):
                raise ValueError("prototype connection exceeds a joint velocity limit")
        if self.motion_generator is not None:
            _project_mimics(positions, self.motion_generator.robot)
        if self.require_collision_validation:
            generator = self.motion_generator
            assert generator is not None
            if not generator.supports_joint_trajectory_validation:
                raise ValueError("prototype collision validation is unavailable")
            endpoint = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            obstacles = context.scene.collision_obstacle_poses(
                batch_size=1, device=positions.device, dtype=positions.dtype
            )
            valid = generator.validate_joint_trajectory(
                positions[:, :, list(endpoint.joint_ids)],
                control_part=endpoint.control_part,
                obstacle_poses=obstacles,
            )
            if not bool(valid.all()):
                raise ValueError("prototype path failed collision validation")
        dt = torch.full(
            positions.shape[:2],
            context.require_control_dt(),
            dtype=positions.dtype,
            device=positions.device,
        )
        dt[:, 0] = 0
        trajectory = TimedTrajectory.from_positions(
            positions, env_ids=context.env_ids, dt=dt
        )
        grasp_pose = None
        if request.skill_id == "pick_up" and self.prototype.object_to_tcp is not None:
            grasp_pose = object_pose @ torch.as_tensor(
                np.array(self.prototype.object_to_tcp, copy=True),
                dtype=positions.dtype,
                device=positions.device,
            ).unsqueeze(0)
        return InstantiatedSkillMotion(
            trajectory=trajectory,
            success=torch.ones(1, dtype=torch.bool, device=positions.device),
            segment_lengths=segment_lengths,
            initial_qpos=initial_qpos,
            source_id=unit.source.source_id,
            source_revision=unit.source.source_revision,
            unit_id=unit.source.unit_id,
            origin=unit.source.origin,
            authority=self.authority,
            gripper_mode=self.gripper_mode,
            initial_state_origin=(
                unit.report.initial_state_origin
                if self.authority == "qpos"
                else "current_observation"
            ),
            pose_origin=(
                unit.source.pose_origin
                if unit.world_tcp is not None
                else "fk_derived_at_execution"
            ),
            qpos_semantics=unit.source.qpos_semantics,
            source_command_semantics=unit.source.command_semantics,
            fk_position_error_m=fk_position_error_m,
            fk_rotation_error_rad=fk_rotation_error_rad,
            calibration_revision=unit.source.calibration_revision,
            annotation_revision=unit.source.annotation_revision,
            prototype_sha256=self.prototype_sha256,
            source_time_map=unit.report.time_map,
            source_sample_offset=positions.shape[1] - len(unit.timestamps),
            ik_smoothing_iterations=smoothing_iterations,
            ik_hold_smoothing_iterations=hold_smoothing_iterations,
            object_supported_regrasp=supported_regrasp,
            semantic_anchor_position_correction_m=anchor_position_correction,
            semantic_anchor_rotation_correction_rad=anchor_rotation_correction,
            collision_evidence=(
                "sampled_passed" if self.require_collision_validation else "unverified"
            ),
            phase_permissions={
                phase.name: phase.allowed_operators for phase in unit.phases
            },
            grasp_pose=grasp_pose,
            joint_names=unit.joint_names,
        )
