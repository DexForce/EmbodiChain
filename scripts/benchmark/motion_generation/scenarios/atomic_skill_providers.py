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

"""Atomic Task case providers for OpenDoor, Pour, AxisAlign and PushObject.

Each provider freezes the scene, the grasp and the poses the skill passes
through while the case manifest is built, so every planner is scored on the
same task.  Task success is read from the replayed scene: the hinge angle at
the end of the opening segment, the peak object rotation while pouring, the
held object's axis after alignment, and the pushed object's final pose.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    Affordance,
    AxisAlignAffordance,
    AxisAlignGoal,
    AxisAlignOptions,
    GraspGoal,
    MotionPolicy,
    ObjectSemantics,
    ObservedArticulationJointState,
    OpenDoorAffordance,
    OpenDoorGoal,
    OpenDoorOptions,
    PickUpOptions,
    PourGoal,
    PourOptions,
    PushObjectGoal,
    PushObjectOptions,
    SceneSnapshot,
)
from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

from ..atomic_skill import AtomicSkillCaseProvider
from ..config import SuiteCfg, TrackCfg
from ..contracts import PhysicalEvaluation, ReplayObservationWindow
from ..evaluation import StageEvidence, evaluate_skill_stage_batch
from ..models import BenchmarkCase, CaseOutcome
from .atomic_task import (
    _TOP_DOWN_ROTATION,
    _ExecutionObservation,
    _atomic_case_id,
    _canonical_case_qpos,
    _case_name,
    _case_pose,
    _difficulty,
    _float_vector,
    _gripper_holding,
    _gripper_released,
    _merge_stage_evidence_batches,
    _minimum_height_above_initial_support,
    _motion_valid_mask,
    _object_pose_success,
    _randomization_parameters,
    _randomized_vector_batch,
    _replay_trace,
    _signed_rotation_about_axis,
    _snapshot_case_qpos,
    _stage_evidence_batch,
    register_atomic_skill_provider,
)

if TYPE_CHECKING:
    from ..planners.base import PlannerAdapter
    from .atomic_objects import AtomicObjectHandle
    from .atomic_task import AtomicTaskScenario

__all__: list[str] = []

_TASK_ROTATION_TOLERANCE_RAD = math.radians(5.0)
_TASK_POSITION_TOLERANCE_M = 0.01


class _OpenDoorEvaluator:
    """Evaluate hinge engagement, opening, and measured gripper release."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        actuation = _replay_trace(observation, "door_actuation")
        release = _replay_trace(observation, "door_release")
        if actuation is None or actuation.articulation_joint is None:
            return evaluate_skill_stage_batch(
                "open_door", tuple({} for _ in range(case.batch_size))
            )
        joint = actuation.articulation_joint
        displacement = joint[-1] - joint[0]
        grasped = displacement.abs() > float(
            case.case_parameters.get("contact_minimum_joint_delta", 1.0e-5)
        )
        opened_error = (
            joint[-1] - float(case.case_parameters["target_joint_position"])
        ).abs()
        batches: list[Sequence[Mapping[str, StageEvidence]]] = [
            _stage_evidence_batch(
                stage_id="grasped",
                window="grasp_end",
                measurements={"grasp_established": grasped},
                method="hinge_motion_proxy",
                source="measured_proxy",
            ),
            _stage_evidence_batch(
                stage_id="opened",
                window="actuation_end",
                measurements={"hinge_angle_error_rad": opened_error},
                method="hinge_segment_end",
            ),
        ]
        if release is not None:
            part = next(iter(observation.gripper_open_qpos), "")
            released = _gripper_released(release, observation, part)
            if released is not None:
                batches.append(
                    _stage_evidence_batch(
                        stage_id="released",
                        window="release_end",
                        measurements={"release_established": released},
                        method="gripper_joint_state",
                    )
                )
        return evaluate_skill_stage_batch(
            "open_door", _merge_stage_evidence_batches(*batches)
        )


class _PourEvaluator:
    """Evaluate peak signed object rotation and support-plane clearance."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        trace = _replay_trace(observation, "pour_actuation")
        if trace is None or trace.object_pose is None:
            return evaluate_skill_stage_batch(
                "pour", tuple({} for _ in range(case.batch_size))
            )
        poses = trace.object_pose
        axis = torch.as_tensor(
            case.case_parameters["object_rotation_axis"],
            dtype=poses.dtype,
            device=poses.device,
        )
        axis = axis / torch.linalg.vector_norm(axis)
        reference = poses[0, :, :3, :3]
        world_axis = reference @ axis
        angles = torch.stack(
            [
                _signed_rotation_about_axis(reference, pose[:, :3, :3], world_axis)
                for pose in poses
            ]
        )
        sign = float(case.case_parameters.get("object_rotation_sign", 1.0))
        peak = (angles * sign).amax(dim=0)
        error = (peak - abs(float(case.case_parameters["rotate_angle_rad"]))).abs()
        min_height = _minimum_height_above_initial_support(trace, case)
        if min_height is None:
            return evaluate_skill_stage_batch(
                "pour", tuple({} for _ in range(case.batch_size))
            )
        return evaluate_skill_stage_batch(
            "pour",
            _stage_evidence_batch(
                stage_id="poured",
                window="actuation_interval",
                measurements={
                    "peak_signed_axis_angle_error_rad": error,
                    "min_height_above_support_m": min_height,
                },
                method="object_pose_trace",
            ),
        )


class _AxisAlignEvaluator:
    """Evaluate grasp proxy, aligned object axis, and terminal retention."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        actuation = _replay_trace(observation, "axis_align_actuation")
        settle = _replay_trace(observation, "axis_align_settle")
        if actuation is None or actuation.object_pose is None:
            return evaluate_skill_stage_batch(
                "axis_align", tuple({} for _ in range(case.batch_size))
            )
        poses = actuation.object_pose
        internal = torch.as_tensor(
            case.case_parameters["internal_axis"],
            dtype=poses.dtype,
            device=poses.device,
        )
        target = torch.as_tensor(
            case.case_parameters["target_axis"],
            dtype=poses.dtype,
            device=poses.device,
        )
        initial_axis = torch.nn.functional.normalize(
            poses[0, :, :3, :3] @ internal, dim=-1
        )
        final_axis = torch.nn.functional.normalize(
            poses[-1, :, :3, :3] @ internal, dim=-1
        )
        moved = torch.arccos(
            torch.clamp((initial_axis * final_axis).sum(dim=-1), -1.0, 1.0)
        )
        aligned_error = torch.arccos(torch.clamp(final_axis @ target, -1.0, 1.0))
        actuation_height = _minimum_height_above_initial_support(actuation, case)
        if actuation_height is None:
            return evaluate_skill_stage_batch(
                "axis_align", tuple({} for _ in range(case.batch_size))
            )
        batches: list[Sequence[Mapping[str, StageEvidence]]] = [
            _stage_evidence_batch(
                stage_id="grasped",
                window="grasp_end",
                measurements={"grasp_established": moved > 1.0e-5},
                method="object_axis_motion_proxy",
                source="measured_proxy",
            ),
            _stage_evidence_batch(
                stage_id="aligned",
                window="actuation_end",
                measurements={
                    "axis_angle_error_rad": aligned_error,
                    "min_height_above_support_m": actuation_height,
                },
                method="object_pose_trace",
            ),
        ]
        if settle is not None:
            min_height = _minimum_height_above_initial_support(settle, case)
            part = next(iter(observation.gripper_open_qpos), "")
            retained = _gripper_holding(settle, observation, part)
            if min_height is not None and retained is not None:
                batches.append(
                    _stage_evidence_batch(
                        stage_id="held",
                        window="terminal_settle",
                        measurements={
                            "object_retained": retained,
                            "min_height_above_support_m": min_height,
                        },
                        method="object_and_gripper_settle_trace",
                    )
                )
        return evaluate_skill_stage_batch(
            "axis_align", _merge_stage_evidence_batches(*batches)
        )


def _unit_vector(value: object, *, name: str, default: Sequence[float]) -> list[float]:
    """Resolve a non-zero 3-vector and normalize it."""
    vector = _float_vector(value, name=name, length=3, default=default)
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 1.0e-6:
        raise ValueError(f"{name} must be non-zero.")
    return [item / norm for item in vector]


def _object_semantics(
    handle: AtomicObjectHandle, affordance: Affordance
) -> ObjectSemantics:
    """Return benchmark semantics bound to one configured rigid object."""
    return ObjectSemantics(
        affordance=affordance,
        geometry={},
        properties={"benchmark_object_id": handle.object_id},
        label=handle.object_id,
        entity_id=handle.entity.uid,
    )


def _primary_control_parts(scenario: "AtomicTaskScenario", skill: str) -> dict:
    """Bind the scenario arm and gripper to a single-arm skill slot."""
    if scenario.end_effector_part is None:
        raise RuntimeError(f"{skill} requires a configured end-effector control part.")
    return {
        "primary": {
            "motion": scenario.control_part,
            "grasp": scenario.end_effector_part,
        }
    }


def _translate(pose: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Translate world poses by a broadcastable world-frame offset."""
    translated = pose.clone()
    translated[..., :3, 3] += offset
    return translated


def _settle_object_pose(
    scenario: "AtomicTaskScenario",
    handle: AtomicObjectHandle,
    config: Mapping[str, object],
    *,
    seed: int,
    stream: int,
) -> tuple[torch.Tensor, int]:
    """Install the configured object pose and return it after gravity settles."""
    settle_steps = config.get(
        "pre_action_settle_steps", handle.config.get("settle_steps", 2)
    )
    if type(settle_steps) is not int or settle_steps < 0:
        raise ValueError("pre_action_settle_steps must be a non-negative integer.")
    pose = scenario.randomize_object_pose(
        handle,
        config,
        seed=seed,
        stream=stream,
        settle_steps=settle_steps,
    )
    return pose, settle_steps


def _set_pre_pick_start(
    scenario: "AtomicTaskScenario",
    object_pose: torch.Tensor,
    config: Mapping[str, object],
    *,
    skill: str,
) -> None:
    """Start above the object with the gripper open, as the PickUp cases do."""
    arm_start = scenario.robot.get_qpos(name=scenario.control_part)
    pre_pick_pose = scenario.robot.compute_fk(
        arm_start, name=scenario.control_part, to_matrix=True
    ).clone()
    pre_pick_pose[:, :2, 3] = object_pose[:, :2, 3]
    pre_pick_pose[:, 2, 3] = float(config.get("pre_pick_height_m", 0.36))
    success, pre_pick_qpos = scenario.robot.compute_ik(
        pose=pre_pick_pose,
        joint_seed=arm_start,
        name=scenario.control_part,
    )
    if not bool(torch.as_tensor(success).all().item()):
        raise RuntimeError(
            f"Independent IK rejected the {skill} start {_case_name(config)!r}."
        )
    scenario.set_robot_start(_canonical_case_qpos(pre_pick_qpos), open_gripper=True)


def _fixed_grasp(
    object_pose: torch.Tensor,
    config: Mapping[str, object],
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Return the configured fixed-rotation grasp at the object origin."""
    rotation = torch.tensor(
        config.get("grasp_rotation", _TOP_DOWN_ROTATION),
        dtype=object_pose.dtype,
        device=object_pose.device,
    )
    if rotation.shape != (3, 3):
        raise ValueError("grasp_rotation must be a 3x3 matrix.")
    grasp = object_pose[:1].clone()
    grasp[:, :3, :3] = rotation
    grasp[:, :3, 3] += _randomized_vector_batch(
        _float_vector(
            config.get("grasp_offset_m"),
            name="grasp_offset_m",
            length=3,
            default=(0.0, 0.0, 0.0),
        ),
        config,
        jitter_name="grasp_offset_jitter_m",
        seed=seed,
        stream=stream,
        batch_size=1,
        dtype=grasp.dtype,
        device=grasp.device,
    )
    return grasp


def _grasp_source(config: Mapping[str, object]) -> str:
    """Validate the configured grasp source."""
    source = str(config.get("grasp_source", "antipodal"))
    if source not in {"antipodal", "fixed"}:
        raise ValueError("grasp_source must be 'antipodal' or 'fixed'.")
    return source


def _grasp_parameters(config: Mapping[str, object]) -> dict[str, object]:
    """Return the antipodal sampling settings recorded with a case."""
    return {
        "grasp_source": _grasp_source(config),
        "grasp_sample_count": int(config.get("grasp_sample_count", 10_000)),
        "grasp_max_candidates": int(config.get("grasp_max_candidates", 128)),
        "grasp_alignment_max_angle_deg": float(
            config.get("grasp_alignment_max_angle_deg", 10.0)
        ),
    }


def _articulation_target(
    config: Mapping[str, object], skill: str
) -> tuple[str, str, str]:
    """Return the configured articulation, link and joint identifiers."""
    values = []
    for key in ("articulation", "target_link", "target_joint"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{skill} cases must define a non-empty {key}.")
        values.append(value)
    return values[0], values[1], values[2]


def _case_identity(
    track: TrackCfg,
    skill_id: str,
    config: Mapping[str, object],
    *,
    seed: int,
    batch_size: int,
) -> str:
    """Return the Atomic Task case id for one configured case."""
    return _atomic_case_id(
        track.id,
        skill_id,
        _case_name(config),
        batch_size=batch_size,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# OpenDoor
# ---------------------------------------------------------------------------


def _door_approach_direction_local(affordance: OpenDoorAffordance) -> torch.Tensor:
    """Mirror OpenDoor: approach against the positive hinge-opening tangent."""
    axis = affordance.rotation_axis.to(dtype=torch.float32)
    axis = axis / torch.linalg.vector_norm(axis)
    origin = torch.tensor(affordance.axis_origin, dtype=torch.float32)
    handle_center = affordance.mesh_vertices.to(dtype=torch.float32).mean(dim=0)
    radial = handle_center - origin
    radial = radial - torch.dot(radial, axis) * axis
    if torch.linalg.vector_norm(radial) <= 1.0e-6:
        raise ValueError("Door-handle center must not lie on the resolved hinge axis.")
    tangent = torch.linalg.cross(axis, radial)
    tangent = tangent / torch.linalg.vector_norm(tangent)
    return -affordance.opening_direction * tangent


def _door_opened_poses(
    link_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    rotation_axis: torch.Tensor,
    axis_origin: Sequence[float],
    hinge_rotation: torch.Tensor,
    waypoint_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror OpenDoor: rotate the handle about the hinge and carry the grasp."""
    device = link_pose.device
    axis = rotation_axis.to(device=device, dtype=torch.float32)
    axis = axis / torch.linalg.vector_norm(axis)
    fractions = torch.linspace(
        1.0 / waypoint_count, 1.0, waypoint_count, dtype=torch.float32, device=device
    )
    angles = hinge_rotation.to(device=device, dtype=torch.float32)[:, None]
    angles = angles * fractions[None]
    rotations = (
        torch.eye(4, dtype=torch.float32, device=device)
        .reshape(1, 1, 4, 4)
        .repeat(link_pose.shape[0], waypoint_count, 1, 1)
    )
    rotations[:, :, :3, :3] = axis_angle_to_rotation_matrix(angles[:, :, None] * axis)
    origin = torch.tensor(axis_origin, dtype=torch.float32, device=device)
    to_origin = torch.eye(4, dtype=torch.float32, device=device)
    from_origin = torch.eye(4, dtype=torch.float32, device=device)
    to_origin[:3, 3] = origin
    from_origin[:3, 3] = -origin
    local_rotations = torch.matmul(
        torch.matmul(to_origin.reshape(1, 1, 4, 4), rotations),
        from_origin.reshape(1, 1, 4, 4),
    )
    opened_link = torch.matmul(link_pose[:, None], local_rotations)
    link_to_eef = torch.bmm(pose_inv(link_pose), grasp_pose)
    return opened_link, torch.matmul(opened_link, link_to_eef[:, None])


def _door_poses(
    link_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    affordance: OpenDoorAffordance,
    hinge_rotation: torch.Tensor,
    options: OpenDoorOptions,
) -> torch.Tensor:
    """Return approach, grasp, every opening keyframe and retract poses."""
    approach_local = _door_approach_direction_local(affordance).to(
        device=link_pose.device
    )
    approach_world = torch.matmul(link_pose[:, :3, :3], approach_local)
    approach = _translate(grasp_pose, -approach_world * options.approach_distance)
    opened_link, opened_eef = _door_opened_poses(
        link_pose,
        grasp_pose,
        affordance.rotation_axis,
        affordance.axis_origin,
        hinge_rotation,
        options.door_waypoint_count,
    )
    rotated_approach = torch.matmul(opened_link[:, -1, :3, :3], approach_local)
    retract = _translate(
        opened_eef[:, -1], -rotated_approach * options.retract_distance
    )
    return torch.cat(
        (approach[:, None], grasp_pose[:, None], opened_eef, retract[:, None]), dim=1
    )


def _door_screening_poses(poses: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Keep approach, grasp, the arc midpoint and end, and retract for screening.

    The full keyframe chain is still solved once for the selected grasp.
    """
    arc_count = poses.shape[1] - 3
    indices = (0, 1, 2 + arc_count // 2, 2 + arc_count - 1, poses.shape[1] - 1)
    return tuple(poses[:, index] for index in indices)


def _door_affordance(case: BenchmarkCase, device: torch.device) -> OpenDoorAffordance:
    """Rebuild the frozen handle and hinge semantics of one OpenDoor case."""
    parameters = case.case_parameters
    limits = parameters["joint_limits"]
    return OpenDoorAffordance(
        mesh_vertices=torch.tensor(
            parameters["mesh_vertices"], dtype=torch.float32, device=device
        ),
        mesh_triangles=torch.tensor(
            parameters["mesh_triangles"], dtype=torch.long, device=device
        ),
        rotation_axis=torch.tensor(
            parameters["rotation_axis"], dtype=torch.float32, device=device
        ),
        axis_origin=tuple(parameters["axis_origin"]),
        joint_name=str(parameters["target_joint"]),
        joint_limits=None if limits is None else (float(limits[0]), float(limits[1])),
        opening_direction=int(parameters["opening_direction"]),
    )


class _OpenDoorCases(AtomicSkillCaseProvider):
    """Grasp a door handle, rotate the door about its hinge, and let go."""

    skill_id = "open_door"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        articulation_id, target_link, target_joint = _articulation_target(
            config, "OpenDoor"
        )
        handle = scenario.activate_articulation(articulation_id)
        scenario.validate_articulation_target(
            handle,
            target_link=target_link,
            target_joint=target_joint,
            joint_type="revolute",
        )
        scenario.restore_base_robot()
        articulation_initial_pose = scenario.randomize_articulation_pose(
            handle, config, seed=seed, stream=62
        )
        scenario.randomize_robot_start(config, seed=seed, stream=61)
        start_qpos, full_start_qpos = _snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        articulation_initial_qpos = handle.entity.get_qpos().clone()
        opening_direction = int(config.get("opening_direction", 1))
        affordance = OpenDoorAffordance.from_articulation(
            handle.entity,
            target_link,
            hinge_joint_name=target_joint,
            opening_direction=opening_direction,
        )
        if affordance.joint_limits is None:
            raise ValueError(f"Hinge {target_joint!r} must expose position limits.")
        lower, upper = affordance.joint_limits
        closed = lower if opening_direction > 0 else upper
        open_angle = float(config.get("open_angle_rad", math.radians(60.0)))
        open_fraction = open_angle / (upper - lower)
        if not 0.0 < open_fraction <= 1.0:
            raise ValueError(
                f"open_angle_rad must lie in (0, {upper - lower:.4f}] for "
                f"hinge {target_joint!r}."
            )
        target_position = closed + opening_direction * open_angle
        hinge_index = handle.entity.joint_names.index(target_joint)
        hinge_position = articulation_initial_qpos[:, hinge_index].to(
            device=start_qpos.device, dtype=torch.float32
        )
        hinge_rotation = target_position - hinge_position
        if not bool((hinge_rotation * opening_direction > 1.0e-4).all().item()):
            raise ValueError(
                "OpenDoor cases must start with the hinge short of the target."
            )

        options = OpenDoorOptions(
            hand_interp_steps=int(config.get("hand_interp_steps", 12)),
            door_waypoint_count=int(config.get("door_waypoint_count", 20)),
            approach_distance=float(config.get("approach_distance_m", 0.1)),
            retract_distance=float(config.get("retract_distance_m", 0.1)),
        )
        link_pose = handle.link_pose(target_link).to(device=start_qpos.device)
        mesh_vertices = affordance.mesh_vertices.to(
            device=link_pose.device, dtype=torch.float32
        )
        mesh_triangles = affordance.mesh_triangles.to(
            device=link_pose.device, dtype=torch.long
        )
        approach_local = _door_approach_direction_local(affordance).to(
            device=link_pose.device
        )
        grasp = _grasp_parameters(config)
        grasp_pose = scenario.screen_articulation_grasp(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            target_pose=link_pose,
            approach_direction=torch.matmul(link_pose[:, :3, :3], approach_local),
            start_qpos=start_qpos,
            seed=seed,
            n_sample=int(grasp["grasp_sample_count"]),
            max_candidates=int(grasp["grasp_max_candidates"]),
            alignment_max_angle_deg=float(grasp["grasp_alignment_max_angle_deg"]),
            waypoint_poses=lambda pose: _door_screening_poses(
                _door_poses(link_pose, pose, affordance, hinge_rotation, options)
            ),
        )
        targets = _door_poses(
            link_pose, grasp_pose, affordance, hinge_rotation, options
        )
        references = scenario.solve_reference_qpos(start_qpos, targets)
        limits = affordance.joint_limits
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=_case_identity(
                track, self.skill_id, config, seed=seed, batch_size=batch_size
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=targets.shape[1],
            path_shape="approach_grasp_open_retract",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=articulation_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 240)),
                "door_target_pose": link_pose.detach().cpu().tolist(),
                "grasp_pose": grasp_pose.detach().cpu().tolist(),
                "mesh_vertices": mesh_vertices.detach().cpu().tolist(),
                "mesh_triangles": mesh_triangles.detach().cpu().tolist(),
                "rotation_axis": affordance.rotation_axis.detach().cpu().tolist(),
                "axis_origin": list(affordance.axis_origin),
                "joint_limits": [float(limits[0]), float(limits[1])],
                "opening_direction": opening_direction,
                **grasp,
                "hand_interp_steps": options.hand_interp_steps,
                "door_waypoint_count": options.door_waypoint_count,
                "approach_distance_m": options.approach_distance,
                "retract_distance_m": options.retract_distance,
                "open_angle_rad": open_angle,
                "open_fraction": open_fraction,
                "target_joint_position": target_position,
                "hinge_angle_tolerance_rad": float(
                    config.get(
                        "hinge_angle_tolerance_rad", _TASK_ROTATION_TOLERANCE_RAD
                    )
                ),
                "articulation_initial_pose": articulation_initial_pose.detach()
                .cpu()
                .tolist(),
                "articulation_translation_offsets_m": (
                    articulation_initial_pose[:, :3, 3]
                    - handle.initial_pose[:, :3, 3].to(
                        device=articulation_initial_pose.device,
                        dtype=articulation_initial_pose.dtype,
                    )
                )
                .detach()
                .cpu()
                .tolist(),
                "articulation_initial_qpos": articulation_initial_qpos.detach()
                .cpu()
                .tolist(),
                "articulation_entity_id": handle.entity.uid,
                "hinge_joint_index": hinge_index,
                "target_link": target_link,
                "target_joint": target_joint,
                "target_entity_id": f"{handle.entity.uid}:{target_link}",
                "effect_joint_sign": float(opening_direction),
                "task_success_scope": "physical_hinge_angle_at_open_end",
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        device = scenario.robot.device
        semantics = ObjectSemantics(
            affordance=_door_affordance(case, device),
            geometry={},
            entity_id=str(case.case_parameters["target_entity_id"]),
            label="door_handle",
        )
        return scenario.require_engine().make_invocation(
            self.skill_id,
            OpenDoorGoal(
                semantics,
                _case_pose(case, "door_target_pose", device=device),
                open_fraction=float(case.case_parameters["open_fraction"]),
            ),
            control_parts=_primary_control_parts(scenario, "OpenDoor"),
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=OpenDoorOptions(
                hand_interp_steps=int(case.case_parameters["hand_interp_steps"]),
                door_waypoint_count=int(case.case_parameters["door_waypoint_count"]),
                approach_distance=float(case.case_parameters["approach_distance_m"]),
                retract_distance=float(case.case_parameters["retract_distance_m"]),
            ),
        )

    def initial_scene(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> SceneSnapshot:
        """Observe the frozen hinge angle that OpenDoor plans its rotation from."""
        qpos = torch.tensor(
            case.case_parameters["articulation_initial_qpos"],
            dtype=torch.float32,
            device=scenario.robot.device,
        )
        index = int(case.case_parameters["hinge_joint_index"])
        return SceneSnapshot(
            timestamp=0.0,
            version=0,
            articulation_joints={
                (
                    str(case.case_parameters["articulation_entity_id"]),
                    str(case.case_parameters["target_joint"]),
                ): ObservedArticulationJointState(qpos[:, index : index + 1])
            },
        )

    def frozen_grasps(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Make OpenDoor sample the manifest handle grasp."""
        if scenario.end_effector_part is None:
            raise RuntimeError("OpenDoor requires a configured end-effector part.")
        device = torch.device("cpu")
        return {
            scenario.end_effector_part: (
                _case_pose(case, "door_target_pose", device=device),
                _case_pose(case, "grasp_pose", device=device),
            )
        }

    def articulation_effect_segment(self, case: BenchmarkCase) -> str:
        """Score the hinge at the end of the segment that rotates the door."""
        del case
        return "open"

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect hinge actuation and gripper release samples."""
        del case
        actuation = compiled.segment(0, "open")
        release = compiled.segment(0, "release")
        return (
            ReplayObservationWindow(
                "door_actuation",
                actuation.start,
                actuation.stop,
                frozenset({"articulation_joint"}),
            ),
            ReplayObservationWindow(
                "door_release",
                release.start,
                release.stop,
                frozenset({"gripper_qpos"}),
            ),
        )

    def physical_evaluator(self) -> _OpenDoorEvaluator:
        """Return the v1 OpenDoor evaluator."""
        return _OpenDoorEvaluator()

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del scenario, compiled
        hinge = observation.articulation_joint_effect_end
        if hinge is None:
            return torch.zeros_like(observation.execution_success), (
                "articulation_effect_miss"
            )
        error = (hinge - float(case.case_parameters["target_joint_position"])).abs()
        tolerance = float(case.case_parameters["hinge_angle_tolerance_rad"])
        return (
            observation.execution_success
            & _motion_valid_mask(motion_outcomes, device=hinge.device)
            & torch.isfinite(error)
            & (error <= tolerance),
            "articulation_effect_miss",
        )


# ---------------------------------------------------------------------------
# Pour
# ---------------------------------------------------------------------------


def _pour_poses(
    object_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    approach: torch.Tensor,
    *,
    pre_grasp_distance: float,
    lift_height: float,
    internal_axis: torch.Tensor,
    rotate_angle: float,
) -> torch.Tensor:
    """Return PickUp poses, the poured pose and the return to the lift pose."""
    pre_grasp = _translate(grasp_pose, -approach * pre_grasp_distance)
    lift = grasp_pose.clone()
    lift[:, 2, 3] += lift_height
    object_to_eef = torch.bmm(pose_inv(object_pose), grasp_pose)
    lifted_object = torch.bmm(lift, pose_inv(object_to_eef))
    world_axis = torch.matmul(lifted_object[:, :3, :3], internal_axis)
    rotation = axis_angle_to_rotation_matrix(world_axis * rotate_angle)
    poured_object = lifted_object.clone()
    poured_object[:, :3, :3] = torch.bmm(rotation, lifted_object[:, :3, :3])
    poured = torch.bmm(poured_object, object_to_eef)
    return torch.stack((pre_grasp, grasp_pose, lift, poured, lift), dim=1)


class _PourCases(AtomicSkillCaseProvider):
    """Pick up an object, rotate it about its own axis, and rotate it back."""

    skill_id = "pour"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        object_id = config.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("Pour cases must reference a non-empty object id.")
        handle = scenario.activate_object(object_id)
        scenario.restore_base_robot()
        object_pose, settle_steps = _settle_object_pose(
            scenario, handle, config, seed=seed, stream=401
        )
        device = object_pose.device
        approach = torch.tensor(
            _unit_vector(
                config.get("approach_direction"),
                name="approach_direction",
                default=(-0.7071, 0.0, -0.7071),
            ),
            dtype=torch.float32,
            device=device,
        )
        internal_axis = torch.tensor(
            _unit_vector(
                config.get("internal_axis"),
                name="internal_axis",
                default=(1.0, 0.0, 0.0),
            ),
            dtype=torch.float32,
            device=device,
        )
        rotate_angle = float(config.get("rotate_angle_rad", math.pi / 4.0))
        if not math.isfinite(rotate_angle) or abs(rotate_angle) <= 1.0e-4:
            raise ValueError("rotate_angle_rad must be finite and non-zero.")
        pre_grasp_distance = float(config.get("pre_grasp_distance_m", 0.15))
        lift_height = float(config.get("lift_height_m", 0.16))

        def poses(pose_object: torch.Tensor, grasp: torch.Tensor) -> torch.Tensor:
            return _pour_poses(
                pose_object,
                grasp,
                approach,
                pre_grasp_distance=pre_grasp_distance,
                lift_height=lift_height,
                internal_axis=internal_axis,
                rotate_angle=rotate_angle,
            )

        screen_start = scenario.robot.get_qpos(name=scenario.control_part).clone()
        grasp = _grasp_parameters(config)
        if grasp["grasp_source"] == "antipodal":
            grasp_pose = scenario.screen_antipodal_grasp(
                handle,
                object_pose,
                approach,
                seed=seed,
                start_qpos=screen_start[:1],
                n_sample=int(grasp["grasp_sample_count"]),
                max_candidates=int(grasp["grasp_max_candidates"]),
                alignment_max_angle_deg=float(grasp["grasp_alignment_max_angle_deg"]),
                waypoint_poses=lambda pose: poses(object_pose[:1], pose).unbind(dim=1),
            )
        else:
            grasp_pose = _fixed_grasp(object_pose, config, seed=seed, stream=402)
        object_to_eef = torch.bmm(pose_inv(object_pose[:1]), grasp_pose)
        grasp_pose = torch.bmm(object_pose, object_to_eef.expand(batch_size, -1, -1))
        targets = poses(object_pose, grasp_pose)
        success, pre_pick_qpos = scenario.robot.compute_ik(
            pose=targets[:, 0], joint_seed=screen_start, name=scenario.control_part
        )
        if not bool(torch.as_tensor(success).all().item()):
            raise RuntimeError(
                f"Independent IK rejected the Pour pre-grasp {_case_name(config)!r}."
            )
        scenario.set_robot_start(_canonical_case_qpos(pre_pick_qpos), open_gripper=True)
        start_qpos, full_start_qpos = _snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        references = scenario.solve_reference_qpos(start_qpos, targets)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=_case_identity(
                track, self.skill_id, config, seed=seed, batch_size=batch_size
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=targets.shape[1],
            path_shape="pick_up_then_pour_and_return",
            start_state_bin="pre_pick",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=handle.object_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 80)),
                "pick_sample_count": int(config.get("pick_sample_count", 120)),
                **grasp,
                "hand_interp_steps": int(config.get("hand_interp_steps", 12)),
                "approach_direction": approach.detach().cpu().tolist(),
                "pre_grasp_distance_m": pre_grasp_distance,
                "lift_height_m": lift_height,
                "pre_action_settle_steps": settle_steps,
                "object_initial_pose": object_pose.detach().cpu().tolist(),
                "grasp_pose": grasp_pose.detach().cpu().tolist(),
                "object_config": dict(handle.config),
                "rotate_angle_rad": rotate_angle,
                "object_rotation_axis": internal_axis.detach().cpu().tolist(),
                "object_rotation_sign": 1.0 if rotate_angle > 0.0 else -1.0,
                "pour_angle_tolerance_rad": float(
                    config.get("pour_angle_tolerance_rad", _TASK_ROTATION_TOLERANCE_RAD)
                ),
                "minimum_object_lift_m": float(
                    config.get("minimum_object_lift_m", 0.04)
                ),
                "task_success_scope": "physical_peak_object_rotation_while_held",
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def prefix_invocations(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> tuple[ActionInvocation, ...]:
        """Acquire the object with the frozen grasp before pouring."""
        device = scenario.robot.device
        handle = scenario.object_handle(case.object_id)
        parameters = case.case_parameters
        semantics = _object_semantics(
            handle,
            AxisAlignAffordance(
                internal_axis=torch.tensor(
                    parameters["object_rotation_axis"],
                    dtype=torch.float32,
                    device=device,
                )
            ),
        )
        return (
            scenario.require_engine().make_invocation(
                "pick_up",
                GraspGoal(
                    semantics=semantics,
                    grasp_xpos=_case_pose(case, "grasp_pose", device=device),
                    object_pose=_case_pose(case, "object_initial_pose", device=device),
                ),
                control_parts=_primary_control_parts(scenario, "Pour"),
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=int(parameters["pick_sample_count"]),
                ),
                skill_options=PickUpOptions(
                    approach_direction=torch.tensor(
                        parameters["approach_direction"],
                        dtype=torch.float32,
                        device=device,
                    ),
                    pre_grasp_distance=float(parameters["pre_grasp_distance_m"]),
                    lift_height=float(parameters["lift_height_m"]),
                    hand_interp_steps=int(parameters["hand_interp_steps"]),
                ),
            ),
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        return scenario.require_engine().make_invocation(
            self.skill_id,
            PourGoal(),
            control_parts=_primary_control_parts(scenario, "Pour"),
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=PourOptions(
                rotate_angle=float(case.case_parameters["rotate_angle_rad"])
            ),
        )

    def lift_segment_start(self, compiled: CompiledTrajectory) -> int | None:
        """Release residual contact impulses once the PickUp lift begins."""
        return compiled.segment(0, "lift").start

    def object_rotation_segment(self, compiled: CompiledTrajectory) -> tuple[int, int]:
        """Measure the held object's rotation over the Pour segment."""
        segment = compiled.segment(1, "pour")
        return segment.start, segment.stop

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect held-object poses only within the pour interval."""
        del case
        segment = compiled.segment(1, "pour")
        return (
            ReplayObservationWindow(
                "pour_actuation",
                segment.start,
                segment.stop,
                frozenset({"object_pose"}),
            ),
        )

    def physical_evaluator(self) -> _PourEvaluator:
        """Return the v1 Pour evaluator."""
        return _PourEvaluator()

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del scenario, compiled
        peak = observation.object_rotation_peak_signed_rad
        lift = observation.object_lift_delta_m
        if peak is None or lift is None:
            return torch.zeros_like(observation.execution_success), (
                "object_rotation_miss"
            )
        error = (peak - float(case.case_parameters["rotate_angle_rad"])).abs()
        return (
            observation.execution_success
            & _motion_valid_mask(motion_outcomes, device=peak.device)
            & torch.isfinite(error)
            & (error <= float(case.case_parameters["pour_angle_tolerance_rad"]))
            & (lift >= float(case.case_parameters["minimum_object_lift_m"])),
            "object_rotation_miss",
        )


# ---------------------------------------------------------------------------
# AxisAlign
# ---------------------------------------------------------------------------

_UPRIGHT_HORIZONTAL_MAX_ABS_Z = 0.5
_UPRIGHT_TARGET_MIN_Z = math.cos(math.pi / 6.0)
_UPRIGHT_GRASP_PRE_ROTATION = math.pi / 4.0


def _axis_alignment_parameters(
    object_pose: torch.Tensor,
    internal_axis: torch.Tensor,
    target_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror AxisAlign: source/target axes and the shortest rotation."""
    device = object_pose.device
    internal = internal_axis.to(device=device, dtype=torch.float32)
    internal = internal / torch.linalg.vector_norm(internal)
    source = torch.nn.functional.normalize(
        torch.matmul(object_pose[:, :3, :3], internal), dim=1
    )
    target = torch.nn.functional.normalize(
        target_axis.to(device=device, dtype=torch.float32)
        .reshape(1, 3)
        .expand(object_pose.shape[0], -1),
        dim=1,
    )
    cross = torch.linalg.cross(source, target, dim=1)
    sin_angle = torch.linalg.vector_norm(cross, dim=1)
    cos_angle = torch.sum(source * target, dim=1).clamp(-1.0, 1.0)
    axis = cross / sin_angle.clamp_min(1.0e-8).unsqueeze(1)
    basis = torch.eye(3, dtype=torch.float32, device=device)
    reference = basis[torch.argmin(torch.abs(source), dim=1)]
    fallback_axis = torch.nn.functional.normalize(
        torch.linalg.cross(source, reference, dim=1), dim=1
    )
    degenerate = sin_angle <= 1.0e-6
    axis = torch.where(degenerate.unsqueeze(1), fallback_axis, axis)
    angle = torch.atan2(sin_angle, cos_angle)
    opposite = degenerate & (cos_angle < 0.0)
    angle = torch.where(opposite, torch.full_like(angle, torch.pi), angle)
    return source, target, axis, angle


def _symmetric_nearest(pose: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Mirror AxisAlign: pick the half-turn grasp variant nearest the start TCP."""

    def angle(rotation: torch.Tensor) -> torch.Tensor:
        relative = reference[:, :3, :3].transpose(-1, -2) @ rotation
        trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1)
        return torch.arccos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))

    symmetric = pose.clone()
    symmetric[:, :3, 0] = -symmetric[:, :3, 0]
    symmetric[:, :3, 1] = -symmetric[:, :3, 1]
    keep = (angle(pose[:, :3, :3]) < angle(symmetric[:, :3, :3]))[:, None, None]
    return torch.where(keep, pose, symmetric)


def _axis_align_poses(
    object_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    approach: torch.Tensor,
    *,
    pre_grasp_distance: float,
    lift_height: float,
    internal_axis: torch.Tensor,
    target_axis: torch.Tensor,
) -> torch.Tensor:
    """Mirror AxisAlign for an explicit grasp: pre-grasp, grasp, lift, aligned."""
    source, target, rotation_axis, angle = _axis_alignment_parameters(
        object_pose, internal_axis, target_axis
    )
    upright = (source[:, 2].abs() <= _UPRIGHT_HORIZONTAL_MAX_ABS_Z) & (
        target[:, 2] >= _UPRIGHT_TARGET_MIN_Z
    )
    pre_rotated = grasp_pose.clone()
    pre_rotated[:, :3, :3] = torch.matmul(
        axis_angle_to_rotation_matrix(-rotation_axis * _UPRIGHT_GRASP_PRE_ROTATION),
        grasp_pose[:, :3, :3],
    )
    grasp = torch.where(upright[:, None, None], pre_rotated, grasp_pose)
    pre_grasp = _translate(grasp, -approach * pre_grasp_distance)
    lift = grasp.clone()
    lift[:, 2, 3] += lift_height
    object_to_eef = torch.bmm(pose_inv(object_pose), grasp)
    lifted_object = torch.bmm(lift, pose_inv(object_to_eef))
    aligned_object = lifted_object.clone()
    aligned_object[:, :3, :3] = torch.bmm(
        axis_angle_to_rotation_matrix(rotation_axis * angle[:, None]),
        lifted_object[:, :3, :3],
    )
    aligned = torch.bmm(aligned_object, object_to_eef)
    return torch.stack((pre_grasp, grasp, lift, aligned), dim=1)


class _AxisAlignCases(AtomicSkillCaseProvider):
    """Grasp an object, lift it, and align one of its axes to a world axis."""

    skill_id = "axis_align"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        object_id = config.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("AxisAlign cases must reference a non-empty object id.")
        handle = scenario.activate_object(object_id)
        scenario.restore_base_robot()
        object_pose, settle_steps = _settle_object_pose(
            scenario, handle, config, seed=seed, stream=501
        )
        device = object_pose.device
        approach = torch.tensor(
            _unit_vector(
                config.get("approach_direction"),
                name="approach_direction",
                default=(0.0, 0.0, -1.0),
            ),
            dtype=torch.float32,
            device=device,
        )
        internal_axis = torch.tensor(
            _unit_vector(
                config.get("internal_axis"),
                name="internal_axis",
                default=(1.0, 0.0, 0.0),
            ),
            dtype=torch.float32,
            device=device,
        )
        target_axis = torch.tensor(
            _unit_vector(
                config.get("target_axis"), name="target_axis", default=(0.0, 0.0, 1.0)
            ),
            dtype=torch.float32,
            device=device,
        )
        pre_grasp_distance = float(config.get("pre_grasp_distance_m", 0.15))
        lift_height = float(config.get("lift_height_m", 0.16))
        _set_pre_pick_start(scenario, object_pose, config, skill="AxisAlign")
        start_qpos, full_start_qpos = _snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        start_pose = scenario.robot.compute_fk(
            start_qpos, name=scenario.control_part, to_matrix=True
        )

        def poses(pose_object: torch.Tensor, grasp: torch.Tensor) -> torch.Tensor:
            return _axis_align_poses(
                pose_object,
                grasp,
                approach,
                pre_grasp_distance=pre_grasp_distance,
                lift_height=lift_height,
                internal_axis=internal_axis,
                target_axis=target_axis,
            )

        _, _, rotation_axis, _ = _axis_alignment_parameters(
            object_pose[:1], internal_axis, target_axis
        )

        def perpendicular_first(
            candidates: torch.Tensor, costs: torch.Tensor
        ) -> torch.Tensor:
            # AxisAlign prefers grasps whose finger axis is perpendicular to the
            # rotation axis; grasp cost orders the equally perpendicular ones.
            error = torch.abs(
                torch.nn.functional.normalize(candidates[:, :3, 1], dim=1)
                @ rotation_axis[0].to(device=candidates.device)
            )
            by_cost = torch.argsort(costs)
            quantized = torch.round(error[by_cost] / 1.0e-6)
            return by_cost[torch.argsort(quantized, stable=True)]

        grasp = _grasp_parameters(config)
        if grasp["grasp_source"] == "antipodal":
            sampled = scenario.screen_antipodal_grasp(
                handle,
                object_pose,
                approach,
                seed=seed,
                start_qpos=start_qpos[:1],
                n_sample=int(grasp["grasp_sample_count"]),
                max_candidates=int(grasp["grasp_max_candidates"]),
                alignment_max_angle_deg=float(grasp["grasp_alignment_max_angle_deg"]),
                waypoint_poses=lambda pose: poses(
                    object_pose[:1], _symmetric_nearest(pose, start_pose[:1])
                ).unbind(dim=1),
                rank=perpendicular_first,
            )
        else:
            sampled = _fixed_grasp(object_pose, config, seed=seed, stream=502)
        object_to_eef = torch.bmm(pose_inv(object_pose[:1]), sampled)
        grasp_pose = _symmetric_nearest(
            torch.bmm(object_pose, object_to_eef.expand(batch_size, -1, -1)),
            start_pose,
        )
        targets = poses(object_pose, grasp_pose)
        references = scenario.solve_reference_qpos(start_qpos, targets)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=_case_identity(
                track, self.skill_id, config, seed=seed, batch_size=batch_size
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=targets.shape[1],
            path_shape="approach_grasp_lift_align",
            start_state_bin="pre_pick",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=handle.object_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 180)),
                **grasp,
                "hand_interp_steps": int(config.get("hand_interp_steps", 12)),
                "approach_direction": approach.detach().cpu().tolist(),
                "pre_grasp_distance_m": pre_grasp_distance,
                "lift_height_m": lift_height,
                "pre_action_settle_steps": settle_steps,
                "object_initial_pose": object_pose.detach().cpu().tolist(),
                "grasp_pose": grasp_pose.detach().cpu().tolist(),
                "object_config": dict(handle.config),
                "internal_axis": internal_axis.detach().cpu().tolist(),
                "target_axis": target_axis.detach().cpu().tolist(),
                "axis_angle_tolerance_rad": float(
                    config.get("axis_angle_tolerance_rad", _TASK_ROTATION_TOLERANCE_RAD)
                ),
                "minimum_object_lift_m": float(
                    config.get("minimum_object_lift_m", 0.04)
                ),
                "task_success_scope": "physical_object_axis_while_held",
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        device = scenario.robot.device
        parameters = case.case_parameters
        handle = scenario.object_handle(case.object_id)
        semantics = _object_semantics(
            handle,
            AxisAlignAffordance(
                internal_axis=torch.tensor(
                    parameters["internal_axis"], dtype=torch.float32, device=device
                )
            ),
        )
        return scenario.require_engine().make_invocation(
            self.skill_id,
            AxisAlignGoal(
                semantics,
                grasp_xpos=_case_pose(case, "grasp_pose", device=device),
            ),
            control_parts=_primary_control_parts(scenario, "AxisAlign"),
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(parameters["sample_count"]),
            ),
            skill_options=AxisAlignOptions(
                target_axis=torch.tensor(
                    parameters["target_axis"], dtype=torch.float32, device=device
                ),
                approach_direction=torch.tensor(
                    parameters["approach_direction"], dtype=torch.float32, device=device
                ),
                pre_grasp_distance=float(parameters["pre_grasp_distance_m"]),
                lift_height=float(parameters["lift_height_m"]),
                hand_interp_steps=int(parameters["hand_interp_steps"]),
            ),
        )

    def lift_segment_start(self, compiled: CompiledTrajectory) -> int | None:
        """Release residual contact impulses once the held object starts moving."""
        return compiled.segment(0, "manipulate").start

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect alignment motion and the terminal settle interval."""
        del case
        segment = compiled.segment(0, "manipulate")
        waypoint_count = compiled.trajectory.waypoint_count
        return (
            ReplayObservationWindow(
                "axis_align_actuation",
                segment.start,
                segment.stop,
                frozenset({"object_pose"}),
            ),
            ReplayObservationWindow(
                "axis_align_settle",
                waypoint_count,
                waypoint_count,
                frozenset({"object_pose", "gripper_qpos"}),
                include_settle=True,
            ),
        )

    def physical_evaluator(self) -> _AxisAlignEvaluator:
        """Return the v1 AxisAlign evaluator."""
        return _AxisAlignEvaluator()

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del scenario, compiled
        final_pose = observation.final_object_pose
        lift = observation.object_lift_delta_m
        if final_pose is None or lift is None:
            return torch.zeros_like(observation.execution_success), (
                "axis_alignment_miss"
            )
        parameters = case.case_parameters
        internal = torch.tensor(
            parameters["internal_axis"],
            dtype=final_pose.dtype,
            device=final_pose.device,
        )
        target = torch.tensor(
            parameters["target_axis"], dtype=final_pose.dtype, device=final_pose.device
        )
        world_axis = torch.nn.functional.normalize(
            torch.matmul(final_pose[:, :3, :3], internal), dim=1
        )
        error = torch.arccos(torch.clamp(world_axis @ target, -1.0, 1.0))
        return (
            observation.execution_success
            & _motion_valid_mask(motion_outcomes, device=final_pose.device)
            & torch.isfinite(error)
            & (error <= float(parameters["axis_angle_tolerance_rad"]))
            & (lift >= float(parameters["minimum_object_lift_m"])),
            "axis_alignment_miss",
        )


# ---------------------------------------------------------------------------
# PushObject
# ---------------------------------------------------------------------------


class _PushObjectEvaluator:
    """Evaluate contact proxy, planar displacement, and settled support pose."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        contact = _replay_trace(observation, "push_contact")
        actuation = _replay_trace(observation, "push_actuation")
        settle = _replay_trace(observation, "push_settle")
        batches: list[Sequence[Mapping[str, StageEvidence]]] = []
        if contact is not None and contact.object_pose is not None:
            contact_motion = torch.linalg.vector_norm(
                contact.object_pose[-1, :, :3, 3] - contact.object_pose[0, :, :3, 3],
                dim=-1,
            )
            batches.append(
                _stage_evidence_batch(
                    stage_id="contacted",
                    window="contact_end",
                    measurements={
                        "contact_established": contact_motion
                        > float(
                            case.case_parameters.get(
                                "contact_motion_threshold_m", 1.0e-5
                            )
                        )
                    },
                    method="object_motion_during_contact",
                    source="measured_proxy",
                )
            )
        if actuation is not None and actuation.object_pose is not None:
            poses = actuation.object_pose
            initial = torch.as_tensor(
                case.case_parameters["object_initial_pose"],
                dtype=poses.dtype,
                device=poses.device,
            )
            target = _case_pose(case, "target_object_pose", device=poses.device).to(
                dtype=poses.dtype
            )
            normal = torch.nn.functional.normalize(target[:, :3, 2], dim=-1)
            desired = target[:, :3, 3] - initial[:, :3, 3]
            desired -= (desired * normal).sum(dim=-1, keepdim=True) * normal
            direction = torch.nn.functional.normalize(desired, dim=-1)
            actual = poses[-1, :, :3, 3] - initial[:, :3, 3]
            residual = target[:, :3, 3] - poses[-1, :, :3, 3]
            residual -= (residual * normal).sum(dim=-1, keepdim=True) * normal
            batches.append(
                _stage_evidence_batch(
                    stage_id="pushed",
                    window="actuation_end",
                    measurements={
                        "signed_displacement_m": (actual * direction).sum(dim=-1),
                        "in_plane_error_m": torch.linalg.vector_norm(residual, dim=-1),
                    },
                    method="object_push_segment_end",
                )
            )
        if settle is not None and settle.object_pose is not None:
            final = settle.object_pose[-1]
            target = _case_pose(case, "target_object_pose", device=final.device).to(
                dtype=final.dtype
            )
            normal = torch.nn.functional.normalize(target[:, :3, 2], dim=-1)
            final_normal = torch.nn.functional.normalize(final[:, :3, 2], dim=-1)
            support_error = (
                ((final[:, :3, 3] - target[:, :3, 3]) * normal).sum(dim=-1).abs()
            )
            batches.append(
                _stage_evidence_batch(
                    stage_id="on_support_plane",
                    window="terminal_settle",
                    measurements={
                        "on_support_plane": support_error
                        <= float(
                            case.case_parameters.get(
                                "support_plane_tolerance_m",
                                _TASK_POSITION_TOLERANCE_M,
                            )
                        ),
                        "tilt_error_rad": torch.arccos(
                            torch.clamp((final_normal * normal).sum(dim=-1), -1.0, 1.0)
                        ),
                    },
                    method="terminal_object_support_pose",
                )
            )
        evidence = (
            _merge_stage_evidence_batches(*batches)
            if batches
            else tuple({} for _ in range(case.batch_size))
        )
        return evaluate_skill_stage_batch("push_object", evidence)


def _push_poses(
    object_pose: torch.Tensor,
    target_pose: torch.Tensor,
    *,
    contact_distance: float,
    approach_height: float,
    retract_height: float,
    push_overshoot: float,
    contact_frame_to_eef: torch.Tensor,
    object_contact_offset: torch.Tensor,
) -> torch.Tensor:
    """Mirror PushObject: approach, contact, pushed and retract TCP poses."""
    normal = torch.nn.functional.normalize(target_pose[:, :3, 2], dim=1)
    displacement = target_pose[:, :3, 3] - object_pose[:, :3, 3]
    planar = displacement - (displacement * normal).sum(dim=1, keepdim=True) * normal
    distance = torch.linalg.vector_norm(planar, dim=1, keepdim=True)
    if bool((distance <= 1.0e-6).any().item()):
        raise ValueError("PushObject cases need a non-zero in-plane displacement.")
    direction = planar / distance
    frame = torch.eye(4, dtype=torch.float32, device=object_pose.device).repeat(
        object_pose.shape[0], 1, 1
    )
    frame[:, :3, 2] = -normal
    x_axis = target_pose[:, :3, 0]
    x_axis = x_axis - (x_axis * normal).sum(dim=1, keepdim=True) * normal
    frame[:, :3, 0] = torch.nn.functional.normalize(x_axis, dim=1)
    frame[:, :3, 1] = torch.linalg.cross(frame[:, :3, 2], frame[:, :3, 0], dim=1)
    contact_center = object_pose[:, :3, 3] + torch.matmul(
        object_pose[:, :3, :3],
        object_contact_offset.to(device=object_pose.device, dtype=torch.float32),
    )
    frame[:, :3, 3] = contact_center - direction * contact_distance
    contact = torch.matmul(frame, contact_frame_to_eef.to(device=frame.device))
    approach = _translate(contact, normal * approach_height)
    pushed = _translate(contact, direction * (distance + push_overshoot))
    retract = _translate(pushed, normal * retract_height)
    return torch.stack((approach, contact, pushed, retract), dim=1)


class _PushObjectCases(AtomicSkillCaseProvider):
    """Close the gripper and push a rigid object to a target on its support."""

    skill_id = "push_object"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        object_id = config.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("PushObject cases must reference a non-empty object id.")
        handle = scenario.activate_object(object_id)
        scenario.restore_base_robot()
        object_pose, settle_steps = _settle_object_pose(
            scenario, handle, config, seed=seed, stream=601
        )
        scenario.randomize_robot_start(config, seed=seed, stream=602)
        start_qpos, full_start_qpos = _snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        target_pose = object_pose.clone()
        target_pose[:, :3, 3] += _randomized_vector_batch(
            _float_vector(
                config.get("target_object_offset_m"),
                name="target_object_offset_m",
                length=3,
                default=(0.0, 0.10, 0.0),
            ),
            config,
            jitter_name="target_object_offset_jitter_m",
            seed=seed,
            stream=603,
            batch_size=object_pose.shape[0],
            dtype=object_pose.dtype,
            device=object_pose.device,
        )
        calibration = torch.tensor(
            config.get("contact_frame_to_eef", torch.eye(4).tolist()),
            dtype=torch.float32,
            device=object_pose.device,
        )
        if calibration.shape != (4, 4):
            raise ValueError("contact_frame_to_eef must be a 4x4 matrix.")
        contact_offset = torch.tensor(
            _float_vector(
                config.get("object_contact_offset_m"),
                name="object_contact_offset_m",
                length=3,
                default=(0.0, 0.0, 0.0),
            ),
            dtype=torch.float32,
            device=object_pose.device,
        )
        options = PushObjectOptions(
            hand_interp_steps=int(config.get("hand_interp_steps", 12)),
            approach_height=float(config.get("approach_height_m", 0.1)),
            retract_height=float(config.get("retract_height_m", 0.1)),
            contact_distance=float(config.get("contact_distance_m", 0.03)),
            push_overshoot=float(config.get("push_overshoot_m", 0.0)),
            object_contact_offset=contact_offset,
            contact_frame_to_eef=calibration,
        )
        targets = _push_poses(
            object_pose,
            target_pose,
            contact_distance=options.contact_distance,
            approach_height=options.approach_height,
            retract_height=options.retract_height,
            push_overshoot=options.push_overshoot,
            contact_frame_to_eef=calibration,
            object_contact_offset=contact_offset,
        )
        references = scenario.solve_reference_qpos(start_qpos, targets)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=_case_identity(
                track, self.skill_id, config, seed=seed, batch_size=batch_size
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=targets.shape[1],
            path_shape="approach_contact_push_retract",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=handle.object_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 160)),
                "hand_interp_steps": options.hand_interp_steps,
                "approach_height_m": options.approach_height,
                "retract_height_m": options.retract_height,
                "contact_distance_m": options.contact_distance,
                "push_overshoot_m": options.push_overshoot,
                "contact_frame_to_eef": calibration.detach().cpu().tolist(),
                "object_contact_offset_m": contact_offset.detach().cpu().tolist(),
                "pre_action_settle_steps": settle_steps,
                "object_initial_pose": object_pose.detach().cpu().tolist(),
                "target_object_pose": target_pose.detach().cpu().tolist(),
                "object_config": dict(handle.config),
                "object_position_threshold_m": float(
                    config.get(
                        "object_position_threshold_m", _TASK_POSITION_TOLERANCE_M
                    )
                ),
                "object_tilt_threshold_rad": float(
                    config.get(
                        "object_tilt_threshold_rad", _TASK_ROTATION_TOLERANCE_RAD
                    )
                ),
                "contact_motion_threshold_m": float(
                    config.get("contact_motion_threshold_m", 1.0e-5)
                ),
                "support_plane_tolerance_m": float(
                    config.get("support_plane_tolerance_m", _TASK_POSITION_TOLERANCE_M)
                ),
                "task_success_scope": "physical_object_pose_after_push",
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        device = scenario.robot.device
        parameters = case.case_parameters
        handle = scenario.object_handle(case.object_id)
        return scenario.require_engine().make_invocation(
            self.skill_id,
            PushObjectGoal(
                _object_semantics(handle, Affordance()),
                target_pose=_case_pose(case, "target_object_pose", device=device),
            ),
            control_parts=_primary_control_parts(scenario, "PushObject"),
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(parameters["sample_count"]),
            ),
            skill_options=PushObjectOptions(
                hand_interp_steps=int(parameters["hand_interp_steps"]),
                approach_height=float(parameters["approach_height_m"]),
                retract_height=float(parameters["retract_height_m"]),
                contact_distance=float(parameters["contact_distance_m"]),
                push_overshoot=float(parameters["push_overshoot_m"]),
                object_contact_offset=torch.tensor(
                    parameters["object_contact_offset_m"],
                    dtype=torch.float32,
                    device=device,
                ),
                contact_frame_to_eef=torch.tensor(
                    parameters["contact_frame_to_eef"],
                    dtype=torch.float32,
                    device=device,
                ),
            ),
        )

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect contact, push, and terminal support-plane evidence."""
        del case
        contact = compiled.segment(0, "contact")
        push = compiled.segment(0, "push")
        waypoint_count = compiled.trajectory.waypoint_count
        return (
            ReplayObservationWindow(
                "push_contact",
                contact.start,
                contact.stop,
                frozenset({"object_pose"}),
            ),
            ReplayObservationWindow(
                "push_actuation",
                push.start,
                push.stop,
                frozenset({"object_pose"}),
            ),
            ReplayObservationWindow(
                "push_settle",
                waypoint_count,
                waypoint_count,
                frozenset({"object_pose"}),
                include_settle=True,
            ),
        )

    def physical_evaluator(self) -> _PushObjectEvaluator:
        """Return the v1 PushObject evaluator."""
        return _PushObjectEvaluator()

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del compiled
        return (
            _object_pose_success(scenario, case, observation, motion_outcomes),
            "object_goal_miss",
        )


register_atomic_skill_provider("open_door", _OpenDoorCases)
register_atomic_skill_provider("pour", _PourCases)
register_atomic_skill_provider("axis_align", _AxisAlignCases)
register_atomic_skill_provider("push_object", _PushObjectCases)
