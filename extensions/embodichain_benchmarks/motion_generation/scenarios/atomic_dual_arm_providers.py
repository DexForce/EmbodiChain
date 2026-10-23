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

"""Atomic Skill case providers for HandOver and the coordinated dual-arm skills.

Coordinated skills derive their key poses internally: the handover point,
the receiving grasp, the shared lift.  While the case manifest is built, each
provider compiles the skill once with the scenario's reference generator and
records every pose the skill requests for each arm.  Those poses depend on the
skill and the scene, not on the planner, so they become the frozen per-arm
waypoints every benchmarked planner must pass.  Grasps that the skill samples
without a fixed seed are frozen from the same reference compile.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AtomicActionEngine,
    ControlPartCommandProfile,
    CoordinatedPickGoal,
    CoordinatedPickmentOptions,
    CoordinatedPlacementGoal,
    CoordinatedPlacementOptions,
    GraspGoal,
    HandOverGoal,
    HandOverOptions,
    MotionPolicy,
    ObjectSemantics,
    PickUpOptions,
    create_simulation_atomic_action_engine,
)
from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory
from embodichain.lab.sim.motion.planners import ToppraPlanOptions
from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod
from embodichain.toolkits.graspkit import GraspPoseGenerator
from embodichain.utils.math import pose_inv

from ..atomic_skill import AtomicSkillCaseProvider
from ..config import SuiteCfg, TrackCfg
from ..contracts import PhysicalEvaluation, ReplayObservationWindow
from ..evaluation import StageEvidence, evaluate_skill_stage_batch
from ..models import BenchmarkCase, CaseOutcome
from ..contracts import ExecutionObservation
from ._atomic_pose_sampling import apply_task_frame_randomization, _pose_settings

from .atomic_case_utils import (
    apply_target_rotation_randomization,
    atomic_case_id,
    case_name,
    case_pose,
    difficulty,
    float_vector,
    randomized_vector_batch,
    randomization_parameters,
    snapshot_case_qpos,
)
from .atomic_evaluation import (
    gripper_holding,
    gripper_released,
    merge_stage_evidence_batches,
    minimum_height_above_initial_support,
    pose_rotation_error,
    replay_trace,
    stage_evidence_batch,
    trace_validity,
)
from .atomic_grasps import parallel_jaw_sampler
from .atomic_policy import with_plan_options
from ..atomic_skill import register_atomic_skill_provider

if TYPE_CHECKING:
    from ..planners.base import PlannerAdapter
    from .atomic_objects import AtomicObjectHandle
    from .atomic_skill import AtomicSkillScenario

__all__: list[str] = []

# A parallel-jaw gripper closing on an object stops at the object's width; one
# that misses runs to its grasp command.  The tutorial PGI holding the plastic
# tray stops about 3.4 mm short of its 0.026 command and reaches the command
# within 0.1 mm when it closes on nothing.
_GRIPPER_HOLD_GAP = 0.001


class _HandOverEvaluator:
    """Evaluate measured pickup, transfer, handover, and placement stages."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        parameters = case.case_parameters
        parts = dict(parameters["control_parts"])
        source = str(parts["source"]["grasp"])
        destination = str(parts["destination"]["grasp"])
        pickup = replay_trace(observation, "handover_pickup")
        transfer = replay_trace(observation, "handover_transfer")
        exchange = replay_trace(observation, "handover_exchange")
        settle = replay_trace(observation, "handover_settle")
        batches: list[Sequence[Mapping[str, StageEvidence]]] = []
        if pickup is not None:
            grasped = gripper_holding(
                pickup, observation, source, gap=_GRIPPER_HOLD_GAP
            )
            if grasped is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(pickup, observation=observation),
                        stage_id="grasped",
                        window="grasp_end",
                        measurements={"grasp_established": grasped},
                        method="source_gripper_joint_state",
                        source="measured_proxy",
                    )
                )
        if transfer is not None:
            source_held = gripper_holding(
                transfer, observation, source, gap=_GRIPPER_HOLD_GAP
            )
            destination_held = gripper_holding(
                transfer, observation, destination, gap=_GRIPPER_HOLD_GAP
            )
            min_height = minimum_height_above_initial_support(transfer, case)
            if (
                source_held is not None
                and destination_held is not None
                and min_height is not None
            ):
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(transfer, observation=observation),
                        stage_id="transferred",
                        window="transfer_end",
                        measurements={
                            "both_grasps_established": source_held & destination_held,
                            "min_height_above_support_m": min_height,
                        },
                        method="dual_gripper_and_object_trace",
                        source="measured_proxy",
                    )
                )
        if exchange is not None:
            giver_released = gripper_released(exchange, observation, source)
            receiver_retained = gripper_holding(
                exchange, observation, destination, gap=_GRIPPER_HOLD_GAP
            )
            min_height = minimum_height_above_initial_support(exchange, case)
            if (
                giver_released is not None
                and receiver_retained is not None
                and min_height is not None
            ):
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(exchange, observation=observation),
                        stage_id="handed_over",
                        window="handover_end",
                        measurements={
                            "giver_released": giver_released,
                            "receiver_retained": receiver_retained,
                            "min_height_above_support_m": min_height,
                        },
                        method="dual_gripper_and_object_trace",
                        source="measured_proxy",
                    )
                )
        if settle is not None and settle.object_pose is not None:
            target = case_pose(
                case, "target_object_pose", device=settle.object_pose.device
            ).to(dtype=settle.object_pose.dtype)
            final = settle.object_pose[-1]
            xy_error = torch.linalg.vector_norm(
                final[:, :2, 3] - target[:, :2, 3], dim=-1
            )
            signed_z_error = final[:, 2, 3] - target[:, 2, 3]
            min_height = minimum_height_above_initial_support(settle, case)
            if min_height is not None:
                receiver_released = gripper_released(settle, observation, destination)
                if receiver_released is not None:
                    batches.append(
                        stage_evidence_batch(
                            valid_mask=trace_validity(settle, observation=observation),
                            stage_id="receiver_released",
                            window="terminal_settle",
                            measurements={"release_established": receiver_released},
                            method="receiver_gripper_joint_state",
                            source="measured_proxy",
                        )
                    )
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(settle, observation=observation),
                        stage_id="placed",
                        window="terminal_settle",
                        measurements={
                            "xy_error_m": xy_error,
                            "signed_z_error_m": signed_z_error,
                            "min_height_above_support_m": min_height,
                        },
                        method="terminal_object_pose_trace",
                    )
                )
        evidence = (
            merge_stage_evidence_batches(*batches)
            if batches
            else tuple({} for _ in range(case.batch_size))
        )
        return evaluate_skill_stage_batch("hand_over", evidence)


class _CoordinatedPickmentEvaluator:
    """Evaluate dual grasp, lift, synchronized move, and retention."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        parts = dict(case.case_parameters["control_parts"])
        left = str(parts["left"]["grasp"])
        right = str(parts["right"]["grasp"])
        grasp = replay_trace(observation, "dual_pick_grasp")
        lift = replay_trace(observation, "dual_pick_lift")
        move = replay_trace(observation, "dual_pick_move")
        settle = replay_trace(observation, "dual_pick_settle")
        batches: list[Sequence[Mapping[str, StageEvidence]]] = []
        if grasp is not None:
            left_held = gripper_holding(grasp, observation, left, gap=_GRIPPER_HOLD_GAP)
            right_held = gripper_holding(
                grasp, observation, right, gap=_GRIPPER_HOLD_GAP
            )
            if left_held is not None and right_held is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(grasp, observation=observation),
                        stage_id="dual_grasped",
                        window="grasp_end",
                        measurements={
                            "left_grasp_established": left_held,
                            "right_grasp_established": right_held,
                        },
                        method="dual_gripper_joint_state",
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
        if move is not None and move.object_pose is not None:
            final = move.object_pose[-1]
            target = case_pose(case, "target_object_pose", device=final.device).to(
                dtype=final.dtype
            )
            min_height = minimum_height_above_initial_support(move, case)
            if min_height is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(move, observation=observation),
                        stage_id="moved",
                        window="actuation_end",
                        measurements={
                            "position_error_m": torch.linalg.vector_norm(
                                final[:, :3, 3] - target[:, :3, 3], dim=-1
                            ),
                            "rotation_error_rad": pose_rotation_error(final, target),
                            "min_height_above_support_m": min_height,
                        },
                        method="object_move_trace",
                    )
                )
        if settle is not None and settle.object_pose is not None:
            left_held = gripper_holding(
                settle, observation, left, gap=_GRIPPER_HOLD_GAP
            )
            right_held = gripper_holding(
                settle, observation, right, gap=_GRIPPER_HOLD_GAP
            )
            min_height = minimum_height_above_initial_support(settle, case)
            final = settle.object_pose[-1]
            target = case_pose(case, "target_object_pose", device=final.device).to(
                dtype=final.dtype
            )
            if min_height is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(settle, observation=observation),
                        stage_id="settled",
                        window="terminal_settle",
                        measurements={
                            "position_error_m": torch.linalg.vector_norm(
                                final[:, :3, 3] - target[:, :3, 3], dim=-1
                            ),
                            "rotation_error_rad": pose_rotation_error(final, target),
                            "min_height_above_support_m": min_height,
                        },
                        method="terminal_object_pose_trace",
                    )
                )
                if left_held is not None and right_held is not None:
                    batches.append(
                        stage_evidence_batch(
                            valid_mask=trace_validity(settle, observation=observation),
                            stage_id="held",
                            window="terminal_settle",
                            measurements={
                                "object_retained": left_held & right_held,
                                "min_height_above_support_m": min_height,
                            },
                            method="dual_gripper_and_object_settle_trace",
                            source="measured_proxy",
                        )
                    )
        evidence = (
            merge_stage_evidence_batches(*batches)
            if batches
            else tuple({} for _ in range(case.batch_size))
        )
        return evaluate_skill_stage_batch("coordinated_pickment", evidence)


def _coordinated_placement_alignment_target(
    case: BenchmarkCase, *, device: torch.device
) -> torch.Tensor:
    """Return the relative object pose commanded before placement release.

    Coordinated placement applies each object's height offset in world Z.  The
    support can be rotated, so the resulting relative displacement must be
    computed from the two offset absolute targets rather than added in the
    support's local frame.
    """
    parameters = case.case_parameters
    placing = case_pose(case, "placing_target_pose", device=device).clone()
    support = case_pose(case, "support_target_pose", device=device).clone()
    placing[:, 2, 3] += float(parameters["placing_height_offset_m"])
    support[:, 2, 3] += float(parameters["support_height_offset_m"])
    return torch.bmm(pose_inv(support), placing)


class _CoordinatedPlacementEvaluator:
    """Evaluate relative alignment, role-aware release, and settled stability."""

    def evaluate(
        self,
        case: BenchmarkCase,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[PhysicalEvaluation, ...]:
        del motion_outcomes
        parameters = case.case_parameters
        parts = dict(parameters["control_parts"])
        placing_part = str(parts["placing"]["grasp"])
        support_part = str(parts["support"]["grasp"])
        support_id = str(parameters["support_object_id"])
        resting_target = case_pose(
            case,
            "placing_relative_target_pose",
            device=observation.execution_success.device,
        )
        aligned = replay_trace(observation, "coordinated_place_alignment")
        release = replay_trace(observation, "coordinated_place_release")
        settle = replay_trace(observation, "coordinated_place_settle")
        batches: list[Sequence[Mapping[str, StageEvidence]]] = []
        if (
            aligned is not None
            and aligned.object_pose is not None
            and support_id in aligned.scene_object_poses
        ):
            placing = aligned.object_pose[-1]
            support = aligned.scene_object_poses[support_id][-1]
            relative = torch.bmm(pose_inv(support), placing)
            alignment_target = _coordinated_placement_alignment_target(
                case, device=relative.device
            ).to(dtype=relative.dtype)
            min_height = minimum_height_above_initial_support(aligned, case)
            if min_height is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(aligned, observation=observation),
                        stage_id="aligned",
                        window="actuation_end",
                        measurements={
                            "relative_position_error_m": torch.linalg.vector_norm(
                                relative[:, :3, 3] - alignment_target[:, :3, 3],
                                dim=-1,
                            ),
                            "relative_rotation_error_rad": pose_rotation_error(
                                relative, alignment_target
                            ),
                            "min_height_above_support_m": min_height,
                        },
                        method="relative_object_pose_trace",
                    )
                )
        if release is not None:
            placing_released = gripper_released(release, observation, placing_part)
            support_retained = gripper_holding(
                release, observation, support_part, gap=_GRIPPER_HOLD_GAP
            )
            if placing_released is not None and support_retained is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(release, observation=observation),
                        stage_id="released",
                        window="release_end",
                        measurements={
                            "placing_release_established": placing_released,
                            "support_retained": support_retained,
                        },
                        method="role_gripper_joint_state",
                        source="measured_proxy",
                    )
                )
        if (
            settle is not None
            and settle.object_pose is not None
            and support_id in settle.scene_object_poses
            and settle.object_pose.shape[0] >= 2
        ):
            relative = torch.matmul(
                pose_inv(settle.scene_object_poses[support_id]), settle.object_pose
            )
            relative_start = relative[:1].expand_as(relative)
            final_relative = relative[-1]
            local_target = resting_target.to(
                device=final_relative.device, dtype=final_relative.dtype
            )
            min_height = minimum_height_above_initial_support(settle, case)
            if min_height is not None:
                batches.append(
                    stage_evidence_batch(
                        valid_mask=trace_validity(settle, observation=observation),
                        stage_id="stable",
                        window="settle_interval",
                        measurements={
                            "relative_position_error_m": torch.linalg.vector_norm(
                                final_relative[:, :3, 3] - local_target[:, :3, 3],
                                dim=-1,
                            ),
                            "relative_rotation_error_rad": pose_rotation_error(
                                final_relative, local_target
                            ),
                            "position_drift_m": torch.linalg.vector_norm(
                                relative[:, :, :3, 3] - relative_start[:, :, :3, 3],
                                dim=-1,
                            ).amax(dim=0),
                            "rotation_drift_rad": pose_rotation_error(
                                relative, relative_start
                            ).amax(dim=0),
                            "min_height_above_support_m": min_height,
                        },
                        method="relative_object_settle_drift_trace",
                    )
                )
        evidence = (
            merge_stage_evidence_batches(*batches)
            if batches
            else tuple({} for _ in range(case.batch_size))
        )
        return evaluate_skill_stage_batch("coordinated_placement", evidence)


# ---------------------------------------------------------------------------
# Reference compile
# ---------------------------------------------------------------------------


class _RecordingGraspPoseGenerator(GraspPoseGenerator):
    """Forward grasp requests and log the object pose of each request."""

    def __init__(
        self,
        inner: GraspPoseGenerator,
        part: str,
        log: list[tuple[str, torch.Tensor]],
    ) -> None:
        self._inner = inner
        self._part = part
        self._log = log

    def get_valid_grasp_poses(self, **kwargs: object) -> list:
        """Log the requested object poses and forward the request."""
        self._log.append((self._part, kwargs["obj_poses"].detach().clone()))
        return self._inner.get_valid_grasp_poses(**kwargs)

    def get_best_grasp_poses(self, **kwargs: object) -> tuple:
        """Log the requested object poses and forward the request."""
        self._log.append((self._part, kwargs["obj_poses"].detach().clone()))
        return self._inner.get_best_grasp_poses(**kwargs)


@dataclass
class _ReferencePlan:
    """Poses a skill requested while being compiled once."""

    compiled: CompiledTrajectory
    planned_poses: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    solved_poses: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    grasp_requests: list[tuple[str, torch.Tensor]] = field(default_factory=list)

    def arm_poses(self, part: str) -> torch.Tensor:
        """Return one arm's requested poses in order as ``(B, N, 4, 4)``.

        Poses sent to the motion generator take precedence.  A skill that
        never calls it, such as CoordinatedPickment, is described by the IK
        targets it solves itself.
        """
        poses = self.planned_poses.get(part) or self.solved_poses.get(part)
        if not poses:
            raise RuntimeError(f"The reference compile requested no pose for {part!r}.")
        return torch.stack(poses, dim=1)


@contextmanager
def _record_requests(scenario: "AtomicSkillScenario", plan: _ReferencePlan):
    """Record generator targets and stand-alone IK targets during a compile."""
    generator = scenario.reference_motion_generator()
    robot = scenario.robot
    original_generate = generator.generate
    original_ik = robot.compute_ik
    inside_generate = False

    def generate(target_states, options=None):
        nonlocal inside_generate
        part = getattr(options, "control_part", None)
        for state in target_states:
            pose = state.xpos
            if pose is None and state.qpos is not None:
                pose = robot.compute_fk(state.qpos, name=part, to_matrix=True)
            if pose is not None and part is not None:
                plan.planned_poses.setdefault(part, []).append(pose.detach().clone())
        inside_generate = True
        try:
            return original_generate(target_states, options=options)
        finally:
            inside_generate = False

    def compute_ik(*args, **kwargs):
        if not inside_generate and "pose" in kwargs and "name" in kwargs:
            plan.solved_poses.setdefault(kwargs["name"], []).append(
                kwargs["pose"].detach().clone()
            )
        return original_ik(*args, **kwargs)

    generator.generate = generate
    robot.compute_ik = compute_ik
    try:
        yield
    finally:
        generator.generate = original_generate
        robot.compute_ik = original_ik


def _compile_reference(
    scenario: "AtomicSkillScenario",
    build: Callable[[AtomicActionEngine], tuple[ActionInvocation, ...]],
    *,
    scene_entities: Sequence[object],
    grasp_generators: Mapping[str, GraspPoseGenerator] | None = None,
    record_grasp_parts: Sequence[str] = (),
) -> _ReferencePlan:
    """Compile a skill once and record the poses it requests for each arm."""
    if scenario.simulation is None or scenario.suite is None:
        raise RuntimeError("Atomic Skill runtime is not configured.")
    plan = _ReferencePlan(compiled=None)  # type: ignore[arg-type]
    generators = dict(grasp_generators or {})
    for part in record_grasp_parts:
        generators[part] = _RecordingGraspPoseGenerator(
            generators[part], part, plan.grasp_requests
        )
    engine = create_simulation_atomic_action_engine(
        motion_generator=scenario.reference_motion_generator(),
        scene_entities=tuple(scene_entities),
        control_profiles={
            part: ControlPartCommandProfile.joint_positions(
                open=open_state, grasp=grasp
            )
            for part, (open_state, grasp) in scenario._configured_grippers().items()
        },
        grasp_pose_generators=generators or None,
    )
    plan_opts = ToppraPlanOptions(
        sample_method=TrajectorySampleMethod.QUANTITY,
        sample_interval=scenario.suite.protocol.sample_interval,
    )
    invocations = tuple(with_plan_options(item, plan_opts) for item in build(engine))
    context = engine.initial_context(
        control_dt=float(scenario.simulation.sim_config.physics_dt)
    )
    with _record_requests(scenario, plan):
        plan.compiled = engine.compile(invocations, context=context)
    return plan


def _antipodal_semantics(handle: AtomicObjectHandle) -> ObjectSemantics:
    """Return whole-object antipodal semantics for one configured object."""
    from .._atomic_fixtures import (
        create_antipodal_semantics,
    )

    return create_antipodal_semantics(handle.entity, label=handle.object_id)


def _control_parts(config: Mapping[str, object], slots: Sequence[str]) -> dict:
    """Resolve per-slot arm and gripper parts, defaulting to left then right."""
    defaults = {
        slot: {"motion": f"{side}_arm", "grasp": f"{side}_eef"}
        for slot, side in zip(slots, ("left", "right"), strict=True)
    }
    configured = config.get("control_parts", {})
    if not isinstance(configured, Mapping):
        raise TypeError("control_parts must map skill slots to arm/gripper parts.")
    parts = {
        slot: dict(defaults[slot], **dict(configured.get(slot, {}))) for slot in slots
    }
    for slot, values in parts.items():
        if set(values) != {"motion", "grasp"}:
            raise ValueError(f"control_parts.{slot} needs exactly motion and grasp.")
    return parts


def _arm_waypoints(
    scenario: "AtomicSkillScenario", plan: _ReferencePlan, arms: Sequence[str]
) -> tuple[torch.Tensor, dict[str, list]]:
    """Split recorded poses into the primary arm and every other arm."""
    if scenario.control_part not in arms:
        raise ValueError(
            f"The robot control part {scenario.control_part!r} must be one of {arms}."
        )
    primary = plan.arm_poses(scenario.control_part)
    others = {
        arm: plan.arm_poses(arm).detach().cpu().tolist()
        for arm in arms
        if arm != scenario.control_part
    }
    return primary, others


def _settled_pose(
    scenario: "AtomicSkillScenario",
    handle: AtomicObjectHandle,
    config: Mapping[str, object],
    *,
    seed: int,
    stream: int,
) -> tuple[torch.Tensor, int]:
    """Install an object's configured pose and return it after it settles."""
    settle_steps = config.get(
        "pre_action_settle_steps", handle.config.get("settle_steps", 50)
    )
    if type(settle_steps) is not int or settle_steps < 0:
        raise ValueError("pre_action_settle_steps must be a non-negative integer.")
    pose = scenario.randomize_object_pose(
        handle, config, seed=seed, stream=stream, settle_steps=settle_steps
    )
    return pose, settle_steps


def _randomized_target_pose(
    object_pose: torch.Tensor,
    config: Mapping[str, object],
    *,
    default_offset: Sequence[float],
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Translate each row by its deterministic target offset."""
    target_pose = object_pose.clone()
    target_pose[:, :3, 3] += randomized_vector_batch(
        float_vector(
            config.get("target_object_offset_m"),
            name="target_object_offset_m",
            length=3,
            default=default_offset,
        ),
        config,
        jitter_name="target_object_offset_jitter_m",
        seed=seed,
        stream=stream,
        batch_size=object_pose.shape[0],
        device=object_pose.device,
        dtype=object_pose.dtype,
    )
    apply_target_rotation_randomization(
        target_pose,
        config,
        seed=seed,
        stream=stream + 1,
    )
    return target_pose


def _support_randomization_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Map support-object overrides onto the generic object jitter contract."""
    support_config = dict(config)
    if "_pose_batch" in config:
        settings = _pose_settings(config)
        support_config["_pose_batch"] = replace(
            settings,
            object_translation_jitter_m=settings.support_object_translation_jitter_m,
            object_yaw_jitter_rad=settings.support_object_yaw_jitter_rad,
            object_roll_pitch_jitter_rad=settings.support_object_roll_pitch_jitter_rad,
        )
    support_config.pop("object_position_jitter_m", None)
    support_config.pop("object_yaw_jitter_rad", None)
    support_config.pop("object_roll_pitch_jitter_rad", None)
    if "support_object_position_jitter_m" in config:
        support_config["object_position_jitter_m"] = config[
            "support_object_position_jitter_m"
        ]
    if "support_object_yaw_jitter_rad" in config:
        support_config["object_yaw_jitter_rad"] = config[
            "support_object_yaw_jitter_rad"
        ]
    if "support_object_roll_pitch_jitter_rad" in config:
        support_config["object_roll_pitch_jitter_rad"] = config[
            "support_object_roll_pitch_jitter_rad"
        ]
    return support_config


def _settled_placement_poses(
    scenario: "AtomicSkillScenario",
    placing: AtomicObjectHandle,
    support: AtomicObjectHandle,
    config: Mapping[str, object],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Install both randomized object batches and settle the pair together."""
    settle_steps = config.get("pre_action_settle_steps", 50)
    if type(settle_steps) is not int or settle_steps < 0:
        raise ValueError("pre_action_settle_steps must be a non-negative integer.")
    scenario.randomize_object_pose(
        placing, config, seed=seed, stream=901, settle_steps=0
    )
    scenario.randomize_object_pose(
        support,
        _support_randomization_config(config),
        seed=seed,
        stream=903,
        settle_steps=0,
    )
    if scenario.simulation is not None and settle_steps:
        scenario.simulation.update(step=settle_steps)
    placing_pose = placing.entity.get_local_pose(to_matrix=True).clone()
    support_pose = support.entity.get_local_pose(to_matrix=True).clone()
    for handle in (placing, support):
        handle.entity.clear_dynamics()
    return placing_pose, support_pose, settle_steps


def _case(
    provider: AtomicSkillCaseProvider,
    scenario: "AtomicSkillScenario",
    suite: SuiteCfg,
    track: TrackCfg,
    config: Mapping[str, object],
    *,
    seed: int,
    batch_size: int,
    path_shape: str,
    object_id: str,
    targets: torch.Tensor,
    start_qpos: torch.Tensor,
    full_start_qpos: torch.Tensor,
    parameters: dict[str, object],
) -> BenchmarkCase:
    """Assemble a coordinated case whose primary waypoints were recorded."""
    return BenchmarkCase(
        suite_version=suite.suite_version,
        track=track.id,
        scenario_id=provider.skill_id,
        case_id=atomic_case_id(
            track.id,
            provider.skill_id,
            case_name(config),
            batch_size=batch_size,
            seed=seed,
        ),
        seed=seed,
        batch_size=batch_size,
        num_waypoints=targets.shape[1],
        path_shape=path_shape,
        start_state_bin="pre_action",
        start_qpos=start_qpos,
        target_waypoints=targets,
        reference_qpos=scenario.solve_reference_qpos(start_qpos, targets),
        robot_id=suite.robot.id,
        skill_id=provider.skill_id,
        object_id=object_id,
        task_difficulty=difficulty(config),
        primary_success="physical_skill_success",
        full_start_qpos=full_start_qpos,
        case_parameters={
            **parameters,
            "randomization": randomization_parameters(config, seed=seed),
            "difficulty_factors": dict(config.get("difficulty_factors", {})),
        },
    )


# ---------------------------------------------------------------------------
# HandOver
# ---------------------------------------------------------------------------


def _hand_over_options(parameters: Mapping[str, object]) -> HandOverOptions:
    return HandOverOptions(
        pre_grasp_distance=float(parameters["pre_grasp_distance_m"]),
        lift_height=float(parameters["lift_height_m"]),
        hand_interp_steps=int(parameters["hand_interp_steps"]),
        hold_steps=int(parameters["hold_steps"]),
        retreat_steps=int(parameters["retreat_steps"]),
        retreat_distance=float(parameters["retreat_distance_m"]),
    )


class _HandOverCases(AtomicSkillCaseProvider):
    """Pick an object with one arm, hand it to the other, and place it."""

    skill_id = "hand_over"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicSkillScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        object_id = config.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("HandOver cases must reference a non-empty object id.")
        handle = scenario.activate_object(object_id)
        scenario.restore_base_robot()
        object_pose, settle_steps = _settled_pose(
            scenario, handle, config, seed=seed, stream=701
        )
        scenario.open_grippers()
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        target_pose = _randomized_target_pose(
            object_pose,
            config,
            default_offset=(0.0, -0.22, 0.06),
            seed=seed,
            stream=703,
        )
        parts = _control_parts(config, ("source", "destination"))
        parameters: dict[str, object] = {
            "sample_count": int(config.get("sample_count", 220)),
            "pre_grasp_distance_m": float(config.get("pre_grasp_distance_m", 0.08)),
            "lift_height_m": float(config.get("lift_height_m", 0.15)),
            "hand_interp_steps": int(config.get("hand_interp_steps", 10)),
            "hold_steps": int(config.get("hold_steps", 4)),
            "retreat_steps": int(config.get("retreat_steps", 20)),
            "retreat_distance_m": float(config.get("retreat_distance_m", 0.10)),
            "control_parts": parts,
        }
        grasp_parts = [values["grasp"] for values in parts.values()]
        sampler = parallel_jaw_sampler(int(config.get("grasp_sample_count", 10_000)))
        semantics = _antipodal_semantics(handle)
        reference = _compile_reference(
            scenario,
            lambda engine: (
                engine.make_invocation(
                    self.skill_id,
                    HandOverGoal(semantics, target_pose=target_pose),
                    control_parts=parts,
                    motion_policy=MotionPolicy(
                        strategy="motion_gen",
                        sample_count=int(parameters["sample_count"]),
                    ),
                    skill_options=_hand_over_options(parameters),
                ),
            ),
            scene_entities=(handle.entity,),
            grasp_generators={part: sampler for part in grasp_parts},
            record_grasp_parts=grasp_parts,
        )
        # HandOver samples one grasp per gripper: the pickup grasp on the
        # settled object, then the receiving grasp on the predicted handover
        # pose.  Each arm's first request ends at its grasp.
        frozen: dict[str, list] = {}
        for part, obj_poses in reference.grasp_requests:
            if part in frozen:
                continue
            arm = next(v["motion"] for v in parts.values() if v["grasp"] == part)
            grasp = reference.arm_poses(arm)[:, 1]
            frozen[part] = [obj_poses.cpu().tolist(), grasp.cpu().tolist()]
        if set(frozen) != set(grasp_parts):
            raise RuntimeError("The reference HandOver did not sample both grasps.")
        targets, other_arms = _arm_waypoints(
            scenario, reference, [values["motion"] for values in parts.values()]
        )
        return _case(
            self,
            scenario,
            suite,
            track,
            config,
            seed=seed,
            batch_size=batch_size,
            path_shape="pick_up_hand_over_place",
            object_id=handle.object_id,
            targets=targets,
            start_qpos=start_qpos,
            full_start_qpos=full_start_qpos,
            parameters={
                **parameters,
                "pre_action_settle_steps": settle_steps,
                "object_initial_pose": object_pose.detach().cpu().tolist(),
                "target_object_pose": target_pose.detach().cpu().tolist(),
                "frozen_grasps": frozen,
                "arm_waypoints": other_arms,
                "reference_plan_success": reference.compiled.plan_success.tolist(),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        handle = scenario.object_handle(case.object_id)
        return scenario.require_engine().make_invocation(
            self.skill_id,
            HandOverGoal(
                _antipodal_semantics(handle),
                target_pose=case_pose(
                    case, "target_object_pose", device=scenario.robot.device
                ),
            ),
            control_parts=dict(case.case_parameters["control_parts"]),
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=_hand_over_options(case.case_parameters),
        )

    def frozen_grasps(
        self, scenario: "AtomicSkillScenario", case: BenchmarkCase
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Replay the reference compile's pickup and receiving grasps."""
        del scenario
        return {
            part: (
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(grasp, dtype=torch.float32),
            )
            for part, (target, grasp) in dict(
                case.case_parameters["frozen_grasps"]
            ).items()
        }

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect each physical handover phase and terminal settle."""
        del case
        pickup = compiled.segment(0, "pickup_close")
        transfer_start = compiled.segment(0, "pickup_transport")
        transfer_end = compiled.segment(0, "receive_close")
        exchange = compiled.segment(0, "handover_release")
        waypoint_count = compiled.trajectory.waypoint_count
        return (
            ReplayObservationWindow(
                "handover_pickup",
                pickup.start,
                pickup.stop,
                frozenset({"gripper_qpos"}),
            ),
            ReplayObservationWindow(
                "handover_transfer",
                transfer_start.start,
                transfer_end.stop,
                frozenset({"object_pose", "gripper_qpos"}),
            ),
            ReplayObservationWindow(
                "handover_exchange",
                exchange.start,
                exchange.stop,
                frozenset({"object_pose", "gripper_qpos"}),
            ),
            ReplayObservationWindow(
                "handover_settle",
                waypoint_count,
                waypoint_count,
                frozenset({"object_pose", "gripper_qpos"}),
                include_settle=True,
            ),
        )

    def physical_evaluator(self) -> _HandOverEvaluator:
        """Return the v1 HandOver evaluator."""
        return _HandOverEvaluator()


# ---------------------------------------------------------------------------
# CoordinatedPickment
# ---------------------------------------------------------------------------


def _left_to_right_direction(scenario: "AtomicSkillScenario", parts: Mapping) -> list:
    """Return the world direction from the left arm base to the right one."""
    robot = scenario.robot
    left = robot.cfg.solver_cfg[parts["left"]["motion"]].root_link_name
    right = robot.cfg.solver_cfg[parts["right"]["motion"]].root_link_name
    left_base = robot.get_link_pose(link_name=left, env_ids=[0], to_matrix=True)
    right_base = robot.get_link_pose(link_name=right, env_ids=[0], to_matrix=True)
    direction = (right_base[0, :3, 3] - left_base[0, :3, 3]).to(dtype=torch.float32)
    return (direction / direction.norm().clamp_min(1.0e-6)).cpu().tolist()


def _pickment_options(
    parameters: Mapping[str, object], device: torch.device
) -> CoordinatedPickmentOptions:
    return CoordinatedPickmentOptions(
        object_motion_keyframes=int(parameters["object_motion_keyframes"]),
        pre_grasp_distance=float(parameters["pre_grasp_distance_m"]),
        lift_height=float(parameters["lift_height_m"]),
        hand_interp_steps=int(parameters["hand_interp_steps"]),
        hold_steps=int(parameters["hold_steps"]),
        left_to_right_arm_direction=torch.tensor(
            parameters["left_to_right_arm_direction"],
            dtype=torch.float32,
            device=device,
        ),
        middle_empty_ratio=float(parameters["middle_empty_ratio"]),
        grasp_seed=int(parameters["grasp_seed"]),
    )


class _CoordinatedPickmentCases(AtomicSkillCaseProvider):
    """Grasp one object with both arms, lift it and move it to a target pose.

    CoordinatedPickment solves its keyframes with robot IK and interpolates
    them itself; it never calls the motion generator, so every planner gets
    the same trajectory.  The case still measures the skill's physical result.
    """

    skill_id = "coordinated_pickment"
    requires_gripper = True
    samples_grasps = True

    def generate_case(
        self,
        scenario: "AtomicSkillScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        object_id = config.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("CoordinatedPickment cases must reference an object id.")
        handle = scenario.activate_object(object_id)
        scenario.restore_base_robot()
        object_pose, settle_steps = _settled_pose(
            scenario, handle, config, seed=seed, stream=801
        )
        scenario.open_grippers()
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        target_pose = _randomized_target_pose(
            object_pose,
            config,
            default_offset=(-0.12, -0.03, 0.108),
            seed=seed,
            stream=803,
        )
        parts = _control_parts(config, ("left", "right"))
        parameters: dict[str, object] = {
            "sample_count": int(config.get("sample_count", 96)),
            "object_motion_keyframes": int(config.get("object_motion_keyframes", 6)),
            "pre_grasp_distance_m": float(config.get("pre_grasp_distance_m", 0.11)),
            "lift_height_m": float(config.get("lift_height_m", 0.10)),
            "hand_interp_steps": int(config.get("hand_interp_steps", 10)),
            "hold_steps": int(config.get("hold_steps", 4)),
            "middle_empty_ratio": float(config.get("middle_empty_ratio", 0.7)),
            "grasp_seed": int(config.get("grasp_seed", 17_393)),
            "grasp_sample_count": int(config.get("grasp_sample_count", 10_000)),
            "left_to_right_arm_direction": _left_to_right_direction(scenario, parts),
            "control_parts": parts,
        }
        sampler = parallel_jaw_sampler(int(parameters["grasp_sample_count"]))
        semantics = _antipodal_semantics(handle)
        reference = _compile_reference(
            scenario,
            lambda engine: (
                engine.make_invocation(
                    self.skill_id,
                    CoordinatedPickGoal(
                        semantics=semantics,
                        object_target_pose=target_pose,
                        object_initial_pose=object_pose,
                    ),
                    control_parts=parts,
                    motion_policy=MotionPolicy(
                        strategy="motion_gen",
                        sample_count=int(parameters["sample_count"]),
                    ),
                    skill_options=_pickment_options(parameters, object_pose.device),
                ),
            ),
            scene_entities=(handle.entity,),
            grasp_generators={values["grasp"]: sampler for values in parts.values()},
        )
        targets, other_arms = _arm_waypoints(
            scenario, reference, [values["motion"] for values in parts.values()]
        )
        return _case(
            self,
            scenario,
            suite,
            track,
            config,
            seed=seed,
            batch_size=batch_size,
            path_shape="dual_grasp_lift_move",
            object_id=handle.object_id,
            targets=targets,
            start_qpos=start_qpos,
            full_start_qpos=full_start_qpos,
            parameters={
                **parameters,
                "pre_action_settle_steps": settle_steps,
                "object_initial_pose": object_pose.detach().cpu().tolist(),
                "target_object_pose": target_pose.detach().cpu().tolist(),
                "arm_waypoints": other_arms,
                "reference_plan_success": reference.compiled.plan_success.tolist(),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        device = scenario.robot.device
        handle = scenario.object_handle(case.object_id)
        return scenario.require_engine().make_invocation(
            self.skill_id,
            CoordinatedPickGoal(
                semantics=_antipodal_semantics(handle),
                object_target_pose=case_pose(case, "target_object_pose", device=device),
                object_initial_pose=case_pose(
                    case, "object_initial_pose", device=device
                ),
            ),
            control_parts=dict(case.case_parameters["control_parts"]),
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=_pickment_options(case.case_parameters, device),
        )

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect dual-grasp, lift, move, and terminal retention evidence."""
        del case
        close = compiled.segment(0, "close")
        lift = compiled.segment(0, "lift")
        move = compiled.segment(0, "move")
        waypoint_count = compiled.trajectory.waypoint_count
        return (
            ReplayObservationWindow(
                "dual_pick_grasp",
                close.start,
                close.stop,
                frozenset({"gripper_qpos"}),
            ),
            ReplayObservationWindow(
                "dual_pick_lift",
                lift.start,
                lift.stop,
                frozenset({"object_pose"}),
            ),
            ReplayObservationWindow(
                "dual_pick_move",
                move.start,
                move.stop,
                frozenset({"object_pose"}),
            ),
            ReplayObservationWindow(
                "dual_pick_settle",
                waypoint_count,
                waypoint_count,
                frozenset({"object_pose", "gripper_qpos"}),
                include_settle=True,
            ),
        )

    def physical_evaluator(self) -> _CoordinatedPickmentEvaluator:
        """Return the v1 CoordinatedPickment evaluator."""
        return _CoordinatedPickmentEvaluator()


# ---------------------------------------------------------------------------
# CoordinatedPlacement
# ---------------------------------------------------------------------------


def _placement_pose_row(
    placing_pose: torch.Tensor,
    support_pose: torch.Tensor,
    placing_vertices: torch.Tensor,
    support_vertices: torch.Tensor,
    local_min: torch.Tensor,
    local_max: torch.Tensor,
    config: Mapping[str, object],
    *,
    placing_target_frame_pose: torch.Tensor | None = None,
    support_target_frame_pose: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build one row of coordinated-placement grasps and targets."""
    from . import _placement_geometry as placement_geometry

    device = placing_pose.device
    placing_grasp = placement_geometry.build_flat_object_grasp_pose(
        placing_pose,
        placing_vertices,
        local_min,
        local_max,
        device,
        z_clearance=float(config.get("placing_grasp_z_clearance_m", 0.018)),
    )
    support_grasp = placement_geometry.build_pan_handle_grasp_pose(
        support_pose,
        support_vertices,
        device,
        z_clearance=float(config.get("support_grasp_z_clearance_m", 0.0)),
    )
    if placing_target_frame_pose is None:
        placing_target_frame_pose = placing_pose
    if support_target_frame_pose is None:
        support_target_frame_pose = support_pose
    support_target = placement_geometry.build_support_object_target_pose(
        support_target_frame_pose, device
    )
    placing_target = placement_geometry.build_placing_object_target_pose(
        placing_target_frame_pose,
        placing_vertices,
        support_vertices,
        support_target,
        device,
    )
    return {
        "placing_grasp_pose": placing_grasp,
        "support_grasp_pose": support_grasp,
        "placing_target_pose": placing_target,
        "support_target_pose": support_target,
    }


def _placement_geometry(
    placing: AtomicObjectHandle,
    support: AtomicObjectHandle,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load local meshes and bounds used by the tutorial pose builders."""
    from .._atomic_fixtures import (
        compute_local_bounds,
        get_local_vertices,
    )

    placing_vertices = get_local_vertices(placing.entity).to(device)
    support_vertices = get_local_vertices(support.entity).to(device)
    local_min, local_max = compute_local_bounds(placing_vertices)
    return placing_vertices, support_vertices, local_min, local_max


def _placement_poses(
    placing: AtomicObjectHandle,
    support: AtomicObjectHandle,
    placing_pose: torch.Tensor,
    support_pose: torch.Tensor,
    config: Mapping[str, object],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build row-specific grasps and targets for placing one object on another."""
    device = placing_pose.device
    placing_vertices, support_vertices, local_min, local_max = _placement_geometry(
        placing, support, device
    )
    task_frame_pose = torch.eye(4, dtype=placing_pose.dtype, device=device).repeat(
        placing_pose.shape[0], 1, 1
    )
    apply_task_frame_randomization(task_frame_pose, config, seed=seed)
    world_to_task_frame = pose_inv(task_frame_pose)
    placing_target_frame_pose = torch.bmm(world_to_task_frame, placing_pose)
    support_target_frame_pose = torch.bmm(world_to_task_frame, support_pose)
    rows = [
        _placement_pose_row(
            placing_pose[row],
            support_pose[row],
            placing_vertices,
            support_vertices,
            local_min,
            local_max,
            config,
            placing_target_frame_pose=placing_target_frame_pose[row],
            support_target_frame_pose=support_target_frame_pose[row],
        )
        for row in range(placing_pose.shape[0])
    ]
    poses = {name: torch.stack([row[name] for row in rows]) for name in rows[0]}
    for name in ("placing_target_pose", "support_target_pose"):
        poses[name] = torch.bmm(task_frame_pose, poses[name])
    target_offset = randomized_vector_batch(
        float_vector(
            config.get("target_object_offset_m"),
            name="target_object_offset_m",
            length=3,
            default=(0.0, 0.0, 0.0),
        ),
        config,
        jitter_name="target_object_offset_jitter_m",
        seed=seed,
        stream=905,
        batch_size=placing_pose.shape[0],
        device=device,
        dtype=placing_pose.dtype,
    )
    poses["placing_target_pose"][:, :3, 3] += target_offset
    poses["support_target_pose"][:, :3, 3] += target_offset
    apply_target_rotation_randomization(
        poses["placing_target_pose"],
        config,
        seed=seed,
        stream=907,
    )
    return poses


def _pick_invocation(
    engine: AtomicActionEngine,
    handle: AtomicObjectHandle,
    grasp_pose: torch.Tensor,
    object_pose: torch.Tensor,
    parts: Mapping[str, str],
    parameters: Mapping[str, object],
    prefix: str,
) -> ActionInvocation:
    """Build the explicit-grasp PickUp that loads one arm before placement."""
    from embodichain.lab.sim.atomic_actions import Affordance

    return engine.make_invocation(
        "pick_up",
        GraspGoal(
            semantics=ObjectSemantics(
                affordance=Affordance(),
                geometry={},
                properties={"benchmark_object_id": handle.object_id},
                label=handle.object_id,
                entity_id=handle.entity.uid,
            ),
            grasp_xpos=grasp_pose,
            object_pose=object_pose,
        ),
        control_parts={"primary": dict(parts)},
        motion_policy=MotionPolicy(
            strategy="motion_gen",
            sample_count=int(parameters[f"{prefix}_pick_sample_count"]),
        ),
        skill_options=PickUpOptions(
            pre_grasp_distance=float(parameters["pick_pre_grasp_distance_m"]),
            lift_height=float(parameters[f"{prefix}_pick_lift_height_m"]),
            hand_interp_steps=int(parameters[f"{prefix}_pick_hand_interp_steps"]),
        ),
    )


def _placement_invocation(
    engine: AtomicActionEngine,
    parameters: Mapping[str, object],
    poses: Mapping[str, torch.Tensor],
) -> ActionInvocation:
    """Build the coordinated placement of the held object onto the held support."""
    return engine.make_invocation(
        "coordinated_placement",
        CoordinatedPlacementGoal(
            placing_object_target_pose=poses["placing_target_pose"],
            support_object_target_pose=poses["support_target_pose"],
            placing_height_offset=float(parameters["placing_height_offset_m"]),
            support_height_offset=float(parameters["support_height_offset_m"]),
            release=True,
        ),
        control_parts=dict(parameters["control_parts"]),
        motion_policy=MotionPolicy(
            strategy="motion_gen",
            sample_count=int(parameters["sample_count"]),
        ),
        skill_options=CoordinatedPlacementOptions(
            release=True,
            placing_height_offset=float(parameters["placing_height_offset_m"]),
            support_height_offset=float(parameters["support_height_offset_m"]),
            lift_height=float(parameters["lift_height_m"]),
            hand_interp_steps=int(parameters["hand_interp_steps"]),
            hold_steps=int(parameters["hold_steps"]),
            retreat_steps=int(parameters["retreat_steps"]),
        ),
    )


class _CoordinatedPlacementCases(AtomicSkillCaseProvider):
    """Pick an object and a support, then place the object onto the support.

    Each arm first runs an explicit-grasp PickUp, as in the tutorial, so the
    placement starts with both objects held.  Success compares the placed
    object's measured pose relative to the support with the planned one.
    """

    skill_id = "coordinated_placement"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicSkillScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        placing_id = config.get("object")
        support_id = config.get("support_object")
        if not isinstance(placing_id, str) or not isinstance(support_id, str):
            raise ValueError(
                "CoordinatedPlacement cases need object and support_object ids."
            )
        placing, support = scenario.activate_objects((placing_id, support_id))
        scenario.restore_base_robot()
        placing_pose, support_pose, settle_steps = _settled_placement_poses(
            scenario, placing, support, config, seed=seed
        )
        scenario.open_grippers()
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        poses = _placement_poses(
            placing, support, placing_pose, support_pose, config, seed=seed
        )
        parts = _control_parts(config, ("placing", "support"))
        parameters: dict[str, object] = {
            "sample_count": int(config.get("sample_count", 120)),
            "placing_height_offset_m": float(
                config.get("placing_height_offset_m", 0.1)
            ),
            "support_height_offset_m": float(
                config.get("support_height_offset_m", 0.0)
            ),
            "lift_height_m": float(config.get("lift_height_m", 0.10)),
            "hand_interp_steps": int(config.get("hand_interp_steps", 10)),
            "hold_steps": int(config.get("hold_steps", 6)),
            "retreat_steps": int(config.get("retreat_steps", 18)),
            "pick_pre_grasp_distance_m": float(
                config.get("pick_pre_grasp_distance_m", 0.12)
            ),
            "placing_pick_sample_count": int(
                config.get("placing_pick_sample_count", 100)
            ),
            "placing_pick_lift_height_m": float(
                config.get("placing_pick_lift_height_m", 0.12)
            ),
            "placing_pick_hand_interp_steps": int(
                config.get("placing_pick_hand_interp_steps", 10)
            ),
            "support_pick_sample_count": int(
                config.get("support_pick_sample_count", 130)
            ),
            "support_pick_lift_height_m": float(
                config.get("support_pick_lift_height_m", 0.10)
            ),
            "support_pick_hand_interp_steps": int(
                config.get("support_pick_hand_interp_steps", 32)
            ),
            "control_parts": parts,
        }

        def invocations(engine: AtomicActionEngine) -> tuple[ActionInvocation, ...]:
            return (
                _pick_invocation(
                    engine,
                    placing,
                    poses["placing_grasp_pose"],
                    placing_pose,
                    parts["placing"],
                    parameters,
                    "placing",
                ),
                _pick_invocation(
                    engine,
                    support,
                    poses["support_grasp_pose"],
                    support_pose,
                    parts["support"],
                    parameters,
                    "support",
                ),
                _placement_invocation(engine, parameters, poses),
            )

        reference = _compile_reference(
            scenario,
            invocations,
            scene_entities=(placing.entity, support.entity),
        )
        targets, other_arms = _arm_waypoints(
            scenario, reference, [values["motion"] for values in parts.values()]
        )
        relative_target = torch.bmm(
            pose_inv(poses["support_target_pose"]), poses["placing_target_pose"]
        )
        return _case(
            self,
            scenario,
            suite,
            track,
            config,
            seed=seed,
            batch_size=batch_size,
            path_shape="pick_both_then_place_on_support",
            object_id=placing.object_id,
            targets=targets,
            start_qpos=start_qpos,
            full_start_qpos=full_start_qpos,
            parameters={
                **parameters,
                "support_object_id": support.object_id,
                "pre_action_settle_steps": settle_steps,
                "object_initial_pose": placing_pose.detach().cpu().tolist(),
                "scene_object_poses": {
                    support.object_id: support_pose.detach().cpu().tolist()
                },
                **{name: pose.detach().cpu().tolist() for name, pose in poses.items()},
                "placing_relative_target_pose": relative_target.detach().cpu().tolist(),
                "arm_waypoints": other_arms,
                "reference_plan_success": reference.compiled.plan_success.tolist(),
            },
        )

    def prefix_invocations(
        self, scenario: "AtomicSkillScenario", case: BenchmarkCase
    ) -> tuple[ActionInvocation, ...]:
        """Load each arm with the frozen explicit-grasp PickUp."""
        device = scenario.robot.device
        parameters = case.case_parameters
        engine = scenario.require_engine()
        parts = dict(parameters["control_parts"])
        support_id = str(parameters["support_object_id"])
        return (
            _pick_invocation(
                engine,
                scenario.object_handle(case.object_id),
                case_pose(case, "placing_grasp_pose", device=device),
                case_pose(case, "object_initial_pose", device=device),
                parts["placing"],
                parameters,
                "placing",
            ),
            _pick_invocation(
                engine,
                scenario.object_handle(support_id),
                case_pose(case, "support_grasp_pose", device=device),
                torch.tensor(
                    parameters["scene_object_poses"][support_id],
                    dtype=torch.float32,
                    device=device,
                ),
                parts["support"],
                parameters,
                "support",
            ),
        )

    def build_invocation(
        self,
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        device = scenario.robot.device
        return _placement_invocation(
            scenario.require_engine(),
            case.case_parameters,
            {
                name: case_pose(case, name, device=device)
                for name in ("placing_target_pose", "support_target_pose")
            },
        )

    def lift_segment_start(self, compiled: CompiledTrajectory) -> int | None:
        """Release residual contact impulses once the first PickUp lifts."""
        return compiled.segment(0, "lift").start

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect relative alignment, release, and stability evidence."""
        support_id = str(case.case_parameters["support_object_id"])
        alignment = compiled.segment(2, "approach")
        release = compiled.segment(2, "release")
        waypoint_count = compiled.trajectory.waypoint_count
        pose_channels = frozenset({"object_pose", "scene_object_pose"})
        return (
            ReplayObservationWindow(
                "coordinated_place_alignment",
                alignment.start,
                alignment.stop,
                pose_channels,
                scene_object_ids=(support_id,),
            ),
            ReplayObservationWindow(
                "coordinated_place_release",
                release.start,
                release.stop,
                frozenset({"gripper_qpos"}),
            ),
            ReplayObservationWindow(
                "coordinated_place_settle",
                waypoint_count,
                waypoint_count,
                pose_channels,
                include_settle=True,
                scene_object_ids=(support_id,),
            ),
        )

    def physical_evaluator(self) -> _CoordinatedPlacementEvaluator:
        """Return the v1 CoordinatedPlacement evaluator."""
        return _CoordinatedPlacementEvaluator()


register_atomic_skill_provider("hand_over", _HandOverCases)
register_atomic_skill_provider("coordinated_pickment", _CoordinatedPickmentCases)
register_atomic_skill_provider("coordinated_placement", _CoordinatedPlacementCases)
