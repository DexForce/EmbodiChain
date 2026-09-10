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

"""PickUp atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import ClassVar, TYPE_CHECKING

import torch

from embodichain.utils import logger
from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    pose_inv,
    quat_error_magnitude,
    quat_from_matrix,
)

from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    require_shared_task_state_key,
    split_joint_trajectory_at_pose,
)
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.candidates import (
    AtomicCandidateBatch,
    AtomicCandidateSelection,
    _input_fingerprint,
    _value_fingerprint,
)
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    OPEN_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction, ObjectSemantics
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import (
    ObjectActionGoal,
    PoseGoalValue,
    _resolve_object_pose,
    collect_scene_dependencies,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    TimedTrajectory,
    normalize_success_mask,
)
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy
from embodichain.lab.sim.atomic_actions.requirements import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    split_three_segments,
    translate_pose_world,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)

if TYPE_CHECKING:
    from embodichain.toolkits.graspkit.candidates import GraspCandidateBatch


_CANDIDATE_POSITION_TOLERANCE = 1.0e-3
_CANDIDATE_ROTATION_TOLERANCE = 1.0e-2
_CANDIDATE_MAX_JOINT_STEP = 0.5


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class PickUpCandidateBatch(AtomicCandidateBatch):
    """All evaluated grasp/roll choices and their certified joint anchors.

    Pose fields have shape ``(E,K,4,4)`` and ``qpos_anchors`` has shape
    ``(E,K,3,arm_dof)`` in pre-grasp, grasp, lift order. Certificates apply only
    to the current invocation input, not another environment or reset.
    """

    grasp_ids: tuple[tuple[str, ...], ...]
    variant_ids: torch.Tensor
    object_to_eef: torch.Tensor
    pre_grasp_poses: torch.Tensor
    grasp_poses: torch.Tensor
    lift_poses: torch.Tensor
    qpos_anchors: torch.Tensor
    downstream_qpos: tuple[torch.Tensor, ...] = ()
    evaluation_fingerprint: str = ""

    def __post_init__(self) -> None:
        AtomicCandidateBatch.__post_init__(self)
        shape = self.valid_mask.shape
        if self.skill_id != "pick_up":
            raise ValueError("PickUp candidates must have skill_id='pick_up'.")
        if self.variant_ids.shape != shape or self.variant_ids.dtype != torch.long:
            raise ValueError("variant_ids must be int64 with shape (E,K).")
        if len(self.grasp_ids) != shape[0] or any(
            len(row) != shape[1] for row in self.grasp_ids
        ):
            raise ValueError("grasp_ids must have E rows and K columns.")
        object.__setattr__(
            self, "grasp_ids", tuple(tuple(row) for row in self.grasp_ids)
        )
        for name in ("object_to_eef", "pre_grasp_poses", "grasp_poses", "lift_poses"):
            value = getattr(self, name)
            if value.shape != (*shape, 4, 4) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite (E,K,4,4) poses.")
        if (
            self.qpos_anchors.ndim != 4
            or self.qpos_anchors.shape[:3] != (*shape, 3)
            or not torch.isfinite(self.qpos_anchors).all()
        ):
            raise ValueError("qpos_anchors must be finite with shape (E,K,3,A).")
        for name in (
            "variant_ids",
            "object_to_eef",
            "pre_grasp_poses",
            "grasp_poses",
            "lift_poses",
            "qpos_anchors",
        ):
            value = getattr(self, name)
            if value.device != self.env_ids.device:
                raise ValueError("PickUp candidate tensors must share a device.")
            object.__setattr__(self, name, value.detach().clone())
        downstream = tuple(value.detach().clone() for value in self.downstream_qpos)
        if any(
            value.shape != (*shape, self.qpos_anchors.shape[-1])
            or value.device != self.env_ids.device
            or not torch.isfinite(value).all()
            for value in downstream
        ):
            raise ValueError("downstream_qpos must contain finite (E,K,A) tensors.")
        object.__setattr__(self, "downstream_qpos", downstream)
        fingerprint = _value_fingerprint(
            self.env_ids,
            self.valid_mask,
            self.costs,
            self.candidate_ids,
            self.grasp_ids,
            self.variant_ids,
            self.object_to_eef,
            self.pre_grasp_poses,
            self.grasp_poses,
            self.lift_poses,
            self.qpos_anchors,
            self.downstream_qpos,
            self.input_fingerprint,
        )
        if self.evaluation_fingerprint and self.evaluation_fingerprint != fingerprint:
            raise ValueError(
                "Evaluated candidate payload was modified; enumerate again."
            )
        object.__setattr__(self, "evaluation_fingerprint", fingerprint)


@dataclass(frozen=True, slots=True, eq=False)
class GraspGoal(ObjectActionGoal):
    """Pickup target with an affordance-selected or supplied grasp pose."""

    grasp_xpos: PoseGoalValue | None = None
    """Optional end-effector grasp pose.

    When omitted, :class:`PickUp` uses the configured fixed object-relative
    grasp when available, otherwise it selects one from the target affordance.
    An explicit tensor or late-bound
    :class:`~embodichain.lab.sim.atomic_actions.goals.SceneEntityPose` skips
    grasp sampling. Late-bound poses also declare the scene dependency used by
    closed-loop execution recovery.
    """

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        if self.grasp_xpos is not None:
            validate_pose_goal(self.grasp_xpos, "grasp_xpos", allow_waypoints=False)


def _validate_single_se3(value: torch.Tensor, name: str) -> None:
    """Validate one finite, proper SE(3) transform."""
    validate_pose_goal(value, name, allow_waypoints=False)
    if value.shape != (4, 4) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be one finite 4x4 transform.")
    transform = value.to(dtype=torch.float64)
    if not torch.allclose(
        transform[3],
        transform.new_tensor((0.0, 0.0, 0.0, 1.0)),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must have bottom row [0, 0, 0, 1].")
    rotation = transform[:3, :3]
    if not torch.allclose(
        rotation.T @ rotation,
        torch.eye(3, dtype=transform.dtype, device=transform.device),
        atol=1.0e-6,
        rtol=0.0,
    ) or not torch.isclose(
        torch.linalg.det(rotation),
        transform.new_tensor(1.0),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must contain a proper SE(3) rotation.")


@dataclass(frozen=True, slots=True, eq=False)
class PickUpOptions(ActionOptions):
    """Per-invocation pickup behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper-close interpolation segment."""

    grasp_settle_steps: int = 0
    """Fully closed hold frames before lifting the end-effector."""

    pick_object_part: str = "center"
    """Name of the object part to pick up (used for grasp pose generation). Currently support [center | top | bottom]."""

    lift_height: float = 0.1
    """Height (m) to lift the end-effector after closing the gripper."""

    pre_grasp_distance: float = 0.15
    """Distance to offset back from the grasp pose along the approach direction."""

    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)
    """World-frame direction from the pre-grasp pose to the grasp pose."""

    approach_alignment_max_angle: float | None = None
    """Optional maximum TCP z-axis deviation from the approach direction."""

    downstream_object_target_poses: tuple[PoseGoalValue, ...] = ()
    """Future object poses that must be reachable with the selected grasp."""

    obj_upright_direction: torch.Tensor | None = None
    """Optional object local direction used to choose the upright grasp rotation."""

    rotate_upright: float | None = None
    """Optional rotation (radians) about the grasp x-axis to apply after grasp selection."""

    grasp_frame_to_eef: torch.Tensor = torch.eye(4, dtype=torch.float32)
    """Canonical grasp-frame to robot end-effector SE(3) calibration."""

    fixed_object_to_eef: torch.Tensor | None = None
    """Optional object-frame to end-effector SE(3) grasp calibration.

    When no explicit goal grasp is supplied, this transform bypasses affordance
    sampling and the sampled-grasp orientation/calibration adjustments.
    """

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if type(self.grasp_settle_steps) is not int or self.grasp_settle_steps < 0:
            raise ValueError("grasp_settle_steps must be a non-negative integer.")
        if not isinstance(self.pick_object_part, str) or not self.pick_object_part:
            raise ValueError("pick_object_part must be a non-empty string.")
        if self.lift_height < 0.0:
            raise ValueError("lift_height must be non-negative.")
        if self.pre_grasp_distance < 0.0:
            raise ValueError("pre_grasp_distance must be non-negative.")
        if self.approach_direction.shape != (3,):
            raise ValueError("approach_direction must have shape (3,).")
        if not torch.isfinite(self.approach_direction).all():
            raise ValueError("approach_direction must contain finite values.")
        if torch.linalg.vector_norm(self.approach_direction) <= 1.0e-6:
            raise ValueError("approach_direction must be non-zero.")
        if self.approach_alignment_max_angle is not None and not (
            0.0 <= self.approach_alignment_max_angle <= math.pi / 2
        ):
            raise ValueError("approach_alignment_max_angle must be in [0, pi / 2].")
        if self.obj_upright_direction is not None and (
            self.obj_upright_direction.shape != (3,)
            or not torch.isfinite(self.obj_upright_direction).all()
        ):
            raise ValueError("obj_upright_direction must be a finite (3,) tensor.")
        _validate_single_se3(self.grasp_frame_to_eef, "grasp_frame_to_eef")
        if self.fixed_object_to_eef is not None:
            _validate_single_se3(self.fixed_object_to_eef, "fixed_object_to_eef")
        object.__setattr__(self, "approach_direction", self.approach_direction.clone())
        object.__setattr__(
            self,
            "grasp_frame_to_eef",
            self.grasp_frame_to_eef.clone(),
        )
        if self.fixed_object_to_eef is not None:
            object.__setattr__(
                self,
                "fixed_object_to_eef",
                self.fixed_object_to_eef.clone(),
            )
        downstream_targets: list[PoseGoalValue] = []
        for index, value in enumerate(self.downstream_object_target_poses):
            validate_pose_goal(
                value,
                f"downstream_object_target_poses[{index}]",
                allow_waypoints=False,
            )
            downstream_targets.append(
                value.clone() if isinstance(value, torch.Tensor) else value.snapshot()
            )
        object.__setattr__(
            self,
            "downstream_object_target_poses",
            tuple(downstream_targets),
        )
        if self.obj_upright_direction is not None:
            object.__setattr__(
                self, "obj_upright_direction", self.obj_upright_direction.clone()
            )


class PickUp(AtomicAction[GraspGoal, PickUpOptions]):
    """Approach a grasp pose, close the gripper, lift."""

    skill_id: ClassVar[str] = "pick_up"
    GoalType: ClassVar[type] = GraspGoal
    OptionsType: ClassVar[type] = PickUpOptions
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset(
                    {
                        BATCH_INVERSE_KINEMATICS_CAPABILITY,
                        CARTESIAN_POSE_CAPABILITY,
                        FORWARD_KINEMATICS_CAPABILITY,
                    }
                ),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
        ),
    )

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
    ) -> tuple[str, ...]:
        """Include the semantic object when it has a stable scene identity."""
        dependencies = set(super()._scene_dependencies(request))
        entity_id = request.goal.semantics.entity_id
        if entity_id is not None:
            dependencies.add(entity_id)
        dependencies.update(
            collect_scene_dependencies(
                request.skill_options.downstream_object_target_poses
            )
        )
        return tuple(sorted(dependencies))

    def _checked_candidate_ik(
        self,
        poses: torch.Tensor,
        seed: torch.Tensor,
        manipulator: JointPositionTarget,
        context: PlanningContext,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[str | None, ...], ...]]:
        """Check solver flags, finite values, limits, and measured FK per choice."""
        if (
            seed.ndim != 3
            or seed.shape[:2] != active.shape
            or seed.shape[-1] != len(manipulator.joint_ids)
            or poses.shape != (*active.shape, 4, 4)
            or active.dtype != torch.bool
        ):
            raise ValueError("Candidate IK inputs have incompatible batch dimensions.")
        for name, value in (("seed", seed), ("poses", poses)):
            if (
                value.device != context.robot.qpos.device
                or value.dtype != context.robot.qpos.dtype
            ):
                raise ValueError(
                    f"Candidate IK {name} must match the context device/dtype."
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"Candidate IK {name} must contain finite values.")
        if active.device != seed.device:
            raise ValueError("Candidate IK active mask must share the seed device.")
        success = torch.zeros_like(active)
        reasons: list[list[str | None]] = [[None] * active.shape[1] for _ in active]
        if not active.any():
            return success, seed.clone(), tuple(tuple(row) for row in reasons)
        env_ids = context.env_ids.tolist()
        limits = self.robot.get_qpos_limits(
            joint_ids=list(manipulator.joint_ids), env_ids=env_ids
        )
        if not isinstance(limits, torch.Tensor) or limits.shape != (
            seed.shape[0],
            seed.shape[-1],
            2,
        ):
            raise ValueError("Candidate planning requires per-row joint limits.")
        if limits.device != seed.device or limits.dtype != seed.dtype:
            raise ValueError("Candidate joint limits must match the seed device/dtype.")
        if torch.isnan(limits).any() or (limits[..., 0] > limits[..., 1]).any():
            raise ValueError("Candidate joint limits must be ordered and non-NaN.")
        if ((seed < limits[:, None, :, 0]) | (seed > limits[:, None, :, 1])).any():
            raise ValueError("Candidate IK safe seed is outside joint limits.")
        safe_poses = self.robot.compute_batch_fk(
            qpos=seed, name=manipulator.control_part, env_ids=env_ids, to_matrix=True
        )
        if (
            not isinstance(safe_poses, torch.Tensor)
            or safe_poses.shape != poses.shape
            or safe_poses.device != seed.device
            or safe_poses.dtype != seed.dtype
            or not torch.isfinite(safe_poses).all()
        ):
            raise ValueError(
                "Candidate safe FK must return finite poses with the requested shape/device/dtype."
            )
        solver_poses = torch.where(active[..., None, None], poses, safe_poses)
        result = self.robot.compute_batch_ik(
            pose=solver_poses,
            name=manipulator.control_part,
            joint_seed=seed,
            env_ids=env_ids,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Candidate IK must return (success, qpos).")
        flags, raw_qpos = result
        if (
            not isinstance(flags, torch.Tensor)
            or flags.shape != active.shape
            or not isinstance(raw_qpos, torch.Tensor)
            or raw_qpos.shape != seed.shape
        ):
            raise ValueError("Candidate IK returned incompatible batch dimensions.")
        if flags.device != seed.device or raw_qpos.device != seed.device:
            raise ValueError("Candidate IK outputs must share the seed device.")
        if not raw_qpos.is_floating_point() or raw_qpos.dtype != seed.dtype:
            raise ValueError("Candidate IK qpos must use the seed floating dtype.")
        if (
            flags.is_complex()
            or not torch.isfinite(flags).all()
            or not ((flags == 0) | (flags == 1)).all()
        ):
            raise ValueError(
                "Candidate IK success flags must be bool or finite numeric 0/1 values."
            )
        flags = flags.to(dtype=torch.bool)
        finite = torch.isfinite(raw_qpos).all(dim=-1)
        success = active & flags & finite
        qpos = torch.where(success[..., None], raw_qpos, seed)
        within_limits = (
            (qpos >= limits[:, None, :, 0]) & (qpos <= limits[:, None, :, 1])
        ).all(dim=-1)
        success &= within_limits
        qpos = torch.where(success[..., None], qpos, seed)
        actual = self.robot.compute_batch_fk(
            qpos=qpos, name=manipulator.control_part, env_ids=env_ids, to_matrix=True
        )
        if not isinstance(actual, torch.Tensor) or actual.shape != poses.shape:
            raise ValueError("Candidate FK returned incompatible batch dimensions.")
        if actual.device != seed.device or actual.dtype != seed.dtype:
            raise ValueError("Candidate FK outputs must match the seed device/dtype.")
        position_error = torch.linalg.vector_norm(
            actual[..., :3, 3] - poses[..., :3, 3], dim=-1
        )
        relative_rotation = actual[..., :3, :3].transpose(-2, -1) @ poses[..., :3, :3]
        cosine = (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
        rotation_error = torch.acos(cosine.clamp(-1.0, 1.0))
        fk_valid = (
            torch.isfinite(actual).all(dim=(-2, -1))
            & (position_error <= _CANDIDATE_POSITION_TOLERANCE)
            & (rotation_error <= _CANDIDATE_ROTATION_TOLERANCE)
        )
        success &= fk_valid
        for row, column in torch.nonzero(active & ~success).tolist():
            reasons[row][column] = (
                "IK_NOT_FOUND"
                if not flags[row, column]
                else (
                    "IK_INVALID_RESULT"
                    if not finite[row, column]
                    else (
                        "JOINT_LIMIT"
                        if not within_limits[row, column]
                        else "FK_MISMATCH"
                    )
                )
            )
        return (
            success,
            torch.where(success[..., None], qpos, seed),
            tuple(tuple(row) for row in reasons),
        )

    def _candidate_pose_variants(
        self,
        grasp_poses: torch.Tensor,
        object_pose: torch.Tensor,
        options: PickUpOptions,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the shared canonical roll, upright and EEF calibration once."""
        mirrored = grasp_poses.clone()
        mirrored[..., :3, :2] *= -1
        selection_variants = torch.stack((grasp_poses, mirrored), dim=2)
        variants = self._upright_adjusted_grasp_poses(
            selection_variants, object_pose, options
        )
        calibration = options.grasp_frame_to_eef.to(
            device=self.device, dtype=variants.dtype
        )
        return selection_variants, variants @ calibration

    def _enumerate_candidates(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
        context: PlanningContext,
        *,
        grasp_candidates: GraspCandidateBatch | None,
        active_mask: torch.Tensor,
    ) -> PickUpCandidateBatch:
        """Retain every geometrically valid grasp and feasible symmetric roll."""
        from embodichain.toolkits.graspkit.candidates import GraspCandidateBatch

        goal = self.require_goal(request)
        options = request.skill_options
        if goal.grasp_xpos is not None or options.fixed_object_to_eef is not None:
            raise ValueError(
                "Candidate enumeration requires a sampled-grasp goal, without fixed/explicit grasp overrides."
            )
        manipulator = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        grasp_target = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        object_pose = _resolve_object_pose(
            goal.semantics, context, name="pickup_object_pose"
        )
        direction = options.approach_direction.to(
            device=self.device, dtype=context.robot.qpos.dtype
        )
        direction = direction / torch.linalg.vector_norm(direction)
        if grasp_candidates is None:
            affordance = goal.semantics.affordance
            if not isinstance(affordance, AntipodalAffordance):
                raise ValueError("Candidate sampling requires an AntipodalAffordance.")
            empty = (object_pose.new_empty((0, 4, 4)), object_pose.new_empty((0,)))
            ragged = [empty for _ in range(context.batch_size)]
            active_rows = torch.nonzero(active_mask).flatten()
            if active_rows.numel():
                generator = self.planning_services.grasp_pose_generator(
                    grasp_target.target_id
                )
                sampled = generator.get_valid_grasp_poses(
                    mesh_vertices=affordance.mesh_vertices,
                    mesh_triangles=affordance.mesh_triangles,
                    obj_poses=object_pose[active_rows],
                    approach_direction=direction,
                    obj_longest_axis=(
                        None
                        if options.pick_object_part == "center"
                        else direction.new_tensor((0.0, 0.0, 1.0))
                    ),
                    is_positive_part=options.pick_object_part != "bottom",
                )
                if len(sampled) != active_rows.numel():
                    raise ValueError(
                        "Grasp sampler must return one result per input row."
                    )
                for index, row in enumerate(active_rows.tolist()):
                    ragged[row] = sampled[index]
            grasp_candidates = GraspCandidateBatch.from_ragged(
                ragged, object_poses=object_pose
            )
        if not isinstance(grasp_candidates, GraspCandidateBatch):
            raise TypeError("grasp_candidates must be GraspCandidateBatch.")
        grasp_candidates = replace(grasp_candidates)
        if (
            grasp_candidates.poses.shape[0] != context.batch_size
            or grasp_candidates.poses.device != self.device
        ):
            raise ValueError(
                "Grasp candidates must be bound to the real E rows/device."
            )
        if grasp_candidates.frame != "local_arena":
            raise ValueError("Atomic grasp candidates must use the local_arena frame.")
        raw = grasp_candidates.poses.to(dtype=context.robot.qpos.dtype)
        _, variants = self._candidate_pose_variants(raw, object_pose, options)
        rows, grasps = raw.shape[:2]
        columns = grasps * 2
        grasp_poses = variants.reshape(rows, columns, 4, 4)
        pre_grasp = grasp_poses.clone()
        pre_grasp[..., :3, 3] -= direction * options.pre_grasp_distance
        lift = grasp_poses.clone()
        lift[..., 2, 3] += options.lift_height
        valid = (
            grasp_candidates.valid_mask[..., None].expand(-1, -1, 2)
            & active_mask[:, None, None]
        ).reshape(rows, columns)
        alignment = self._approach_alignment_mask(variants, options, direction).reshape(
            rows, columns
        )
        reasons: list[list[str | None]] = [[None] * columns for _ in range(rows)]
        for row in range(rows):
            for col in range(columns):
                if not active_mask[row]:
                    reasons[row][col] = "INACTIVE"
                elif not valid[row, col]:
                    reasons[row][col] = "grasp:INVALID_GEOMETRY"
                elif not alignment[row, col]:
                    reasons[row][col] = "grasp:APPROACH_ALIGNMENT"
        valid &= alignment
        start = context.robot.qpos[:, list(manipulator.joint_ids)]
        seed = start[:, None].expand(-1, columns, -1).clone()
        anchors: list[torch.Tensor] = []
        downstream: list[torch.Tensor] = []
        object_to_eef = pose_inv(object_pose)[:, None] @ grasp_poses
        targets = [("pre_grasp", pre_grasp), ("grasp", grasp_poses), ("lift", lift)]
        for index, target in enumerate(options.downstream_object_target_poses):
            pose = resolve_pose_target(
                resolve_pose_goal(target, context, name=f"downstream[{index}]"),
                num_envs=rows,
                device=self.device,
            )
            targets.append((f"downstream_{index}", pose[:, None] @ object_to_eef))
        for stage_index, (stage, poses) in enumerate(targets):
            stage_valid, seed, stage_reasons = self._checked_candidate_ik(
                poses, seed, manipulator, context, valid
            )
            for row, col in torch.nonzero(valid & ~stage_valid).tolist():
                reasons[row][col] = f"{stage}:{stage_reasons[row][col]}"
            valid &= stage_valid
            (anchors if stage_index < 3 else downstream).append(seed.clone())
        variant_ids = (
            torch.arange(2, device=self.device).repeat(grasps).expand(rows, -1)
        )
        grasp_ids = tuple(
            tuple(value for value in row for _ in range(2))
            for row in grasp_candidates.grasp_ids
        )
        candidate_ids = tuple(
            tuple(f"{value}/roll-{index % 2}" for index, value in enumerate(row))
            for row in grasp_ids
        )
        costs = (
            grasp_candidates.costs[..., None].expand(-1, -1, 2).reshape(rows, columns)
        )
        return PickUpCandidateBatch(
            skill_id=self.skill_id,
            env_ids=context.env_ids,
            valid_mask=valid,
            costs=torch.where(valid, costs, torch.inf),
            candidate_ids=candidate_ids,
            failure_stages=tuple(tuple(row) for row in reasons),
            input_fingerprint=_input_fingerprint(request, context),
            grasp_ids=grasp_ids,
            variant_ids=variant_ids,
            object_to_eef=object_to_eef,
            pre_grasp_poses=pre_grasp,
            grasp_poses=grasp_poses,
            lift_poses=lift,
            qpos_anchors=torch.stack(anchors, dim=2),
            downstream_qpos=tuple(downstream),
        )

    def _plan_candidate(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
        selection: AtomicCandidateSelection,
        context: PlanningContext,
        *,
        active_mask: torch.Tensor,
    ) -> ActionPlan:
        """Build a Cartesian pickup around the selected, certified IK anchors."""
        candidates = selection.candidates
        if not isinstance(candidates, PickUpCandidateBatch):
            raise TypeError("PickUp requires PickUpCandidateBatch selections.")
        if candidates.input_fingerprint != _input_fingerprint(request, context):
            raise ValueError(
                "Candidate evaluation is stale for this invocation/context; enumerate again at its current projected start."
            )
        if request.motion_policy.strategy != "ik_interp":
            raise NotImplementedError(
                "Selected PickUp candidates currently require ik_interp for solved-branch preservation."
            )
        options = request.skill_options
        binding = request.binding
        motion = binding.endpoint("primary", "motion")
        grasp = binding.endpoint("primary", "grasp")
        manipulator = motion.require_target(JointPositionTarget)
        hand = grasp.require_target(JointPositionTarget)
        task_key = require_shared_task_state_key(
            motion, grasp, participant="PickUp primary participant"
        )
        rows = torch.arange(context.batch_size, device=self.device)
        indices = selection.indices.clamp_min(0)
        capacity = candidates.valid_mask.shape[1]
        selected_ids = tuple(
            (
                candidates.candidate_ids[row][int(indices[row])]
                if active_mask[row] and capacity
                else None
            )
            for row in range(context.batch_size)
        )
        reasons: list[str | None] = [None] * context.batch_size
        success = active_mask.clone()
        if capacity:
            success &= candidates.valid_mask[rows, indices]
            for row in range(context.batch_size):
                if not active_mask[row]:
                    reasons[row] = "INACTIVE"
                elif not success[row]:
                    col = int(indices[row])
                    reason = candidates.failure_reasons[row][col] or "IK_NOT_FOUND"
                    stage = candidates.failure_stages[row][col] or "grasp"
                    reasons[row] = f"{stage}:{reason}"
        else:
            success[:] = False
            reasons = [
                "grasp:NO_CANDIDATES" if active_mask[row] else "INACTIVE"
                for row in range(context.batch_size)
            ]
        if not success.any():
            return self.build_plan(
                request,
                context,
                success=success,
                trajectory=TimedTrajectory.from_uniform_step(
                    context.robot.qpos[:, None],
                    env_ids=context.env_ids,
                    step_dt=context.require_control_dt(),
                ),
                diagnostics=PlannerDiagnostics(
                    backend="selected_pickup_ik",
                    metadata={
                        "candidate_ids": selected_ids,
                        "candidate_failure_reasons": tuple(reasons),
                    },
                ),
            )
        arm_ids = list(manipulator.joint_ids)
        hand_ids = list(hand.joint_ids)
        anchors = candidates.qpos_anchors[rows, indices]
        pre_pose = candidates.pre_grasp_poses[rows, indices]
        grasp_pose = candidates.grasp_poses[rows, indices]
        lift_pose = candidates.lift_poses[rows, indices]
        n_approach, n_close, n_lift = split_three_segments(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
            first_segment_name="approach",
            third_segment_name="lift",
        )
        n_transit = max(2, n_approach // 2)
        n_reach = n_approach - n_transit
        if n_reach < 2 or n_lift < 2:
            raise ValueError(
                "Selected pickup sample_count must provide two transit, approach and lift samples each."
            )
        start = context.robot.qpos[:, arm_ids]
        weights = torch.linspace(
            0.0, 1.0, n_transit, device=self.device, dtype=start.dtype
        )
        transit = torch.lerp(
            start[:, None], anchors[:, 0, None], weights[None, :, None]
        )

        def cartesian_segment(
            name: str,
            begin_pose: torch.Tensor,
            end_pose: torch.Tensor,
            begin_qpos: torch.Tensor,
            end_qpos: torch.Tensor,
            count: int,
        ) -> torch.Tensor:
            nonlocal success
            values = [begin_qpos]
            for waypoint in range(1, count):
                pose = begin_pose.clone()
                pose[:, :3, 3] = torch.lerp(
                    begin_pose[:, :3, 3], end_pose[:, :3, 3], waypoint / (count - 1)
                )
                previous = values[-1]
                if waypoint == count - 1:
                    # The terminal qpos is the selected branch's owned anchor,
                    # not a freshly selected IK solution at the same pose.
                    qpos = end_qpos.clone()
                    stage_valid = success.clone()
                    stage_reasons = tuple((None,) for _ in range(context.batch_size))
                else:
                    stage_valid, qpos_values, stage_reasons = (
                        self._checked_candidate_ik(
                            pose[:, None],
                            previous[:, None],
                            manipulator,
                            context,
                            success[:, None],
                        )
                    )
                    stage_valid = stage_valid[:, 0]
                    qpos = qpos_values[:, 0]
                continuity = (
                    torch.amax(torch.abs(qpos - previous), dim=-1)
                    <= _CANDIDATE_MAX_JOINT_STEP
                )
                for row in (
                    torch.nonzero(success & ~(stage_valid & continuity))
                    .flatten()
                    .tolist()
                ):
                    code = stage_reasons[row][0] or "BRANCH_DISCONTINUITY"
                    reasons[row] = f"{name}:{code}"
                success &= stage_valid & continuity
                values.append(torch.where(success[:, None], qpos, previous))
            return torch.stack(values, dim=1)

        reach = cartesian_segment(
            "approach", pre_pose, grasp_pose, anchors[:, 0], anchors[:, 1], n_reach
        )
        lift = cartesian_segment(
            "lift", grasp_pose, lift_pose, anchors[:, 1], anchors[:, 2], n_lift
        )
        open_qpos = grasp.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        closed_qpos = grasp.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        close_weights = torch.linspace(
            0.0, 1.0, n_close, device=self.device, dtype=start.dtype
        )
        close = torch.lerp(
            open_qpos[:, None], closed_qpos[:, None], close_weights[None, :, None]
        )
        settle = options.grasp_settle_steps
        total = n_transit + n_reach + n_close + settle + n_lift
        full = context.robot.qpos[:, None].expand(-1, total, -1).clone()
        approach_end = n_transit + n_reach
        close_end = approach_end + n_close
        lift_start = close_end + settle
        full[:, :n_transit, arm_ids] = transit
        full[:, :n_transit, hand_ids] = torch.lerp(
            context.robot.qpos[:, None, hand_ids],
            open_qpos[:, None],
            weights[None, :, None],
        )
        full[:, n_transit:approach_end, arm_ids] = reach
        full[:, n_transit:approach_end, hand_ids] = open_qpos[:, None]
        full[:, approach_end:lift_start, arm_ids] = anchors[:, 1, None]
        full[:, approach_end:close_end, hand_ids] = close
        full[:, close_end:lift_start, hand_ids] = closed_qpos[:, None]
        full[:, lift_start:, arm_ids] = lift
        full[:, lift_start:, hand_ids] = closed_qpos[:, None]
        limits = self.robot.get_qpos_limits(env_ids=context.env_ids.tolist()).to(
            self.device
        )
        if limits.shape != (context.batch_size, self.robot_dof, 2):
            raise ValueError("Selected trajectory requires full-robot joint limits.")
        within_limits = (
            (full >= limits[:, None, :, 0]) & (full <= limits[:, None, :, 1])
        ).all(dim=(1, 2))
        continuous = (
            torch.abs(full[:, 1:, arm_ids] - full[:, :-1, arm_ids])
            <= _CANDIDATE_MAX_JOINT_STEP
        ).all(dim=(1, 2))
        for row in (
            torch.nonzero(success & ~(within_limits & continuous)).flatten().tolist()
        ):
            reasons[row] = (
                "trajectory:JOINT_LIMIT"
                if not within_limits[row]
                else "trajectory:BRANCH_DISCONTINUITY"
            )
        success &= within_limits & continuous
        held = HeldObjectState(
            semantics=request.goal.semantics,
            object_to_eef=candidates.object_to_eef[rows, indices],
            grasp_xpos=grasp_pose,
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                full, env_ids=context.env_ids, step_dt=context.require_control_dt()
            ),
            expected_effects=StateDelta(
                held_object_updates={task_key: held},
                coordinated_held_object_updates={
                    key: None
                    for key in context.task.coordinated_held_objects
                    if task_key in key
                },
            ),
            segment_lengths={
                "transit": n_transit,
                "approach": n_reach,
                "close": n_close + settle,
                "lift": n_lift,
            },
            scene_dependency_monitor_until=(
                {}
                if request.goal.semantics.entity_id is None
                else {request.goal.semantics.entity_id: approach_end}
            ),
            diagnostics=PlannerDiagnostics(
                backend="selected_pickup_ik",
                metadata={
                    "candidate_ids": selected_ids,
                    "candidate_failure_reasons": tuple(reasons),
                    "selected_joint_branch_preserved": True,
                    "max_joint_step": _CANDIDATE_MAX_JOINT_STEP,
                },
            ),
        )

    def _get_full_pickup_trajectory(
        self,
        grasp_xpos: torch.Tensor,
        start_arm_qpos: torch.Tensor,
        last_qpos: torch.Tensor,
        motion_policy: MotionPolicy,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
        manipulator: JointPositionTarget,
        end_effector: JointPositionTarget,
        hand_open_qpos: torch.Tensor,
        hand_grasp_qpos: torch.Tensor,
        interpolation_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        pre_grasp_xpos = translate_pose_world(
            grasp_xpos, -approach_direction * options.pre_grasp_distance
        )

        n_approach, n_close, n_lift = split_three_segments(
            motion_policy.sample_count,
            options.hand_interp_steps,
            first_segment_name="approach",
            third_segment_name="lift",
        )

        lift_xpos = translate_pose_world(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * options.lift_height,
        )
        motion_options = motion_policy.to_motion_gen_options(
            start_qpos=start_arm_qpos,
            control_part=manipulator.control_part,
            sample_count=n_approach + n_lift,
            interpolation_dt=interpolation_dt,
        )
        if motion_policy.strategy == "motion_gen":
            motion_options.sample_count = None
        motion_result = self.motion_generator.generate(
            build_pose_plan_states(
                torch.stack([pre_grasp_xpos, grasp_xpos, lift_xpos], dim=1)
            ),
            options=motion_options,
        )
        assert isinstance(motion_result.success, torch.Tensor)
        assert motion_result.positions is not None
        approach_arm, lift_arm = split_joint_trajectory_at_pose(
            motion_result.positions,
            grasp_xpos,
            robot=self.robot,
            control_part=manipulator.control_part,
            first_sample_count=n_approach,
            second_sample_count=n_lift,
        )
        grasp_arm_qpos = approach_arm[:, -1, :]
        is_success = motion_result.success

        hand_close_path = interpolate_hand_qpos(
            hand_open_qpos, hand_grasp_qpos, n_waypoints=n_close
        )
        n_settle = options.grasp_settle_steps
        close_start = n_approach
        settle_start = close_start + n_close
        lift_start = settle_start + n_settle
        full = torch.empty(
            (
                self.num_envs,
                lift_start + n_lift,
                self.robot_dof,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = last_qpos.unsqueeze(1)
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        full[:, :n_approach, arm_joint_ids] = approach_arm
        full[:, :n_approach, hand_joint_ids] = hand_open_qpos.unsqueeze(1)
        full[:, close_start:settle_start, arm_joint_ids] = grasp_arm_qpos.unsqueeze(1)
        full[:, close_start:settle_start, hand_joint_ids] = hand_close_path
        if n_settle:
            full[:, settle_start:lift_start, arm_joint_ids] = grasp_arm_qpos.unsqueeze(
                1
            )
            full[:, settle_start:lift_start, hand_joint_ids] = (
                hand_grasp_qpos.unsqueeze(1)
            )
        full[:, lift_start:, arm_joint_ids] = lift_arm
        full[:, lift_start:, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
        return (
            is_success,
            full,
            {
                "approach": n_approach,
                "close": n_close + n_settle,
                "lift": n_lift,
            },
        )

    def _plan(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan approach, close, and lift segments without committing attachment."""
        target = self.require_goal(request)
        options = replace(
            request.skill_options,
            downstream_object_target_poses=tuple(
                resolve_pose_goal(
                    downstream_target,
                    context,
                    name=f"downstream_object_target_poses[{index}]",
                )
                for index, downstream_target in enumerate(
                    request.skill_options.downstream_object_target_poses
                )
            ),
        )
        approach_direction = options.approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        approach_direction = approach_direction / torch.linalg.vector_norm(
            approach_direction
        )
        binding = request.binding
        motion = binding.endpoint("primary", "motion")
        grasp = binding.endpoint("primary", "grasp")
        manipulator = motion.require_target(JointPositionTarget)
        end_effector = grasp.require_target(JointPositionTarget)
        task_state_key = require_shared_task_state_key(
            motion,
            grasp,
            participant="PickUp primary participant",
        )
        hand_open_qpos = grasp.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_grasp_qpos = grasp.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        state = context
        sem = target.semantics
        object_pose = _resolve_object_pose(
            sem,
            context,
            name="pickup_object_pose",
        )
        if (
            target.grasp_xpos is None
            and options.fixed_object_to_eef is None
            and not isinstance(sem.affordance, AntipodalAffordance)
        ):
            raise ValueError(
                "PickUp requires an AntipodalAffordance when neither grasp_xpos "
                "nor fixed_object_to_eef is set."
            )
        start_arm_qpos = arm_qpos_from_state(
            state,
            list(manipulator.joint_ids),
        )
        if target.grasp_xpos is None:
            if options.fixed_object_to_eef is None:
                is_success, grasp_xpos = self._resolve_grasp_pose(
                    sem,
                    object_pose,
                    start_arm_qpos,
                    manipulator,
                    end_effector.target_id,
                    options,
                    approach_direction,
                )
            else:
                object_to_eef = options.fixed_object_to_eef.to(
                    device=self.device,
                    dtype=object_pose.dtype,
                )
                grasp_xpos = torch.matmul(object_pose, object_to_eef)
                is_success = torch.ones(
                    self.num_envs,
                    dtype=torch.bool,
                    device=self.device,
                )
        else:
            grasp_xpos = resolve_pose_target(
                resolve_pose_goal(target.grasp_xpos, context, name="grasp_xpos"),
                num_envs=self.num_envs,
                device=self.device,
            )
            if options.rotate_upright is not None:
                grasp_xpos = self._upright_adjusted_grasp_poses(
                    grasp_xpos,
                    object_pose,
                    options,
                )
            is_success = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        grasp_success = normalize_success_mask(
            is_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Grasp-pose success",
        )
        if not grasp_success.any():
            logger.log_warning("PickUp failed to resolve a grasp pose.")
            return self.failed_plan(
                request, context, message="Failed to resolve a grasp pose."
            )

        trajectory_success, full, segment_lengths = self._get_full_pickup_trajectory(
            grasp_xpos,
            start_arm_qpos,
            state.last_qpos,
            request.motion_policy,
            options,
            approach_direction,
            manipulator,
            end_effector,
            hand_open_qpos,
            hand_grasp_qpos,
            context.require_control_dt(),
        )
        success_mask = grasp_success & normalize_success_mask(
            trajectory_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Pick-up trajectory success",
        )

        object_to_eef = torch.bmm(pose_inv(object_pose), grasp_xpos)
        held = HeldObjectState(
            semantics=sem, object_to_eef=object_to_eef, grasp_xpos=grasp_xpos
        )
        coordinated_updates = {
            key: None for key in state.coordinated_held_objects if task_state_key in key
        }
        return self.build_plan(
            request,
            context,
            success=success_mask,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(
                held_object_updates={task_state_key: held},
                coordinated_held_object_updates=coordinated_updates,
            ),
            segment_lengths=segment_lengths,
            # Once the approach is dispatched the object can move because of
            # contact or grasping. That self-induced motion must not look like
            # an external dynamic-goal update.
            scene_dependency_monitor_until=(
                {}
                if sem.entity_id is None
                else {sem.entity_id: segment_lengths["approach"]}
            ),
        )

    def _resolve_grasp_pose(
        self,
        semantics: ObjectSemantics,
        object_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        manipulator: JointPositionTarget,
        grasp_target_id: str,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        affordance = semantics.affordance
        if not isinstance(affordance, AntipodalAffordance):
            raise ValueError("PickUp grasp sampling requires AntipodalAffordance.")
        generator = self.planning_services.grasp_pose_generator(grasp_target_id)
        obj_longest_axis = None
        is_positive_part = True
        if options.pick_object_part != "center":
            obj_longest_axis = torch.tensor(
                [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
            )
            is_positive_part = options.pick_object_part == "top"
        grasp_poses_result = generator.get_valid_grasp_poses(
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=object_pose,
            approach_direction=approach_direction,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=is_positive_part,
        )
        num_envs = object_pose.shape[0]
        n_max_pose = max(r[0].shape[0] for r in grasp_poses_result)
        safe_pose = self.robot.compute_fk(
            qpos=start_qpos,
            name=manipulator.control_part,
            to_matrix=True,
        )
        if n_max_pose == 0:
            return (
                torch.zeros(num_envs, dtype=torch.bool, device=self.device),
                safe_pose,
            )
        grasp_xpos_padding = safe_pose[:, None].expand(-1, n_max_pose, -1, -1).clone()
        candidate_mask = torch.zeros(
            (num_envs, n_max_pose), dtype=torch.bool, device=self.device
        )
        grasp_cost_padding = torch.full(
            (num_envs, n_max_pose),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(num_envs):
            n_pose = grasp_poses_result[i][0].shape[0]
            if n_pose == 0:
                continue
            grasp_poses = grasp_poses_result[i][0].to(
                device=self.device, dtype=grasp_xpos_padding.dtype
            )
            grasp_costs = grasp_poses_result[i][1].to(
                device=self.device, dtype=torch.float32
            )
            valid = torch.isfinite(grasp_costs) & torch.isfinite(grasp_poses).all(
                dim=(-2, -1)
            )
            # Validate finite transforms before any rotation math or batch IK.
            # Invalid candidates remain masked even if the safe placeholder is
            # itself reachable by the robot.
            checked_poses = torch.where(
                valid[:, None, None], grasp_poses, safe_pose[i]
            ).to(dtype=torch.float64)
            rotation = checked_poses[:, :3, :3]
            valid &= torch.isclose(
                checked_poses[:, 3],
                checked_poses.new_tensor((0.0, 0.0, 0.0, 1.0)),
                atol=1.0e-6,
                rtol=0.0,
            ).all(dim=-1)
            valid &= torch.isclose(
                rotation.transpose(-2, -1) @ rotation,
                torch.eye(3, dtype=rotation.dtype, device=rotation.device),
                atol=1.0e-6,
                rtol=0.0,
            ).all(dim=(-2, -1))
            valid &= torch.isclose(
                torch.linalg.det(rotation),
                rotation.new_tensor(1.0),
                atol=1.0e-6,
                rtol=0.0,
            )
            grasp_xpos_padding[i, :n_pose] = torch.where(
                valid[:, None, None], grasp_poses, safe_pose[i]
            )
            grasp_cost_padding[i, :n_pose] = torch.where(valid, grasp_costs, torch.inf)
            candidate_mask[i, :n_pose] = valid
        grasp_xpos_padding, ik_success = self._select_feasible_grasp_variants(
            grasp_xpos_padding,
            start_qpos,
            object_pose,
            manipulator,
            options,
            approach_direction,
        )
        grasp_cost_masked = torch.where(
            candidate_mask & ik_success, grasp_cost_padding, torch.inf
        )
        best_cost, best_idx = grasp_cost_masked.min(dim=1)
        is_success = torch.isfinite(best_cost)
        best_grasp_xpos = grasp_xpos_padding[
            torch.arange(num_envs, device=self.device), best_idx
        ]
        return is_success, torch.where(
            is_success[:, None, None], best_grasp_xpos, safe_pose
        )

    def _select_feasible_grasp_variants(
        self,
        grasp_xpos: torch.Tensor,
        start_qpos: torch.Tensor,
        object_poses: torch.Tensor,
        manipulator: JointPositionTarget,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose a TCP-roll variant with a feasible pickup and transport path."""
        num_envs, n_pose = grasp_xpos.shape[:2]
        selection_variants, grasp_variants = self._candidate_pose_variants(
            grasp_xpos, object_poses, options
        )
        grasp_frame_to_eef = options.grasp_frame_to_eef.to(
            device=self.device,
            dtype=grasp_variants.dtype,
        )
        pre_grasp_variants = grasp_variants.clone()
        pre_grasp_variants[..., :3, 3] -= (
            approach_direction * options.pre_grasp_distance
        )
        lift_variants = grasp_variants.clone()
        lift_variants[..., :3, 3] += torch.tensor(
            [0.0, 0.0, options.lift_height],
            dtype=grasp_variants.dtype,
            device=self.device,
        )

        pre_grasp_success, pre_grasp_qpos = self._compute_batch_candidate_ik(
            pre_grasp_variants, start_qpos, manipulator
        )
        grasp_success, grasp_qpos = self._compute_batch_candidate_ik(
            grasp_variants, pre_grasp_qpos, manipulator
        )
        lift_success, lift_qpos = self._compute_batch_candidate_ik(
            lift_variants, grasp_qpos, manipulator
        )
        alignment_success = self._approach_alignment_mask(
            grasp_variants, options, approach_direction
        )
        pickup_success = (
            alignment_success & pre_grasp_success & grasp_success & lift_success
        )
        downstream_success_counts: list[list[int]] = []
        object_to_eef_variants = torch.matmul(
            pose_inv(object_poses)[:, None, None], grasp_variants
        )
        # MoveHeldObject begins after the lift, so screen its target from the
        # same joint state that the execution stream will use.
        downstream_seed = lift_qpos
        for object_target_pose in options.downstream_object_target_poses:
            object_target_pose = object_target_pose.to(
                device=self.device, dtype=torch.float32
            )
            if object_target_pose.shape == (4, 4):
                object_target_pose = object_target_pose.unsqueeze(0).repeat(
                    num_envs, 1, 1
                )
            if object_target_pose.shape != (num_envs, 4, 4):
                raise ValueError(
                    "downstream_object_target_poses entries must have shape "
                    f"(4, 4) or ({num_envs}, 4, 4), but got "
                    f"{object_target_pose.shape}."
                )
            downstream_eef_variants = torch.matmul(
                object_target_pose[:, None, None], object_to_eef_variants
            )
            downstream_success, downstream_seed = self._compute_batch_candidate_ik(
                downstream_eef_variants, downstream_seed, manipulator
            )
            pickup_success &= downstream_success
            downstream_success_counts.append(pickup_success.sum(dim=(1, 2)).tolist())
        if not pickup_success.any(dim=(1, 2)).all():
            logger.log_warning(
                "PickUp found no candidate with a feasible vertical pickup path: "
                f"aligned={alignment_success.sum(dim=(1, 2)).tolist()}, "
                f"pre_grasp={pre_grasp_success.sum(dim=(1, 2)).tolist()}, "
                f"grasp={(pre_grasp_success & grasp_success).sum(dim=(1, 2)).tolist()}, "
                f"lift={(pre_grasp_success & grasp_success & lift_success).sum(dim=(1, 2)).tolist()}, "
                f"downstream={downstream_success_counts}."
            )

        start_xpos = self.robot.compute_fk(
            qpos=start_qpos,
            name=manipulator.control_part,
            to_matrix=True,
        )
        start_quat = quat_from_matrix(start_xpos[:, :3, :3])
        # Preserve the established preference between symmetric roll variants;
        # use the upright-adjusted pose only for feasibility and execution.
        selection_eef_variants = torch.matmul(
            selection_variants,
            grasp_frame_to_eef,
        )
        variant_quat = quat_from_matrix(selection_eef_variants[..., :3, :3])
        start_quat = start_quat[:, None, None, :].expand_as(variant_quat)
        rotation_error = quat_error_magnitude(
            variant_quat.reshape(-1, 4),
            start_quat.reshape(-1, 4),
        ).reshape(num_envs, n_pose, 2)
        feasible_rotation_error = torch.where(
            pickup_success,
            rotation_error,
            torch.full_like(rotation_error, torch.inf),
        )
        best_variant_idx = feasible_rotation_error.argmin(dim=2)

        env_idx = torch.arange(num_envs, device=self.device)[:, None]
        pose_idx = torch.arange(n_pose, device=self.device)[None, :]
        selected_grasp_xpos = grasp_variants[env_idx, pose_idx, best_variant_idx]
        ik_success = pickup_success[env_idx, pose_idx, best_variant_idx]
        return selected_grasp_xpos, ik_success

    def _approach_alignment_mask(
        self,
        grasp_poses: torch.Tensor,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidates whose final TCP z-axis follows the approach direction."""
        max_angle = options.approach_alignment_max_angle
        if options.rotate_upright is not None or max_angle is None:
            return torch.ones(
                grasp_poses.shape[:3], dtype=torch.bool, device=grasp_poses.device
            )
        grasp_z = torch.nn.functional.normalize(grasp_poses[..., :3, 2], dim=-1)
        alignment = torch.sum(grasp_z * approach_direction, dim=-1)
        return alignment >= math.cos(float(max_angle))

    def _compute_batch_candidate_ik(
        self,
        poses: torch.Tensor,
        joint_seed: torch.Tensor,
        manipulator: JointPositionTarget,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve candidate IK poses while preserving the candidate dimensions."""
        num_envs, n_pose, n_variant = poses.shape[:3]
        flat_poses = poses.reshape(num_envs, n_pose * n_variant, 4, 4)
        if joint_seed.dim() == 2:
            joint_seed = joint_seed[:, None, None, :].expand(-1, n_pose, n_variant, -1)
        manipulator_dof = len(manipulator.joint_ids)
        flat_seed = joint_seed.reshape(num_envs, n_pose * n_variant, manipulator_dof)
        is_success, qpos = self.robot.compute_batch_ik(
            pose=flat_poses,
            name=manipulator.control_part,
            joint_seed=flat_seed,
        )
        success = is_success.to(device=self.device, dtype=torch.bool)
        success = success.reshape(num_envs, n_pose * n_variant)
        success &= torch.isfinite(qpos).all(dim=-1)
        # A failed solver may return NaNs or an arbitrary finite configuration.
        # Keep its last valid seed so it cannot contaminate the next stage's IK.
        qpos = torch.where(success[..., None], qpos, flat_seed)
        return (
            success.reshape(
                num_envs,
                n_pose,
                n_variant,
            ),
            qpos.reshape(num_envs, n_pose, n_variant, manipulator_dof),
        )

    def _upright_adjusted_grasp_poses(
        self,
        grasp_xpos: torch.Tensor,
        object_pose: torch.Tensor,
        options: PickUpOptions,
    ) -> torch.Tensor:
        """Return grasp poses after the optional upright-in-place roll adjustment."""
        if options.rotate_upright is None:
            return grasp_xpos

        if options.obj_upright_direction is None:
            upright_direction = torch.tensor(
                [0, 0, 1], dtype=torch.float32, device=self.device
            )
        else:
            upright_direction = options.obj_upright_direction.to(
                device=self.device, dtype=torch.float32
            )
        obj_upright = torch.matmul(object_pose[:, :3, :3], upright_direction)
        adjusted_grasp_xpos = grasp_xpos.clone()
        grasp_ry = adjusted_grasp_xpos[..., :3, 1]
        object_axes = obj_upright.reshape(
            obj_upright.shape[0], *([1] * (grasp_ry.ndim - 2)), 3
        )
        dot_result = (grasp_ry * object_axes).sum(dim=-1)
        revert_flag = torch.where(dot_result < 0, -1.0, 1.0)
        grasp_rx = adjusted_grasp_xpos[..., :3, 0]
        rota_axis_angle = options.rotate_upright * revert_flag[..., None] * grasp_rx
        rota_offset = axis_angle_to_rotation_matrix(
            rota_axis_angle.reshape(-1, 3)
        ).reshape(*rota_axis_angle.shape[:-1], 3, 3)
        adjusted_grasp_xpos[..., :3, :3] = torch.matmul(
            rota_offset, adjusted_grasp_xpos[..., :3, :3]
        )
        return adjusted_grasp_xpos


__all__ = ["GraspGoal", "PickUp", "PickUpCandidateBatch", "PickUpOptions"]
