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

"""Host-owned sampled collision validation for tabletop Real2Sim plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.data_pipeline.real2sim import SkillMotionPrototype
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.motion.expansion.real2sim_collision import (
    BoxCollisionBody,
    CollisionPhase,
    PlaneCollisionBody,
    SampledCollisionReport,
    SampledRobotCollisionChecker,
)
from embodichain.lab.sim.motion.expansion.operators import validate_motion_samples

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import (
        ActionPlan,
        AtomicActionEngine,
        PlanTransform,
        PlanningContext,
        ResolvedActionRequest,
    )
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.task_program.runtime.expansion import TaskProgramPlanRequest

__all__ = ["TabletopPrototypeCollisionValidator", "TabletopCollisionGuardFactory"]


def _support_errors(
    object_pose: np.ndarray,
    target_pose: np.ndarray,
    object_body: BoxCollisionBody,
    target_body: BoxCollisionBody,
) -> tuple[float, float]:
    """Measure bottom-to-support height and footprint overhang in metres."""
    object_shape = np.asarray(object_pose, dtype=np.float64) @ object_body.local_pose
    target_shape = np.asarray(target_pose, dtype=np.float64) @ target_body.local_pose
    if (
        object_shape.shape != (4, 4)
        or target_shape.shape != (4, 4)
        or not np.isfinite(object_shape).all()
        or not np.isfinite(target_shape).all()
        or not np.allclose(target_shape[:3, 2], (0, 0, 1), atol=1e-5, rtol=0)
    ):
        raise ValueError("support measurement requires finite horizontal box poses")
    relative = np.linalg.inv(target_shape) @ object_shape
    extent = np.abs(relative[:3, :3]) @ np.asarray(object_body.half_extents)
    height = abs(float(relative[2, 3] - extent[2] - target_body.half_extents[2]))
    overhang = float(
        np.maximum(
            np.abs(relative[:2, 3]) + extent[:2] - target_body.half_extents[:2],
            0.0,
        ).max()
    )
    return height, overhang


class TabletopPrototypeCollisionValidator:
    """Check one source-backed Pick, transport or Place plan at every sample.

    The collision model and prototype identities are validated when the host
    constructs the validator. Pick assumes the object remains at its observed
    pose until lift; later samples use the grasp relation at lift entry.
    Transport uses the verified held relation. Place follows that relation
    through release and places the object on a horizontal support for retract.
    These are geometric predictions, not a dynamics or continuous-collision
    guarantee. Hosts must still verify actual task outcomes and object states.
    """

    def __init__(
        self,
        *,
        robot: Robot,
        checker: SampledRobotCollisionChecker,
        prototypes: Sequence[SkillMotionPrototype],
        object_uid: str,
        target_uid: str,
        ground_uid: str,
        finger_links: Sequence[str],
        finger_joints: Sequence[str],
        closing_directions: Sequence[int],
        base_link: str,
        contact_depth_m: float,
        base_ground_depth_m: float,
    ) -> None:
        if not isinstance(checker, SampledRobotCollisionChecker):
            raise TypeError("checker must be a SampledRobotCollisionChecker")
        if not callable(getattr(robot, "compute_batch_fk", None)):
            raise TypeError("robot must provide batch FK")
        sources = tuple(prototypes)
        if not sources or any(
            not isinstance(item, SkillMotionPrototype) for item in sources
        ):
            raise ValueError("prototypes must contain prepared skill motions")
        units = {
            (
                item.unit.source.source_id,
                item.unit.source.source_revision,
                item.unit.source.unit_id,
            ): item
            for item in sources
        }
        if len(units) != len(sources):
            raise ValueError("prototype source, revision and unit must be unique")
        bodies = {body.uid: body for body in checker.bodies}
        if set(bodies) != {object_uid, target_uid, ground_uid}:
            raise ValueError(
                "tabletop validation must cover exactly the declared bodies"
            )
        if (
            not isinstance(bodies[object_uid], BoxCollisionBody)
            or bodies[object_uid].is_static
        ):
            raise ValueError("manipulated object must be a dynamic physical box")
        if (
            not isinstance(bodies[target_uid], BoxCollisionBody)
            or not bodies[target_uid].is_static
        ):
            raise ValueError("target must be a static physical box")
        if not isinstance(bodies[ground_uid], PlaneCollisionBody):
            raise ValueError("ground must be a declared fixed plane")
        fingers = tuple(finger_links)
        if (
            not fingers
            or len(set(fingers)) != len(fingers)
            or any(not isinstance(link, str) or not link for link in fingers)
        ):
            raise ValueError("finger_links must contain unique URDF link names")
        joints = tuple(finger_joints)
        directions = tuple(closing_directions)
        robot_names = tuple(getattr(robot, "joint_names", ()))
        if (
            len(joints) != len(fingers)
            or len(set(joints)) != len(joints)
            or any(name not in robot_names for name in joints)
            or len(directions) != len(joints)
            or any(
                type(direction) is not int or direction not in (-1, 1)
                for direction in directions
            )
        ):
            raise ValueError(
                "every finger link needs an explicit joint and closing direction"
            )
        for link, joint in zip(fingers, joints):
            checker.validate_prismatic_link(link, joint)
        if not isinstance(base_link, str) or not base_link:
            raise ValueError("base_link must be a URDF link name")
        for value in (contact_depth_m, base_ground_depth_m):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError("contact depth limits must be finite and nonnegative")
        for prototype in sources:
            checker.validate_source_geometry(
                prototype.unit.source,
                object_uid=object_uid,
                target_uid=(
                    target_uid
                    if prototype.unit.source.target_pose is not None
                    else None
                ),
            )
        self._robot = robot
        self._checker = checker
        self._sources = units
        self._object = bodies[object_uid]
        self._target = bodies[target_uid]
        self._object_uid, self._target_uid, self._ground_uid = (
            object_uid,
            target_uid,
            ground_uid,
        )
        self._fingers, self._base_link = fingers, base_link
        self._finger_indices = tuple(robot_names.index(name) for name in joints)
        self._closing_directions = directions
        self._contact_depth = float(contact_depth_m)
        self._base_ground_depth = float(base_ground_depth_m)
        self._reports: list[tuple[str, SampledCollisionReport]] = []
        self._last_release_target: np.ndarray | None = None

    @property
    def reports(self) -> tuple[tuple[str, SampledCollisionReport], ...]:
        """Return nominal and variant validation results in execution order."""
        return tuple(self._reports)

    def phase_policy(
        self,
        skill_id: str,
        phase_name: str,
        start: int,
        stop: int,
        *,
        object_supported_by_target: bool = False,
    ) -> CollisionPhase:
        """Describe permitted contacts for a declared skill interval.

        A subsequent Pick can start on the target support. Its object/support
        contact is allowed only when the caller has verified that initial
        support relation, and remains subject to the configured depth bound.

        Args:
            skill_id: Atomic skill owning the phase.
            phase_name: Declared skill phase name.
            start: First included sample.
            stop: First excluded sample.
            object_supported_by_target: Whether the current object footprint
                has been verified on the target support.

        Returns:
            Phase-specific allowed contact pairs and penetration limits.
        """
        if type(object_supported_by_target) is not bool:
            raise TypeError("object_supported_by_target must be a boolean")
        supported = {
            "pick_up": {"approach", "close", "lift"},
            "move_held_object": {"transport"},
            "place": {"approach", "release", "retract"},
        }
        if skill_id not in supported or phase_name not in supported[skill_id]:
            raise ValueError("unknown tabletop collision phase")
        allowed = {
            (
                "robot:" + self._base_link,
                "body:" + self._ground_uid,
            ): self._base_ground_depth,
        }
        if skill_id == "pick_up":
            allowed[("body:" + self._object_uid, "body:" + self._ground_uid)] = (
                self._contact_depth
            )
            if object_supported_by_target:
                allowed[("body:" + self._object_uid, "body:" + self._target_uid)] = (
                    self._contact_depth
                )
        if skill_id != "pick_up" or phase_name in {"close", "lift"}:
            for finger in self._fingers:
                allowed[("robot:" + finger, "body:" + self._object_uid)] = (
                    self._contact_depth
                )
        if skill_id == "place" and phase_name in {"release", "retract"}:
            allowed[("body:" + self._object_uid, "body:" + self._target_uid)] = (
                self._contact_depth
            )
        return CollisionPhase(phase_name, start, stop, allowed)

    def validate_observed(
        self,
        joint_positions: object,
        *,
        object_poses: object,
        target_pose: object,
        phases: Sequence[CollisionPhase],
    ) -> SampledCollisionReport:
        """Check measured robot and object poses using explicit phase ranges.

        The target is fixed for the episode; the object must supply a separate
        measured world pose for every robot sample. No grip projection or
        inferred object motion is applied to these observations.
        """
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.detach().cpu().numpy()
        joints = np.asarray(joint_positions)
        if joints.ndim != 2 or len(joints) == 0:
            raise ValueError("observed joints must have shape (T, joints)")
        if isinstance(object_poses, torch.Tensor):
            object_poses = object_poses.detach().cpu().numpy()
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError("fixed target must have one 4x4 pose")
        return self._checker.validate(
            joints,
            body_poses={
                self._object_uid: object_poses,
                self._target_uid: np.broadcast_to(target, (len(joints), 4, 4)),
                self._ground_uid: np.broadcast_to(np.eye(4), (len(joints), 4, 4)),
            },
            phases=phases,
        )

    @staticmethod
    def _scene_pose(context: PlanningContext, uid: str) -> np.ndarray:
        if uid not in context.scene.entities:
            raise ValueError(f"collision scene is missing {uid!r}")
        pose = context.scene.entities[uid].pose.detach().cpu().numpy()
        if pose.shape == (1, 4, 4):
            pose = pose[0]
        if pose.shape != (4, 4):
            raise ValueError("collision scene requires one rigid pose per entity")
        return np.array(pose, dtype=np.float64, copy=True)

    def object_supported_by_target(self, context: PlanningContext) -> bool:
        """Check the current object's footprint and height on the fixed support.

        Args:
            context: Current B=1 planning observation.

        Returns:
            Whether the object is supported within the declared contact depth.
        """
        if context.batch_size != 1:
            raise ValueError("support verification requires B=1")
        height, overhang = _support_errors(
            self._scene_pose(context, self._object_uid),
            self._scene_pose(context, self._target_uid),
            self._object,
            self._target,
        )
        return height <= self._contact_depth and overhang <= self._contact_depth

    def validate(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        plan: ActionPlan,
    ) -> SampledCollisionReport:
        """Check source identity, contact permissions and sampled geometry."""
        if context.batch_size != 1 or plan.joint_trajectory is None:
            raise ValueError("tabletop collision validation requires a B=1 joint plan")
        if request.skill_id != plan.skill_id or plan.skill_id not in {
            "pick_up",
            "move_held_object",
            "place",
        }:
            raise ValueError("unsupported tabletop collision skill")
        self._last_release_target = None
        source = plan.diagnostics.metadata
        prototype = self._sources.get(
            (
                source.get("source_id"),
                source.get("source_revision"),
                source.get("unit_id"),
            )
        )
        if prototype is None or prototype.unit.source.skill_id != plan.skill_id:
            raise ValueError("plan does not match a geometry-qualified source")
        expected = {
            "pick_up": ("approach", "close", "lift"),
            "move_held_object": ("transport",),
            "place": ("approach", "release", "retract"),
        }[plan.skill_id]
        if tuple(segment.name for segment in plan.segments) != expected:
            raise ValueError("tabletop collision plan has unsupported phases")
        trajectory = plan.joint_trajectory
        positions = trajectory.positions
        endpoint = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        tcp = self._robot.compute_batch_fk(
            qpos=positions[:, :, list(endpoint.joint_ids)],
            name=endpoint.control_part,
            to_matrix=True,
        )
        if not isinstance(tcp, torch.Tensor) or tcp.shape != (
            1,
            trajectory.waypoint_count,
            4,
            4,
        ):
            raise ValueError("robot FK must provide one TCP pose per plan sample")
        tcp_values = tcp[0].detach().cpu().numpy().astype(np.float64)
        count = trajectory.waypoint_count
        object_pose = self._scene_pose(context, self._object_uid)
        target_pose = self._scene_pose(context, self._target_uid)
        shape_target = target_pose @ self._target.local_pose
        if not np.allclose(shape_target[:3, 2], (0, 0, 1), atol=1e-5, rtol=0):
            raise ValueError("placement support must be horizontal")
        object_poses = np.repeat(object_pose[None], count, axis=0)
        initial_height, initial_overhang = _support_errors(
            object_pose, target_pose, self._object, self._target
        )
        initially_supported = (
            initial_height <= self._contact_depth
            and initial_overhang <= self._contact_depth
        )
        if plan.skill_id == "pick_up":
            start = plan.segments[-1].start
            object_to_tcp = np.linalg.inv(object_pose) @ tcp_values[start]
            object_poses[start:] = tcp_values[start:] @ np.linalg.inv(object_to_tcp)
        else:
            task_key = request.binding.endpoint("primary", "motion").task_state_key
            held = context.get_held_object(task_key)
            if held is None:
                raise ValueError(
                    "held-object collision validation needs a verified grasp relation"
                )
            relation = held.object_to_eef.detach().cpu().numpy()
            if relation.shape == (1, 4, 4):
                relation = relation[0]
            if relation.shape != (4, 4):
                raise ValueError("held relation must contain one transform")
            object_poses = tcp_values @ np.linalg.inv(relation)
            if plan.skill_id == "place":
                self._last_release_target = np.array(
                    object_poses[plan.segments[1].start], copy=True
                )
                retract = plan.segments[-1].start
                released = object_poses[retract - 1].copy()
                object_shape = released @ self._object.local_pose
                object_half_height = float(
                    np.abs(object_shape[2, :3]) @ np.asarray(self._object.half_extents)
                )
                support_top = shape_target[2, 3] + self._target.half_extents[2]
                released[2, 3] += support_top - (
                    object_shape[2, 3] - object_half_height
                )
                object_poses[retract:] = released
        # Grasp targets are actuator commands. Once a finger reaches the
        # object, the physical finger can stop before that target. Check a
        # contact-limited geometric path, and require measured effect and
        # collision evidence separately after execution.
        collision_positions = positions[0].detach().cpu().numpy().astype(np.float64)
        collision_positions = np.array(collision_positions, copy=True)
        if plan.skill_id == "pick_up":
            close = plan.segments[1]
            start = collision_positions[close.start].copy()
            end = collision_positions[close.stop - 1].copy()
            non_fingers = [
                index
                for index in range(collision_positions.shape[1])
                if index not in self._finger_indices
            ]
            if not np.allclose(
                collision_positions[close.start : close.stop, non_fingers],
                start[non_fingers],
                atol=1e-5,
                rtol=0,
            ):
                raise ValueError(
                    "gripper contact projection requires a stationary close phase"
                )
            for link, index, direction in zip(
                self._fingers, self._finger_indices, self._closing_directions
            ):
                if direction * (end[index] - start[index]) <= 0:
                    raise ValueError(
                        "gripper command does not follow its declared closing direction"
                    )
                sweep_end = start.copy()
                sweep_end[index] = end[index]
                fraction = self._checker.first_link_body_contact(
                    start,
                    sweep_end,
                    link=link,
                    body_uid=self._object_uid,
                    body_pose=object_pose,
                )
                contact = start[index] + fraction * (end[index] - start[index])
                samples = collision_positions[close.start :, index]
                collision_positions[close.start :, index] = (
                    np.minimum(samples, contact)
                    if direction == 1
                    else np.maximum(samples, contact)
                )
        else:
            current = context.robot.qpos[0].detach().cpu().numpy()
            for index, direction in zip(self._finger_indices, self._closing_directions):
                samples = collision_positions[:, index]
                collision_positions[:, index] = (
                    np.minimum(samples, current[index])
                    if direction == 1
                    else np.maximum(samples, current[index])
                )
        scene_poses = {
            self._object_uid: object_poses,
            self._target_uid: np.broadcast_to(target_pose, (count, 4, 4)),
            self._ground_uid: np.broadcast_to(np.eye(4), (count, 4, 4)),
        }
        phases = [
            self.phase_policy(
                plan.skill_id,
                segment.name,
                segment.start,
                segment.stop,
                object_supported_by_target=initially_supported,
            )
            for segment in plan.segments
        ]
        result = self._checker.validate(
            collision_positions, body_poses=scene_poses, phases=phases
        )
        self._reports.append((str(source["unit_id"]), result))
        return result

    def __call__(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        plan: ActionPlan,
    ) -> bool:
        """Implement the Task Program plan-transform collision callback."""
        return self.validate(request, context, plan).accepted


class TabletopCollisionGuardFactory:
    """Validate both the source plan and any episode-owned plan transform.

    Args:
        validator: Phase-aware sampled tabletop collision validator.
        delegate: Optional call-scoped expansion transform factory.
        velocity_limits: Optional full-joint speed limits.
        acceleration_limits: Optional full-joint acceleration limits.
        ik_smoothing_trigger_fraction: Optional nominal IK headroom threshold;
            only free phases authorized for task-space variation are adjusted.
    """

    def __init__(
        self,
        validator: TabletopPrototypeCollisionValidator,
        delegate: object | None = None,
        *,
        velocity_limits: Sequence[float] | None = None,
        acceleration_limits: Sequence[float] | None = None,
        ik_smoothing_trigger_fraction: float | None = None,
    ) -> None:
        if not isinstance(validator, TabletopPrototypeCollisionValidator):
            raise TypeError("validator must be a tabletop collision validator")
        if delegate is not None and not callable(
            getattr(delegate, "create_plan_transform", None)
        ):
            raise TypeError("delegate must create call-scoped plan transforms")
        self.validator = validator
        self.delegate = delegate
        if ik_smoothing_trigger_fraction is not None and (
            isinstance(ik_smoothing_trigger_fraction, bool)
            or not isinstance(ik_smoothing_trigger_fraction, (int, float))
            or not np.isfinite(ik_smoothing_trigger_fraction)
            or not 0 < ik_smoothing_trigger_fraction < 1
        ):
            raise ValueError(
                "ik_smoothing_trigger_fraction must be between zero and one"
            )
        self.ik_smoothing_trigger_fraction = ik_smoothing_trigger_fraction
        if (velocity_limits is None) != (acceleration_limits is None):
            raise ValueError(
                "velocity and acceleration limits must be supplied together"
            )
        self._velocity_limits = (
            None
            if velocity_limits is None
            else torch.as_tensor(tuple(velocity_limits), dtype=torch.float64)
        )
        self._acceleration_limits = (
            None
            if acceleration_limits is None
            else torch.as_tensor(tuple(acceleration_limits), dtype=torch.float64)
        )
        self._segments: dict[int, tuple[tuple[str, int, int], ...]] = {}
        self._motion_targets: dict[int, tuple[str, tuple[int, ...]]] = {}
        self._motion_metrics: dict[int, dict[str, float]] = {}
        self._phase_samples: dict[int, tuple[tuple[str, int, int], ...]] = {}
        self._release_targets: dict[int, np.ndarray] = {}

    @property
    def segments(self) -> Mapping[int, tuple[tuple[str, int, int], ...]]:
        """Return the final validated plan's phase boundaries by occurrence."""
        from types import MappingProxyType

        return MappingProxyType(dict(self._segments))

    @property
    def motion_targets(self) -> Mapping[int, tuple[str, tuple[int, ...]]]:
        """Return the validated motion endpoint for each call occurrence."""
        from types import MappingProxyType

        return MappingProxyType(dict(self._motion_targets))

    @property
    def release_targets(self) -> Mapping[int, np.ndarray]:
        """Return owned world-frame object release targets by Place occurrence."""
        from types import MappingProxyType

        return MappingProxyType(
            {index: value.copy() for index, value in self._release_targets.items()}
        )

    @property
    def motion_metrics(self) -> Mapping[int, Mapping[str, float]]:
        """Return checked finite-difference ratios for final call plans."""
        from types import MappingProxyType

        return MappingProxyType(
            {
                index: MappingProxyType(dict(values))
                for index, values in self._motion_metrics.items()
            }
        )

    @property
    def phase_samples(self) -> Mapping[int, tuple[tuple[str, int, int], ...]]:
        """Return nominal and final sample counts for each semantic phase."""
        from types import MappingProxyType

        return MappingProxyType(dict(self._phase_samples))

    def create_plan_transform(
        self, request: TaskProgramPlanRequest, *, engine: AtomicActionEngine
    ) -> PlanTransform:
        """Build one call's validator and optional expansion transform."""
        downstream = (
            None
            if self.delegate is None
            else self.delegate.create_plan_transform(request, engine=engine)
        )

        def checked(resolved, context, plan):
            motion_metrics = None
            if (
                self._velocity_limits is not None
                and self._acceleration_limits is not None
            ):
                trajectory = plan.joint_trajectory
                if trajectory is None or trajectory.batch_size != 1:
                    raise ValueError(
                        "motion limit validation requires a B=1 joint trajectory"
                    )
                velocity, acceleration = (
                    self._velocity_limits.clone(),
                    self._acceleration_limits.clone(),
                )
                if velocity.shape != (trajectory.robot_dof,) or acceleration.shape != (
                    trajectory.robot_dof,
                ):
                    raise ValueError("motion limits must cover the full robot layout")
                endpoint = resolved.binding.endpoint(
                    "primary", "motion"
                ).require_target(JointPositionTarget)
                ids = list(endpoint.joint_ids)
                if resolved.motion_policy.velocity_limit is not None:
                    velocity[ids] = velocity[ids].clamp(
                        max=resolved.motion_policy.velocity_limit
                    )
                if resolved.motion_policy.acceleration_limit is not None:
                    acceleration[ids] = acceleration[ids].clamp(
                        max=resolved.motion_policy.acceleration_limit
                    )
                limits = engine.robot.get_qpos_limits()
                limits = limits[0] if limits.ndim == 3 else limits
                q = trajectory.positions[0]
                if (
                    limits.shape != (q.shape[1], 2)
                    or not bool(torch.isfinite(limits).all())
                    or bool(((q < limits[:, 0]) | (q > limits[:, 1])).any())
                ):
                    raise ValueError("source plan violates full-robot joint limits")
                check = validate_motion_samples(
                    q,
                    trajectory.dt[0],
                    velocity_limits=velocity,
                    acceleration_limits=acceleration,
                )
                motion_metrics = dict(check.checks[0].metrics)
                if not check.accepted:
                    raise ValueError(
                        f"source plan exceeds sampled motion limits: {motion_metrics}"
                    )
            report = self.validator.validate(resolved, context, plan)
            if not report.accepted:
                raise ValueError(
                    f"collision at sample {report.first_failure_sample} in "
                    f"{report.first_failure_phase}: {report.first_failure_pair}: "
                    f"{report.reason}"
                )
            metadata = dict(plan.diagnostics.metadata)
            metadata.update(
                collision_evidence="host_verified",
                collision_scope="sampled_tabletop_geometry_with_gripper_contact_projection",
                geometry_compatibility="asset_identity_matched",
            )
            if motion_metrics is not None:
                metadata["sampled_motion_limits"] = motion_metrics
            return replace(
                plan, diagnostics=replace(plan.diagnostics, metadata=metadata)
            )

        def guard(resolved, context, plan):
            nominal = checked(resolved, context, plan)
            final = (
                nominal
                if downstream is None
                else checked(resolved, context, downstream(resolved, context, nominal))
            )
            if tuple(item.name for item in nominal.segments) != tuple(
                item.name for item in final.segments
            ):
                raise ValueError("expanded plan changed the semantic phase identities")
            self._phase_samples[request.analysis_call_index] = tuple(
                (before.name, before.stop - before.start, after.stop - after.start)
                for before, after in zip(nominal.segments, final.segments)
            )
            self._segments[request.analysis_call_index] = tuple(
                (item.name, item.start, item.stop) for item in final.segments
            )
            endpoint = resolved.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            self._motion_targets[request.analysis_call_index] = (
                endpoint.control_part,
                tuple(endpoint.joint_ids),
            )
            if final.skill_id == "place":
                if self.validator._last_release_target is None:
                    raise ValueError("validated Place has no object release target")
                self._release_targets[request.analysis_call_index] = (
                    self.validator._last_release_target.copy()
                )
            metrics = final.diagnostics.metadata.get("sampled_motion_limits")
            if metrics is not None:
                self._motion_metrics[request.analysis_call_index] = dict(metrics)
            return final

        return guard
