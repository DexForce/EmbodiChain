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

"""Scene grounding and measured contract enforcement for demonstration skills."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.data_pipeline.real2sim import DemoSkillArtifact, retarget_demo_skill
from .bindings import JointPositionTarget
from .nominal import InstantiatedSkillMotion
from .plans import TimedTrajectory, SkillPlan
from .state import PlanningContext
from .invocation import ResolvedSkillRequest
from .verification import EffectVerificationRequest, EffectVerificationResult

__all__ = [
    "DemoSkillObservation",
    "DemoSkillEvidenceProvider",
    "DemoSkillMonitor",
    "DemoSkillMotionProvider",
]


@dataclass(frozen=True)
class DemoSkillObservation:
    """Owned measurements aligned to one robot observation and physical row.

    Missing contact pairs mean unavailable evidence, including for no-contact
    requirements. ``metrics`` must come from a task monitor of measured state.
    """

    timestamp: float
    env_id: int
    poses: Mapping[str, np.ndarray]
    contacts: Mapping[tuple[str, str], bool] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.timestamp)
            or self.timestamp < 0
            or type(self.env_id) is not int
            or self.env_id < 0
        ):
            raise ValueError("observation requires a finite timestamp and physical row")
        from embodichain.data_pipeline.real2sim.normalize import _rigid

        poses = {}
        for name, value in self.poses.items():
            owned = np.array(_rigid(value, name), copy=True)
            owned.setflags(write=False)
            poses[name] = owned
        if any(type(value) is not bool for value in self.contacts.values()):
            raise ValueError("contact observations must be measured booleans")
        if any(not np.isfinite(value) for value in self.metrics.values()):
            raise ValueError("observed metrics must be finite")
        object.__setattr__(self, "poses", MappingProxyType(poses))
        object.__setattr__(self, "contacts", MappingProxyType(dict(self.contacts)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


class DemoSkillEvidenceProvider(Protocol):
    """Host-owned acquisition port; no stepping or planned values as evidence."""

    def observe(self, context: PlanningContext) -> DemoSkillObservation:
        """Acquire one synchronized measured frame for the active row."""


class DemoSkillMonitor:
    """Latch failures and require every declared event and success predicate.

    Args:
        artifact: Frozen event, phase and predicate declarations.
        evidence_provider: Independent measured observation port.
    """

    def __init__(
        self, artifact: DemoSkillArtifact, evidence_provider: DemoSkillEvidenceProvider
    ) -> None:
        self.artifact, self.provider = artifact, evidence_provider
        self.spec = artifact.execution_spec
        self.results: dict[str, str] = {
            item.check_id: "not_run" for item in self.spec.constraints
        }
        self.events: set[str] = set()
        self.sample_count = len(artifact.execution_timestamps)
        self.last_sample = -1
        self.last_timestamp = -1.0
        self.terminal_passed = False
        self.failure: str | None = None
        self.static_pose_guards: dict[str, np.ndarray] = {}
        self.observations: list[dict[str, object]] = []

    def observe(
        self,
        context: PlanningContext,
        sample_index: int | None,
        *,
        terminal: bool = False,
    ) -> None:
        """Validate fresh measured state and latch failures before the next command.

        Args:
            context: Exact robot observation and physical-row identity.
            sample_index: Last acknowledged local plan sample, None at entry.
            terminal: Whether the execution session is at its effect boundary.
        """
        if self.failure is not None:
            raise ValueError(self.failure)
        frame = self.provider.observe(context)
        if (
            type(frame) is not DemoSkillObservation
            or context.batch_size != 1
            or frame.env_id != int(context.env_ids[0])
            or abs(frame.timestamp - context.robot.timestamp) > 1e-8
        ):
            raise ValueError("DemoSkill evidence is not aligned to the observation")
        if frame.timestamp < self.last_timestamp:
            raise ValueError("DemoSkill observation clock moved backwards")
        for name, initial in self.static_pose_guards.items():
            current = frame.poses.get(name)
            if (
                current is None
                or np.linalg.norm(current[:3, 3] - initial[:3, 3])
                > self.spec.scene_position_tolerance_m
                or Rotation.from_matrix(current[:3, :3] @ initial[:3, :3].T).magnitude()
                > self.spec.scene_rotation_tolerance_rad
            ):
                self.failure = f"DemoSkill static scene dependency changed: {name}"
                raise ValueError(self.failure)
        index = -1 if sample_index is None else sample_index
        if index > self.last_sample and frame.timestamp <= self.last_timestamp:
            raise ValueError(
                "an advanced DemoSkill sample requires fresh physical evidence"
            )
        if index < self.last_sample:
            raise ValueError("DemoSkill execution cannot rewind source events")
        if index >= self.sample_count:
            raise ValueError("executed sample exceeds the frozen DemoSkill timeline")
        if index > self.last_sample + 1:
            raise ValueError("DemoSkill execution skipped an unobserved plan sample")
        phase = next(
            (phase for phase in self.spec.phases if phase.start <= index < phase.stop),
            None,
        )
        due_events = {
            event.event_id for event in self.spec.events if event.sample_index == index
        }
        for item in self.spec.constraints:
            due = (
                (item.scope == "entry" and index == -1)
                or (item.scope == "terminal" and terminal)
                or (item.scope == "event" and item.event_id in due_events)
                or (
                    item.scope == "throughout"
                    and phase is not None
                    and (not item.phase_names or phase.name in item.phase_names)
                )
            )
            if not due:
                continue
            passed = self._predicate(item, frame)
            self.results[item.check_id] = "passed" if passed else "failed"
            if not passed:
                self.failure = f"DemoSkill constraint failed: {item.check_id}"
                raise ValueError(self.failure)
        self.events.update(due_events)
        if index != self.last_sample or terminal:
            self.observations.append(
                {
                    "sample_index": index,
                    "timestamp": frame.timestamp,
                    "events": sorted(due_events),
                    "checks": dict(self.results),
                    "metrics": dict(frame.metrics),
                }
            )
        self.last_sample, self.last_timestamp = index, frame.timestamp
        if terminal:
            if (
                index != self.sample_count - 1
                or self.events != {event.event_id for event in self.spec.events}
                or any(value != "passed" for value in self.results.values())
            ):
                self.failure = (
                    "DemoSkill completed without all required measured evidence"
                )
                raise ValueError(self.failure)
            self.terminal_passed = True

    def _predicate(self, item, frame: DemoSkillObservation) -> bool:
        if item.kind in {"contact", "no_contact"}:
            pair = tuple(sorted((item.subject, item.other)))
            if pair not in frame.contacts:
                raise ValueError(f"contact evidence unavailable: {item.check_id}")
            return frame.contacts[pair] == (item.kind == "contact")
        if item.kind == "metric":
            if item.check_id not in frame.metrics:
                raise ValueError(f"task metric unavailable: {item.check_id}")
            return item.minimum <= frame.metrics[item.check_id] <= item.maximum
        poses = {"world": np.eye(4), **frame.poses}
        if item.subject not in poses or item.other not in poses:
            raise ValueError(f"pose evidence unavailable: {item.check_id}")
        measured = np.linalg.inv(poses[item.other]) @ poses[item.subject]
        reference = np.asarray(item.reference_pose).reshape(4, 4)
        return (
            np.linalg.norm(measured[:3, 3] - reference[:3, 3])
            <= item.position_tolerance_m
            and Rotation.from_matrix(measured[:3, :3] @ reference[:3, :3].T).magnitude()
            <= item.rotation_tolerance_rad
        )


class DemoSkillMotionProvider:
    """Instantiate same-robot motion and validate its measured execution.

    Args:
        artifact: One immutable exemplar with explicit adaptation authority.
        role_bindings: Logical role to scene entity identifiers.
        robot_geometry_sha256: Identity captured from the live robot model.
        role_geometry_sha256: Identities captured from live physical role geometry.
        evidence_provider: Fresh measured contact/pose/task evidence.
        motion_generator: Engine-owned generator for Cartesian phases.
        plan_validator: Optional full-plan collision validator; required for an
            admitted physical expansion job, supplied by its source adapter.
    """

    def __init__(
        self,
        artifact: DemoSkillArtifact,
        *,
        role_bindings: Mapping[str, str],
        robot_geometry_sha256: str,
        role_geometry_sha256: Mapping[str, str],
        evidence_provider: DemoSkillEvidenceProvider,
        motion_generator: object,
        plan_validator: Callable[[TimedTrajectory], bool] | None = None,
    ) -> None:
        self.artifact = artifact
        self.role_bindings = MappingProxyType(dict(role_bindings))
        self.robot_geometry_sha256 = robot_geometry_sha256
        self.role_geometry_sha256 = dict(role_geometry_sha256)
        self.evidence_provider = evidence_provider
        self.motion_generator = motion_generator
        self.plan_validator = plan_validator
        self.monitor = DemoSkillMonitor(artifact, evidence_provider)
        self.motion: InstantiatedSkillMotion | None = None

    def instantiate(
        self, request: ResolvedSkillRequest, context: PlanningContext
    ) -> InstantiatedSkillMotion:
        """Ground one frozen nominal path without stepping or changing task state.

        Args:
            request: Bound ExecuteDemoSkill invocation.
            context: Current measured robot and scene state.

        Returns:
            Full-robot timed motion preserving recorded gripper samples.
        """
        from .primitives.execute_demo_skill import DemoSkillGoal
        from embodichain.lab.sim.motion.planners.utils import PlanState, MoveType

        goal = request.goal
        if (
            type(goal) is not DemoSkillGoal
            or goal.asset_sha256 != self.artifact.content_sha256
            or dict(goal.role_bindings) != dict(self.role_bindings)
        ):
            raise ValueError("DemoSkill invocation does not match the pinned source")
        if context.batch_size != 1 or self.motion is not None:
            raise ValueError(
                "DemoSkill sources are fresh single-row, single-attempt values"
            )
        if (
            self.robot_geometry_sha256
            != self.artifact.demonstration.robot_geometry_sha256
        ):
            raise ValueError("DemoSkill requires the recorded robot geometry")
        spec = self.artifact.execution_spec
        if not np.isclose(
            context.require_control_dt(),
            self.artifact.execution_timestamps[1]
            - self.artifact.execution_timestamps[0],
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError("DemoSkill control clock differs from the environment")
        initial = torch.as_tensor(
            np.array(
                self.artifact.initial_qpos
                if self.artifact.reference is None
                else (
                    self.artifact.reference.qpos[0]
                    if self.artifact.reference.initial_qpos is None
                    else self.artifact.reference.initial_qpos
                )
            ),
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        ).unsqueeze(0)
        entry_matches = torch.allclose(
            initial, context.robot.qpos, atol=spec.entry_tolerance, rtol=0
        )
        if not entry_matches and not spec.allow_entry_connection:
            raise ValueError("DemoSkill entry posture differs from the recorded state")
        roles = {}
        for role, entity in self.role_bindings.items():
            pose = context.scene.entities[entity].pose
            roles[role] = pose.reshape(4, 4).detach().cpu().numpy()
        self._frozen_role_poses = {
            name: np.array(pose, copy=True) for name, pose in roles.items()
        }
        self._role_uids = dict(goal.role_uids)
        fk_position_error = None
        fk_rotation_error = None
        targets = retarget_demo_skill(
            self.artifact, roles, geometry_sha256=self.role_geometry_sha256
        )
        path = torch.as_tensor(
            np.array(self.artifact.execution_qpos),
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        ).unsqueeze(0)
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        generator = self.motion_generator
        for phase in spec.phases:
            if phase.authority != "eef_pose":
                continue
            assert targets is not None
            seed = (
                context.robot.qpos[:, list(arm.joint_ids)]
                if phase.start == 0
                else path[:, phase.start - 1, list(arm.joint_ids)]
            )
            # Fixed-grid sequential IK preserves timestamps and source gripper values.
            states = [
                PlanState(
                    move_type=MoveType.EEF_MOVE,
                    xpos=torch.as_tensor(
                        targets[index], device=path.device, dtype=path.dtype
                    ).unsqueeze(0),
                )
                for index in range(phase.start, phase.stop)
            ]
            options = request.motion_policy.to_motion_gen_options(
                start_qpos=seed, control_part=arm.control_part
            )
            from dataclasses import replace

            options = replace(
                options,
                strategy="ik_interp",
                interpolation_mode="fixed_grid",
                interpolation_dt=context.require_control_dt(),
                sample_count=len(states) + 1,
                ik_seed_mode="sequential",
                velocity_limit=None,
                acceleration_limit=None,
            )
            result = generator.generate(states, options=options)
            if (
                result.positions is None
                or not bool(result.success.all())
                or result.positions.shape[1] != len(states) + 1
            ):
                raise ValueError("DemoSkill fixed-grid Cartesian IK failed")
            path[:, phase.start : phase.stop, list(arm.joint_ids)] = result.positions[
                :, 1:
            ]
        if any(phase.authority == "eef_pose" for phase in spec.phases):
            actual = generator.robot.compute_batch_fk(
                path[:, :, list(arm.joint_ids)],
                name=arm.control_part,
                env_ids=context.env_ids.tolist(),
                to_matrix=True,
            )
            if (
                actual is None
                or actual.shape != (1, path.shape[1], 4, 4)
                or not bool(torch.isfinite(actual).all())
            ):
                raise ValueError("DemoSkill independent FK is unavailable")
            actual = actual[0].detach().cpu().numpy()
            for phase in spec.phases:
                if phase.authority != "eef_pose":
                    continue
                observed = actual[phase.start : phase.stop]
                requested = targets[phase.start : phase.stop]
                position = np.linalg.norm(
                    observed[:, :3, 3] - requested[:, :3, 3], axis=1
                )
                rotation = Rotation.from_matrix(
                    observed[:, :3, :3] @ np.swapaxes(requested[:, :3, :3], 1, 2)
                ).magnitude()
                fk_position_error = max(fk_position_error or 0.0, float(position.max()))
                fk_rotation_error = max(fk_rotation_error or 0.0, float(rotation.max()))
                if np.any(position > 0.002) or np.any(rotation > np.deg2rad(2)):
                    raise ValueError(
                        "DemoSkill Cartesian IK exceeds its independent TCP error tube"
                    )
        connection_count = 0
        if not entry_matches:
            if self.plan_validator is None:
                raise ValueError(
                    "entry connection requires an independent complete-plan collision validator"
                )
            from dataclasses import replace
            from embodichain.data_pipeline.real2sim import DemoSkillPhase

            target = PlanState(
                move_type=MoveType.JOINT_MOVE, qpos=path[:, 0, list(arm.joint_ids)]
            )
            options = request.motion_policy.to_motion_gen_options(
                start_qpos=context.robot.qpos[:, list(arm.joint_ids)],
                control_part=arm.control_part,
                interpolation_dt=context.require_control_dt(),
            )
            result = generator.generate([target], options=options)
            if result.positions is None or not bool(result.success.all()):
                raise ValueError("DemoSkill free-space entry planning failed")
            connection_count = result.positions.shape[1] - 1
            connection = (
                context.robot.qpos[:, None, :].expand(-1, connection_count, -1).clone()
            )
            connection[:, :, list(arm.joint_ids)] = result.positions[:, :-1]
            path = torch.cat((connection, path), dim=1)
            spec.phases = (
                DemoSkillPhase("__entry_connection__", 0, connection_count),
            ) + tuple(
                replace(
                    phase,
                    start=phase.start + connection_count,
                    stop=phase.stop + connection_count,
                )
                for phase in spec.phases
            )
            spec.events = tuple(
                replace(event, sample_index=event.sample_index + connection_count)
                for event in spec.events
            )
            self.monitor.spec = spec
            self.monitor.sample_count = path.shape[1]
        self._nominal_spec = spec
        timed = TimedTrajectory.from_uniform_step(
            path, env_ids=context.env_ids, step_dt=context.require_control_dt()
        )
        if self.plan_validator is not None and not self.plan_validator(timed):
            raise ValueError("DemoSkill full-plan collision validation failed")
        self.monitor.static_pose_guards = {
            role.role_id: np.array(roles[role.role_id], copy=True)
            for role in spec.roles
            if role.reference_kind == "static"
        }
        self.monitor.observe(context, None)
        self.motion = InstantiatedSkillMotion(
            trajectory=timed,
            success=torch.ones(1, dtype=torch.bool, device=path.device),
            segment_lengths={
                phase.name: phase.stop - phase.start for phase in spec.phases
            },
            initial_qpos=context.robot.qpos,
            source_id=self.artifact.demonstration.source_id,
            source_revision=self.artifact.demonstration.source_revision,
            unit_id=self.artifact.demonstration.unit_id,
            origin=self.artifact.demonstration.origin,
            authority="hybrid",
            collision_evidence=(
                "sampled_passed" if self.plan_validator is not None else "unverified"
            ),
            pose_origin=self.artifact.demonstration.pose_origin or "unknown",
            fk_position_error_m=fk_position_error,
            fk_rotation_error_rad=fk_rotation_error,
            qpos_semantics=(
                "measured_state"
                if self.artifact.demonstration.origin == "measured"
                else "design_reference"
            ),
            source_command_semantics=(
                (
                    "design_reference"
                    if self.artifact.demonstration.command_alignment
                    == "sample_aligned_reference"
                    else "actual_submission"
                )
                if self.artifact.demonstration.commands is not None
                else None
            ),
            calibration_revision=self.artifact.demonstration.calibration_revision,
            annotation_revision=self.artifact.demonstration.annotation_revision,
            prototype_sha256=self.artifact.content_sha256,
            source_time_map=(
                self.artifact.report.time_map
                if self.artifact.reference is None
                else self.artifact.reference.time_map
            ),
            source_sample_offset=connection_count,
            phase_permissions={
                phase.name: tuple(op for op in phase.allowed_operators)
                for phase in spec.phases
            },
            joint_names=self.artifact.joint_names,
        )
        return self.motion

    def bind_execution_plan(self, plan: SkillPlan) -> None:
        """Freeze the actual plan and remap monitors after authorized expansion.

        Args:
            plan: Engine-validated plan after the selected transformation.
        """
        from dataclasses import replace

        if self.motion is None or plan.joint_trajectory is None:
            raise ValueError("DemoSkill plan must have an instantiated nominal path")
        original = self.motion.trajectory.positions
        candidate = plan.joint_trajectory.positions
        spec = self._nominal_spec
        mapping = plan.diagnostics.metadata.get("nominal_time_map")
        if mapping is None:
            if candidate.shape != original.shape:
                raise ValueError(
                    "changed DemoSkill timing requires an explicit time map"
                )
            event_map = {index: index for index in range(original.shape[1])}
        else:
            event_map = dict(
                zip(mapping["source_event_indices"], mapping["execution_event_indices"])
            )
            if len(candidate[0]) != len(mapping["left"]):
                raise ValueError(
                    "DemoSkill timing map does not cover the executed plan"
                )
        phases = {segment.name: segment for segment in plan.segments}
        if set(phases) != {phase.name for phase in spec.phases}:
            raise ValueError("DemoSkill expansion changed its internal phases")
        for phase in spec.phases:
            segment = phases[phase.name]
            protected = phase.kind != "free" or not phase.allowed_operators
            if protected and not torch.allclose(
                candidate[:, segment.start : segment.stop],
                original[:, phase.start : phase.stop],
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("DemoSkill expansion modified a protected phase")
        for event in spec.events:
            for old_index in range(
                max(0, event.sample_index - 1),
                min(original.shape[1], event.sample_index + 2),
            ):
                new_index = event_map.get(old_index)
                if new_index is None or not torch.allclose(
                    candidate[:, new_index], original[:, old_index], atol=1e-7, rtol=0
                ):
                    raise ValueError("DemoSkill expansion modified an event stencil")
        self.monitor.spec.phases = tuple(
            replace(phase, start=phases[phase.name].start, stop=phases[phase.name].stop)
            for phase in spec.phases
        )
        self.monitor.spec.events = tuple(
            replace(event, sample_index=event_map[event.sample_index])
            for event in spec.events
        )
        self.monitor.sample_count = len(candidate[0])
        self.executed_time_map = mapping

    def observe_execution(
        self, context: PlanningContext, sample_index: int | None, *, terminal: bool
    ) -> None:
        """Validate measured evidence before the runner submits its next frame.

        Args:
            context: Fresh measured robot and scene observation.
            sample_index: Last acknowledged sample in the installed plan.
            terminal: Whether the session is waiting for its terminal effect.
        """
        if sample_index is None and self.monitor.last_sample >= 0:
            sample_index = self.monitor.last_sample
        self.monitor.observe(context, sample_index, terminal=terminal)

    def owns_effect(self, request: object) -> bool:
        """Select only this invocation's explicit DemoSkill effect boundary.

        Args:
            request: Pending core verification request.

        Returns:
            Whether the request selects the DemoSkill terminal requirement.
        """
        return (
            isinstance(request, EffectVerificationRequest)
            and request.skill_id == "execute_demo_skill"
            and request.effect_verification is not None
            and request.effect_verification.kind == "demo_skill"
        )

    def verify_effect(
        self, context: PlanningContext, request: EffectVerificationRequest
    ) -> EffectVerificationResult:
        """Confirm the terminal boundary after all measured predicates pass.

        Args:
            context: Fresh observation correlated with the pending request.
            request: Core request whose identity must be retained.

        Returns:
            Source-contract approval; the runner also preserves any external
            verifier required for nonempty symbolic state changes.
        """
        if not self.owns_effect(request) or not self.monitor.terminal_passed:
            raise ValueError("DemoSkill terminal evidence has not passed")
        return EffectVerificationResult(
            verification_id=request.verification_id,
            success_mask=request.env_mask,
            failure_mask=torch.zeros_like(request.env_mask),
            invalidation_mask=torch.zeros_like(request.env_mask),
            retry_mask=torch.zeros_like(request.env_mask),
        )

    def execution_report(self) -> dict[str, object]:
        """Return owned measured qualification and source lineage for persistence."""
        return {
            "asset_sha256": self.artifact.content_sha256,
            "asset_id": self.artifact.execution_spec.asset_id,
            "role_bindings": dict(self.role_bindings),
            "role_simulation_uids": dict(getattr(self, "_role_uids", {})),
            "frozen_role_poses": {
                name: pose.tolist()
                for name, pose in getattr(self, "_frozen_role_poses", {}).items()
            },
            "source_id": self.artifact.demonstration.source_id,
            "source_revision": self.artifact.demonstration.source_revision,
            "recording_sha256": self.artifact.demonstration.recording_sha256,
            "recording_sample_offset": self.artifact.demonstration.source_sample_offset,
            "exemplar_id": self.artifact.demonstration.unit_id,
            "origin": self.artifact.demonstration.origin,
            "nominal_origin": (
                "adapted_nominal"
                if self.artifact.reference is not None
                else "normalized_source"
            ),
            "preparation": (
                None
                if self.artifact.reference is None
                else json.loads(self.artifact.reference.preparation_json)
            ),
            "terminal_passed": self.monitor.terminal_passed,
            "checks": dict(self.monitor.results),
            "events": sorted(self.monitor.events),
            "observations": [dict(item) for item in self.monitor.observations],
            "nominal_time_map": getattr(self, "executed_time_map", None),
            "motion_provenance": (
                None if self.motion is None else self.motion.provenance()
            ),
        }
