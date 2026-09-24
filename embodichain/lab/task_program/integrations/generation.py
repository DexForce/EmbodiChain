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

"""Source-neutral generation adapters for grounded Task Program calls."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    ActionPlanTemplateAdapter,
    AtomicActionEngine,
    PlanTransform,
    PlannerDiagnostics,
    ResolvedActionRequest,
    TimedTrajectory,
)
from embodichain.lab.sim.motion.expansion import (
    CandidateCoordinator,
    CandidateSpec,
    GenerationSession,
    SceneCase,
    SourceContext,
    TrajectoryGenerationJobCfg,
    TrajectoryTemplate,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.task_program.runtime.generation import TaskProgramPlanRequest

__all__ = [
    "TaskProgramSourceAdapter",
    "TaskProgramGenerationRecord",
    "TaskProgramCandidatePlanTransformFactory",
]


@dataclass(frozen=True)
class TaskProgramSourceAdapter:
    """Expose a Task Program's grounded Atomic plan through SourceAdapter."""

    atomic_adapter: ActionPlanTemplateAdapter
    kind: ClassVar[str] = "task_program"

    def __post_init__(self) -> None:
        if not isinstance(self.atomic_adapter, ActionPlanTemplateAdapter):
            raise TypeError("atomic_adapter must be an ActionPlanTemplateAdapter")

    def export_template(
        self,
        source: ActionPlan,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        """Delegate one grounded Atomic plan to the common template adapter.

        Args:
            source: Grounded Atomic plan produced for one Task Program call.
            context: Coordinator-owned source and scene identity.

        Returns:
            Complete phase-annotated trajectory template.
        """
        return self.atomic_adapter.export_template(source, context=context)


@dataclass(frozen=True, slots=True)
class TaskProgramGenerationRecord:
    """Owned logical candidates generated for one Task Program planning call."""

    workflow_id: str
    workflow_call_index: int
    candidate_index: int
    selected_candidate_id: str
    candidates: tuple[CandidateSpec, ...]
    templates: tuple[TrajectoryTemplate, ...]

    def __post_init__(self) -> None:
        if type(self.workflow_id) is not str or not self.workflow_id:
            raise ValueError("workflow_id must be a nonempty string")
        for name in ("workflow_call_index", "candidate_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            type(self.selected_candidate_id) is not str
            or not self.selected_candidate_id
        ):
            raise ValueError("selected_candidate_id must be a nonempty string")
        candidates = tuple(self.candidates)
        templates = tuple(self.templates)
        if not candidates or len(candidates) != len(templates):
            raise ValueError("candidates and templates must be nonempty and aligned")
        if not all(isinstance(value, CandidateSpec) for value in candidates):
            raise TypeError("candidates must contain CandidateSpec values")
        if not all(isinstance(value, TrajectoryTemplate) for value in templates):
            raise TypeError("templates must contain TrajectoryTemplate values")
        if self.selected_candidate_id not in {
            candidate.identity.candidate_id for candidate in candidates
        }:
            raise ValueError("selected_candidate_id is absent from candidates")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "templates",
            tuple(replace(template) for template in templates),
        )

    def snapshot(self) -> TaskProgramGenerationRecord:
        """Return a record with independently owned trajectory tensors."""
        return TaskProgramGenerationRecord(
            workflow_id=self.workflow_id,
            workflow_call_index=self.workflow_call_index,
            candidate_index=self.candidate_index,
            selected_candidate_id=self.selected_candidate_id,
            candidates=self.candidates,
            templates=self.templates,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-compatible candidate identity and factor provenance."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_call_index": self.workflow_call_index,
            "candidate_index": self.candidate_index,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_ids": [
                candidate.identity.candidate_id for candidate in self.candidates
            ],
            "geometry_family_ids": [
                candidate.identity.geometry_family_id for candidate in self.candidates
            ],
            "trajectory_variants": [
                dict(candidate.trajectory_variant) for candidate in self.candidates
            ],
        }


class TaskProgramCandidatePlanTransformFactory:
    """Generate and select same-grid candidates for configured Atomic calls."""

    def __init__(
        self,
        profile: TrajectoryGenerationJobCfg,
        *,
        candidate_index: int,
        program_id: str,
        integration_id: str,
        robot_profile_id: str,
    ) -> None:
        if not isinstance(profile, TrajectoryGenerationJobCfg):
            raise TypeError("profile must be a TrajectoryGenerationJobCfg")
        if type(candidate_index) is not int or candidate_index < 0:
            raise ValueError("candidate_index must be a non-negative integer")
        for value, name in (
            (program_id, "program_id"),
            (integration_id, "integration_id"),
            (robot_profile_id, "robot_profile_id"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        owned = TrajectoryGenerationJobCfg.from_mapping(profile.to_dict())
        if owned.source.kind != "task_program":
            raise ValueError("profile source.kind must be task_program")
        if owned.source.source_id != program_id:
            raise ValueError("profile source_id must match program_id")
        if candidate_index >= owned.augmentation.max_variants_per_reference:
            raise ValueError("candidate_index is outside generated variants")
        if owned.affordance.enabled:
            raise ValueError(
                "the T4 Task Program transform requires Affordance disabled"
            )
        if owned.augmentation.factors.timing.enabled:
            raise ValueError("the T4 Task Program transform does not support timing")
        self._profile = owned
        self._candidate_index = candidate_index
        self._program_id = program_id
        self._integration_id = integration_id
        self._robot_profile_id = robot_profile_id
        self._records: list[TaskProgramGenerationRecord] = []

    @property
    def records(self) -> tuple[TaskProgramGenerationRecord, ...]:
        """Return generation records in planning-call order."""
        return tuple(record.snapshot() for record in self._records)

    def create_plan_transform(
        self,
        request: TaskProgramPlanRequest,
        *,
        engine: AtomicActionEngine,
    ) -> PlanTransform | None:
        """Return a candidate transform for the selected skill only."""
        if not isinstance(request, TaskProgramPlanRequest):
            raise TypeError("request must be a TaskProgramPlanRequest")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine")
        workflow_program = request.workflow_id.split("/", 1)[0]
        if workflow_program != self._program_id:
            raise ValueError("workflow ID does not match profile source_id")
        if request.invocation.skill_id != self._profile.source.template_id:
            return None

        def transform(
            resolved: ResolvedActionRequest,
            context: PlanningContext,
            plan: ActionPlan,
        ) -> ActionPlan:
            return self._transform(request, engine, resolved, context, plan)

        return transform

    def _transform(
        self,
        call_request: TaskProgramPlanRequest,
        engine: AtomicActionEngine,
        resolved: ResolvedActionRequest,
        context: PlanningContext,
        plan: ActionPlan,
    ) -> ActionPlan:
        if context.batch_size != 1:
            raise ValueError("the T4 Task Program transform supports B=1")
        robot = engine.robot
        joint_names = tuple(robot.joint_names)
        limits = robot.body_data.qpos_limits
        if not isinstance(limits, torch.Tensor):
            raise TypeError("robot qpos_limits must be a tensor")
        if limits.ndim == 3:
            limits = limits[0]
        if limits.shape != (len(joint_names), 2):
            raise ValueError("robot qpos_limits must match the full joint order")
        initial_digest = hashlib.sha256(
            context.robot.qpos.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        case = SceneCase(
            scene_case_id=self._integration_id,
            initial_state_id="initial_" + initial_digest,
            scene_signature=f"{self._integration_id}:scene:{context.scene.version}",
            task_id=self._program_id,
            robot_profile_id=self._robot_profile_id,
        )
        session = GenerationSession(self._profile)
        session.register_case(case, limits, joint_names=joint_names)
        invocation_id = resolved.invocation_id or resolved.skill_id
        source_context = SourceContext(
            source_id=self._profile.source.source_id,
            source_revision=self._profile.source.source_revision,
            unit_id=(
                f"{call_request.workflow_id}:"
                f"{call_request.workflow_call_index}:{invocation_id}"
            ),
            scene_case=case,
            control_dt=context.require_control_dt(),
        )
        atomic_adapter = ActionPlanTemplateAdapter(
            joint_names=joint_names,
            phase_permissions=self._profile.source.phase_permissions,
            phase_kinds=self._profile.source.phase_kinds,
            validator_id=self._profile.validation.validator_id,
        )
        coordinator = CandidateCoordinator(
            self._profile,
            session=session,
            source_adapter=TaskProgramSourceAdapter(atomic_adapter),
            source_context=source_context,
            joint_limits=limits,
            backend_id=plan.diagnostics.backend,
        )
        items = coordinator.enqueue_source(
            plan,
            count=self._profile.augmentation.max_variants_per_reference,
        )
        if self._candidate_index >= len(items):
            raise ValueError(
                "candidate_index is outside candidates that survived generation"
            )
        selected = items[self._candidate_index]
        candidates = tuple(item.spec for item in items)
        templates = tuple(item.template for item in items)
        record = TaskProgramGenerationRecord(
            workflow_id=call_request.workflow_id,
            workflow_call_index=call_request.workflow_call_index,
            candidate_index=self._candidate_index,
            selected_candidate_id=selected.spec.identity.candidate_id,
            candidates=candidates,
            templates=templates,
        )
        metadata = dict(plan.diagnostics.metadata)
        metadata["generation"] = record.to_metadata()
        source_plan = replace(
            plan,
            diagnostics=PlannerDiagnostics(
                backend=plan.diagnostics.backend,
                messages=plan.diagnostics.messages,
                metadata=metadata,
                failure=plan.diagnostics.failure,
            ),
        )
        trajectory = TimedTrajectory.from_positions(
            selected.template.positions.unsqueeze(0),
            env_ids=context.env_ids,
            dt=selected.template.dt.unsqueeze(0),
        )
        rebuilt = engine.rebuild_plan_from_trajectory(
            resolved,
            context,
            source_plan,
            trajectory,
        )
        self._records.append(record)
        return rebuilt
