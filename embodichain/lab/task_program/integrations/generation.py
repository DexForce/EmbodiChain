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
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordanceSamplingContext,
)
from embodichain.lab.sim.motion.expansion import (
    CandidateRecipe,
    CandidateCoordinator,
    CandidateSpec,
    CombinedGenerationProfile,
    GenerationSession,
    SceneCase,
    SourceContext,
    TemplateSourceAdapter,
    TrajectoryPhase,
    TrajectoryGenerationJobCfg,
    TrajectoryTemplate,
    enumerate_candidate_recipes,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.task_program.runtime.generation import TaskProgramPlanRequest

__all__ = [
    "TaskProgramSourceAdapter",
    "TaskProgramGenerationRecord",
    "TaskProgramCandidatePlanTransformFactory",
    "CombinedTaskProgramCandidatePlanTransformFactory",
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
    recipe_index: int | None = None
    cycle_index: int | None = None
    affordance_requested: int | None = None
    trajectory_requested: int | None = None
    trajectory_selected: int | None = None
    fallback_count: int = 0
    env_id: int | None = None

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
        for name in (
            "recipe_index",
            "cycle_index",
            "affordance_requested",
            "trajectory_requested",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (self.cycle_index is None) != (self.recipe_index is None):
            raise ValueError("cycle_index and recipe_index must be provided together")
        if self.trajectory_selected is not None and (
            type(self.trajectory_selected) is not int or self.trajectory_selected < 0
        ):
            raise ValueError(
                "trajectory_selected must be a non-negative integer or None"
            )
        if type(self.fallback_count) is not int or self.fallback_count < 0:
            raise ValueError("fallback_count must be a non-negative integer")
        if self.env_id is not None and (
            type(self.env_id) is not int or self.env_id < 0
        ):
            raise ValueError("env_id must be a non-negative integer or None")
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
            recipe_index=self.recipe_index,
            cycle_index=self.cycle_index,
            affordance_requested=self.affordance_requested,
            trajectory_requested=self.trajectory_requested,
            trajectory_selected=self.trajectory_selected,
            fallback_count=self.fallback_count,
            env_id=self.env_id,
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
            "recipe_index": self.recipe_index,
            "cycle_index": self.cycle_index,
            "affordance_requested": self.affordance_requested,
            "trajectory_requested": self.trajectory_requested,
            "trajectory_selected": self.trajectory_selected,
            "fallback_count": self.fallback_count,
            "env_id": self.env_id,
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


def _combined_phase_declarations(
    plan: ActionPlan,
    operators: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Declare the built-in Pick/Place phase contract from Atomic segments."""
    permissions: dict[str, tuple[str, ...]] = {}
    kinds: dict[str, str] = {}
    contact_names = {"close", "open", "release", "contact", "grasp"}
    editable_names = {"lift", "back", "retract", "transport"}
    for segment in plan.segments:
        name = segment.name
        kinds[name] = "contact" if name in contact_names else "free"
        permissions[name] = operators if name in editable_names else ()
    if not permissions:
        raise ValueError("combined generation requires named Atomic trajectory phases")
    return permissions, kinds


class CombinedTaskProgramCandidatePlanTransformFactory:
    """Apply one deterministic combined recipe to a repeated Task Program.

    Pick calls receive their recipe-selected Affordance branch before Atomic
    planning. Place calls are expanded after planning on the existing control
    grid. Unknown Atomic phase names remain replay-only and are never edited.
    """

    def __init__(
        self,
        profile: CombinedGenerationProfile,
        *,
        candidate_index: int,
        program_id: str,
        integration_id: str,
        robot_profile_id: str,
    ) -> None:
        if not isinstance(profile, CombinedGenerationProfile):
            raise TypeError("profile must be a CombinedGenerationProfile")
        if type(candidate_index) is not int or candidate_index < 0:
            raise ValueError("candidate_index must be a non-negative integer")
        for value, name in (
            (program_id, "program_id"),
            (integration_id, "integration_id"),
            (robot_profile_id, "robot_profile_id"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if profile.source.source_id != program_id:
            raise ValueError("profile source_id must match program_id")
        recipes = enumerate_candidate_recipes(profile)
        if candidate_index >= len(recipes):
            raise ValueError("candidate_index is outside combined recipes")
        self._profile = profile
        self._candidate_index = candidate_index
        self._recipes = recipes
        self._recipe = recipes[candidate_index]
        self._program_id = program_id
        self._integration_id = integration_id
        self._robot_profile_id = robot_profile_id
        self._records: list[TaskProgramGenerationRecord] = []

    def _recipe_for_row(self, row_index: int) -> CandidateRecipe:
        """Map a physical batch row to a deterministic logical recipe."""
        return self._recipes[(self._candidate_index + row_index) % len(self._recipes)]

    @property
    def records(self) -> tuple[TaskProgramGenerationRecord, ...]:
        """Return Place generation records in planning-call order."""
        return tuple(record.snapshot() for record in self._records)

    def prepare_planning_context(
        self,
        request: TaskProgramPlanRequest,
        context: PlanningContext,
        *,
        engine: AtomicActionEngine,
    ) -> PlanningContext:
        """Attach the recipe's Affordance branch before the Pick is planned."""
        if not isinstance(request, TaskProgramPlanRequest):
            raise TypeError("request must be a TaskProgramPlanRequest")
        if not isinstance(context, PlanningContext):
            raise TypeError("context must be a PlanningContext")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine")
        if request.workflow_id.split("/", 1)[0] != self._program_id:
            raise ValueError("workflow ID does not match profile source_id")
        cycle_index = request.workflow_call_index // 2
        if cycle_index >= len(self._recipes[0].cycle_schedule):
            raise ValueError("workflow call index exceeds the three-cycle recipe")
        if request.invocation.skill_id not in {"pick", "pick_up"}:
            return context
        recipes = tuple(self._recipe_for_row(row) for row in range(context.batch_size))
        return replace(
            context,
            affordance_sampling=AffordanceSamplingContext(
                count=self._profile.affordance.branches_per_family,
                seed=self._recipes[0].visual_seed,
                episode_id=self._candidate_index,
                attempt_id=0,
                branch_overrides=tuple(
                    recipe.cycle_schedule[cycle_index].affordance_requested
                    for recipe in recipes
                ),
            ),
        )

    def create_plan_transform(
        self,
        request: TaskProgramPlanRequest,
        *,
        engine: AtomicActionEngine,
    ) -> PlanTransform | None:
        """Return the Place transform; Pick uses the pre-plan hook only."""
        if not isinstance(request, TaskProgramPlanRequest):
            raise TypeError("request must be a TaskProgramPlanRequest")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine")
        if request.workflow_id.split("/", 1)[0] != self._program_id:
            raise ValueError("workflow ID does not match profile source_id")
        if request.invocation.skill_id not in {"place"}:
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
            return self._transform_batch(
                call_request,
                engine,
                resolved,
                context,
                plan,
            )
        cycle_index = call_request.workflow_call_index // 2
        if cycle_index >= len(self._recipe.cycle_schedule):
            raise ValueError("workflow call index exceeds the three-cycle recipe")
        robot = engine.robot
        joint_names = tuple(robot.joint_names)
        limits = robot.body_data.qpos_limits
        if not isinstance(limits, torch.Tensor):
            raise TypeError("robot qpos_limits must be a tensor")
        if limits.ndim == 3:
            limits = limits[0]
        if limits.shape != (len(joint_names), 2):
            raise ValueError("robot qpos_limits must match the full joint order")
        permissions, kinds = _combined_phase_declarations(
            plan,
            self._profile.trajectory.spatial.operators,
        )
        payload = {
            "source": {
                "kind": "task_program",
                "source_id": self._program_id,
                "source_revision": self._profile.source.source_revision,
                "unit_scope": "action",
                "template_id": "place",
                "phase_permissions": permissions,
                "phase_kinds": kinds,
            },
            "augmentation": {
                "seed": self._recipe.visual_seed % (2**63),
                "max_variants_per_reference": self._profile.trajectory.variants_per_family,
                "factors": {
                    "spatial": {
                        "enabled": self._profile.trajectory.spatial.enabled,
                        "method": self._profile.trajectory.spatial.operators,
                        "joint_offset_scale": self._profile.trajectory.spatial.joint_offset_scale,
                        "via_count": self._profile.trajectory.spatial.via_count,
                    }
                },
            },
            "affordance": {"enabled": False},
            "scheduling": {
                "candidate_budget": self._profile.trajectory.variants_per_family,
            },
            "execution": {"max_inflight": 1},
            "observation": {"enabled": False, "profiles": []},
        }
        legacy = TrajectoryGenerationJobCfg.from_mapping(payload)
        digest = hashlib.sha256(
            context.robot.qpos.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        case = SceneCase(
            scene_case_id=self._integration_id,
            initial_state_id="initial_" + digest,
            scene_signature=f"{self._integration_id}:scene:{context.scene.version}",
            task_id=self._program_id,
            robot_profile_id=self._robot_profile_id,
        )
        session = GenerationSession(legacy)
        session.register_case(case, limits, joint_names=joint_names)
        source_context = SourceContext(
            source_id=self._program_id,
            source_revision=self._profile.source.source_revision,
            unit_id=(
                f"{self._recipe.candidate_id}:cycle:{cycle_index}:"
                f"{call_request.workflow_call_index}"
            ),
            scene_case=case,
            control_dt=context.require_control_dt(),
        )
        adapter = ActionPlanTemplateAdapter(
            joint_names=joint_names,
            phase_permissions=permissions,
            phase_kinds=kinds,
            validator_id="task_success",
        )
        coordinator = CandidateCoordinator(
            legacy,
            session=session,
            source_adapter=TaskProgramSourceAdapter(adapter),
            source_context=source_context,
            joint_limits=limits,
            backend_id=plan.diagnostics.backend,
        )
        items = coordinator.enqueue_source(
            plan,
            count=self._profile.trajectory.variants_per_family,
        )
        if not items:
            raise ValueError("combined trajectory generation produced no survivors")
        requested = self._recipe.cycle_schedule[cycle_index].trajectory_requested
        selected_index = min(requested, len(items) - 1)
        selected = items[selected_index]
        record = TaskProgramGenerationRecord(
            workflow_id=call_request.workflow_id,
            workflow_call_index=call_request.workflow_call_index,
            candidate_index=selected_index,
            selected_candidate_id=selected.spec.identity.candidate_id,
            candidates=tuple(item.spec for item in items),
            templates=tuple(item.template for item in items),
            recipe_index=self._recipe.recipe_index,
            cycle_index=cycle_index,
            affordance_requested=self._recipe.cycle_schedule[
                cycle_index
            ].affordance_requested,
            trajectory_requested=requested,
            trajectory_selected=selected_index,
            fallback_count=max(0, requested - selected_index),
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
        rebuilt = engine.rebuild_plan_from_trajectory(
            resolved,
            context,
            source_plan,
            TimedTrajectory.from_positions(
                selected.template.positions.unsqueeze(0),
                env_ids=context.env_ids,
                dt=selected.template.dt.unsqueeze(0),
            ),
        )
        self._records.append(record)
        return rebuilt

    def _transform_batch(
        self,
        call_request: TaskProgramPlanRequest,
        engine: AtomicActionEngine,
        resolved: ResolvedActionRequest,
        context: PlanningContext,
        plan: ActionPlan,
    ) -> ActionPlan:
        """Expand each physical row against its own deterministic recipe."""
        cycle_index = call_request.workflow_call_index // 2
        if cycle_index >= len(self._recipes[0].cycle_schedule):
            raise ValueError("workflow call index exceeds the three-cycle recipe")
        trajectory = plan.joint_trajectory
        if trajectory is None:
            raise ValueError("combined batch generation requires a joint trajectory")
        robot = engine.robot
        joint_names = tuple(robot.joint_names)
        limits = robot.body_data.qpos_limits
        if not isinstance(limits, torch.Tensor):
            raise TypeError("robot qpos_limits must be a tensor")
        if limits.ndim == 3:
            limits = limits[0]
        if limits.shape != (len(joint_names), 2):
            raise ValueError("robot qpos_limits must match the full joint order")
        permissions, kinds = _combined_phase_declarations(
            plan,
            self._profile.trajectory.spatial.operators,
        )
        positions: list[torch.Tensor] = []
        selected_dts: list[torch.Tensor] = []
        row_records: list[dict[str, object]] = []
        row_templates: list[TrajectoryTemplate] = []
        row_specs: list[CandidateSpec] = []
        for row in range(context.batch_size):
            recipe = self._recipe_for_row(row)
            payload = {
                "source": {
                    "kind": "task_program",
                    "source_id": self._program_id,
                    "source_revision": self._profile.source.source_revision,
                    "unit_scope": "action",
                    "template_id": "place",
                    "phase_permissions": permissions,
                    "phase_kinds": kinds,
                },
                "augmentation": {
                    "seed": recipe.visual_seed % (2**63),
                    "max_variants_per_reference": self._profile.trajectory.variants_per_family,
                    "factors": {
                        "spatial": {
                            "enabled": self._profile.trajectory.spatial.enabled,
                            "method": self._profile.trajectory.spatial.operators,
                            "joint_offset_scale": self._profile.trajectory.spatial.joint_offset_scale,
                            "via_count": self._profile.trajectory.spatial.via_count,
                        }
                    },
                },
                "affordance": {"enabled": False},
                "scheduling": {
                    "candidate_budget": self._profile.trajectory.variants_per_family,
                },
                "execution": {"max_inflight": 1},
                "observation": {"enabled": False, "profiles": []},
            }
            legacy = TrajectoryGenerationJobCfg.from_mapping(payload)
            digest = hashlib.sha256(
                context.robot.qpos[row].detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            case = SceneCase(
                scene_case_id=self._integration_id,
                initial_state_id=f"initial_{digest}",
                scene_signature=f"{self._integration_id}:scene:{context.scene.version}",
                task_id=self._program_id,
                robot_profile_id=self._robot_profile_id,
            )
            session = GenerationSession(legacy)
            session.register_case(case, limits, joint_names=joint_names)
            phases = tuple(
                TrajectoryPhase(
                    segment.name,
                    segment.start,
                    segment.stop,
                    kind=kinds[segment.name],
                    allowed_operators=permissions[segment.name],
                )
                for segment in plan.segments
            )
            template = TrajectoryTemplate(
                source_id=self._program_id,
                source_revision=self._profile.source.source_revision,
                template_id=f"{recipe.candidate_id}:place:{cycle_index}:{row}",
                joint_names=joint_names,
                positions=trajectory.positions[row],
                dt=trajectory.dt[row],
                phases=phases,
                allowed_operators=tuple(
                    dict.fromkeys(
                        operator
                        for values in permissions.values()
                        for operator in values
                    )
                ),
                validator_id="task_success",
                controlled_joint_indices=tuple(range(len(joint_names))),
            )
            source_context = SourceContext(
                source_id=self._program_id,
                source_revision=self._profile.source.source_revision,
                unit_id=template.template_id,
                scene_case=case,
                control_dt=context.require_control_dt(),
            )
            coordinator = CandidateCoordinator(
                legacy,
                session=session,
                source_adapter=TemplateSourceAdapter(),
                source_context=source_context,
                joint_limits=limits,
                backend_id=plan.diagnostics.backend,
            )
            items = coordinator.enqueue_source(
                template,
                count=self._profile.trajectory.variants_per_family,
            )
            if not items:
                raise ValueError(f"row {row} produced no trajectory survivors")
            requested = recipe.cycle_schedule[cycle_index].trajectory_requested
            selected_index = min(requested, len(items) - 1)
            selected = items[selected_index]
            positions.append(selected.template.positions)
            selected_dts.append(selected.template.dt)
            row_templates.append(selected.template)
            row_specs.append(selected.spec)
            row_records.append(
                {
                    "row_index": row,
                    "recipe_index": recipe.recipe_index,
                    "candidate_id": selected.spec.identity.candidate_id,
                    "affordance_requested": recipe.cycle_schedule[
                        cycle_index
                    ].affordance_requested,
                    "trajectory_requested": requested,
                    "trajectory_selected": selected_index,
                    "fallback_count": max(0, requested - selected_index),
                }
            )
        metadata = dict(plan.diagnostics.metadata)
        metadata["generation"] = {"rows": row_records}
        source_plan = replace(
            plan,
            diagnostics=PlannerDiagnostics(
                backend=plan.diagnostics.backend,
                messages=plan.diagnostics.messages,
                metadata=metadata,
                failure=plan.diagnostics.failure,
            ),
        )
        rebuilt = engine.rebuild_plan_from_trajectory(
            resolved,
            context,
            source_plan,
            TimedTrajectory.from_positions(
                torch.stack(positions),
                env_ids=context.env_ids,
                dt=torch.stack(selected_dts),
            ),
        )
        for row, (spec, template, record) in enumerate(
            zip(row_specs, row_templates, row_records, strict=True)
        ):
            self._records.append(
                TaskProgramGenerationRecord(
                    workflow_id=call_request.workflow_id,
                    workflow_call_index=call_request.workflow_call_index,
                    candidate_index=int(record["trajectory_selected"]),
                    selected_candidate_id=spec.identity.candidate_id,
                    candidates=(spec,),
                    templates=(template,),
                    recipe_index=int(record["recipe_index"]),
                    cycle_index=cycle_index,
                    affordance_requested=int(record["affordance_requested"]),
                    trajectory_requested=int(record["trajectory_requested"]),
                    trajectory_selected=int(record["trajectory_selected"]),
                    fallback_count=int(record["fallback_count"]),
                    env_id=row,
                )
            )
        return rebuilt
