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

"""Source-neutral expansion adapters for grounded Task Program calls."""

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
    CombinedExpansionProfile,
    ExpansionSession,
    SceneCase,
    SourceContext,
    TemplateSourceAdapter,
    TrajectoryExpansionCfg,
    TrajectoryPhase,
    TrajectoryTemplate,
    enumerate_candidate_recipes,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions._expansion_phases import (
    FREE_PHASES,
    _canonical_samples,
    _free_joint_indices,
)
from embodichain.lab.task_program.runtime.expansion import TaskProgramPlanRequest

__all__ = [
    "TaskProgramSourceAdapter",
    "TaskProgramExpansionRecord",
    "CombinedTaskProgramExpansionFactory",
]


def _combined_augmentation_payload(
    profile: CombinedExpansionProfile,
    seed: int,
) -> dict[str, object]:
    """Return the augmentation settings every combined transform shares.

    The profile's deduplication tolerance travels with the offset scale it has
    to stay below: with the contract default, a profile that declares a
    conservative amplitude would have every proposal measured as a duplicate of
    the reference and would keep the nominal variant alone.

    Args:
        profile: Combined episode profile that owns the trajectory settings.
        seed: Local sampling seed for this recipe's expansion streams.

    Returns:
        JSON-compatible augmentation settings.
    """
    if not isinstance(profile, CombinedExpansionProfile):
        raise TypeError("profile must be a CombinedExpansionProfile")
    spatial = profile.trajectory.spatial
    payload: dict[str, object] = {
        "seed": seed,
        "max_variants_per_reference": profile.trajectory.variants_per_family,
        "factors": {
            "spatial": {
                "enabled": spatial.enabled,
                "method": spatial.operators,
                "joint_offset_scale": spatial.joint_offset_scale,
                "via_count": spatial.via_count,
            }
        },
    }
    if spatial.dedup_tolerance is not None:
        payload["coverage"] = {
            "joint_dedup_normalized_tol": spatial.dedup_tolerance,
        }
    return payload


def _motion_joint_indices(resolved: ResolvedActionRequest) -> tuple[int, ...]:
    """Limit residuals to explicitly bound motion endpoints, including dual arms."""
    indices = tuple(
        dict.fromkeys(
            index
            for endpoint in resolved.binding.endpoints
            if endpoint.endpoint_id == "motion"
            for index in endpoint.joint_ids
        )
    )
    if not indices:
        raise ValueError("Trajectory expansion requires bound motion joint indices")
    return indices


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
class TaskProgramExpansionRecord:
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

    def snapshot(self) -> TaskProgramExpansionRecord:
        """Return a record with independently owned trajectory tensors."""
        return TaskProgramExpansionRecord(
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


def _combined_phase_declarations(
    plan: ActionPlan,
    operators: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Use the same free/contact boundary as the standalone Atomic host."""
    permissions: dict[str, tuple[str, ...]] = {}
    kinds: dict[str, str] = {}
    editable_names = FREE_PHASES.get(plan.skill_id)
    if editable_names is None:
        raise ValueError(f"No expansion permissions for skill {plan.skill_id!r}")
    for segment in plan.segments:
        free = segment.name in editable_names
        kinds[segment.name] = "free" if free else "contact"
        permissions[segment.name] = (
            operators if free and segment.stop - segment.start >= 3 else ()
        )
    if not permissions:
        raise ValueError("combined expansion requires named Atomic trajectory phases")
    return permissions, kinds


class CombinedTaskProgramExpansionFactory:
    """Apply one deterministic combined recipe to a Task Program call.

    Selected calls receive their Affordance branch before Atomic planning.
    All skills share one row-local transform; explicit phase declarations may
    narrow the sample permissions for specialized deployments.
    """

    def __init__(
        self,
        profile: CombinedExpansionProfile,
        *,
        candidate_index: int,
        recipe_indices: tuple[int, ...] | None = None,
        program_id: str,
        integration_id: str,
        robot_profile_id: str,
    ) -> None:
        if not isinstance(profile, CombinedExpansionProfile):
            raise TypeError("profile must be a CombinedExpansionProfile")
        if type(candidate_index) is not int or candidate_index < 0:
            raise ValueError("candidate_index must be a non-negative integer")
        if recipe_indices is not None:
            recipe_indices = tuple(recipe_indices)
            if not recipe_indices or any(
                type(index) is not int or index < 0 for index in recipe_indices
            ):
                raise ValueError("recipe_indices must contain non-negative integers")
            # CollectionSelection validates logical selections for uniqueness;
            # physical padding rows may intentionally repeat a recipe.
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
        if recipe_indices is not None and any(
            index >= len(recipes) for index in recipe_indices
        ):
            raise ValueError("recipe_indices contain an out-of-range recipe")
        self._profile = profile
        self._candidate_index = candidate_index
        self._recipe_indices = recipe_indices
        self._recipes = recipes
        self._recipe = recipes[candidate_index]
        self._program_id = program_id
        self._integration_id = integration_id
        self._robot_profile_id = robot_profile_id
        self._records: list[TaskProgramExpansionRecord] = []

    def _recipe_for_row(self, row_index: int) -> CandidateRecipe:
        """Map a physical batch row to a deterministic logical recipe."""
        if self._recipe_indices is not None:
            if row_index >= len(self._recipe_indices):
                raise ValueError("recipe_indices do not cover the active batch")
            return self._recipes[self._recipe_indices[row_index]]
        return self._recipes[(self._candidate_index + row_index) % len(self._recipes)]

    def _augmentation_payload(self, seed: int) -> dict[str, object]:
        """Return the augmentation settings every combined transform shares.

        Args:
            seed: Local sampling seed for this recipe's expansion streams.

        Returns:
            JSON-compatible augmentation settings.
        """
        return _combined_augmentation_payload(self._profile, seed)

    @property
    def records(self) -> tuple[TaskProgramExpansionRecord, ...]:
        """Return expansion records in planning-call order."""
        return tuple(record.snapshot() for record in self._records)

    def prepare_planning_context(
        self,
        request: TaskProgramPlanRequest,
        context: PlanningContext,
        *,
        engine: AtomicActionEngine,
    ) -> PlanningContext:
        """Attach the selected call's Affordance branch before Atomic planning."""
        if not isinstance(request, TaskProgramPlanRequest):
            raise TypeError("request must be a TaskProgramPlanRequest")
        if not isinstance(context, PlanningContext):
            raise TypeError("context must be a PlanningContext")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine")
        if request.workflow_id.split("/", 1)[0] != self._program_id:
            raise ValueError("workflow ID does not match profile source_id")
        call_index = request.analysis_call_index
        if call_index >= len(self._recipes[0].call_schedule):
            raise ValueError("workflow call index exceeds the recipe call schedule")
        if call_index not in self._profile.source.affordance_call_indices:
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
                    recipe.call_schedule[call_index].affordance_requested
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
        """Return the transform for calls selected by the episode schedule."""
        if not isinstance(request, TaskProgramPlanRequest):
            raise TypeError("request must be a TaskProgramPlanRequest")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine")
        if request.workflow_id.split("/", 1)[0] != self._program_id:
            raise ValueError("workflow ID does not match profile source_id")
        call_index = request.analysis_call_index
        if call_index not in self._profile.source.trajectory_call_indices:
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
        """Expand one generic Task Program call through the new profile schema."""
        trajectory = plan.joint_trajectory
        if trajectory is None or trajectory.batch_size != context.batch_size:
            raise ValueError("generic expansion requires a matching joint trajectory")
        robot = engine.robot
        joint_names = tuple(robot.joint_names)
        limits = robot.body_data.qpos_limits
        if not isinstance(limits, torch.Tensor):
            raise TypeError("robot qpos_limits must be a tensor")
        if limits.ndim == 3:
            limits = limits[0]
        if limits.shape != (len(joint_names), 2):
            raise ValueError("robot qpos_limits must match the full joint order")
        if self._profile.source.phase_permissions:
            permissions = self._profile.source.phase_permissions
            kinds = self._profile.source.phase_kinds
            declared = {segment.name for segment in plan.segments}
            if declared != set(permissions) or declared != set(kinds):
                raise ValueError(
                    "generic expansion phase declarations must exactly match the plan"
                )
        else:
            permissions, kinds = _combined_phase_declarations(
                plan,
                self._profile.trajectory.spatial.operators,
            )
        invocation_id = resolved.invocation_id or resolved.skill_id
        positions: list[torch.Tensor] = []
        dts: list[torch.Tensor] = []
        row_records: list[dict[str, object]] = []
        row_specs: list[tuple[CandidateSpec, ...]] = []
        row_templates: list[tuple[TrajectoryTemplate, ...]] = []
        for row in range(context.batch_size):
            recipe = self._recipe_for_row(row)
            qpos = context.robot.qpos[row]
            digest = hashlib.sha256(
                qpos.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            case = SceneCase(
                scene_case_id=self._integration_id,
                initial_state_id="initial_" + digest,
                scene_signature=f"{self._integration_id}:scene:{context.scene.version}",
                task_id=self._program_id,
                robot_profile_id=self._robot_profile_id,
            )
            payload = {
                "source": {
                    "kind": "task_program",
                    "source_id": self._program_id,
                    "source_revision": self._profile.source.source_revision,
                    "unit_scope": "action",
                    "template_id": self._profile.source.template_id,
                    "phase_permissions": permissions,
                    "phase_kinds": kinds,
                },
                "augmentation": self._augmentation_payload(
                    recipe.visual_seed % (2**63)
                ),
                "affordance": {"enabled": False},
                "scheduling": {
                    "candidate_budget": self._profile.trajectory.variants_per_family,
                },
                "execution": {"max_inflight": 1},
                "observation": {"enabled": False, "profiles": []},
            }
            expansion_cfg = TrajectoryExpansionCfg.from_mapping(
                {key: value for key, value in payload.items() if key != "source"}
            )
            session = ExpansionSession(expansion_cfg)
            session.register_case(case, limits, joint_names=joint_names)
            source_context = SourceContext(
                source_id=self._program_id,
                source_revision=self._profile.source.source_revision,
                unit_id=(f"{recipe.candidate_id}:{invocation_id}:row:{row}"),
                scene_case=case,
                control_dt=context.require_control_dt(),
            )
            canonical_qpos, canonical_dt, boundaries, restore = _canonical_samples(
                trajectory.positions[row],
                trajectory.dt[row],
            )
            phases = tuple(
                TrajectoryPhase(
                    segment.name,
                    int(boundaries[segment.start]),
                    int(boundaries[segment.stop]),
                    kind=kinds[segment.name],
                    allowed_operators=(
                        tuple(permissions[segment.name])
                        if int(boundaries[segment.stop] - boundaries[segment.start])
                        >= 3
                        else ()
                    ),
                )
                for segment in plan.segments
            )
            template = TrajectoryTemplate(
                source_id=self._program_id,
                source_revision=self._profile.source.source_revision,
                template_id=source_context.unit_id,
                joint_names=joint_names,
                positions=canonical_qpos,
                dt=canonical_dt,
                phases=phases,
                allowed_operators=tuple(
                    dict.fromkeys(
                        operator
                        for phase in phases
                        for operator in phase.allowed_operators
                    )
                ),
                validator_id="task_success",
                controlled_joint_indices=(
                    _free_joint_indices(
                        canonical_qpos, phases, _motion_joint_indices(resolved)
                    )
                    if plan.skill_id == "hand_over"
                    else _motion_joint_indices(resolved)
                ),
            )
            coordinator = CandidateCoordinator(
                expansion_cfg,
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
                raise ValueError(f"generic expansion row {row} produced no candidates")
            call_index = call_request.analysis_call_index
            if call_index >= len(recipe.call_schedule):
                raise ValueError("workflow call index exceeds the recipe call schedule")
            requested = recipe.call_schedule[call_index].trajectory_requested
            if requested >= len(items):
                raise ValueError(
                    f"Recipe {recipe.recipe_index} requests trajectory variant {requested}, "
                    f"but only {len(items)} survived for call {call_index}"
                )
            selected_index = requested
            selected = items[selected_index]
            restored = (
                trajectory.positions[row].clone()
                if selected_index == 0
                else selected.template.positions[restore].to(trajectory.positions)
            )
            # Protected samples and gripper columns retain their exact source
            # values, including near-equal zero-time joins.
            motion_ids = template.controlled_joint_indices
            protected_ids = [
                index for index in range(len(joint_names)) if index not in motion_ids
            ]
            restored[:, protected_ids] = trajectory.positions[row, :, protected_ids]
            for segment in plan.segments:
                restored[segment.start] = trajectory.positions[row, segment.start]
                restored[segment.stop - 1] = trajectory.positions[row, segment.stop - 1]
                if not permissions[segment.name]:
                    restored[segment.start : segment.stop] = trajectory.positions[
                        row, segment.start : segment.stop
                    ]
            positions.append(restored)
            dts.append(trajectory.dt[row].clone())
            row_specs.append(tuple(item.spec for item in items))
            row_templates.append(tuple(item.template for item in items))
            row_records.append(
                {
                    "row_index": row,
                    "recipe_index": recipe.recipe_index,
                    "candidate_id": selected.spec.identity.candidate_id,
                    "trajectory_requested": requested,
                    "trajectory_selected": selected_index,
                    "fallback_count": max(0, requested - selected_index),
                    "affordance_requested": (
                        recipe.call_schedule[call_index].affordance_requested
                        if call_index in self._profile.source.affordance_call_indices
                        else None
                    ),
                }
            )
        metadata = dict(plan.diagnostics.metadata)
        metadata["expansion"] = {"rows": row_records}
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
                dt=torch.stack(dts),
            ),
        )
        for row, (specs, templates, record) in enumerate(
            zip(row_specs, row_templates, row_records, strict=True)
        ):
            self._records.append(
                TaskProgramExpansionRecord(
                    workflow_id=call_request.workflow_id,
                    workflow_call_index=call_request.analysis_call_index,
                    candidate_index=int(record["trajectory_selected"]),
                    selected_candidate_id=str(record["candidate_id"]),
                    candidates=specs,
                    templates=templates,
                    recipe_index=int(record["recipe_index"]),
                    cycle_index=call_request.analysis_call_index // 2,
                    affordance_requested=record["affordance_requested"],
                    trajectory_requested=int(record["trajectory_requested"]),
                    trajectory_selected=int(record["trajectory_selected"]),
                    fallback_count=int(record["fallback_count"]),
                    env_id=int(context.env_ids[row].item()),
                )
            )
        return rebuilt
