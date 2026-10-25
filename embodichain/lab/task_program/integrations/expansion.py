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

from embodichain.lab.sim.atomic_skills import (
    SkillPlan,
    SkillPlanTemplateAdapter,
    AtomicSkillEngine,
    PlanTransform,
    ResolvedSkillRequest,
)
from embodichain.lab.sim.atomic_skills.affordance_sampling import (
    AffordanceSamplingContext,
)
from embodichain.lab.sim.motion.expansion import (
    CandidateRecipe,
    CandidateSpec,
    CombinedExpansionProfile,
    SceneCase,
    SourceContext,
    TrajectoryExpansionCfg,
    TrajectoryTemplate,
    enumerate_candidate_recipes,
)
from embodichain.lab.sim.atomic_skills.state import PlanningContext
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


@dataclass(frozen=True)
class TaskProgramSourceAdapter:
    """Expose a Task Program's grounded Atomic plan through SourceAdapter."""

    atomic_adapter: SkillPlanTemplateAdapter
    kind: ClassVar[str] = "task_program"

    def __post_init__(self) -> None:
        if not isinstance(self.atomic_adapter, SkillPlanTemplateAdapter):
            raise TypeError("atomic_adapter must be an SkillPlanTemplateAdapter")

    def export_template(
        self,
        source: SkillPlan,
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
        engine: AtomicSkillEngine,
    ) -> PlanningContext:
        """Attach the selected call's Affordance branch before Atomic planning."""
        if not isinstance(request, TaskProgramPlanRequest):
            raise TypeError("request must be a TaskProgramPlanRequest")
        if not isinstance(context, PlanningContext):
            raise TypeError("context must be a PlanningContext")
        if not isinstance(engine, AtomicSkillEngine):
            raise TypeError("engine must be an AtomicSkillEngine")
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
        engine: AtomicSkillEngine,
    ) -> PlanTransform | None:
        """Return the transform for calls selected by the episode schedule."""
        if not isinstance(request, TaskProgramPlanRequest):
            raise TypeError("request must be a TaskProgramPlanRequest")
        if not isinstance(engine, AtomicSkillEngine):
            raise TypeError("engine must be an AtomicSkillEngine")
        if request.workflow_id.split("/", 1)[0] != self._program_id:
            raise ValueError("workflow ID does not match profile source_id")
        call_index = request.analysis_call_index
        if call_index not in self._profile.source.trajectory_call_indices:
            return None

        def transform(
            resolved: ResolvedSkillRequest,
            context: PlanningContext,
            plan: SkillPlan,
        ) -> SkillPlan:
            return self._transform(request, engine, resolved, context, plan)

        return transform

    def _transform(
        self,
        call_request: TaskProgramPlanRequest,
        engine: AtomicSkillEngine,
        resolved: ResolvedSkillRequest,
        context: PlanningContext,
        plan: SkillPlan,
    ) -> SkillPlan:
        """Delegate row-local adaptation and rebuilding to the Atomic layer."""
        from embodichain.lab.sim.atomic_skills._plan_expansion import (
            _PlanExpansionSpec,
            _expand_plan,
        )

        call_index = call_request.analysis_call_index
        specs = []
        for row in range(context.batch_size):
            recipe = self._recipe_for_row(row)
            if call_index >= len(recipe.call_schedule):
                raise ValueError("workflow call index exceeds the recipe call schedule")
            schedule = recipe.call_schedule[call_index]
            digest = hashlib.sha256(
                context.robot.qpos[row].detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            case = SceneCase(
                scene_case_id=self._integration_id,
                initial_state_id="initial_" + digest,
                scene_signature=f"{self._integration_id}:scene:{context.scene.version}",
                task_id=self._program_id,
                robot_profile_id=self._robot_profile_id,
            )
            count = self._profile.trajectory.variants_per_family
            cfg = TrajectoryExpansionCfg.from_mapping(
                {
                    "augmentation": self._augmentation_payload(
                        recipe.visual_seed % (2**63)
                    ),
                    "affordance": {"enabled": False},
                    "scheduling": {"candidate_budget": count},
                    "execution": {"max_inflight": 1},
                    "observation": {"enabled": False, "profiles": []},
                }
            )
            source = SourceContext(
                self._program_id,
                self._profile.source.source_revision,
                f"{recipe.candidate_id}:{resolved.invocation_id or resolved.skill_id}:call:{call_index}",
                case,
                context.require_control_dt(),
            )
            specs.append(
                _PlanExpansionSpec(
                    source,
                    cfg,
                    schedule.trajectory_requested,
                    count,
                    {
                        "recipe_index": recipe.recipe_index,
                        "affordance_requested": (
                            schedule.affordance_requested
                            if call_index
                            in self._profile.source.affordance_call_indices
                            else None
                        ),
                    },
                )
            )
        rebuilt, results = _expand_plan(
            engine,
            resolved,
            context,
            plan,
            specs,
            phase_permissions=self._profile.source.phase_permissions or None,
            phase_kinds=self._profile.source.phase_kinds or None,
        )
        for result in results:
            if not result.items:
                continue
            record = result.metadata
            self._records.append(
                TaskProgramExpansionRecord(
                    workflow_id=call_request.workflow_id,
                    workflow_call_index=call_index,
                    candidate_index=int(record["trajectory_selected"]),
                    selected_candidate_id=str(record["candidate_id"]),
                    candidates=tuple(item.spec for item in result.items),
                    templates=tuple(item.template for item in result.items),
                    recipe_index=int(record["recipe_index"]),
                    cycle_index=call_index // 2,
                    affordance_requested=record["affordance_requested"],
                    trajectory_requested=int(record["trajectory_requested"]),
                    trajectory_selected=int(record["trajectory_selected"]),
                    fallback_count=0,
                    env_id=int(context.env_ids[result.row_index]),
                )
            )
        return rebuilt
