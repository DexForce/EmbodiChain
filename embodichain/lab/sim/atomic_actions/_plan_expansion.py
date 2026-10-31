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

"""Shared same-grid Atomic plan expansion for direct and Task Program hosts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateCoordinator,
    CandidateWorkItem,
    ExpansionSession,
    SourceContext,
    TemplateSourceAdapter,
    TrajectoryExpansionCfg,
    TrajectoryTemplate,
    validate_motion_limits,
)
from ._expansion_phases import FREE_PHASES
from .plans import ActionPlan, TimedTrajectory
from .core import PlannerDiagnostics
from .trajectory_adapter import ActionPlanTemplateAdapter

if TYPE_CHECKING:
    from .engine import AtomicActionEngine
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext

__all__: list[str] = []


@dataclass(frozen=True)
class _PlanExpansionSpec:
    """Logical candidate request, independent of its physical batch row."""

    source_context: SourceContext
    cfg: TrajectoryExpansionCfg
    requested_index: int
    count: int
    metadata: Mapping[str, object]
    kinematics_factory: Callable[[TrajectoryTemplate], Callable] | None = None
    protect_derivative_stencils: bool = False
    geometry_family_id: str | None = None
    velocity_limits: torch.Tensor | None = None
    acceleration_limits: torch.Tensor | None = None
    label_proposal_ordinal: bool = False
    max_generation_attempts: int | None = None
    editable_joint_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _PlanExpansionResult:
    """Accepted candidates and actual row-local lineage."""

    row_index: int
    metadata: Mapping[str, object]
    items: tuple[CandidateWorkItem, ...]


def _expand_plan(
    engine: AtomicActionEngine,
    request: ResolvedActionRequest,
    context: PlanningContext,
    plan: ActionPlan,
    specs: Sequence[_PlanExpansionSpec],
    *,
    phase_permissions: Mapping[str, Sequence[str]] | None = None,
    phase_kinds: Mapping[str, str] | None = None,
) -> tuple[ActionPlan, tuple[_PlanExpansionResult, ...]]:
    """Adapt, expand, restore and rebuild a plan on its exact original grid."""
    trajectory = plan.joint_trajectory
    if trajectory is None or trajectory.batch_size != len(specs):
        raise ValueError(
            "expansion requires one logical request per joint trajectory row"
        )
    joint_names = tuple(engine.robot.joint_names)
    limits = engine.robot.body_data.qpos_limits
    if not isinstance(limits, torch.Tensor):
        raise TypeError("robot qpos_limits must be a tensor")
    if limits.ndim == 2:
        limits = limits.unsqueeze(0)
    if limits.shape[-2:] != (len(joint_names), 2) or limits.shape[0] not in (
        1,
        len(specs),
    ):
        raise ValueError("robot qpos_limits must match the full joint order and batch")
    motion_ids = tuple(
        dict.fromkeys(
            joint_id
            for endpoint in request.binding.endpoints
            if endpoint.endpoint_id == "motion"
            for joint_id in endpoint.joint_ids
        )
    )
    if not motion_ids:
        raise ValueError("trajectory expansion requires explicitly bound motion joints")
    names = {segment.name for segment in plan.segments}
    if phase_permissions is not None and names != set(phase_permissions):
        raise ValueError("phase declarations must exactly match the Atomic plan")
    if phase_kinds is not None and names != set(phase_kinds):
        raise ValueError("phase declarations must exactly match the Atomic plan")
    default_free = FREE_PHASES.get(plan.skill_id)
    if default_free is None:
        if phase_permissions is None and any(spec.requested_index for spec in specs):
            raise ValueError(
                "custom skills require explicit phase declarations for expansion"
            )
        # Registered custom actions retain their trusted explicit source contract.
        default_free = frozenset(phase_permissions or ())
    positions = trajectory.positions.clone()
    results = []
    for row, spec in enumerate(specs):
        record = dict(spec.metadata)
        record.update(
            row_index=row,
            env_id=int(context.env_ids[row]),
            trajectory_requested=spec.requested_index,
            trajectory_selected=0,
            operator="none",
            phases=[],
        )
        if not bool(plan.plan_success[row]):
            record["status"] = "planning_failed"
            results.append(_PlanExpansionResult(row, record, ()))
            continue
        factors = spec.cfg.augmentation.factors
        operators = (
            tuple(factors.spatial.method) if factors.spatial.enabled else ()
        ) + (("task_space_residual",) if factors.task_space.enabled else ())
        if not operators and spec.requested_index:
            raise ValueError(
                f"variant {record.get('variant_id', record.get('recipe_index', row))} "
                f"requests trajectory variant {spec.requested_index}, but only 1 survived "
                f"for {plan.skill_id}"
            )
        source_permissions = plan.diagnostics.metadata.get(
            "prototype_phase_permissions"
        )
        if source_permissions is not None and (
            not isinstance(source_permissions, Mapping)
            or (source_permissions and set(source_permissions) != names)
        ):
            raise ValueError("prototype phase permissions must match the action plan")
        permissions = {
            name: (
                tuple(
                    operator
                    for operator in (
                        phase_permissions[name]
                        if phase_permissions is not None
                        else operators
                    )
                    if operator in operators
                    and (
                        source_permissions is None
                        or operator in source_permissions.get(name, ())
                    )
                )
                if name in default_free
                and (phase_kinds is None or phase_kinds[name] == "free")
                else ()
            )
            for name in names
        }
        kinds = {name: "free" if permissions[name] else "contact" for name in names}
        selected_joint_ids = (
            motion_ids
            if spec.editable_joint_indices is None
            else spec.editable_joint_indices
        )
        if (
            not selected_joint_ids
            or len(set(selected_joint_ids)) != len(selected_joint_ids)
            or set(selected_joint_ids) - set(motion_ids)
        ):
            raise ValueError(
                "editable joint indices must be a nonempty subset of the motion endpoint"
            )
        adapter = ActionPlanTemplateAdapter(
            joint_names=joint_names,
            phase_permissions=permissions,
            phase_kinds=kinds,
            controlled_joint_indices=selected_joint_ids,
        )
        template = adapter.export_template(
            plan, context=spec.source_context, row_index=row
        )
        if spec.protect_derivative_stencils:
            phases = tuple(
                (
                    replace(
                        phase,
                        start_index=phase.start_index + 1,
                        stop_index=phase.stop_index - 1,
                    )
                    if phase.allowed_operators
                    and phase.stop_index - phase.start_index >= 5
                    else replace(phase, allowed_operators=())
                )
                for phase in template.phases
            )
            template = replace(
                template,
                phases=phases,
                allowed_operators=tuple(
                    dict.fromkeys(
                        operator
                        for phase in phases
                        for operator in phase.allowed_operators
                    )
                ),
            )
        if len(template.dt) < 2:
            record["status"] = "protected"
            results.append(_PlanExpansionResult(row, record, ()))
            continue
        editable = any(phase.allowed_operators for phase in template.phases)
        if source_permissions is not None and spec.requested_index and not editable:
            raise ValueError(
                "prototype does not authorize the requested trajectory expansion"
            )
        requested = spec.requested_index if editable else 0
        count = spec.count if editable else 1
        cfg = spec.cfg
        if not editable or set(operators) != set(template.allowed_operators):
            payload = cfg.to_dict()
            spatial = payload["augmentation"]["factors"]["spatial"]
            selected_spatial = [
                operator
                for operator in template.allowed_operators
                if operator in factors.spatial.method
            ]
            if not editable or not selected_spatial:
                spatial["enabled"] = False
            else:
                spatial["method"] = selected_spatial
            if "task_space_residual" not in template.allowed_operators:
                payload["augmentation"]["factors"]["task_space"]["enabled"] = False
            cfg = TrajectoryExpansionCfg.from_mapping(payload)
        row_limits = limits[0 if limits.shape[0] == 1 else row].detach().cpu()
        session = ExpansionSession(cfg)
        session.register_case(
            spec.source_context.scene_case, row_limits, joint_names=joint_names
        )
        coordinator = CandidateCoordinator(
            cfg,
            session=session,
            source_adapter=TemplateSourceAdapter(),
            source_context=spec.source_context,
            joint_limits=row_limits,
            backend_id=plan.diagnostics.backend,
            kinematics=(
                None
                if spec.kinematics_factory is None
                else spec.kinematics_factory(template)
            ),
            geometry_family_id=spec.geometry_family_id,
            velocity_limits=spec.velocity_limits,
            acceleration_limits=spec.acceleration_limits,
            label_proposal_ordinal=spec.label_proposal_ordinal,
            max_generation_attempts=spec.max_generation_attempts,
        )
        items = coordinator.enqueue_source(template, count=count)
        statistics = coordinator.last_generation_statistics
        if requested >= len(items):
            baseline = ""
            if (
                spec.velocity_limits is not None
                and spec.acceleration_limits is not None
            ):
                check = validate_motion_limits(
                    template,
                    velocity_limits=spec.velocity_limits,
                    acceleration_limits=spec.acceleration_limits,
                ).checks[0]
                baseline = f"; nominal_motion_limits={check.status}, ratios={dict(check.metrics)}"
            raise ValueError(
                f"variant {record.get('variant_id', record.get('recipe_index', row))} "
                f"requests trajectory variant {requested}, but only {len(items)} survived "
                f"for {plan.skill_id}; generation={statistics}{baseline}"
            )
        selected = items[requested]
        if requested:
            positions[row] = adapter.restore_positions(
                plan, selected.template.positions, row_index=row
            )
        record.update(
            trajectory_selected=requested,
            operator=selected.spec.trajectory_variant.get("spatial_operator", "none"),
            candidate_id=selected.spec.identity.candidate_id,
            phases=(
                [phase.phase_id for phase in template.phases if phase.allowed_operators]
                if requested
                else []
            ),
            fallback_count=0,
            generation=statistics,
        )
        results.append(_PlanExpansionResult(row, record, items))
    metadata = dict(plan.diagnostics.metadata)
    if any(result.metadata.get("trajectory_selected", 0) for result in results):
        # Source-path evidence does not certify newly perturbed samples.
        metadata["source_collision_evidence"] = metadata.get(
            "collision_evidence", "unverified"
        )
        metadata["collision_evidence"] = "unverified"
        metadata["collision_scope"] = "not_checked"
    metadata["expansion"] = {
        "rows": [dict(result.metadata) for result in results if result.items]
    }
    source = replace(
        plan,
        diagnostics=PlannerDiagnostics(
            backend=plan.diagnostics.backend,
            messages=plan.diagnostics.messages,
            metadata=metadata,
            failure=plan.diagnostics.failure,
        ),
    )
    rebuilt = engine.rebuild_plan_from_trajectory(
        request,
        context,
        source,
        TimedTrajectory.from_positions(
            positions, env_ids=context.env_ids, dt=trajectory.dt.clone()
        ),
    )
    return rebuilt, tuple(results)
