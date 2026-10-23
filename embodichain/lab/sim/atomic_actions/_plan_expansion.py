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

from collections.abc import Mapping, Sequence
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
        operators = tuple(spec.cfg.augmentation.factors.spatial.method)
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
                )
                if name in default_free
                and (phase_kinds is None or phase_kinds[name] == "free")
                else ()
            )
            for name in names
        }
        kinds = {name: "free" if permissions[name] else "contact" for name in names}
        adapter = ActionPlanTemplateAdapter(
            joint_names=joint_names,
            phase_permissions=permissions,
            phase_kinds=kinds,
            controlled_joint_indices=motion_ids,
        )
        template = adapter.export_template(
            plan, context=spec.source_context, row_index=row
        )
        if len(template.dt) < 2:
            record["status"] = "protected"
            results.append(_PlanExpansionResult(row, record, ()))
            continue
        editable = any(phase.allowed_operators for phase in template.phases)
        requested = spec.requested_index if editable else 0
        count = spec.count if editable else 1
        cfg = spec.cfg
        if not editable or set(operators) != set(template.allowed_operators):
            payload = cfg.to_dict()
            spatial = payload["augmentation"]["factors"]["spatial"]
            if not editable:
                spatial["enabled"] = False
            else:
                spatial["method"] = list(template.allowed_operators)
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
        )
        items = coordinator.enqueue_source(template, count=count)
        if requested >= len(items):
            raise ValueError(
                f"variant {record.get('variant_id', record.get('recipe_index', row))} "
                f"requests trajectory variant {requested}, but only {len(items)} survived "
                f"for {plan.skill_id}"
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
        )
        results.append(_PlanExpansionResult(row, record, items))
    metadata = dict(plan.diagnostics.metadata)
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
