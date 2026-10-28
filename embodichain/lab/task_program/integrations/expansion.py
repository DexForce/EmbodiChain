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
from collections.abc import Callable, Mapping, Sequence
import math
from typing import ClassVar

import torch

from embodichain.data_pipeline.real2sim import TrajectoryTimeMap
from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    ActionPlanTemplateAdapter,
    AtomicActionEngine,
    PlanTransform,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
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
    MappedRetimeResult,
    retime_with_time_map,
    validate_motion_limits,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.task_program.runtime.expansion import TaskProgramPlanRequest

__all__ = [
    "TaskProgramSourceAdapter",
    "TaskProgramExpansionRecord",
    "CombinedTaskProgramExpansionFactory",
    "TaskProgramMappedRetimeFactory",
    "TaskProgramMappedRetimeRecord",
    "TaskProgramSameGridExpansionFactory",
    "TaskProgramSameGridExpansionRecord",
]


def _phase_allowlist(
    value: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("phase_allowlist must map skill IDs to phase names")
    result = {}
    for skill, phases in value.items():
        if (
            not isinstance(skill, str)
            or not skill.strip()
            or isinstance(phases, (str, bytes))
            or not isinstance(phases, Sequence)
        ):
            raise ValueError(
                "phase_allowlist must map skill IDs to phase-name sequences"
            )
        values = tuple(phases)
        if any(not isinstance(name, str) or not name.strip() for name in values) or len(
            set(values)
        ) != len(values):
            raise ValueError("phase_allowlist phase names must be nonempty and unique")
        result[skill] = values
    return result


def _headroom_joint_selection(
    positions: torch.Tensor,
    dt: torch.Tensor,
    *,
    velocity_limits: torch.Tensor,
    acceleration_limits: torch.Tensor,
    joint_ids: tuple[int, ...],
    fraction: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Exclude joints whose nominal finite differences leave little margin."""
    if (
        positions.ndim != 2
        or dt.shape != (positions.shape[0],)
        or positions.shape[0] < 3
    ):
        raise ValueError("headroom selection requires at least three timed samples")
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("motion headroom fraction must be between zero and one")
    q = positions.detach().to(device="cpu", dtype=torch.float64)
    intervals = dt.detach().to(device="cpu", dtype=torch.float64)
    velocity, acceleration = velocity_limits.to(q), acceleration_limits.to(q)
    if (
        velocity.shape != (q.shape[1],)
        or acceleration.shape != (q.shape[1],)
        or not bool(torch.isfinite(q).all())
        or not bool(torch.isfinite(intervals).all())
        or intervals[0] != 0
        or bool((intervals[1:] <= 0).any())
    ):
        raise ValueError(
            "headroom selection requires finite full-joint positions and intervals"
        )
    if (
        not bool(torch.isfinite(velocity).all())
        or not bool(torch.isfinite(acceleration).all())
        or not bool((velocity > 0).all())
        or not bool((acceleration > 0).all())
    ):
        raise ValueError("headroom limits must be positive and finite")
    speed = (q[1:] - q[:-1]) / intervals[1:, None]
    accel = (speed[1:] - speed[:-1]) / ((intervals[2:] + intervals[1:-1])[:, None] / 2)
    ratios = torch.maximum(
        (speed.abs() / velocity).amax(dim=0), (accel.abs() / acceleration).amax(dim=0)
    )
    if (
        not joint_ids
        or len(set(joint_ids)) != len(joint_ids)
        or any(
            type(index) is not int or not 0 <= index < q.shape[1] for index in joint_ids
        )
    ):
        raise ValueError("headroom joints must be unique full-layout indices")
    editable = tuple(index for index in joint_ids if float(ratios[index]) < fraction)
    locked = tuple(index for index in joint_ids if index not in editable)
    if not editable:
        raise ValueError(
            "no authorized motion joint has the requested derivative headroom"
        )
    return editable, locked


@dataclass(frozen=True)
class TaskProgramSameGridExpansionRecord:
    """Call-level identity and the accepted same-grid expansion template."""

    workflow_id: str
    call_index: int
    operator: str
    seed: int
    candidate_id: str
    geometry_family_id: str
    source_unit_id: str
    template: TrajectoryTemplate
    collision_evidence: str
    variant_ordinal: int
    attempted: int
    rejected: tuple[tuple[str, int], ...]
    locked_joint_indices: tuple[int, ...] = ()
    motion_headroom_fraction: float | None = None


class TaskProgramSameGridExpansionFactory:
    """Apply one episode-owned spatial recipe to selected source-backed calls.

    Existing trajectory operators produce the proposal. The skill and source
    independently restrict editable phases, and the first and last two
    samples of each phase are retained to protect discrete derivative stencils.
    Full-plan rebuilding retains the original time grid. Sampled joint motion
    limits and a host-owned, contact-aware collision validator gate the result.
    An optional ``phase_allowlist`` further restricts eligible phases per skill.

    For task-space deformation, ``kinematics_factory`` receives the resolved
    request, planning context and exported template and must return the
    existing expansion kinematics callback for its controlled joint order.
    An optional ``motion_headroom_fraction`` narrows joint-residual edits to
    joints whose nominal velocity and acceleration ratios are below that
    fraction. The locked joints remain exact and are recorded; candidates
    still pass the unchanged full-path motion and collision checks.
    """

    def __init__(
        self,
        *,
        operator: str,
        seed: int,
        geometry_family_id: str,
        initial_state_id: str,
        velocity_limits: Sequence[float],
        acceleration_limits: Sequence[float],
        call_indices: Sequence[int],
        joint_offset_scale: float = 0.002,
        dedup_tolerance: float = 0.0005,
        max_translation_m: float = 0.005,
        max_generation_attempts: int = 64,
        motion_headroom_fraction: float | None = None,
        phase_allowlist: Mapping[str, Sequence[str]] | None = None,
        kinematics_factory: (
            Callable[
                [ResolvedActionRequest, PlanningContext, TrajectoryTemplate], Callable
            ]
            | None
        ) = None,
        collision_validator: (
            Callable[[ResolvedActionRequest, PlanningContext, ActionPlan], bool] | None
        ) = None,
        diagnostic_unchecked: bool = False,
    ) -> None:
        if operator not in ("joint_residual", "task_space_residual"):
            raise ValueError("unsupported same-grid operator")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative signed 64-bit integer")
        if type(max_generation_attempts) is not int or max_generation_attempts < 2:
            raise ValueError("max_generation_attempts must be at least two")
        if motion_headroom_fraction is not None and (
            isinstance(motion_headroom_fraction, bool)
            or not isinstance(motion_headroom_fraction, (int, float))
            or not math.isfinite(motion_headroom_fraction)
            or not 0 < motion_headroom_fraction < 1
        ):
            raise ValueError("motion_headroom_fraction must be between zero and one")
        if motion_headroom_fraction is not None and operator != "joint_residual":
            raise ValueError(
                "motion headroom selection applies only to joint residuals"
            )
        if not isinstance(geometry_family_id, str) or not geometry_family_id.strip():
            raise ValueError(
                "geometry_family_id must name the episode's geometry baseline"
            )
        if not isinstance(initial_state_id, str) or not initial_state_id.strip():
            raise ValueError("initial_state_id must name the episode's initial state")
        indices = tuple(call_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(index) is not int or index < 0 for index in indices)
        ):
            raise ValueError("call_indices must contain unique nonnegative indices")
        if type(diagnostic_unchecked) is not bool:
            raise TypeError("diagnostic_unchecked must be a boolean")
        if collision_validator is not None and not callable(collision_validator):
            raise TypeError("collision_validator must be callable")
        if collision_validator is None and not diagnostic_unchecked:
            raise ValueError(
                "a phase-aware collision validator is required outside diagnostics"
            )
        if operator == "task_space_residual" and not callable(kinematics_factory):
            raise ValueError("task-space expansion requires a kinematics factory")
        if kinematics_factory is not None and not callable(kinematics_factory):
            raise TypeError("kinematics_factory must be callable")
        self._velocity_limits = TaskProgramMappedRetimeFactory._limits(
            velocity_limits, "velocity_limits"
        )
        self._acceleration_limits = TaskProgramMappedRetimeFactory._limits(
            acceleration_limits, "acceleration_limits"
        )
        self._cfg = TrajectoryExpansionCfg.from_mapping(
            {
                "augmentation": {
                    "seed": seed,
                    "max_variants_per_reference": 2,
                    "factors": {
                        "spatial": {
                            "enabled": operator == "joint_residual",
                            "method": ["joint_residual"],
                            "joint_offset_scale": joint_offset_scale,
                        },
                        "task_space": {
                            "enabled": operator == "task_space_residual",
                            "max_translation_m": max_translation_m,
                            "max_rotation_rad": 0.0,
                        },
                    },
                    "coverage": {"joint_dedup_normalized_tol": dedup_tolerance},
                },
                "affordance": {"enabled": False},
                "scheduling": {"candidate_budget": 2},
                "execution": {"max_inflight": 1},
                "observation": {"enabled": False, "profiles": []},
            }
        )
        self._operator = operator
        self._seed = seed
        self._geometry_family_id = geometry_family_id
        self._initial_state_id = initial_state_id
        self._max_generation_attempts = max_generation_attempts
        self._motion_headroom_fraction = motion_headroom_fraction
        self._phase_allowlist = _phase_allowlist(phase_allowlist)
        self._call_indices = frozenset(indices)
        self._kinematics_factory = kinematics_factory
        self._collision_validator = collision_validator
        self._records: list[TaskProgramSameGridExpansionRecord] = []

    @property
    def records(self) -> tuple[TaskProgramSameGridExpansionRecord, ...]:
        """Return successfully rebuilt call-level candidates."""
        return tuple(
            replace(record, template=replace(record.template))
            for record in self._records
        )

    def create_plan_transform(
        self, request: TaskProgramPlanRequest, *, engine: AtomicActionEngine
    ) -> PlanTransform | None:
        """Select the spatial transform for one compiled call occurrence."""
        if not isinstance(request, TaskProgramPlanRequest) or not isinstance(
            engine, AtomicActionEngine
        ):
            raise TypeError("request and engine must be Task Program and Atomic values")
        if request.analysis_call_index not in self._call_indices:
            return None

        def transform(
            resolved: ResolvedActionRequest, context: PlanningContext, plan: ActionPlan
        ) -> ActionPlan:
            from embodichain.lab.sim.atomic_actions._plan_expansion import (
                _PlanExpansionSpec,
                _expand_plan,
            )

            if context.batch_size != 1 or plan.joint_trajectory is None:
                raise ValueError(
                    "same-grid Task Program expansion requires a B=1 trajectory"
                )
            names = tuple(engine.robot.joint_names)
            if self._velocity_limits.shape != (
                len(names),
            ) or self._acceleration_limits.shape != (len(names),):
                raise ValueError("expansion limits must match the full robot layout")
            source = plan.diagnostics.metadata
            source_id, revision, unit = (
                source.get("source_id"),
                source.get("source_revision"),
                source.get("unit_id"),
            )
            if not all(
                isinstance(value, str) and value
                for value in (source_id, revision, unit)
            ) or not isinstance(source.get("prototype_phase_permissions"), Mapping):
                raise ValueError(
                    "same-grid expansion requires a source-backed plan and phase permissions"
                )
            case = SceneCase(
                self._geometry_family_id,
                self._initial_state_id,
                self._geometry_family_id,
                request.workflow_id.split("/", 1)[0],
                "task_program_robot",
            )
            source_context = SourceContext(
                source_id, revision, unit, case, context.require_control_dt()
            )
            if self._phase_allowlist is not None and set(
                self._phase_allowlist.get(plan.skill_id, ())
            ) - {segment.name for segment in plan.segments}:
                raise ValueError(
                    "phase_allowlist contains a phase absent from the action plan"
                )
            kinematics = (
                None
                if self._kinematics_factory is None
                else lambda template: self._kinematics_factory(
                    resolved, context, template
                )
            )
            velocity, acceleration = (
                self._velocity_limits.clone(),
                self._acceleration_limits.clone(),
            )
            motion_ids = tuple(
                dict.fromkeys(
                    index
                    for endpoint in resolved.binding.endpoints
                    if endpoint.endpoint_id == "motion"
                    for index in endpoint.joint_ids
                )
            )
            if resolved.motion_policy.velocity_limit is not None:
                velocity[list(motion_ids)] = velocity[list(motion_ids)].clamp(
                    max=resolved.motion_policy.velocity_limit
                )
            if resolved.motion_policy.acceleration_limit is not None:
                acceleration[list(motion_ids)] = acceleration[list(motion_ids)].clamp(
                    max=resolved.motion_policy.acceleration_limit
                )
            editable_ids, locked_ids = motion_ids, ()
            if self._motion_headroom_fraction is not None:
                editable_ids, locked_ids = _headroom_joint_selection(
                    plan.joint_trajectory.positions[0],
                    plan.joint_trajectory.dt[0],
                    velocity_limits=velocity,
                    acceleration_limits=acceleration,
                    joint_ids=motion_ids,
                    fraction=self._motion_headroom_fraction,
                )
            rebuilt, rows = _expand_plan(
                engine,
                resolved,
                context,
                plan,
                (
                    _PlanExpansionSpec(
                        source_context,
                        self._cfg,
                        1,
                        2,
                        {"recipe_index": self._seed},
                        kinematics_factory=kinematics,
                        protect_derivative_stencils=True,
                        geometry_family_id=self._geometry_family_id,
                        velocity_limits=velocity,
                        acceleration_limits=acceleration,
                        label_proposal_ordinal=True,
                        max_generation_attempts=self._max_generation_attempts,
                        editable_joint_indices=editable_ids,
                    ),
                ),
                phase_permissions=(
                    None
                    if self._phase_allowlist is None
                    else {
                        segment.name: (
                            (self._operator,)
                            if segment.name
                            in self._phase_allowlist.get(plan.skill_id, ())
                            else ()
                        )
                        for segment in plan.segments
                    }
                ),
            )
            if len(rows) != 1 or len(rows[0].items) < 2:
                raise ValueError(
                    "same-grid expansion produced no non-nominal candidate"
                )
            if locked_ids and not torch.equal(
                rebuilt.joint_trajectory.positions[:, :, list(locked_ids)],
                plan.joint_trajectory.positions[:, :, list(locked_ids)],
            ):
                raise ValueError("same-grid expansion changed a headroom-locked joint")
            selected = rows[0].items[1]
            if not validate_motion_limits(
                selected.template,
                velocity_limits=velocity,
                acceleration_limits=acceleration,
            ).accepted:
                raise ValueError("same-grid expansion violates sampled motion limits")
            evidence = "unverified"
            if self._collision_validator is not None:
                accepted = self._collision_validator(resolved, context, rebuilt)
                if type(accepted) is not bool or not accepted:
                    raise ValueError("expanded plan failed host collision validation")
                evidence = "host_verified"
            metadata = dict(rebuilt.diagnostics.metadata)
            metadata["collision_evidence"] = evidence
            metadata["collision_scope"] = (
                "host_declared" if evidence == "host_verified" else "not_checked"
            )
            metadata["same_grid_expansion"] = {
                "operator": self._operator,
                "seed": self._seed,
                "call_index": request.analysis_call_index,
                "collision_evidence": evidence,
                "generation": rows[0].metadata["generation"],
                "locked_joint_indices": locked_ids,
                "motion_headroom_fraction": self._motion_headroom_fraction,
            }
            rebuilt = replace(
                rebuilt, diagnostics=replace(rebuilt.diagnostics, metadata=metadata)
            )
            self._records.append(
                TaskProgramSameGridExpansionRecord(
                    request.workflow_id,
                    request.analysis_call_index,
                    self._operator,
                    self._seed,
                    selected.spec.identity.candidate_id,
                    selected.spec.identity.geometry_family_id,
                    unit,
                    selected.template,
                    evidence,
                    int(selected.spec.trajectory_variant["variant_ordinal"]),
                    int(rows[0].metadata["generation"]["attempted"]),
                    tuple(sorted(rows[0].metadata["generation"]["rejected"].items())),
                    locked_ids,
                    self._motion_headroom_fraction,
                )
            )
            return rebuilt

        return transform


@dataclass(frozen=True)
class TaskProgramMappedRetimeRecord:
    """One call's plan-grid mapping; runtime holds are recorded separately.

    The map relates the rebuilt command plan to its nominal plan. It does not
    claim that each environment step advanced the plan: effect gates and
    tracking can introduce additional submitted hold commands.
    """

    workflow_id: str
    call_index: int
    duration_scale: float
    source_unit_id: str
    source_sample_offset: int
    result: MappedRetimeResult
    collision_evidence: str


class TaskProgramMappedRetimeFactory:
    """Retime selected external-source calls through complete plan rebuilding.

    This is a programmatic, B=1 integration. The episode owner chooses the
    scale and eligible call occurrences once; the factory intersects each
    prototype's permissions with its skill's free phases. A host collision
    validator must understand the whole rebuilt plan, including intended
    contacts and attached objects. Omitting it is permitted only for explicit
    diagnostics and never certifies a candidate for recording.

    Args:
        duration_scale: Scale for eligible phase interiors.
        velocity_limits: Explicit positive finite full-layout speed limits.
        acceleration_limits: Explicit positive finite full-layout acceleration
            limits; the motion policy may further narrow them.
        call_indices: Compiled call occurrences to retime.
        phase_allowlist: Optional per-skill phase names narrowing source and skill permissions.
        collision_validator: Host-owned validator for the complete rebuilt plan.
        diagnostic_unchecked: Permit execution without that validator, recording
            its collision status as unverified.
    """

    def __init__(
        self,
        *,
        duration_scale: float,
        velocity_limits: Sequence[float],
        acceleration_limits: Sequence[float],
        call_indices: Sequence[int],
        phase_allowlist: Mapping[str, Sequence[str]] | None = None,
        collision_validator: (
            Callable[[ResolvedActionRequest, PlanningContext, ActionPlan], bool] | None
        ) = None,
        diagnostic_unchecked: bool = False,
    ) -> None:
        if (
            isinstance(duration_scale, bool)
            or not isinstance(duration_scale, (int, float))
            or not math.isfinite(duration_scale)
            or duration_scale <= 0
        ):
            raise ValueError("duration_scale must be finite and positive")
        self._velocity_limits = self._limits(velocity_limits, "velocity_limits")
        self._acceleration_limits = self._limits(
            acceleration_limits, "acceleration_limits"
        )
        indices = tuple(call_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(index) is not int or index < 0 for index in indices)
        ):
            raise ValueError("call_indices must contain unique nonnegative indices")
        if type(diagnostic_unchecked) is not bool:
            raise TypeError("diagnostic_unchecked must be a boolean")
        if collision_validator is not None and not callable(collision_validator):
            raise TypeError("collision_validator must be callable")
        if collision_validator is None and not diagnostic_unchecked:
            raise ValueError(
                "a phase-aware collision validator is required outside diagnostics"
            )
        self._duration_scale = float(duration_scale)
        self._phase_allowlist = _phase_allowlist(phase_allowlist)
        self._call_indices = frozenset(indices)
        self._collision_validator = collision_validator
        self._records: list[TaskProgramMappedRetimeRecord] = []

    @staticmethod
    def _limits(values: Sequence[float], name: str) -> torch.Tensor:
        try:
            owned = tuple(values)
            if any(isinstance(value, bool) for value in owned):
                raise ValueError(f"{name} must contain positive finite values")
            result = torch.as_tensor(owned, dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain positive finite values") from error
        if (
            result.ndim != 1
            or not result.numel()
            or not bool(torch.isfinite(result).all())
            or not bool((result > 0).all())
        ):
            raise ValueError(f"{name} must contain positive finite values")
        return result.clone()

    @property
    def records(self) -> tuple[TaskProgramMappedRetimeRecord, ...]:
        """Return the mappings for the successfully rebuilt calls."""
        result = []
        for record in self._records:
            source = record.result.time_map
            mapping = TrajectoryTimeMap(
                source.left,
                source.right,
                source.weight,
                source.source_event_indices,
                source.execution_event_indices,
                source.max_event_error_s,
            )
            result.append(
                replace(
                    record,
                    result=MappedRetimeResult(replace(record.result.template), mapping),
                )
            )
        return tuple(result)

    def create_plan_transform(
        self, request: TaskProgramPlanRequest, *, engine: AtomicActionEngine
    ) -> PlanTransform | None:
        """Select a retiming transform by compiled call occurrence."""
        if not isinstance(request, TaskProgramPlanRequest) or not isinstance(
            engine, AtomicActionEngine
        ):
            raise TypeError("request and engine must be Task Program and Atomic values")
        if request.analysis_call_index not in self._call_indices:
            return None

        def transform(
            resolved: ResolvedActionRequest, context: PlanningContext, plan: ActionPlan
        ) -> ActionPlan:
            from embodichain.lab.sim.atomic_actions._expansion_phases import FREE_PHASES

            if context.batch_size != 1 or plan.joint_trajectory is None:
                raise ValueError(
                    "mapped Task Program retiming requires a B=1 trajectory"
                )
            names = tuple(engine.robot.joint_names)
            if self._velocity_limits.shape != (
                len(names),
            ) or self._acceleration_limits.shape != (len(names),):
                raise ValueError("retiming limits must match the full robot layout")
            source = plan.diagnostics.metadata
            permissions = source.get("prototype_phase_permissions")
            segment_names = {segment.name for segment in plan.segments}
            if (
                self._phase_allowlist is not None
                and set(self._phase_allowlist.get(plan.skill_id, ())) - segment_names
            ):
                raise ValueError(
                    "phase_allowlist contains a phase absent from the action plan"
                )
            if (
                not isinstance(permissions, Mapping)
                or set(permissions) != segment_names
            ):
                raise ValueError("retiming requires exact prototype phase permissions")
            if any(
                isinstance(permissions[name], (str, bytes))
                or not isinstance(permissions[name], (tuple, list))
                or any(not isinstance(value, str) for value in permissions[name])
                for name in segment_names
            ):
                raise ValueError(
                    "prototype phase permissions must contain operator sequences"
                )
            free = FREE_PHASES.get(plan.skill_id, frozenset())
            selected = {
                name
                for name in segment_names
                if name in free
                and "retime" in permissions[name]
                and (
                    self._phase_allowlist is None
                    or name in self._phase_allowlist.get(plan.skill_id, ())
                )
            }
            if not selected:
                raise ValueError("prototype and skill do not authorize retiming")
            motion_ids = tuple(
                dict.fromkeys(
                    index
                    for endpoint in resolved.binding.endpoints
                    if endpoint.endpoint_id == "motion"
                    for index in endpoint.joint_ids
                )
            )
            if not motion_ids:
                raise ValueError("retiming requires a bound motion endpoint")
            adapter = ActionPlanTemplateAdapter(
                joint_names=names,
                phase_permissions={
                    name: (("retime",) if name in selected else ())
                    for name in segment_names
                },
                phase_kinds={
                    name: ("free" if name in selected else "contact")
                    for name in segment_names
                },
                controlled_joint_indices=motion_ids,
            )
            source_id = source.get("source_id")
            revision = source.get("source_revision")
            unit = source.get("unit_id")
            if not all(
                isinstance(item, str) and item for item in (source_id, revision, unit)
            ):
                raise ValueError(
                    "retiming requires the original prototype source identity"
                )
            source_offset = source.get("source_sample_offset", 0)
            if (
                type(source_offset) is not int
                or not 0 <= source_offset < plan.joint_trajectory.waypoint_count
            ):
                raise ValueError("prototype connection offset is invalid")
            digest = hashlib.sha256(
                context.robot.qpos[0].detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            case = SceneCase(
                scene_case_id=request.workflow_id,
                initial_state_id=f"initial_{digest}",
                scene_signature=f"{request.workflow_id}:scene:{context.scene.version}",
                task_id=request.workflow_id.split("/", 1)[0],
                robot_profile_id="task_program_robot",
            )
            template = adapter.export_template(
                plan,
                context=SourceContext(
                    source_id, revision, unit, case, context.require_control_dt()
                ),
            )
            count = plan.joint_trajectory.waypoint_count
            protected = tuple(
                {source_offset}
                | {
                    index
                    for index in plan.scene_dependency_monitor_until.values()
                    if index < count
                }
            )
            mapped = retime_with_time_map(
                template,
                duration_scale=self._duration_scale,
                control_dt=context.require_control_dt(),
                protected_indices=protected,
            )
            limits = engine.robot.body_data.qpos_limits
            if not isinstance(limits, torch.Tensor) or limits.ndim not in (2, 3):
                raise ValueError("robot joint position limits are unavailable")
            limits = limits[0] if limits.ndim == 3 else limits
            velocity = self._velocity_limits.clone()
            acceleration = self._acceleration_limits.clone()
            if resolved.motion_policy.velocity_limit is not None:
                velocity[list(motion_ids)] = velocity[list(motion_ids)].clamp(
                    max=resolved.motion_policy.velocity_limit
                )
            if resolved.motion_policy.acceleration_limit is not None:
                acceleration[list(motion_ids)] = acceleration[list(motion_ids)].clamp(
                    max=resolved.motion_policy.acceleration_limit
                )
            rebuilt = engine.rebuild_plan_from_mapped_trajectory(
                resolved,
                context,
                plan,
                template,
                mapped,
                joint_limits=limits,
                velocity_limits=velocity,
                acceleration_limits=acceleration,
            )
            evidence = "unverified"
            if self._collision_validator is not None:
                accepted = self._collision_validator(resolved, context, rebuilt)
                if type(accepted) is not bool or not accepted:
                    raise ValueError("rebuilt plan failed host collision validation")
                evidence = "host_verified"
            metadata = dict(rebuilt.diagnostics.metadata)
            metadata["source_collision_evidence"] = metadata.get(
                "collision_evidence", "unverified"
            )
            metadata["collision_evidence"] = evidence
            metadata["collision_scope"] = (
                "host_declared" if evidence == "host_verified" else "not_checked"
            )
            metadata["mapped_retime"] = {
                "duration_scale": self._duration_scale,
                "call_index": request.analysis_call_index,
                "collision_evidence": evidence,
            }
            rebuilt = replace(
                rebuilt, diagnostics=replace(rebuilt.diagnostics, metadata=metadata)
            )
            self._records.append(
                TaskProgramMappedRetimeRecord(
                    request.workflow_id,
                    request.analysis_call_index,
                    self._duration_scale,
                    unit,
                    source_offset,
                    mapped,
                    evidence,
                )
            )
            return rebuilt

        return transform


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
        if profile.trajectory.timing.enabled:
            raise ValueError(
                "combined Task Program timing recipes are not wired; use the "
                "explicit mapped retime factory instead"
            )
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
        """Delegate row-local adaptation and rebuilding to the Atomic layer."""
        from embodichain.lab.sim.atomic_actions._plan_expansion import (
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
