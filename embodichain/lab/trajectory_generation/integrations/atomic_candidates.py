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

"""Bounded, planning-only affordance branches over real replica rows.

The source compiles independent branches; it never steps physics or declares a
grasp successful in the real world. Contact-aware validation and collection
remain separate host responsibilities.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
import math
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    GenerationSession,
    MotionSnapshot,
    TrajectoryGenerationJobCfg,
    TrajectoryPhase,
    ValidationCheck,
    ValidationResult,
)
from embodichain.toolkits.graspkit import GraspCandidateBatch
from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        AtomicActionEngine,
        CompiledTrajectory,
        PlanningContext,
    )
    from embodichain.lab.trajectory_generation.replicas import (
        CandidateSlotAssignment,
        SceneReplicaPool,
    )

__all__ = [
    "AtomicCandidateGenerationCfg",
    "AtomicCandidateRejection",
    "AtomicGenerationResult",
    "AtomicTrajectoryGenerator",
]


@configclass
class AtomicCandidateGenerationCfg:
    """Limits for the initial single-PickUp affordance source.

    Counts are upper bounds, not promises of feasible output. Only real
    environment-row batching is supported. Alternate IK seeds, resampling,
    held-object suffixes and independent candidate-capacity backends are
    deliberately rejected until they have separate qualification.
    """

    max_grasps_per_case: int = 16
    max_output_trajectories: int = 8
    max_proposals: int = 64
    max_waypoints: int = 10000
    max_output_bytes: int = 268435456
    max_rejection_records: int = 256
    max_wall_time_s: float = 120.0
    variants: str = "feasible_rolls"
    batch_mode: str = "env_rows"
    pool_mode: str = "grouped_replicas"
    ik_max_attempts_per_candidate: int = 1
    grasp_resample_rounds: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "max_grasps_per_case",
            "max_output_trajectories",
            "max_proposals",
            "max_waypoints",
            "max_output_bytes",
            "max_rejection_records",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if (
            type(self.max_wall_time_s) not in (int, float)
            or not math.isfinite(self.max_wall_time_s)
            or self.max_wall_time_s <= 0
        ):
            raise ValueError("max_wall_time_s must be positive and finite")
        if self.max_output_trajectories > self.max_proposals:
            raise ValueError("max_proposals cannot be smaller than the output target")
        if self.variants not in ("original", "feasible_rolls"):
            raise ValueError("variants must be original or feasible_rolls")
        if self.batch_mode != "env_rows" or self.pool_mode != "grouped_replicas":
            raise ValueError("only env_rows / grouped_replicas is implemented")
        if (
            type(self.ik_max_attempts_per_candidate) is not int
            or self.ik_max_attempts_per_candidate != 1
        ):
            raise ValueError("alternate IK attempts are not implemented; use 1")
        if (
            type(self.grasp_resample_rounds) is not int
            or self.grasp_resample_rounds != 0
        ):
            raise ValueError("automatic grasp resampling is not implemented; use 0")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AtomicCandidateGenerationCfg:
        """Decode source settings without accepting unknown keys.

        Args:
            values: Flat source configuration mapping.

        Returns:
            Validated configuration.
        """
        if not isinstance(values, Mapping):
            raise TypeError("source configuration must be a mapping")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown atomic candidate settings: {sorted(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True)
class AtomicCandidateRejection:
    """Lightweight first-failure audit; no failed trajectory is retained."""

    source_index: int
    grasp_id: str | None
    variant_id: int | None
    candidate_id: str | None
    invocation_id: str | None
    stage: str
    reason_code: str
    attempt_id: int = 0


@dataclass(frozen=True)
class AtomicGenerationResult:
    """Compact planning output and bounded audit, never physical success proof.

    ``planning_checks`` and ``branch_metadata`` use exactly the trajectory row
    order. Collision checks exist only when supplied by a trusted validator.
    """

    trajectories: CandidateTrajectoryBatch
    planning_checks: tuple[ValidationResult, ...]
    branch_metadata: tuple[Mapping[str, object], ...]
    rejections: tuple[AtomicCandidateRejection, ...]
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        count = self.trajectories.positions.shape[0]
        if len(self.planning_checks) != count or len(self.branch_metadata) != count:
            raise ValueError("checks and provenance must align with output rows")
        object.__setattr__(self, "planning_checks", tuple(self.planning_checks))
        object.__setattr__(
            self,
            "branch_metadata",
            tuple(MappingProxyType(dict(item)) for item in self.branch_metadata),
        )
        object.__setattr__(self, "rejections", tuple(self.rejections))
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))


def _empty(qpos: torch.Tensor, names: tuple[str, ...]) -> CandidateTrajectoryBatch:
    return CandidateTrajectoryBatch(
        positions=qpos.new_empty((0, 0, len(names))),
        dt=torch.empty((0, 0), device=qpos.device, dtype=torch.float64),
        valid_length=torch.empty(0, device=qpos.device, dtype=torch.int64),
        identities=(),
        joint_names=names,
        source_row_indices=torch.empty(0, dtype=torch.int64, device=qpos.device),
    )


def _pack(
    rows: Sequence[CandidateTrajectoryBatch],
    *,
    qpos: torch.Tensor,
    names: tuple[str, ...],
) -> CandidateTrajectoryBatch:
    if not rows:
        return _empty(qpos, names)
    horizon = max(int(row.valid_length[0]) for row in rows)
    positions = qpos.new_empty((len(rows), horizon, len(names)))
    intervals = torch.zeros(
        (len(rows), horizon), dtype=torch.float64, device=qpos.device
    )
    lengths = torch.empty(len(rows), dtype=torch.int64, device=qpos.device)
    for index, row in enumerate(rows):
        length = int(row.valid_length[0])
        lengths[index] = length
        positions[index, :length] = row.positions[0, :length]
        positions[index, length:] = row.positions[0, length - 1]
        intervals[index, :length] = row.dt[0, :length]
    return CandidateTrajectoryBatch(
        positions,
        intervals,
        lengths,
        tuple(row.identities[0] for row in rows),
        names,
        tuple(row.phases[0] for row in rows),
        tuple(row.factors[0] for row in rows),
        torch.cat([row.source_row_indices for row in rows]),
    )


def _export_row(
    compiled: CompiledTrajectory,
    slot: int,
    identity: CandidateIdentity,
    source_row: int,
    robot: object,
) -> CandidateTrajectoryBatch:
    """Merge zero-time anchors without inventing time; protect contact phases."""
    qpos = compiled.trajectory.positions[slot].clone()
    dt = compiled.trajectory.dt[slot].double().clone()
    if not len(qpos) or dt[0] != 0:
        raise ValueError("compiled sequence must have a zero-time initial sample")
    keep = torch.ones(len(qpos), dtype=torch.bool, device=qpos.device)
    for index in range(1, len(qpos)):
        if dt[index] == 0:
            if not torch.allclose(qpos[index], qpos[index - 1], atol=1e-5, rtol=0):
                raise ValueError("zero-time action boundary is discontinuous")
            keep[index] = False
    # Couplings are command geometry, never separately actuated labels.
    if set(robot.mimic_ids) & set(robot.mimic_parents):
        raise ValueError("candidate export requires direct active-to-mimic coupling")
    for child, parent, multiplier, offset in zip(
        robot.mimic_ids,
        robot.mimic_parents,
        robot.mimic_multipliers,
        robot.mimic_offsets,
    ):
        qpos[:, child] = qpos[:, parent] * multiplier + offset
    phases = []
    offset = 0
    for action_index, plan in enumerate(compiled.action_plans):
        for segment in plan.segments:
            start = int(keep[: offset + segment.start].sum())
            stop = int(keep[: offset + segment.stop].sum())
            if stop > start:
                phases.append(
                    TrajectoryPhase(
                        f"{action_index}:{plan.skill_id}:{segment.name}",
                        start,
                        stop,
                        kind="contact" if plan.skill_id == "pick_up" else "free",
                    )
                )
        offset += plan.joint_trajectory.waypoint_count
    qpos, dt = qpos[keep], dt[keep]
    return CandidateTrajectoryBatch(
        qpos[None],
        dt[None],
        torch.tensor([len(qpos)], device=qpos.device),
        (identity,),
        tuple(robot.joint_names),
        (tuple(phases),),
        source_row_indices=torch.tensor([source_row], device=qpos.device),
    )


class AtomicTrajectoryGenerator:
    """Compile affordance candidates in waves over verified real replica slots.

    The initial supported sequence is optional MoveEndEffector/MoveJoints
    prefixes followed by one PickUp. Prefix targets are caller-owned physical
    E-row goals; candidate enumeration happens after their projected endpoint.
    Held-object suffixes are rejected because their projected collision worlds
    require a separate adapter. The engine is borrowed and never closed here.

    Args:
        engine: Engine owning the physical robot and atomic planners.
        replica_pool: Verified source-to-physical-slot mapping.
        validator: Optional trusted per-row planning validator. A session shared
            with collection requires this callback and passing path_collision.
        source_id: Stable logical source name.
        source_revision: Caller-owned revision covering task/skill configuration.
    """

    def __init__(
        self,
        engine: AtomicActionEngine,
        *,
        replica_pool: SceneReplicaPool,
        validator: (
            Callable[[CandidateTrajectoryBatch, MotionSnapshot], ValidationResult]
            | None
        ) = None,
        source_id: str = "atomic_affordance",
        source_revision: str = "v1",
    ) -> None:
        for value in (source_id, source_revision):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("source identity and revision must be nonempty")
        if validator is not None and not callable(validator):
            raise TypeError("validator must be callable")
        self.engine, self.replica_pool = engine, replica_pool
        self.validator = validator
        self.source_id, self.source_revision = source_id, source_revision

    def generate(
        self,
        *,
        invocations: Sequence[ActionInvocation],
        context: PlanningContext,
        candidate_inputs: Mapping[tuple[str, str], GraspCandidateBatch],
        cfg: AtomicCandidateGenerationCfg | None = None,
        session: GenerationSession | None = None,
    ) -> AtomicGenerationResult:
        """Generate complete successful branches, returning fewer rows on rejection.

        Args:
            invocations: Fixed prefixes and one final PickUp with a nonempty ID.
            context: Frozen full physical initial state shared by every wave.
            candidate_inputs: One (PickUp invocation ID, primary) entry. Grasp
                rows are in the replica pool's canonical source order, with
                poses in local-arena coordinates.
            cfg: Bounded source settings; omitted uses defaults.
            session: Optional collection-owned session. Required checks must
                pass before its candidates become ready. Without this argument,
                an internal session provides identity and rejection accounting
                only; returned trajectories remain planning-only exports.

        Returns:
            Compact (B_out, N_max, D_full) trajectories, timing and failure audit.

        Raises:
            ValueError: For incompatible contracts or unsupported configuration.
            NotImplementedError: For unsupported skill sequences/backends.
            RuntimeError: For backend errors or stale replica bindings.
        """
        cfg = AtomicCandidateGenerationCfg.from_mapping(
            (cfg or AtomicCandidateGenerationCfg()).to_dict()
        )
        deadline = time.monotonic() + cfg.max_wall_time_s
        invocations = tuple(invocations)
        if (
            not invocations
            or invocations[-1].skill_id != "pick_up"
            or any(
                item.skill_id not in ("move_end_effector", "move_joints")
                for item in invocations[:-1]
            )
        ):
            raise NotImplementedError(
                "supported source: MoveEndEffector/MoveJoints prefixes then one PickUp"
            )
        pickup = invocations[-1]
        if not pickup.invocation_id:
            raise ValueError("candidate PickUp must have an explicit invocation_id")
        if context.task.held_objects or context.task.coordinated_held_objects:
            raise NotImplementedError(
                "the initial affordance source requires unheld objects"
            )
        if invocations[:-1]:
            from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget

            arm = pickup.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            for prefix in invocations[:-1]:
                target = prefix.binding.endpoint("primary", "motion").require_target(
                    JointPositionTarget
                )
                if (
                    target.control_part != arm.control_part
                    or target.joint_ids != arm.joint_ids
                ):
                    raise NotImplementedError(
                        "prefixes must use the same PickUp arm; projected solver-root changes are unsupported"
                    )
        if any(item.motion_policy.strategy != "ik_interp" for item in invocations):
            raise NotImplementedError(
                "affordance candidate source currently requires ik_interp"
            )
        key = (pickup.invocation_id, "primary")
        if set(candidate_inputs) != {key} or not isinstance(
            candidate_inputs[key], GraspCandidateBatch
        ):
            raise ValueError("provide exactly the PickUp primary GraspCandidateBatch")
        grasps = candidate_inputs[key]
        pool, robot = self.replica_pool, self.engine.robot
        pool.assert_current(context)
        if (
            pool.batch_size != robot.num_instances
            or context.batch_size != pool.batch_size
        ):
            raise ValueError("replicas must cover the complete physical robot batch")
        if grasps.poses.shape[0] != pool.source_count or grasps.frame != "local_arena":
            raise ValueError(
                "grasp rows must match canonical sources in local_arena coordinates"
            )
        if grasps.poses.device != context.robot.qpos.device:
            raise ValueError("grasp and context devices must match")
        if session is not None and self.validator is None:
            raise ValueError(
                "a collection session requires an explicit planning validator"
            )
        owned_session = session is None
        if session is None:
            session = GenerationSession(
                TrajectoryGenerationJobCfg.from_mapping(
                    {
                        "augmentation": {"seed": cfg.seed},
                        "collection": {
                            "max_proposals": cfg.max_proposals,
                            "max_wall_time_s": cfg.max_wall_time_s,
                        },
                    }
                )
            )
        names = tuple(robot.joint_names)
        limits = robot.get_qpos_limits()
        if limits.shape != (pool.batch_size, len(names), 2):
            raise ValueError("robot must supply complete per-row joint limits")
        for source, case in enumerate(pool.cases):
            session.register_case(
                case, limits[pool.source_rows[source]], joint_names=names
            )
        rows, checks, metadata, rejections = [], [], [], []
        counts: Counter[str] = Counter()
        outstanding: dict[str, CandidateIdentity] = {}
        variants = (0,) if cfg.variants == "original" else (0, 1)
        jobs: list[tuple[int, int, int]] = []

        def reject(
            source: int,
            grasp_id: str | None,
            variant: int | None,
            identity: CandidateIdentity | None,
            stage: str,
            reason: str,
            invocation_id: str | None = pickup.invocation_id,
        ) -> None:
            counts[reason] += 1
            if len(rejections) < cfg.max_rejection_records:
                rejections.append(
                    AtomicCandidateRejection(
                        source,
                        grasp_id,
                        variant,
                        None if identity is None else identity.candidate_id,
                        invocation_id,
                        stage,
                        reason,
                    )
                )
            if identity is not None:
                session.release(identity, reason=reason)
                outstanding.pop(identity.candidate_id, None)

        for source in range(pool.source_count):
            populated = 0
            seen = set()
            valid_grasps = []
            for grasp in range(grasps.poses.shape[1]):
                reason = grasps.rejection_reasons[source][grasp]
                if reason == "PADDING":
                    continue
                populated += 1
                if not bool(grasps.valid_mask[source, grasp]):
                    reject(
                        source,
                        grasps.grasp_ids[source][grasp],
                        None,
                        None,
                        "grasp",
                        reason or "INVALID_GRASP",
                    )
                    continue
                valid_grasps.append(grasp)
            valid_grasps.sort(
                key=lambda index: (
                    float(grasps.costs[source, index]),
                    grasps.grasp_ids[source][index],
                )
            )
            selected_count = 0
            for grasp in valid_grasps:
                if grasps.grasp_ids[source][grasp] in seen:
                    counts["DUPLICATE_GRASP"] += 1
                    continue
                seen.add(grasps.grasp_ids[source][grasp])
                if selected_count >= cfg.max_grasps_per_case:
                    counts["GRASP_LIMIT_EXCEEDED"] += 1
                    continue
                selected_count += 1
                jobs.extend((source, grasp, variant) for variant in variants)
            if not populated:
                reject(source, None, None, None, "grasp", "NO_GRASP_CANDIDATES")
        stop_reason = "candidate_exhausted"
        rounds = 0
        pending = list(range(len(jobs)))
        backpressure = False
        try:
            while pending:
                if len(rows) >= cfg.max_output_trajectories:
                    stop_reason = "target_reached"
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "wall_time_exhausted"
                    break
                if (
                    counts["proposed"] >= cfg.max_proposals
                    or not session.remaining_proposals
                ):
                    stop_reason = session.stop_reason or "proposal_budget_exhausted"
                    break
                pool.assert_current(context)
                assignments = next(
                    iter(pool.rounds([jobs[index][0] for index in pending]))
                )
                assignments = assignments[
                    : min(
                        cfg.max_proposals - counts["proposed"],
                        session.remaining_proposals,
                        cfg.max_output_trajectories - len(rows),
                    )
                ]
                wave_jobs = {
                    item.slot_index: jobs[pending[item.candidate_index]]
                    for item in assignments
                }
                consumed = {item.candidate_index for item in assignments}
                pending = [
                    index
                    for ordinal, index in enumerate(pending)
                    if ordinal not in consumed
                ]
                active = torch.zeros(
                    pool.batch_size, dtype=torch.bool, device=context.robot.qpos.device
                )
                poses = torch.eye(
                    4, dtype=grasps.poses.dtype, device=grasps.poses.device
                ).repeat(pool.batch_size, 1, 1, 1)
                costs = grasps.costs.new_full((pool.batch_size, 1), torch.inf)
                widths = (
                    None
                    if grasps.opening_widths is None
                    else grasps.opening_widths.new_zeros((pool.batch_size, 1))
                )
                ids = [(f"inactive:{slot}",) for slot in range(pool.batch_size)]
                indices = torch.full(
                    (pool.batch_size,), -1, dtype=torch.long, device=active.device
                )
                identities: dict[int, CandidateIdentity] = {}
                for assignment in assignments:
                    source, grasp, variant = wave_jobs[assignment.slot_index]
                    slot = assignment.slot_index
                    case = pool.cases[source]
                    identity, _ = session.propose(
                        case.scene_case_id,
                        case.initial_state_id,
                        source_id=self.source_id,
                        source_revision=self.source_revision,
                        template_id=f"{pickup.invocation_id}:{grasps.grasp_ids[source][grasp]}:roll{variant}",
                        operator_id="affordance_grasp",
                    )
                    identities[slot] = identity
                    outstanding[identity.candidate_id] = identity
                    counts["proposed"] += 1
                    active[slot] = True
                    poses[slot, 0], costs[slot, 0] = (
                        grasps.poses[source, grasp],
                        grasps.costs[source, grasp],
                    )
                    if widths is not None:
                        widths[slot, 0] = grasps.opening_widths[source, grasp]
                    ids[slot] = (grasps.grasp_ids[source][grasp],)
                    indices[slot] = variant
                bound = GraspCandidateBatch(
                    poses, costs, active[:, None], widths, tuple(ids)
                )
                evaluated = None

                def select(request, projected, alive):
                    nonlocal evaluated
                    evaluated = self.engine.enumerate_candidates(
                        pickup, projected, grasp_candidates=bound, active_mask=alive
                    )
                    return evaluated.select(indices, active_mask=alive)

                compiled = self.engine.compile(
                    invocations,
                    context,
                    eligible_mask=active,
                    candidate_selections={pickup.invocation_id: select},
                )
                rounds += 1
                for assignment in assignments:
                    source, grasp, variant = wave_jobs[assignment.slot_index]
                    slot, identity = (
                        assignment.slot_index,
                        identities[assignment.slot_index],
                    )
                    if backpressure:
                        reject(
                            source,
                            grasps.grasp_ids[source][grasp],
                            variant,
                            identity,
                            "admission",
                            "READY_BUDGET_EXHAUSTED",
                        )
                        continue
                    if not bool(compiled.plan_success[slot]):
                        stage, reason, failed_invocation = (
                            "path",
                            "PATH_NOT_FOUND",
                            pickup.invocation_id,
                        )
                        for invocation, plan in zip(invocations, compiled.action_plans):
                            if not bool(plan.plan_success[slot]):
                                failed_invocation = invocation.invocation_id
                                if (
                                    invocation is pickup
                                    and evaluated is not None
                                    and not bool(evaluated.valid_mask[slot, variant])
                                ):
                                    stage = (
                                        evaluated.failure_stages[slot][variant] or "ik"
                                    )
                                    reason = (
                                        evaluated.failure_reasons[slot][variant]
                                        or "IK_NOT_FOUND"
                                    )
                                else:
                                    failures = plan.diagnostics.metadata.get(
                                        "candidate_failure_reasons", ()
                                    )
                                    if failures and failures[slot]:
                                        parts = failures[slot].split(":", 1)
                                        stage, reason = (
                                            parts
                                            if len(parts) == 2
                                            else (stage, parts[0])
                                        )
                                    elif plan.diagnostics.failure is not None:
                                        reason = plan.diagnostics.failure.code
                                break
                        reject(
                            source,
                            grasps.grasp_ids[source][grasp],
                            variant,
                            identity,
                            stage,
                            reason,
                            failed_invocation,
                        )
                        continue
                    if compiled.trajectory.waypoint_count > cfg.max_waypoints:
                        reject(
                            source,
                            grasps.grasp_ids[source][grasp],
                            variant,
                            identity,
                            "export",
                            "WAYPOINT_BUDGET_EXCEEDED",
                        )
                        continue
                    row = _export_row(
                        compiled, slot, identity, pool.source_rows[source], robot
                    )
                    q = row.positions[0]
                    if not torch.allclose(
                        q[0], context.robot.qpos[slot], atol=1e-5, rtol=0
                    ):
                        reject(
                            source,
                            grasps.grasp_ids[source][grasp],
                            variant,
                            identity,
                            "export",
                            "INITIAL_QPOS_MISMATCH",
                        )
                        continue
                    bounds = limits[slot].to(q)
                    if not bool(
                        ((q >= bounds[:, 0] - 1e-5) & (q <= bounds[:, 1] + 1e-5)).all()
                    ):
                        reject(
                            source,
                            grasps.grasp_ids[source][grasp],
                            variant,
                            identity,
                            "export",
                            "JOINT_LIMIT_FAILED",
                        )
                        continue
                    validation = ValidationResult(
                        (
                            ValidationCheck("atomic_plan", "passed"),
                            ValidationCheck("joint_limits", "passed"),
                            ValidationCheck("timing", "passed"),
                        )
                    )
                    if self.validator is not None:
                        extra = self.validator(row, pool.source_snapshot(source))
                        if not isinstance(extra, ValidationResult):
                            raise TypeError("validator must return ValidationResult")
                        if not extra.accepted:
                            reject(
                                source,
                                grasps.grasp_ids[source][grasp],
                                variant,
                                identity,
                                "validation",
                                "PATH_VALIDATION_FAILED",
                            )
                            continue
                        validation = ValidationResult(validation.checks + extra.checks)
                    if not owned_session and "path_collision" not in {
                        check.check_id for check in validation.checks
                    }:
                        raise ValueError(
                            "collection admission requires path_collision evidence"
                        )
                    new_horizon = max(
                        [q.shape[0]] + [int(item.valid_length[0]) for item in rows]
                    )
                    padded_bytes = (
                        (len(rows) + 1)
                        * new_horizon
                        * (len(names) * q.element_size() + 8)
                    )
                    if padded_bytes > cfg.max_output_bytes:
                        reject(
                            source,
                            grasps.grasp_ids[source][grasp],
                            variant,
                            identity,
                            "export",
                            "OUTPUT_BYTE_BUDGET_EXCEEDED",
                        )
                        continue
                    if owned_session:
                        session.release(identity, reason="planning_only_export")
                    else:
                        try:
                            session.add_planned(row, validation)
                        except BufferError:
                            backpressure = True
                            stop_reason = "ready_budget_exhausted"
                            reject(
                                source,
                                grasps.grasp_ids[source][grasp],
                                variant,
                                identity,
                                "admission",
                                "READY_BUDGET_EXHAUSTED",
                            )
                            continue
                    outstanding.pop(identity.candidate_id, None)
                    rows.append(row)
                    checks.append(validation)
                    metadata.append(
                        {
                            "source_index": source,
                            "source_row": pool.source_rows[source],
                            "grasp_id": grasps.grasp_ids[source][grasp],
                            "variant_id": variant,
                            "candidate_id": identity.candidate_id,
                            "planning_slot": slot,
                            "invocation_id": pickup.invocation_id,
                            "physical_validation": False,
                        }
                    )
                if backpressure:
                    break
        except Exception:
            for identity in tuple(outstanding.values()):
                session.release(identity, reason="source_error")
            raise
        if len(rows) >= cfg.max_output_trajectories:
            stop_reason = "target_reached"
        summary = {
            "status": (
                "complete"
                if stop_reason == "target_reached"
                else ("partial" if rows else "empty")
            ),
            "stop_reason": stop_reason,
            "requested": cfg.max_output_trajectories,
            "output_count": len(rows),
            "proposed": counts.pop("proposed", 0),
            "rounds": rounds,
            "rejection_counts": dict(counts),
            "physical_validation": False,
        }
        return AtomicGenerationResult(
            _pack(rows, qpos=context.robot.qpos, names=names),
            tuple(checks),
            tuple(metadata),
            tuple(rejections),
            summary,
        )
