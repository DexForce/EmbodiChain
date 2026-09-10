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

"""Distinct affordance choices planned directly over live environment rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions.plans import TimedTrajectory
from embodichain.toolkits.graspkit import GraspCandidateBatch

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        AtomicActionEngine,
        PlanningContext,
    )
    from embodichain.lab.sim.atomic_actions.candidates import AtomicCandidateBatch

__all__ = ["AtomicAffordanceBatch", "plan_affordance_batch"]


@dataclass(frozen=True, slots=True, eq=False)
class AtomicAffordanceBatch:
    """One distinct planned raw grasp per successful physical environment.

    ``trajectory`` retains every physical row on one uniform control grid.
    Unsuccessful rows hold their initial qpos; successful shorter rows hold
    their endpoint after ``valid_length``. ``grasp_poses`` are the selected
    input raw poses, before a primitive's roll/calibration transformations.
    The zero-time sample retains the complete observed joint state, including
    passive constraint residuals; later samples expand the mimic geometry.
    These values certify planning only, not collision or physical task success.
    """

    trajectory: TimedTrajectory
    """Synchronized full-robot replay, shape ``(E,N,D)``; failed rows only hold."""
    success_mask: torch.Tensor
    """Bool ``(E,)`` planning acceptance, never physical task success evidence."""
    grasp_ids: tuple[str | None, ...]
    """Distinct selected raw identities in env-row order; failures have ``None``."""
    candidate_indices: torch.Tensor
    """Evaluated candidate columns, shape ``(E,)``; failures have index ``-1``."""
    grasp_poses: torch.Tensor
    """Input raw local-arena poses ``(E,4,4)``; failures contain identity poses."""
    valid_length: torch.Tensor
    """Unpadded row lengths; failures use 1 for a hold, not an exported success."""
    rejections: tuple[Mapping[str, object], ...]
    """Geometric, IK, path and export failures with physical-row/raw provenance."""
    summary: Mapping[str, object]
    """Planning counts and bounded termination reason; no physics certification."""

    def __post_init__(self) -> None:
        rows = self.trajectory.batch_size
        if self.success_mask.dtype != torch.bool or self.success_mask.shape != (rows,):
            raise ValueError("success_mask must be bool with shape (E,).")
        for name in ("candidate_indices", "valid_length"):
            value = getattr(self, name)
            if value.dtype != torch.long or value.shape != (rows,):
                raise ValueError(f"{name} must be int64 with shape (E,).")
        if self.grasp_poses.shape != (rows, 4, 4):
            raise ValueError("grasp_poses must have shape (E,4,4).")
        for name in (
            "success_mask",
            "candidate_indices",
            "grasp_poses",
            "valid_length",
        ):
            value = getattr(self, name)
            if value.device != self.trajectory.positions.device:
                raise ValueError("Result tensors must share the trajectory device.")
            object.__setattr__(self, name, value.detach().clone())
        if not torch.isfinite(self.grasp_poses).all():
            raise ValueError("grasp_poses must be finite.")
        if len(self.grasp_ids) != rows or any(
            (identity is not None) != bool(self.success_mask[row])
            for row, identity in enumerate(self.grasp_ids)
        ):
            raise ValueError("grasp_ids must identify exactly the successful rows.")
        selected = [identity for identity in self.grasp_ids if identity is not None]
        if len(set(selected)) != len(selected):
            raise ValueError("Successful rows must have distinct raw grasp IDs.")
        if not torch.equal(self.candidate_indices >= 0, self.success_mask):
            raise ValueError("Only successful rows may have a candidate index.")
        if (self.candidate_indices[~self.success_mask] != -1).any():
            raise ValueError("Failed rows must have candidate index -1.")
        if (
            (self.valid_length < 1)
            | (self.valid_length > self.trajectory.waypoint_count)
        ).any():
            raise ValueError("valid_length must lie within the full trajectory.")
        if (self.valid_length[~self.success_mask] != 1).any():
            raise ValueError("Failed rows must have valid_length 1 for their hold.")
        object.__setattr__(self, "trajectory", self.trajectory.snapshot())
        object.__setattr__(self, "grasp_ids", tuple(self.grasp_ids))
        object.__setattr__(
            self,
            "rejections",
            tuple(MappingProxyType(dict(item)) for item in self.rejections),
        )
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))

    @property
    def compact_positions(self) -> torch.Tensor:
        """Successful rows only, shape ``(B_success,N,D)``, in env-row order."""
        if not self.success_mask.any():
            return self.trajectory.positions.new_empty(
                (0, 0, self.trajectory.robot_dof)
            )
        return self.trajectory.positions[self.success_mask].clone()

    @property
    def compact_valid_length(self) -> torch.Tensor:
        """Original successful lengths, excluding endpoint padding."""
        return self.valid_length[self.success_mask].clone()

    @property
    def compact_env_ids(self) -> torch.Tensor:
        """Physical environment IDs aligned with compact trajectory rows."""
        return self.trajectory.env_ids[self.success_mask].clone()

    @property
    def dt(self) -> torch.Tensor:
        """Full physical-row arrival intervals on the shared replay grid."""
        return self.trajectory.dt.clone()


def _match_unique_raw(
    candidates: AtomicCandidateBatch,
    grasp_ids: tuple[tuple[str, ...], ...],
    remaining: torch.Tensor,
    pending: torch.Tensor,
    occupied: set[str],
) -> torch.Tensor:
    """Stable maximum-cardinality row-to-raw matching, with cheapest roll edges."""
    choices: dict[int, list[tuple[str, int]]] = {}
    for row in torch.nonzero(pending).flatten().tolist():
        indices = torch.nonzero(remaining[row]).flatten().tolist()
        indices.sort(
            key=lambda index: (
                float(candidates.costs[row, index]),
                grasp_ids[row][index],
                index,
            )
        )
        seen: set[str] = set()
        choices[row] = []
        for index in indices:
            identity = grasp_ids[row][index]
            if identity not in occupied and identity not in seen:
                choices[row].append((identity, index))
                seen.add(identity)
    owners: dict[str, tuple[int, int]] = {}

    def augment(row: int, visited: set[str]) -> bool:
        for identity, index in choices[row]:
            if identity in visited:
                continue
            visited.add(identity)
            previous = owners.get(identity)
            if previous is None or augment(previous[0], visited):
                owners[identity] = (row, index)
                return True
        return False

    for row in sorted(choices, key=lambda value: (len(choices[value]), value)):
        augment(row, set())
    result = torch.full_like(candidates.env_ids, -1)
    for row, index in owners.values():
        result[row] = index
    return result


def _export_positions(
    trajectory: TimedTrajectory,
    row: int,
    start: torch.Tensor,
    bounds: torch.Tensor,
    couplings: tuple[tuple[int, int, float, float], ...],
    control_dt: float,
) -> tuple[torch.Tensor | None, str | None]:
    """Filter one row without allowing bad qpos or timing to contaminate peers."""
    positions = trajectory.positions[row].detach().clone()
    intervals = trajectory.dt[row]
    if not len(positions):
        return None, "EMPTY_TRAJECTORY"
    if not torch.isfinite(positions).all():
        return None, "NONFINITE_QPOS"
    if not torch.allclose(positions[0], start, atol=1e-5, rtol=0):
        return None, "INITIAL_QPOS_MISMATCH"
    if (
        not torch.isfinite(intervals).all()
        or intervals[0] != 0
        or (intervals < 0).any()
    ):
        return None, "INVALID_TIMING"
    keep = torch.ones(len(positions), dtype=torch.bool, device=positions.device)
    for index in range(1, len(positions)):
        if intervals[index] == 0:
            if not torch.allclose(
                positions[index], positions[index - 1], atol=1e-5, rtol=0
            ):
                return None, "DISCONTINUOUS_ZERO_TIME_BOUNDARY"
            keep[index] = False
        elif not torch.isclose(
            intervals[index], intervals.new_tensor(control_dt), atol=1e-7, rtol=1e-5
        ):
            return None, "NONUNIFORM_CONTROL_DT"
    positions = positions[keep]
    # A live observation can include a passive constraint residual. Keep that
    # measured t=0 state; expand command geometry only at positive arrival times.
    for child, parent, multiplier, offset in couplings:
        positions[1:, child] = positions[1:, parent] * multiplier + offset
    if not torch.isfinite(positions).all():
        return None, "NONFINITE_QPOS"
    if not (
        (positions >= bounds[:, 0] - 1e-5) & (positions <= bounds[:, 1] + 1e-5)
    ).all():
        return None, "JOINT_LIMIT_FAILED"
    return positions, None


def plan_affordance_batch(
    engine: AtomicActionEngine,
    invocation: ActionInvocation,
    context: PlanningContext,
    grasp_candidates: GraspCandidateBatch,
    *,
    active_mask: torch.Tensor | None = None,
    max_rounds: int = 32,
) -> AtomicAffordanceBatch:
    """Plan distinct raw grasps, with bounded row-local refill and roll fallback.

    Each real environment receives at most one successful trajectory. Candidate
    columns never create extra robot instances. The same raw grasp ID can be
    tried with different rolls, but can appear in only one successful row.
    Successes are locked while later compile rounds start other rows from the
    unchanged input context. No simulation stepping or scene restoration occurs.

    Args:
        engine: Borrowed engine with one physical robot row per context row.
        invocation: Single PickUp or Slide using the strict ``ik_interp`` path.
        context: Current observed planning start, with explicit control cadence.
        grasp_candidates: ``(E,G)`` raw grasps in ``local_arena`` coordinates.
            IDs identify the same raw grasp across replicas; rows may also hold
            different IDs, as when replanning a previously selected drawer grip.
        active_mask: Optional bool ``(E,)`` mask; excluded rows remain idle.
        max_rounds: Positive upper bound on complete physical-batch compilations.

    Returns:
        Full-row safe replay and compact success views. Backend exceptions are
        propagated as errors, never reclassified as ordinary IK infeasibility.

    .. attention::
        Primitive candidate planning owns FK, IK and Cartesian-path checks.
        This adapter adds identity matching, export checks and synchronized
        timing; it does not establish collision freedom or physical success.
    """
    if invocation.skill_id not in ("pick_up", "slide"):
        raise ValueError("Affordance batches support only PickUp and Slide.")
    if invocation.motion_policy.strategy != "ik_interp":
        raise ValueError("Affordance batches require the strict ik_interp path.")
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer.")
    if not isinstance(grasp_candidates, GraspCandidateBatch):
        raise TypeError("grasp_candidates must be a GraspCandidateBatch.")
    start = context.robot.qpos
    rows, dof = start.shape
    if engine.robot.num_instances != rows or engine.robot.dof != dof:
        raise ValueError("Context rows/DoF must match the real engine robot.")
    if (
        grasp_candidates.poses.shape[0] != rows
        or grasp_candidates.poses.device != start.device
    ):
        raise ValueError("Raw candidates must match the context physical rows/device.")
    if grasp_candidates.frame != "local_arena":
        raise ValueError("Affordance batches require local_arena raw poses.")
    if active_mask is None:
        active = torch.ones(rows, device=start.device, dtype=torch.bool)
    elif (
        not isinstance(active_mask, torch.Tensor)
        or active_mask.dtype != torch.bool
        or active_mask.shape != (rows,)
        or active_mask.device != start.device
    ):
        raise ValueError("active_mask must be bool (E,) on the context device.")
    else:
        active = active_mask.detach().clone()
    control_dt = context.require_control_dt()
    robot = engine.robot
    coupling_fields = tuple(
        tuple(getattr(robot, name))
        for name in ("mimic_ids", "mimic_parents", "mimic_multipliers", "mimic_offsets")
    )
    if len({len(field) for field in coupling_fields}) != 1 or set(
        coupling_fields[0]
    ) & set(coupling_fields[1]):
        raise ValueError("Export requires aligned direct active-to-mimic coupling.")
    couplings = tuple(zip(*coupling_fields))
    limits = robot.get_qpos_limits().to(start)
    if (
        limits.shape != (rows, dof, 2)
        or torch.isnan(limits).any()
        or (limits[..., 0] > limits[..., 1]).any()
    ):
        raise ValueError("Robot limits must be ordered (E,D,2) bounds without NaN.")
    success = torch.zeros_like(active)
    selected_indices = torch.full((rows,), -1, device=start.device, dtype=torch.long)
    selected_poses = torch.eye(
        4, device=start.device, dtype=grasp_candidates.poses.dtype
    ).repeat(rows, 1, 1)
    selected_ids: list[str | None] = [None] * rows
    paths: dict[int, torch.Tensor] = {}
    rejections: list[Mapping[str, object]] = []

    def reject(
        row: int,
        identity: str | None,
        index: int | None,
        stage: str,
        reason: str,
        round_index: int = 0,
    ) -> None:
        rejections.append(
            {
                "env_id": int(context.env_ids[row]),
                "row_index": row,
                "grasp_id": identity,
                "candidate_index": index,
                "stage": stage,
                "reason_code": reason,
                "round": round_index,
            }
        )

    for row in torch.nonzero(active).flatten().tolist():
        for index in range(grasp_candidates.valid_mask.shape[1]):
            if not grasp_candidates.valid_mask[row, index]:
                reason = grasp_candidates.rejection_reasons[row][index]
                if reason != "PADDING":
                    reject(
                        row,
                        grasp_candidates.grasp_ids[row][index],
                        None,
                        "grasp",
                        reason or "INVALID_GRASP_POSE",
                    )
        if not grasp_candidates.valid_mask[row].any():
            reject(row, None, None, "grasp", "NO_GRASP_CANDIDATES")
    eligible = active & grasp_candidates.valid_mask.any(dim=1)
    rounds = 0
    attempts = 0
    occupied: set[str] = set()
    if eligible.any():
        evaluated = engine.enumerate_candidates(
            invocation, context, grasp_candidates=grasp_candidates, active_mask=eligible
        )
        if (
            not torch.equal(evaluated.env_ids, context.env_ids)
            or evaluated.valid_mask.shape[0] != rows
        ):
            raise ValueError("Evaluated candidates must preserve physical env_ids.")
        identities = getattr(evaluated, "grasp_ids", None)
        if (
            identities is None
            or len(identities) != rows
            or any(len(row) != evaluated.valid_mask.shape[1] for row in identities)
        ):
            raise ValueError("Evaluated candidates must expose aligned raw grasp_ids.")
        raw_lookup = [
            {
                identity: index
                for index, identity in reversed(tuple(enumerate(row)))
                if grasp_candidates.valid_mask[row_index, index]
            }
            for row_index, row in enumerate(grasp_candidates.grasp_ids)
        ]
        remaining = evaluated.valid_mask.clone() & eligible[:, None]
        for row in torch.nonzero(eligible).flatten().tolist():
            for index, identity in enumerate(identities[row]):
                if identity not in raw_lookup[row]:
                    if remaining[row, index]:
                        raise ValueError(
                            "A valid evaluated candidate references an unavailable raw grasp."
                        )
                    continue
                if not remaining[row, index]:
                    reject(
                        row,
                        identity,
                        index,
                        evaluated.failure_stages[row][index] or "ik",
                        evaluated.failure_reasons[row][index] or "IK_NOT_FOUND",
                    )
        for round_index in range(1, max_rounds + 1):
            indices = _match_unique_raw(
                evaluated, identities, remaining, eligible & ~success, occupied
            )
            selected = indices >= 0
            if not selected.any():
                break
            rounds += 1
            attempts += int(selected.sum())
            choice = evaluated.select(indices, active_mask=selected)
            compiled = engine.compile(
                (invocation,),
                context,
                eligible_mask=selected,
                candidate_selections={invocation.invocation_id: choice},
            )
            if (
                compiled.plan_success.shape != (rows,)
                or compiled.plan_success.dtype != torch.bool
            ):
                raise ValueError("Compiled success mask must preserve physical rows.")
            if (
                compiled.trajectory.positions.shape[0] != rows
                or compiled.trajectory.positions.shape[2] != dof
                or not torch.equal(compiled.trajectory.env_ids, context.env_ids)
            ):
                raise ValueError(
                    "Compiled trajectory must preserve physical rows and DoF."
                )
            for row in torch.nonzero(selected).flatten().tolist():
                index = int(indices[row])
                identity = identities[row][index]
                remaining[row, index] = False
                if not compiled.plan_success[row]:
                    stage, reason = "path", "PATH_NOT_FOUND"
                    for plan in compiled.action_plans:
                        if not plan.plan_success[row]:
                            failures = plan.diagnostics.metadata.get(
                                "candidate_failure_reasons", ()
                            )
                            if failures and failures[row]:
                                parts = failures[row].split(":", 1)
                                stage, reason = (
                                    parts if len(parts) == 2 else (stage, parts[0])
                                )
                            elif plan.diagnostics.failure is not None:
                                reason = plan.diagnostics.failure.code
                            break
                    reject(row, identity, index, stage, reason, round_index)
                    continue
                positions, error = _export_positions(
                    compiled.trajectory,
                    row,
                    start[row],
                    limits[row],
                    couplings,
                    control_dt,
                )
                if error is not None:
                    reject(row, identity, index, "export", error, round_index)
                    continue
                success[row] = True
                selected_indices[row] = index
                selected_ids[row] = identity
                selected_poses[row] = grasp_candidates.poses[
                    row, raw_lookup[row][identity]
                ]
                paths[row] = positions
                occupied.add(identity)
    lengths = torch.ones(rows, device=start.device, dtype=torch.long)
    horizon = max((len(value) for value in paths.values()), default=1)
    positions = start[:, None].repeat(1, horizon, 1)
    for row, path in paths.items():
        lengths[row] = len(path)
        positions[row, : len(path)] = path
        positions[row, len(path) :] = path[-1]
    count, requested = int(success.sum()), int(active.sum())
    summary = {
        "status": (
            "complete" if count == requested else ("partial" if count else "empty")
        ),
        "requested": requested,
        "successful": count,
        "physical_env_count": rows,
        "unique_raw_grasps": len(occupied),
        "rounds": rounds,
        "attempts": attempts,
        "max_rounds": max_rounds,
        "stop_reason": (
            "target_reached"
            if count == requested
            else (
                "round_budget_exhausted"
                if rounds == max_rounds
                else "candidates_exhausted"
            )
        ),
        "rejection_counts": dict(Counter(item["reason_code"] for item in rejections)),
        "physical_validation": False,
        "collision_validated": False,
    }
    return AtomicAffordanceBatch(
        trajectory=TimedTrajectory.from_uniform_step(
            positions, env_ids=context.env_ids, step_dt=control_dt
        ),
        success_mask=success,
        grasp_ids=tuple(selected_ids),
        candidate_indices=selected_indices,
        grasp_poses=selected_poses,
        valid_length=lengths,
        rejections=tuple(rejections),
        summary=summary,
    )
