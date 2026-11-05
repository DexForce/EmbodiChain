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

"""Operation-owned bounded Twist recovery, separate from physical sensing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from uuid import uuid4

from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlanningRecoveryOutcome,
)
from embodichain.lab.sim.atomic_actions.policies import PlanningRecovery


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be an exact non-empty identifier.")


def _ordinal(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class TwistPlanningRecovery(PlanningRecovery):
    """Fixed source budgets for measured multi-chunk Twist recovery.

    The 12 contact candidates are logical planning alternatives. The two physical
    fallbacks are additional attempts after measured safe release, not automatic
    engine action retries. Ordinary Twist does not select this declaration.
    """

    kind: str = field(default="twist_bounded", init=False)
    skill_id: str = field(default="twist", init=False)
    revision: int = field(default=1, init=False)
    minimum_sample_count: int = field(default=320, init=False)
    occurrence_angle_budget: float = field(default=math.radians(5.0), init=False)
    """Coverage reference only; actual arcs use explicit TwistOptions.max_twist_angle."""
    minimum_occurrences: int = field(default=8, init=False)
    maximum_occurrences: int = field(default=32, init=False)
    fallback_occurrences: int = field(default=3, init=False)
    maximum_contact_candidates: int = field(default=12, init=False)
    maximum_fallbacks: int = field(default=2, init=False)
    stalled_chunk_limit: int = field(default=2, init=False)
    minimum_contact_progress: float = field(default=1.0e-9, init=False)
    single_environment_only: bool = field(default=True, init=False)


def required_twist_occurrences(
    target_angle: float, policy: TwistPlanningRecovery
) -> int:
    """Resolve the fixed declared occurrence budget for a calibrated turn.

    The coverage reference does not set the executed arc. Explicit variants
    may use a different ``TwistOptions.max_twist_angle`` while retaining
    this same occurrence budget.

    Args:
        target_angle: Finite zero-to-setting angle in radians. Exact Python
            integers/floats are supported; booleans are not angle declarations.
        policy: The explicit source-compatible bounded Twist recovery rule.

    Returns:
        Eight occurrences for an exactly zero target, otherwise 32 for a legal
        nonzero target. Unused occurrences become stationary holds only after
        measured physical acceptance; this helper does not declare completion.

    Raises:
        ValueError: The target exceeds the fixed angle/occurrence budget or is
            not finite. The source ceiling tolerance and three fallback
            occurrences are preserved independently of two physical fallbacks.
        TypeError: A different rule was supplied.
    """
    if type(policy) is not TwistPlanningRecovery:
        raise TypeError("Twist occurrence budgets require TwistPlanningRecovery.")
    if type(target_angle) not in (int, float) or not math.isfinite(target_angle):
        raise ValueError("Twist chunk coverage requires a finite calibrated angle.")
    count = max(
        policy.minimum_occurrences,
        math.ceil(abs(target_angle) / policy.occurrence_angle_budget - 1e-12)
        + policy.fallback_occurrences,
    )
    if count > policy.maximum_occurrences:
        raise ValueError("Twist calibrated turn exceeds the bounded chunk budget.")
    return (
        policy.minimum_occurrences
        if target_angle == 0.0
        else policy.maximum_occurrences
    )


@dataclass(frozen=True, slots=True)
class TwistRecoveryScope:
    """Exact owner of one operation's cursor across its chunk occurrences.

    Args:
        program_fingerprint: Opaque fingerprint supplied by the trusted host.
        operation_id: Stable operation identity, distinct from object identity.
        episode_id: Exact episode instance, replaced when the host resets.
        env_id: Native environment identity, not a transient planning row index.

    Direct Atomic hosts may supply their own immutable program fingerprint;
    no Task Program object or graph is retained by this contract.
    """

    program_fingerprint: str
    operation_id: str
    episode_id: str
    env_id: int

    def __post_init__(self) -> None:
        for name in ("program_fingerprint", "operation_id", "episode_id"):
            _identifier(getattr(self, name), name)
        _ordinal(self.env_id, "env_id")


@dataclass(frozen=True, slots=True)
class TwistRecoverySnapshot:
    """Owned read-only planning inputs from one version of a Twist cursor.

    Candidate rows are ``(index, roll_radians, axial_shift_metres)``. The order
    contains feasible candidates with the currently selected grip axis, while
    candidates retains all feasible axes for measured stalled-contact recovery.
    """

    scope: TwistRecoveryScope
    cursor_id: str
    version: int = 0
    forced_index: int | None = None
    order: tuple[int, ...] = ()
    candidates: tuple[tuple[int, float, float], ...] = ()
    reserved_indices: tuple[int, ...] = ()
    selected_index: int | None = None
    selected_bite_depth: float | None = None
    selected_roll: float | None = None
    fallback_count: int = 0
    locked: bool = False
    stalled_chunks: int = 0
    contact_path: float = 0.0
    goal_complete: bool = False
    last_feedback_timestamp: float | None = None

    def __post_init__(self) -> None:
        if type(self.scope) is not TwistRecoveryScope:
            raise TypeError("scope must be exactly TwistRecoveryScope.")
        _identifier(self.cursor_id, "cursor_id")
        for name in ("version", "fallback_count", "stalled_chunks"):
            _ordinal(getattr(self, name), name)
        for name in ("forced_index", "selected_index"):
            value = getattr(self, name)
            if value is not None:
                _ordinal(value, name)
        for name in ("order", "reserved_indices"):
            value = tuple(getattr(self, name))
            for index in value:
                _ordinal(index, name)
            if len(set(value)) != len(value):
                raise ValueError(f"{name} must not repeat a candidate.")
            object.__setattr__(self, name, value)
        candidates = tuple(tuple(item) for item in self.candidates)
        for item in candidates:
            if len(item) != 3:
                raise ValueError("Candidates require index, roll and axial shift.")
            _ordinal(item[0], "candidate index")
            if any(
                type(value) not in (float, int) or not math.isfinite(value)
                for value in item[1:]
            ):
                raise ValueError("Candidate geometry must be finite.")
        ids = tuple(item[0] for item in candidates)
        if len(set(ids)) != len(ids) or not set(self.order) <= set(ids):
            raise ValueError(
                "Candidate identities and feasible order are inconsistent."
            )
        object.__setattr__(self, "candidates", candidates)
        for name in ("locked", "goal_complete"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        for name in ("selected_bite_depth", "selected_roll", "last_feedback_timestamp"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite or None.")
        if (
            type(self.contact_path) not in (int, float)
            or not math.isfinite(self.contact_path)
            or self.contact_path < 0
        ):
            raise ValueError("contact_path must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class TwistPlanningOutcome(PlanningRecoveryOutcome):
    """Logical selection proposed by one complete Twist planning attempt.

    ``next_state.version`` is exactly one greater than the consumed snapshot.
    Planning may change only candidate selection/order; it cannot invent measured
    progress, spend a physical fallback, or mark the goal complete.
    """

    next_state: TwistRecoverySnapshot

    def __post_init__(self) -> None:
        PlanningRecoveryOutcome.__post_init__(self)
        if (
            self.skill_id != "twist"
            or type(self.next_state) is not TwistRecoverySnapshot
        ):
            raise ValueError("TwistPlanningOutcome requires Twist scope and state.")


@dataclass(frozen=True, slots=True)
class TwistMeasuredFeedback:
    """Fresh external physical assessment for one adopted Twist plan.

    The sensing/acceptance owner supplies validity, contact, angular proximity,
    released stability and cumulative contact path. This value does not sample
    a sensor or infer successful effects from planned motion.
    """

    scope: TwistRecoveryScope
    cursor_id: str
    expected_version: int
    invocation_id: str
    invocation_revision: int
    plan_id: str
    timestamp: float
    valid: bool
    target_contact: bool
    already_reached: bool
    recovery_safe: bool
    contact_path: float
    goal_complete: bool

    def __post_init__(self) -> None:
        if type(self.scope) is not TwistRecoveryScope:
            raise TypeError("scope must be exactly TwistRecoveryScope.")
        for name in ("cursor_id", "invocation_id", "plan_id"):
            _identifier(getattr(self, name), name)
        _ordinal(self.expected_version, "expected_version")
        _ordinal(self.invocation_revision, "invocation_revision")
        for name in ("timestamp", "contact_path"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in (
            "valid",
            "target_contact",
            "already_reached",
            "recovery_safe",
            "goal_complete",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")


@dataclass(frozen=True, slots=True)
class TwistRecoveryDecision:
    """Bounded cursor decision; acceptance is not a task-success certificate."""

    accepted: bool
    fallback_used: bool
    axis_switch: bool
    snapshot: TwistRecoverySnapshot


def _same_grip_axis(first: float, second: float) -> bool:
    return math.isclose(math.remainder(first - second, math.pi), 0.0, abs_tol=1e-6)


def _contact_decision(
    state: TwistRecoverySnapshot,
    rule: TwistPlanningRecovery,
    *,
    target_contact: bool,
    already_reached: bool,
    progress: bool,
    recovery_safe: bool,
) -> tuple[TwistRecoverySnapshot, bool, bool]:
    """Preserve the source contact-loss/stalled-grip transition order."""
    if (
        state.selected_index is not None
        and state.selected_index not in state.reserved_indices
    ):
        state = replace(
            state, reserved_indices=state.reserved_indices + (state.selected_index,)
        )
    if already_reached or target_contact:
        state = replace(
            state,
            locked=True,
            forced_index=state.selected_index,
            stalled_chunks=(
                0 if progress or already_reached else state.stalled_chunks + 1
            ),
        )
        if state.stalled_chunks < rule.stalled_chunk_limit:
            return state, True, False
        if not recovery_safe:
            return state, False, False
        state = replace(state, locked=False)
    elif recovery_safe:
        state = replace(state, locked=False)
    if (
        state.locked
        or not recovery_safe
        or state.fallback_count >= rule.maximum_fallbacks
    ):
        return state, False, False
    next_index = None
    if target_contact and state.selected_roll is not None:
        next_index = next(
            (
                index
                for index, roll, _ in state.candidates
                if index not in state.reserved_indices
                and not _same_grip_axis(roll, state.selected_roll)
            ),
            None,
        )
    if next_index is None:
        next_index = next(
            (index for index in state.order if index not in state.reserved_indices),
            None,
        )
    if next_index is None:
        return state, False, False
    return (
        replace(
            state,
            fallback_count=state.fallback_count + 1,
            forced_index=next_index,
            reserved_indices=state.reserved_indices + (next_index,),
            stalled_chunks=0,
        ),
        True,
        True,
    )


class TwistRecoveryCursor:
    """Host-owned serial cursor for one operation, episode and native environment.

    Planning consumes :meth:`snapshot` and returns a proposal. The host commits
    that logical outcome at the source planning boundary, including failed plans;
    plan adoption and fresh measured feedback are separate operations. Different
    chunk invocation IDs share this cursor only when the host binds them to its
    exact scope. No object-index lookup or process-global cache is used.
    """

    def __init__(self, scope: TwistRecoveryScope) -> None:
        self._state = TwistRecoverySnapshot(scope, uuid4().hex)
        self._rule = TwistPlanningRecovery()
        self._committed: TwistPlanningOutcome | None = None
        self._adopted: tuple[str, int, str] | None = None
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Twist recovery cursor has ended its episode scope.")

    def snapshot(self) -> TwistRecoverySnapshot:
        """Return owned planning data without advancing logical or physical state."""
        self._require_open()
        return deepcopy(self._state)

    def fork_preview(self) -> TwistRecoveryCursor:
        """Create an isolated cursor whose proposals cannot commit to this owner."""
        self._require_open()
        fork = TwistRecoveryCursor(self._state.scope)
        fork._state = replace(self._state, cursor_id=fork._state.cursor_id)
        return fork

    def commit_planning(self, outcome: TwistPlanningOutcome) -> TwistRecoverySnapshot:
        """Commit logical selection even when the corresponding plan failed.

        Args:
            outcome: Typed proposal returned by the current planning attempt.

        Returns:
            Owned newly committed cursor snapshot.
        """
        self._require_open()
        if type(outcome) is not TwistPlanningOutcome:
            raise TypeError("Twist cursor requires exactly TwistPlanningOutcome.")
        proposed = outcome.next_state
        if (
            proposed.scope != self._state.scope
            or proposed.cursor_id != self._state.cursor_id
        ):
            raise ValueError("Twist planning outcome belongs to another cursor scope.")
        if proposed.version != self._state.version + 1:
            raise ValueError("Twist planning outcome has a stale cursor version.")
        for name in (
            "forced_index",
            "reserved_indices",
            "fallback_count",
            "locked",
            "stalled_chunks",
            "contact_path",
            "goal_complete",
            "last_feedback_timestamp",
        ):
            if getattr(proposed, name) != getattr(self._state, name):
                raise ValueError(
                    "Planning cannot change measured Twist recovery state."
                )
        self._state = deepcopy(proposed)
        self._committed = deepcopy(outcome)
        self._adopted = None
        return self.snapshot()

    def adopt_plan(self, plan: ActionPlan) -> None:
        """Bind subsequent measured feedback to the actual successful plan.

        Adoption does not advance candidate or fallback state. Failed proposals
        remain committed logically but cannot authorize physical feedback.
        """
        self._require_open()
        if not isinstance(plan, ActionPlan):
            raise TypeError("Twist adoption requires an ActionPlan.")
        outcome = plan.planning_recovery_outcome
        if type(outcome) is not TwistPlanningOutcome or outcome != self._committed:
            raise ValueError(
                "Twist adoption must identify the last committed proposal."
            )
        if (
            plan.plan_success.numel() != 1
            or not plan.plan_success.all()
            or plan.joint_trajectory is None
            or plan.commands.frame_count == 0
            or plan.commands.env_ids.tolist() != [self._state.scope.env_id]
            or type(plan.recovery_policy.planning_recovery) is not TwistPlanningRecovery
        ):
            raise ValueError(
                "Twist adoption requires one successful native environment and its typed joint plan."
            )
        self._adopted = (
            outcome.invocation_id,
            outcome.invocation_revision,
            outcome.plan_id,
        )

    def commit_measurement(
        self, feedback: TwistMeasuredFeedback
    ) -> TwistRecoveryDecision:
        """Advance physical recovery only from fresh feedback for the adopted plan."""
        self._require_open()
        if type(feedback) is not TwistMeasuredFeedback:
            raise TypeError("Twist feedback must be exactly TwistMeasuredFeedback.")
        if (
            feedback.scope != self._state.scope
            or feedback.cursor_id != self._state.cursor_id
        ):
            raise ValueError("Twist feedback belongs to another cursor scope.")
        if feedback.expected_version != self._state.version:
            raise ValueError("Twist feedback has a stale cursor version.")
        if self._adopted != (
            feedback.invocation_id,
            feedback.invocation_revision,
            feedback.plan_id,
        ):
            raise ValueError("Twist feedback must identify the adopted plan.")
        previous_time = self._state.last_feedback_timestamp
        if previous_time is not None and feedback.timestamp <= previous_time:
            raise ValueError("Twist feedback must follow the previous observation.")
        state, accepted, fallback = self._state, False, False
        if feedback.valid:
            state, accepted, fallback = _contact_decision(
                state,
                self._rule,
                target_contact=feedback.target_contact,
                already_reached=feedback.already_reached or state.goal_complete,
                progress=max(0.0, feedback.contact_path - state.contact_path)
                > self._rule.minimum_contact_progress,
                recovery_safe=feedback.recovery_safe,
            )
        next_roll = next(
            (
                roll
                for index, roll, _ in state.candidates
                if index == state.forced_index
            ),
            state.selected_roll,
        )
        axis_switch = bool(
            fallback
            and next_roll is not None
            and state.selected_roll is not None
            and not _same_grip_axis(next_roll, state.selected_roll)
        )
        self._state = replace(
            state,
            version=state.version + 1,
            contact_path=feedback.contact_path,
            goal_complete=feedback.goal_complete,
            last_feedback_timestamp=feedback.timestamp,
        )
        self._adopted = None
        return TwistRecoveryDecision(accepted, fallback, axis_switch, self.snapshot())

    def reset_ready(self) -> None:
        """Reset at the explicit physical ready boundary, never at every chunk."""
        self._require_open()
        self._state = TwistRecoverySnapshot(
            self._state.scope, self._state.cursor_id, self._state.version + 1
        )
        self._committed = self._adopted = None

    def close(self) -> None:
        """End this episode scope; later planning or feedback must use a new cursor."""
        self._closed = True
        self._committed = self._adopted = None


__all__ = [
    "TwistMeasuredFeedback",
    "TwistPlanningOutcome",
    "TwistPlanningRecovery",
    "TwistRecoveryCursor",
    "TwistRecoveryDecision",
    "TwistRecoveryScope",
    "TwistRecoverySnapshot",
    "required_twist_occurrences",
]
