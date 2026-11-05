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

"""Typed installed-plan facts for bounded prismatic cleanup continuation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable

from .policies import PlanningRecovery
from .plans import PlanningRecoveryOutcome

if TYPE_CHECKING:
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "SlidePlanningRecovery",
    "ArticulationWithdrawalPlanningRecovery",
    "ArticulationParkPlanningRecovery",
    "ArticulationInstalledPlan",
    "ArticulationContinuationContext",
    "ArticulationPlanningOutcome",
    "ArticulationContinuationProvider",
    "capture_installed_articulation_plan",
    "require_articulation_withdrawal_state",
    "articulation_withdrawal_targets",
    "short_articulation_withdrawal_targets",
]


@dataclass(frozen=True, slots=True)
class SlidePlanningRecovery(PlanningRecovery):
    """Fixed B1 source recovery; paired cleanup is an explicit provider fact."""

    kind: str = field(default="slide_bounded", init=False)
    skill_id: str = field(default="slide", init=False)
    revision: int = field(default=1, init=False)
    maximum_frames: int = field(default=390, init=False)
    maximum_free_approaches: int = field(default=2, init=False)
    approach_distances: tuple[float, ...] = field(default=(0.05, 0.03), init=False)
    preshape_aperture_recovery: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        PlanningRecovery.__post_init__(self)
        if type(self.preshape_aperture_recovery) is not bool:
            raise TypeError("preshape_aperture_recovery must be bool.")


@dataclass(frozen=True, slots=True)
class ArticulationWithdrawalPlanningRecovery(PlanningRecovery):
    """Fixed B1 cleanup recovery, conditioned on installed predecessor facts."""

    kind: str = field(default="articulation_withdrawal", init=False)
    skill_id: str = field(default="move_end_effector", init=False)
    revision: int = field(default=1, init=False)
    maximum_frames: int = field(default=390, init=False)


@dataclass(frozen=True, slots=True)
class ArticulationParkPlanningRecovery(PlanningRecovery):
    """Source Park clearance fallback only after an installed declared withdrawal."""

    kind: str = field(default="articulation_park", init=False)
    skill_id: str = field(default="move_joints", init=False)
    revision: int = field(default=1, init=False)


def _identity(value: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("Articulation continuation requires exact identities.")


def _ordinal(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(
            "Articulation continuation versions must be nonnegative integers."
        )


@dataclass(frozen=True, slots=True)
class ArticulationInstalledPlan:
    """One validated installation fact, never a physical task-success verdict."""

    kind: Literal["slide", "withdraw", "park"]
    invocation_id: str
    invocation_revision: int
    planned: bool
    withdrawal_selection_changed: bool
    withdrawal_target_accepted: bool

    def __post_init__(self) -> None:
        _identity(self.invocation_id)
        _ordinal(self.invocation_revision)
        if self.kind not in ("slide", "withdraw", "park") or any(
            type(value) is not bool
            for value in (
                self.planned,
                self.withdrawal_selection_changed,
                self.withdrawal_target_accepted,
            )
        ):
            raise ValueError(
                "Installed articulation facts require exact roles and booleans."
            )


@dataclass(frozen=True, slots=True)
class ArticulationContinuationContext:
    """Immutable same-scope history and declared cleanup grid for one invocation.

    Only the host's actual installation hook may add predecessor facts. Planning
    previews and returned plans do not mutate this value or authorize continuation.
    The opaque scope ID belongs to an explicit program/episode operation owner.
    """

    scope_id: str
    history_version: int
    kind: Literal["slide", "withdraw", "park"]
    invocation_id: str
    withdrawal_sample_count: int | None
    predecessors: tuple[ArticulationInstalledPlan, ...] = ()

    def __post_init__(self) -> None:
        _identity(self.scope_id)
        _identity(self.invocation_id)
        _ordinal(self.history_version)
        if self.kind not in ("slide", "withdraw", "park"):
            raise ValueError("Unknown articulation continuation role.")
        if self.withdrawal_sample_count is not None and (
            type(self.withdrawal_sample_count) is not int
            or self.withdrawal_sample_count < 1
        ):
            raise ValueError(
                "Declared cleanup sample count must be a positive integer."
            )
        if type(self.predecessors) is not tuple or any(
            type(value) is not ArticulationInstalledPlan for value in self.predecessors
        ):
            raise TypeError(
                "Continuation predecessors must be immutable installed-plan facts."
            )

    @property
    def withdrawal_reselected(self) -> bool:
        """Retain source any-installed-successful-Slide selection semantics."""
        return any(
            value.kind == "slide"
            and value.planned
            and value.withdrawal_selection_changed
            for value in self.predecessors
        )

    @property
    def park_after_withdrawal(self) -> bool:
        """Retain source any-installed nominal or accepted recovered withdrawal."""
        return any(
            value.kind == "withdraw"
            and value.planned
            and value.withdrawal_target_accepted
            for value in self.predecessors
        )


@dataclass(frozen=True, slots=True)
class ArticulationPlanningOutcome(PlanningRecoveryOutcome):
    """Uncommitted planning facts; only installation may advance the host history."""

    scope_id: str
    history_version: int
    withdrawal_selection_changed: bool = False
    withdrawal_target_accepted: bool = True

    def __post_init__(self) -> None:
        PlanningRecoveryOutcome.__post_init__(self)
        _identity(self.scope_id)
        _ordinal(self.history_version)
        if any(
            type(value) is not bool
            for value in (
                self.withdrawal_selection_changed,
                self.withdrawal_target_accepted,
            )
        ):
            raise TypeError("Articulation planning flags must be explicit booleans.")


@runtime_checkable
class ArticulationContinuationProvider(Protocol):
    """Return exact invocation-owned facts without installing or mutating a plan."""

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed engine robot."""
        ...

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ArticulationContinuationContext | None:
        """Return declared facts, or None for an unrelated ordinary invocation."""
        ...


def capture_installed_articulation_plan(
    context: ArticulationContinuationContext,
    outcome: ArticulationPlanningOutcome | None,
    *,
    invocation_id: str,
    invocation_revision: int,
    skill_id: str,
    planned: bool,
) -> ArticulationInstalledPlan:
    """Validate exact outcome scope/version at the installation boundary.

    Failed installations may lack an outcome and cannot grant continuation.
    Successful installations require their typed proposal; neither inspecting a
    successful plan nor creating an outcome calls this function automatically.
    """
    if (
        type(context) is not ArticulationContinuationContext
        or type(planned) is not bool
    ):
        raise TypeError(
            "Installation requires typed continuation context and planned mask fact."
        )
    expected_skill = {
        "slide": "slide",
        "withdraw": "move_end_effector",
        "park": "move_joints",
    }[context.kind]
    if invocation_id != context.invocation_id or skill_id != expected_skill:
        raise ValueError(
            "Installed articulation plan belongs to another occurrence or skill."
        )
    _ordinal(invocation_revision)
    if outcome is None:
        if planned:
            raise ValueError(
                "A successful articulation installation requires its planning outcome."
            )
        changed, accepted = False, True
    else:
        if (
            type(outcome) is not ArticulationPlanningOutcome
            or outcome.scope_id != context.scope_id
            or outcome.history_version != context.history_version
            or outcome.invocation_id != invocation_id
            or outcome.invocation_revision != invocation_revision
            or outcome.skill_id != skill_id
        ):
            raise ValueError(
                "Articulation outcome is stale or belongs to another scope."
            )
        changed, accepted = (
            outcome.withdrawal_selection_changed,
            outcome.withdrawal_target_accepted,
        )
    return ArticulationInstalledPlan(
        context.kind, invocation_id, invocation_revision, planned, changed, accepted
    )


def articulation_withdrawal_targets(
    current: torch.Tensor, outward: torch.Tensor
) -> torch.Tensor:
    """Build the original world-up 0.12 m, then outward 0.10 m cleanup targets."""
    raised = current.clone()
    raised[:, 2, 3] += 0.12
    withdrawn = raised.clone()
    withdrawn[:, :3, 3] += 0.10 * outward
    return torch.stack((current, raised, withdrawn), dim=1)


def short_articulation_withdrawal_targets(nominal: torch.Tensor) -> torch.Tensor:
    """Build the source outward-only 0.02 m alternative without changing wrist pose."""
    if nominal.ndim != 4 or nominal.shape[1:] != (3, 4, 4):
        raise ValueError(
            "Short withdrawal requires the owned three-pose cleanup route."
        )
    direction = nominal[:, -1, :3, 3] - nominal[:, -2, :3, 3]
    length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if not torch.isfinite(length).all() or (length <= 0).any():
        raise ValueError("Withdrawal rail direction is unavailable.")
    start = nominal[:, 0]
    end = start.clone()
    end[:, :3, 3] += 0.02 * direction / length
    return torch.stack((start, end), dim=1)


def require_articulation_withdrawal_state(
    joint_positions: torch.Tensor,
    *,
    target: float,
    joint_span: float,
    observed_hand: torch.Tensor | None = None,
    open_hand: torch.Tensor | None = None,
) -> None:
    """Retain the source target/hand-open preconditions before empty-hand cleanup.

    These instantaneous physical checks are not a TaskSpec success judgment.
    Inputs are caller-owned observations and declared commands; no state is read.
    """
    if not math.isfinite(target) or not math.isfinite(joint_span) or joint_span <= 0:
        raise ValueError("Withdrawal requires finite target and positive joint span.")
    tolerance = min(0.03, joint_span * 0.49)
    if (
        not joint_positions.numel()
        or not torch.isfinite(joint_positions).all()
        or ((joint_positions - target).abs() > tolerance).any()
    ):
        raise ValueError("Articulation target was lost before withdrawal.")
    if observed_hand is not None:
        if (
            open_hand is None
            or open_hand.shape != observed_hand.shape
            or not torch.isfinite(open_hand).all()
            or not torch.isfinite(observed_hand).all()
            or (observed_hand - open_hand).abs().amax() > 0.08
        ):
            raise ValueError(
                "The hand has not reached its open posture before withdrawal."
            )
