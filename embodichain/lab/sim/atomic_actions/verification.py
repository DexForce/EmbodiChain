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

"""Typed physical-effect verification boundaries for atomic actions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .effects import StateDelta
from .plans import EffectVerificationRequirement


@dataclass(frozen=True, slots=True, eq=False)
class EffectVerificationRequest:
    """Typed boundary describing a physical effect awaiting verification.

    ``requested_at`` and ``deadline`` use the same timestamp domain as
    :class:`RobotObservation`. Request-mask shrinkage retains both values;
    only a newly installed plan starts a new attempt deadline.
    ``attempt_generation`` is session-local and remains stable when partial
    resolution or row deactivation replaces only the request ID.
    ``failure_invalidation`` is a core-owned removal-only delta; verification
    results may select failed rows on which to apply it but cannot replace it.
    """

    verification_id: int
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    invocation_index: int
    attempt_generation: int
    terminal_segment: str | None
    requested_at: float
    deadline: float
    env_mask: torch.Tensor
    expected_effects: StateDelta
    effect_verification: EffectVerificationRequirement | None = None
    failure_invalidation: StateDelta = field(default_factory=StateDelta)

    def __post_init__(self) -> None:
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        if not isinstance(self.skill_id, str) or not self.skill_id:
            raise ValueError("skill_id must be a non-empty string.")
        if self.invocation_id is not None and (
            not isinstance(self.invocation_id, str) or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        if self.invocation_revision < 0:
            raise ValueError("invocation_revision must be non-negative.")
        if self.invocation_index < 0:
            raise ValueError("invocation_index must be non-negative.")
        if type(self.attempt_generation) is not int or self.attempt_generation < 0:
            raise ValueError("attempt_generation must be a non-negative integer.")
        if self.terminal_segment is not None and (
            not isinstance(self.terminal_segment, str) or not self.terminal_segment
        ):
            raise ValueError("terminal_segment must be a non-empty string or None.")
        if not math.isfinite(self.requested_at) or self.requested_at < 0.0:
            raise ValueError("requested_at must be finite and non-negative.")
        if not math.isfinite(self.deadline) or self.deadline < self.requested_at:
            raise ValueError(
                "deadline must be finite and no earlier than requested_at."
            )
        if not isinstance(self.env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("env_mask must be a 1D bool tensor.")
        if not self.env_mask.any():
            raise ValueError("env_mask must contain at least one requested row.")
        if not isinstance(self.expected_effects, StateDelta):
            raise TypeError("expected_effects must be a StateDelta.")
        if (
            self.effect_verification is not None
            and type(self.effect_verification) is not EffectVerificationRequirement
        ):
            raise TypeError(
                "effect_verification must be exactly "
                "EffectVerificationRequirement or None."
            )
        if self.expected_effects.is_empty and self.effect_verification is None:
            raise ValueError(
                "Effect verification requires expected symbolic effects or an "
                "explicit physical-effect requirement."
            )
        if not isinstance(self.failure_invalidation, StateDelta):
            raise TypeError("failure_invalidation must be a StateDelta.")
        if (
            any(
                value is not None
                for value in self.failure_invalidation.held_object_updates.values()
            )
            or any(
                value is not None
                for value in self.failure_invalidation.coordinated_held_object_updates.values()
            )
            or any(
                value is not None
                for value in self.failure_invalidation.articulation_joint_updates.values()
            )
        ):
            raise ValueError(
                "failure_invalidation may only remove previously verified state."
            )
        object.__setattr__(self, "env_mask", self.env_mask.clone())
        object.__setattr__(self, "expected_effects", self.expected_effects.snapshot())
        object.__setattr__(
            self,
            "failure_invalidation",
            self.failure_invalidation.snapshot(),
        )
        object.__setattr__(
            self,
            "effect_verification",
            (
                None
                if self.effect_verification is None
                else self.effect_verification.snapshot()
            ),
        )

    def snapshot(self) -> EffectVerificationRequest:
        """Return a request snapshot with an independently owned row mask."""
        return EffectVerificationRequest(
            verification_id=self.verification_id,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            invocation_index=self.invocation_index,
            attempt_generation=self.attempt_generation,
            terminal_segment=self.terminal_segment,
            requested_at=self.requested_at,
            deadline=self.deadline,
            env_mask=self.env_mask,
            expected_effects=self.expected_effects,
            effect_verification=self.effect_verification,
            failure_invalidation=self.failure_invalidation,
        )


@dataclass(frozen=True, slots=True, eq=False)
class EffectExpectationResult:
    """Current-observation outcome for one physical state expectation.

    ``inverse_satisfied_mask`` is stronger than contradiction: every clause
    must have reached its explicit inverse band for the monitor's complete
    hysteresis window.  It may therefore be used to retain a pre-existing
    relation during failure reconciliation, while a single contradictory
    clause may not.
    """

    expectation_id: str
    satisfied_mask: torch.Tensor
    contradicted_mask: torch.Tensor
    inverse_satisfied_mask: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.expectation_id) is not str
            or not self.expectation_id
            or self.expectation_id != self.expectation_id.strip()
        ):
            raise ValueError(
                "expectation_id must be a non-empty string without outer whitespace."
            )
        for name in (
            "satisfied_mask",
            "contradicted_mask",
            "inverse_satisfied_mask",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        masks = (
            self.satisfied_mask,
            self.contradicted_mask,
            self.inverse_satisfied_mask,
        )
        if any(mask.shape != masks[0].shape for mask in masks[1:]):
            raise ValueError("Expectation-result masks must have equal shapes.")
        if any(mask.device != masks[0].device for mask in masks[1:]):
            raise ValueError("Expectation-result masks must use the same device.")
        if (self.satisfied_mask & self.contradicted_mask).any():
            raise ValueError("satisfied_mask and contradicted_mask must not overlap.")
        if (self.inverse_satisfied_mask & ~self.contradicted_mask).any():
            raise ValueError(
                "inverse_satisfied_mask must be a subset of contradicted_mask."
            )
        for name in (
            "satisfied_mask",
            "contradicted_mask",
            "inverse_satisfied_mask",
        ):
            object.__setattr__(self, name, getattr(self, name).clone())

    def snapshot(self) -> EffectExpectationResult:
        """Return an independently owned expectation outcome."""
        return EffectExpectationResult(
            expectation_id=self.expectation_id,
            satisfied_mask=self.satisfied_mask,
            contradicted_mask=self.contradicted_mask,
            inverse_satisfied_mask=self.inverse_satisfied_mask,
        )


@dataclass(frozen=True, slots=True, eq=False)
class EffectVerificationResult:
    """Correlated per-environment update for one effect boundary.

    Rows absent from both ``success_mask`` and ``failure_mask`` remain
    unresolved. ``invalidation_mask`` and ``retry_mask`` classify only failed
    rows: the former selects the request's core-owned removal delta, while the
    latter authorizes replay of the same invocation. Failed rows outside the
    retry mask require external recovery. This lets one shared batch barrier
    commit verified rows while other rows continue observing the same physical
    effect.
    """

    verification_id: int
    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    invalidation_mask: torch.Tensor
    retry_mask: torch.Tensor
    expectation_results: tuple[EffectExpectationResult, ...] = ()

    def __post_init__(self) -> None:
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        for name in (
            "success_mask",
            "failure_mask",
            "invalidation_mask",
            "retry_mask",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        masks = (
            self.success_mask,
            self.failure_mask,
            self.invalidation_mask,
            self.retry_mask,
        )
        if any(mask.shape != masks[0].shape for mask in masks[1:]):
            raise ValueError("Effect-result masks must have equal shapes.")
        if any(mask.device != masks[0].device for mask in masks[1:]):
            raise ValueError("Effect-result masks must use the same device.")
        if (self.success_mask & self.failure_mask).any():
            raise ValueError("success_mask and failure_mask must not overlap.")
        if (self.invalidation_mask & ~self.failure_mask).any():
            raise ValueError("invalidation_mask must be a subset of failure_mask.")
        if (self.retry_mask & ~self.failure_mask).any():
            raise ValueError("retry_mask must be a subset of failure_mask.")
        expectation_results = tuple(self.expectation_results)
        if not all(
            type(value) is EffectExpectationResult for value in expectation_results
        ):
            raise TypeError(
                "expectation_results must contain exact EffectExpectationResult values."
            )
        expectation_ids = [value.expectation_id for value in expectation_results]
        if len(set(expectation_ids)) != len(expectation_ids):
            raise ValueError("Effect expectation-result IDs must be unique.")
        if expectation_results:
            expected_success = torch.ones_like(self.success_mask)
            expected_failure = torch.zeros_like(self.failure_mask)
            for value in expectation_results:
                if value.satisfied_mask.shape != self.success_mask.shape:
                    raise ValueError(
                        "Expectation and aggregate result masks must have equal shapes."
                    )
                if value.satisfied_mask.device != self.success_mask.device:
                    raise ValueError(
                        "Expectation and aggregate result masks must use the same device."
                    )
                expected_success &= value.satisfied_mask
                expected_failure |= value.contradicted_mask
            if not torch.equal(self.success_mask, expected_success):
                raise ValueError(
                    "success_mask must equal the conjunction of expectation results."
                )
            if not torch.equal(self.failure_mask, expected_failure):
                raise ValueError(
                    "failure_mask must equal the union of expectation results."
                )
        for name in (
            "success_mask",
            "failure_mask",
            "invalidation_mask",
            "retry_mask",
        ):
            object.__setattr__(self, name, getattr(self, name).clone())
        object.__setattr__(
            self,
            "expectation_results",
            tuple(value.snapshot() for value in expectation_results),
        )


@dataclass(frozen=True, slots=True, eq=False)
class PhaseEffectGateRequest:
    """Correlate a blocking physical-effect check with a segment entry.

    The action's preceding command remains active while the gate is unresolved.
    A gate is scoped to the enclosing action attempt and does not create a
    separate planning, recovery, or timeout budget.

    Args:
        verification_id: Session-local single-use request identity.
        gate_id: Invocation-local stable gate identity.
        skill_id: Registered action skill identity.
        invocation_id: Optional logical invocation correlation identity.
        invocation_revision: Active invocation revision.
        invocation_index: Active invocation position in the session.
        attempt_generation: Installed action-plan attempt generation.
        next_waypoint_index: First command frame blocked by the gate.
        segment_name: Named trajectory segment blocked by the gate.
        requested_at: Request creation time in the observation timestamp domain.
        deadline: Enclosing action deadline in that same timestamp domain.
        env_mask: Active rows that must satisfy the gate together.
    """

    verification_id: int
    gate_id: str
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    invocation_index: int
    attempt_generation: int
    next_waypoint_index: int
    segment_name: str
    requested_at: float
    deadline: float
    env_mask: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "verification_id",
            "invocation_revision",
            "invocation_index",
            "attempt_generation",
            "next_waypoint_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("gate_id", "skill_id", "segment_name"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    f"{name} must be a non-empty string without outer whitespace."
                )
        if self.invocation_id is not None and (
            type(self.invocation_id) is not str or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        if not math.isfinite(self.requested_at) or self.requested_at < 0.0:
            raise ValueError("requested_at must be finite and non-negative.")
        if not math.isfinite(self.deadline) or self.deadline < self.requested_at:
            raise ValueError(
                "deadline must be finite and no earlier than requested_at."
            )
        if not isinstance(self.env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("env_mask must be a one-dimensional bool tensor.")
        if not self.env_mask.any():
            raise ValueError("env_mask must contain at least one gated row.")
        object.__setattr__(self, "env_mask", self.env_mask.clone())

    def snapshot(self) -> PhaseEffectGateRequest:
        """Return an independently owned gate request."""
        return PhaseEffectGateRequest(
            verification_id=self.verification_id,
            gate_id=self.gate_id,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            invocation_index=self.invocation_index,
            attempt_generation=self.attempt_generation,
            next_waypoint_index=self.next_waypoint_index,
            segment_name=self.segment_name,
            requested_at=self.requested_at,
            deadline=self.deadline,
            env_mask=self.env_mask,
        )


@dataclass(frozen=True, slots=True, eq=False)
class PhaseEffectGateResult:
    """Current-observation decision for one blocking segment-entry gate.

    Rows absent from both decision masks remain unresolved. ``retry_mask`` is
    a subset of failed rows for which replaying the enclosing action remains
    valid; no gate outcome mutates verified task state.

    Args:
        verification_id: Identity copied from the consumed gate request.
        gate_id: Stable gate identity copied from the request.
        attempt_generation: Action attempt copied from the request.
        invocation_index: Session invocation index copied from the request.
        next_waypoint_index: Blocked waypoint copied from the request.
        success_mask: Rows whose current evidence satisfies the gate.
        failure_mask: Rows whose current evidence contradicts the gate.
        retry_mask: Failed rows allowed to retry the enclosing action.
        message: Optional physical-failure diagnostic.
    """

    verification_id: int
    gate_id: str
    attempt_generation: int
    invocation_index: int
    next_waypoint_index: int
    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    retry_mask: torch.Tensor
    message: str = ""

    def __post_init__(self) -> None:
        for name in (
            "verification_id",
            "attempt_generation",
            "invocation_index",
            "next_waypoint_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            type(self.gate_id) is not str
            or not self.gate_id
            or self.gate_id != self.gate_id.strip()
        ):
            raise ValueError(
                "gate_id must be a non-empty string without outer whitespace."
            )
        masks = (self.success_mask, self.failure_mask, self.retry_mask)
        for name, value in zip(
            ("success_mask", "failure_mask", "retry_mask"),
            masks,
            strict=True,
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        if any(value.shape != masks[0].shape for value in masks[1:]):
            raise ValueError("Phase-effect gate masks must have equal shapes.")
        if any(value.device != masks[0].device for value in masks[1:]):
            raise ValueError("Phase-effect gate masks must use the same device.")
        if (self.success_mask & self.failure_mask).any():
            raise ValueError("Gate success and failure masks must not overlap.")
        if (self.retry_mask & ~self.failure_mask).any():
            raise ValueError("retry_mask must be a subset of failure_mask.")
        if type(self.message) is not str:
            raise TypeError("message must be a string.")
        for name in ("success_mask", "failure_mask", "retry_mask"):
            object.__setattr__(self, name, getattr(self, name).clone())


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectGuardRequest:
    """Describe the next in-flight command boundary for held-object checks.

    A request is correlated to one installed action-plan attempt and one next
    waypoint. The named segment lets an external verifier select phase-aware
    physical evidence without teaching the execution core skill-specific
    phases. ``deadline`` uses the observation timestamp domain.
    """

    verification_id: int
    skill_id: str
    invocation_id: str | None
    invocation_revision: int
    invocation_index: int
    attempt_generation: int
    next_waypoint_index: int
    segment_name: str
    env_mask: torch.Tensor
    allowed_held_object_relations: tuple[tuple[str, str], ...]
    allowed_coordinated_held_object_relations: tuple[tuple[str, str, str], ...]
    deadline: float

    def __post_init__(self) -> None:
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        if type(self.skill_id) is not str or not self.skill_id:
            raise ValueError("skill_id must be a non-empty string.")
        if self.invocation_id is not None and (
            type(self.invocation_id) is not str or not self.invocation_id
        ):
            raise ValueError("invocation_id must be a non-empty string or None.")
        for name in (
            "invocation_revision",
            "invocation_index",
            "attempt_generation",
            "next_waypoint_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if type(self.segment_name) is not str or not self.segment_name:
            raise ValueError("segment_name must be a non-empty string.")
        if not isinstance(self.env_mask, torch.Tensor):
            raise TypeError("env_mask must be a torch.Tensor.")
        if self.env_mask.dtype != torch.bool or self.env_mask.dim() != 1:
            raise ValueError("env_mask must be a one-dimensional bool tensor.")
        if not self.env_mask.any():
            raise ValueError("env_mask must contain at least one guarded row.")
        held_relations = tuple(self.allowed_held_object_relations)
        if len(set(held_relations)) != len(held_relations) or not all(
            type(value) is tuple
            and len(value) == 2
            and all(type(item) is str and item for item in value)
            for value in held_relations
        ):
            raise ValueError(
                "allowed_held_object_relations must contain unique "
                "(task_state_key, object_id) pairs."
            )
        coordinated_relations = tuple(self.allowed_coordinated_held_object_relations)
        if len(set(coordinated_relations)) != len(coordinated_relations) or not all(
            type(value) is tuple
            and len(value) == 3
            and all(type(item) is str and item for item in value)
            for value in coordinated_relations
        ):
            raise ValueError(
                "allowed_coordinated_held_object_relations must contain unique "
                "(first_key, second_key, object_id) triples."
            )
        if not math.isfinite(self.deadline) or self.deadline < 0.0:
            raise ValueError("deadline must be finite and non-negative.")
        object.__setattr__(self, "env_mask", self.env_mask.clone())
        object.__setattr__(self, "allowed_held_object_relations", held_relations)
        object.__setattr__(
            self,
            "allowed_coordinated_held_object_relations",
            coordinated_relations,
        )

    def snapshot(self) -> HeldObjectGuardRequest:
        """Return an independently owned guard request.

        Returns:
            Request with an independently owned environment mask.
        """
        return HeldObjectGuardRequest(
            verification_id=self.verification_id,
            skill_id=self.skill_id,
            invocation_id=self.invocation_id,
            invocation_revision=self.invocation_revision,
            invocation_index=self.invocation_index,
            attempt_generation=self.attempt_generation,
            next_waypoint_index=self.next_waypoint_index,
            segment_name=self.segment_name,
            env_mask=self.env_mask,
            allowed_held_object_relations=self.allowed_held_object_relations,
            allowed_coordinated_held_object_relations=(
                self.allowed_coordinated_held_object_relations
            ),
            deadline=self.deadline,
        )


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectGuardResult:
    """Correlated in-flight held-object loss and recovery decision.

    ``state_invalidation`` may only remove single-resource or coordinated
    held-object relations. It is applied to ``failure_mask`` before recovery
    planning, so a retry always observes reconciled symbolic state.
    """

    verification_id: int
    object_id: str
    attempt_generation: int
    invocation_index: int
    next_waypoint_index: int
    failure_mask: torch.Tensor
    state_invalidation: StateDelta
    retry_mask: torch.Tensor
    message: str = ""

    def __post_init__(self) -> None:
        if type(self.verification_id) is not int or self.verification_id < 0:
            raise ValueError("verification_id must be a non-negative integer.")
        if type(self.object_id) is not str or not self.object_id:
            raise ValueError("object_id must be a non-empty string.")
        for name in (
            "attempt_generation",
            "invocation_index",
            "next_waypoint_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("failure_mask", "retry_mask"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.dim() != 1:
                raise ValueError(f"{name} must be a one-dimensional bool tensor.")
        if self.failure_mask.shape != self.retry_mask.shape:
            raise ValueError("failure_mask and retry_mask must have equal shapes.")
        if self.failure_mask.device != self.retry_mask.device:
            raise ValueError("failure_mask and retry_mask must use the same device.")
        if (self.retry_mask & ~self.failure_mask).any():
            raise ValueError("retry_mask must be a subset of failure_mask.")
        if not isinstance(self.state_invalidation, StateDelta):
            raise TypeError("state_invalidation must be a StateDelta.")
        if any(
            value is not None
            for value in self.state_invalidation.held_object_updates.values()
        ) or any(
            value is not None
            for value in self.state_invalidation.coordinated_held_object_updates.values()
        ):
            raise ValueError(
                "state_invalidation may only remove held-object relations."
            )
        if self.state_invalidation.articulation_joint_updates:
            raise ValueError(
                "state_invalidation cannot update articulation-joint state."
            )
        has_invalidation = bool(
            self.state_invalidation.held_object_updates
            or self.state_invalidation.coordinated_held_object_updates
        )
        if bool(self.failure_mask.any().item()) != has_invalidation:
            raise ValueError(
                "state_invalidation must contain relation removals exactly when "
                "failure_mask contains failed rows."
            )
        if type(self.message) is not str:
            raise TypeError("message must be a string.")
        object.__setattr__(self, "failure_mask", self.failure_mask.clone())
        object.__setattr__(self, "retry_mask", self.retry_mask.clone())
        object.__setattr__(
            self,
            "state_invalidation",
            self.state_invalidation.snapshot(),
        )


__all__ = [
    "EffectExpectationResult",
    "EffectVerificationRequest",
    "EffectVerificationResult",
    "HeldObjectGuardRequest",
    "HeldObjectGuardResult",
    "PhaseEffectGateRequest",
    "PhaseEffectGateResult",
]
