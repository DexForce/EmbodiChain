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

"""Explicit GenSim state handoff contracts for E1/E2/E6 composition.

The planning context remains immutable.  This module is the only GenSim-owned
boundary used by the migrated actions to resolve a resource, inspect a held
object, and declare the symbolic state that the executor may commit after
physical verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, PlanningFailure
from embodichain.lab.sim.atomic_actions.state import (
    ArticulationJointState,
    HeldObjectState,
    PlanningContext,
)

from .actions_binding import resolve_bound_resource, resolve_shared_resource

__all__ = [
    "GenSimStateHandoff",
    "StagingContract",
    "articulation_effect",
    "held_effect",
    "require_release_effect",
]


@dataclass(frozen=True, slots=True)
class GenSimStateHandoff:
    """Call-local state view owned by one resource binding.

    A handoff never exposes the mutable host task object.  It records the
    resource key and the observed relation needed by the current call; the
    returned ``StateDelta`` is the only state that may cross the call boundary.
    """

    resource: str
    context: PlanningContext

    @classmethod
    def from_request(
        cls, request: Any, context: PlanningContext, *, participant: str
    ) -> GenSimStateHandoff:
        resource = resolve_shared_resource(request, participant=participant)
        return cls(resource=resource, context=context)

    @classmethod
    def from_binding(
        cls, binding: Any, context: PlanningContext, *, participant: str
    ) -> GenSimStateHandoff:
        return cls(
            resource=resolve_bound_resource(binding, participant=participant),
            context=context,
        )

    @property
    def held(self) -> HeldObjectState | None:
        """Return the current relation for this resource."""
        getter = getattr(self.context, "get_held_object", None)
        if getter is not None:
            return getter(self.resource)
        return self.context.task.get_held_object(self.resource)

    def require_held(self, *, entity_id: str | None = None) -> HeldObjectState:
        """Require the declared resource to own exactly one held relation."""
        held = self.held
        exclusive = self.context.task.exclusive_held_object_mask
        if held is None or not exclusive(self.resource).all():
            raise ValueError("GenSim state handoff requires an exclusive held object.")
        if entity_id is not None and held.semantics.entity_id != entity_id:
            raise ValueError(
                f"GenSim state handoff expected {entity_id!r}, "
                f"got {held.semantics.entity_id!r}."
            )
        return held

    def require_free(self, *, participant: str) -> None:
        """Reject reuse of a resource that still owns a held relation."""
        held = self.held
        if held is not None and (held.env_mask is None or held.env_mask.any()):
            raise ValueError(
                f"{participant} cannot reuse resource {self.resource!r} "
                "while it is holding another object."
            )
        for resources, held in getattr(
            self.context.task, "coordinated_held_objects", {}
        ).items():
            if self.resource in resources and (
                held.env_mask is None or held.env_mask.any()
            ):
                raise ValueError(
                    f"{participant} cannot reuse resource {self.resource!r} "
                    "while it owns a coordinated holding."
                )

    def retain(
        self, plan: ActionPlan, held: HeldObjectState | None = None
    ) -> ActionPlan:
        """Declare the observed held relation as the call's terminal effect."""
        relation = self.require_held() if held is None else held
        existing = plan.expected_effects
        if (
            existing.coordinated_held_object_updates
            or existing.articulation_joint_updates
        ):
            raise ValueError(
                "Held-object handoff cannot silently replace coordinated or "
                "articulation effects."
            )
        other_resources = set(existing.held_object_updates) - {self.resource}
        if other_resources:
            raise ValueError(
                "Held-object handoff contains an unrelated resource effect: "
                f"{sorted(other_resources)!r}."
            )
        return replace(
            plan,
            expected_effects=StateDelta(
                held_object_updates={
                    **existing.held_object_updates,
                    self.resource: relation,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class StagingContract:
    """Invocation-owned E2 staging authority for one compiled program."""

    rigid_entity_ids: frozenset[str] = frozenset()
    invocation_ids: frozenset[str] = frozenset()
    retention_sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rigid = frozenset(self.rigid_entity_ids)
        invocations = frozenset(self.invocation_ids)
        sources = {
            str(key): MappingProxyType(dict(value))
            for key, value in dict(self.retention_sources).items()
        }
        if not set(sources).issubset(invocations):
            raise ValueError("Staging retention sources must be bound invocations.")
        object.__setattr__(self, "rigid_entity_ids", rigid)
        object.__setattr__(self, "invocation_ids", invocations)
        object.__setattr__(self, "retention_sources", MappingProxyType(sources))

    @classmethod
    def empty(cls) -> StagingContract:
        return cls()

    def retention_source(self, invocation_id: str | None) -> Mapping[str, Any] | None:
        if invocation_id is None:
            return None
        return self.retention_sources.get(invocation_id)


def held_effect(resource: str, held: HeldObjectState | None) -> StateDelta:
    """Build an explicit attachment/release delta for one resource."""
    if not isinstance(resource, str) or not resource:
        raise ValueError("GenSim handoff resource must be non-empty.")
    return StateDelta(held_object_updates={resource: held})


def require_release_effect(plan: ActionPlan, resource: str) -> ActionPlan:
    """Require a Place plan to release exactly its bound resource."""
    if not plan.plan_success.any():
        return plan
    updates = dict(plan.expected_effects.held_object_updates)
    valid = set(updates) == {resource} and updates[resource] is None
    if valid:
        return plan
    return replace(
        plan,
        plan_success=torch.zeros_like(plan.plan_success),
        diagnostics=replace(
            plan.diagnostics,
            failure=PlanningFailure(
                "place_release_contract_missing",
                retryable=False,
            ),
        ),
    )


def articulation_effect(
    articulation_id: str,
    joint_id: str,
    position: torch.Tensor,
    *,
    env_mask: torch.Tensor | None = None,
) -> StateDelta:
    """Build an explicit verified E6 joint-state delta."""
    return StateDelta(
        articulation_joint_updates={
            (articulation_id, joint_id): ArticulationJointState(
                position=position,
                env_mask=env_mask,
            )
        }
    )
