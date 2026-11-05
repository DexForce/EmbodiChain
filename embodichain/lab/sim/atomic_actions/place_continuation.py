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

"""Canonical action requests for private, occurrence-bound release predictions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext
    from .plans import ActionPlan
    from .planning_probes import PlaceSceneInputs
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "PlaceContinuationRoute",
    "PlaceContinuationSession",
    "PlaceContinuationProvider",
]


@dataclass(frozen=True, slots=True)
class PlaceContinuationRoute:
    """Exact ordered occurrence identities; predictions never authorize effects."""

    source_invocation_id: str
    place_invocation_id: str
    object_id: str
    transport_invocation_id: str | None = None
    requires_full_preview: bool = True

    def __post_init__(self) -> None:
        for name in (
            "source_invocation_id",
            "place_invocation_id",
            "object_id",
            "transport_invocation_id",
        ):
            value = getattr(self, name)
            if name == "transport_invocation_id" and value is None:
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact nonempty identity.")
        identities = [self.source_invocation_id, self.place_invocation_id]
        if self.transport_invocation_id is not None:
            identities.append(self.transport_invocation_id)
        if len(set(identities)) != len(identities):
            raise ValueError("Continuation occurrences must remain distinct.")
        if type(self.requires_full_preview) is not bool:
            raise TypeError("requires_full_preview must be a bool.")


@runtime_checkable
class PlaceContinuationSession(Protocol):
    """Host-owned canonical grounding; Atomic owns prediction and validation.

    Grounding methods must verify exact occurrence, resource and effect-monitor
    contracts before returning requests. They must not acquire physical facts
    from a projected state. The session owns no mutable simulator state.
    """

    @property
    def route(self) -> PlaceContinuationRoute:
        """Return immutable source/transport/release occurrence declarations."""
        ...

    def validate_source(self, context: PlanningContext) -> None:
        """Validate canonical acquisition monitor and expected object/resource."""
        ...

    def resolve_transport(self, context: PlanningContext) -> ResolvedActionRequest:
        """Ground the declared retained transport against the private context."""
        ...

    def resolve_place(self, context: PlanningContext) -> ResolvedActionRequest:
        """Ground the declared release against the private context."""
        ...

    def plan(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ActionPlan:
        """Use the ordinary complete engine path, including policy and transforms."""
        ...

    def place_scene_inputs(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PlaceSceneInputs:
        """Return owned declared scene geometry for emitted release inspection."""
        ...


@runtime_checkable
class PlaceContinuationProvider(Protocol):
    """Per-engine, explicitly bound canonical continuation owner."""

    @property
    def robot(self) -> Robot:
        """Return the exact engine robot."""
        ...

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PlaceContinuationSession | None:
        """Return only this occurrence's session; previews must not recurse."""
        ...
