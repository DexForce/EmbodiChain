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

"""Explicit measured rigid planning facts supplied by trusted physical hosts."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import torch
from embodichain.lab.sim.scene.native_motion_scene import (
    NativeRestSceneQuery,
    NativeRestCollisionPart,
)
from embodichain.lab.sim.atomic_actions.source_native_contact import (
    NativeClosingQuery,
    NativeContactPart,
)
from embodichain.lab.sim.scene.planar_observation import NativePlanarPlacement

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext
    from .core import ObjectSemantics

__all__ = [
    "MeasuredInsidePlanningInputs",
    "MeasuredRigidPlanningSession",
    "MeasuredRigidPlanningProvider",
]


@dataclass(frozen=True, slots=True)
class MeasuredInsidePlanningInputs:
    """Physical geometry/identities for one original inside-release occurrence.

    No task graph or normative success criteria enter this snapshot. Content
    mesh arrays remain source-ordered scaled local vertices; Atomic owns their
    fresh context-relative projection and the unchanged release displacement.
    """

    subject_id: str
    container_id: str
    parent_id: str
    release_clearance: float
    geometry: NativePlanarPlacement
    content_vertices: tuple[tuple[str, torch.Tensor], ...]

    def __post_init__(self) -> None:
        for name in ("subject_id", "container_id", "parent_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    "Measured inside facts require exact entity identities."
                )
        if not isinstance(self.geometry, NativePlanarPlacement):
            raise TypeError(
                "Measured inside geometry requires its physical planar owner."
            )
        owned = []
        names = []
        for name, value in self.content_vertices:
            if (
                type(name) is not str
                or not name
                or name != name.strip()
                or value.ndim != 2
                or value.shape[1] != 3
                or not len(value)
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    "Measured content facts require finite source-local geometry."
                )
            names.append(name)
            owned.append((name, value.detach().clone()))
        if len(names) != len(set(names)):
            raise ValueError("Measured content identities are duplicated.")
        object.__setattr__(self, "content_vertices", tuple(owned))


@runtime_checkable
class MeasuredRigidPlanningSession(Protocol):
    """One same-observation lifetime; Atomic owns policy and plan transformation."""

    subject_id: str
    source_uid: str
    container_id: str | None
    physics_backend: str

    def closing_query(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        semantics: ObjectSemantics,
    ) -> NativeClosingQuery:
        """Capture original revision-four hand geometry using current commands."""
        ...

    def scene_query(self) -> NativeRestSceneQuery:
        """Capture one owned native rest-offset scene at the original observation."""
        ...

    def arm_parts(
        self, control_part: str, hand: Sequence[NativeContactPart]
    ) -> tuple[NativeRestCollisionPart, ...]:
        """Capture sorted source arm collision links excluding this hand."""
        ...

    def inside_inputs(self, context: PlanningContext) -> MeasuredInsidePlanningInputs:
        """Collect required source inside geometry and ordered non-target contents."""
        ...

    def validate_observation(self) -> None:
        """Reject changed robot/native/episode source without refreshing."""
        ...

    def close(self) -> None:
        """Release owned queries only; never close the host Robot/simulation."""
        ...


@runtime_checkable
class MeasuredRigidPlanningProvider(Protocol):
    """Supply exact declared occurrences without implementing an action algorithm."""

    @property
    def robot(self) -> Robot:
        """Return this provider's exact engine-owned Robot."""
        ...

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> MeasuredRigidPlanningSession | None:
        """Bind an explicitly selected source occurrence without motion or simulation steps."""
        ...
