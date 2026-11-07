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

"""Read-only placement inputs and an observed-state joint-prefix planning port."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot

if TYPE_CHECKING:
    from embodichain.lab.sim.motion.planners.utils import PlanResult
    from embodichain.lab.sim.objects import Robot
    from .invocation import ResolvedActionRequest
    from .primitives.place import PlaceGoal, PlaceOptions
    from .state import PlanningContext

__all__ = ["JointPrefixPlanner", "PlaceSceneInputs", "PlaceSceneProvider"]


@runtime_checkable
class JointPrefixPlanner(Protocol):
    """Backend adapter for a fresh-world prefix with observed passive joints."""

    @property
    def robot(self) -> Robot:
        """Return the exact robot whose full observed state the planner consumes."""
        ...

    def plan(
        self,
        *,
        context: PlanningContext,
        control_part: str,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        full_qpos: torch.Tensor,
        control_dt: float,
    ) -> PlanResult:
        """Plan an unchanged joint target from one explicit observation.

        Args:
            context: Exact observed root and scene/version scope.
            control_part: Declared active control group.
            start_qpos: Observed controlled joints, shape ``(1, D)``.
            target_qpos: Unchanged controlled-joint target, shape ``(1, D)``.
            full_qpos: Same-observation full robot state, shape ``(1, J)``.
            control_dt: Positive command interval in seconds.

        Returns:
            Owned native timed samples. The adapter owns backend lifetime and
            world/base projection; it does not select another target or phase.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaceSceneInputs:
    """Owned planning envelopes with explicit floor and intended-support roles.

    These are planning inputs, not contact measurements or physical certificates.
    Consumers read current poses from their PlanningContext.

    Args:
        geometry: Scaled visual envelopes and collision-input components.
        floor_entity_id: Explicit rigid floor owner present in the envelopes.
        support_owner_id: Optional intended-contact owner excluded from payload
            free-space checks. Release/support evidence is still required.
    """

    geometry: SceneMeshSnapshot
    floor_entity_id: str
    support_owner_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, SceneMeshSnapshot):
            raise TypeError("Place geometry must be a SceneMeshSnapshot.")
        for name in ("floor_entity_id", "support_owner_id"):
            value = getattr(self, name)
            if name == "support_owner_id" and value is None:
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact nonempty identity.")
        if self.floor_entity_id not in self.geometry:
            raise ValueError(
                "Place floor requires an explicit captured rigid envelope."
            )
        owners = set(self.geometry) | {
            component.owner_id for component in self.geometry.components
        }
        if self.support_owner_id is not None and self.support_owner_id not in owners:
            raise ValueError("Intended Place support is absent from captured geometry.")
        object.__setattr__(
            self,
            "geometry",
            SceneMeshSnapshot(
                self.geometry,
                components=self.geometry.components,
                collision_roles=self.geometry.collision_roles,
                unavailable=self.geometry.unavailable,
            ),
        )

    def snapshot(self) -> PlaceSceneInputs:
        """Return independent arrays/components for another ownership boundary."""
        return replace(self)


@runtime_checkable
class PlaceSceneProvider(Protocol):
    """Resolve declared planning geometry without Task Program objects."""

    def prepare(
        self,
        request: ResolvedActionRequest[PlaceGoal, PlaceOptions],
        context: PlanningContext,
    ) -> PlaceSceneInputs:
        """Return owned envelopes and exact intended-contact identities.

        Args:
            request: Current resolved Place invocation and declared policies.
            context: Same-timestamp robot and scene observation.

        Returns:
            Planning-only inputs. Missing required geometry raises rather than
            changing the collision oracle or certifying clearance.
        """
        ...
