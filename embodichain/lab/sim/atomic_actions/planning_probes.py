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


"""Shared observed-motion and placement inputs, independent of skill algorithms."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ContextManager, Protocol, runtime_checkable
import torch
from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot
from embodichain.lab.sim.scene.planar_observation import NativePlanarPlacement

if TYPE_CHECKING:
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext
    from .primitives.place import PlaceGoal, PlaceOptions
    from embodichain.lab.sim.motion.planners.utils import PlanResult
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from embodichain.lab.sim.motion.planners.curobo.payload import CuroboPayload
    from embodichain.lab.sim.motion.planners.curobo.observed import (
        CuroboWorldObservation,
    )
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "JointPrefixPlanner",
    "CuroboWorldObservationProvider",
    "ObservedCuroboQueryService",
    "PlaceInsideLanding",
    "PlaceSceneInputs",
    "PlaceSceneProvider",
]


@runtime_checkable
class JointPrefixPlanner(Protocol):
    """Owned backend adapter for a fresh-world, locked-full-state joint prefix."""

    @property
    def robot(self) -> Robot:
        """Return the exact robot whose full observed state this planner consumes."""
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
        """Plan an unchanged joint target with explicit observed passive joints.

        Args:
            context: Exact observed robot root and scene/version scope.
            control_part: Declared active control group.
            start_qpos: Observed controlled-joint start (1, D).
            target_qpos: Exact controlled-joint goal (1, D).
            full_qpos: Same-observation full robot state (1, J).
            control_dt: Positive command interval in seconds.

        Returns:
            Owned native timed samples. The adapter owns backend resources and
            explicit world/base projection; no phase or goal choice occurs here.
        """
        ...


@runtime_checkable
class CuroboWorldObservationProvider(Protocol):
    """Supply full collision-roster poses bound to the current planning context.

    The host includes collision-only links and verifies actual native revision
    during acquisition. It must not substitute a later live world for the
    context's captured root/joint state, or fabricate a native revision from the
    semantic scene version. No program/compiler object is returned.
    """

    def capture(
        self, context: PlanningContext, *, excluded_entity_ids: tuple[str, ...] = ()
    ) -> CuroboWorldObservation:
        """Return complete, owned B1 poses for this exact observation.

        Args:
            context: Borrowed current robot/scene planning observation.
            excluded_entity_ids: Explicit payload identities excluded before any
                pose read. Other collision-roster entries remain mandatory.

        Returns:
            A native-stamped plain world observation; missing scope raises.
        """
        ...


@runtime_checkable
class ObservedCuroboQueryService(Protocol):
    """Optional observed-query capability on the existing joint-prefix service.

    One configured adapter may serve prefix and payload consumers; this does not
    establish a second registry or permit fallback to a different world source.
    Owning skills retain their explicit goal, RNG and backend-limit choices.
    """

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed robot."""
        ...

    def planning_device(self) -> torch.device:
        """Resolve the backend device only when a caller is opening a query."""
        ...

    def open(
        self,
        *,
        context: PlanningContext,
        control_part: str,
        full_qpos: torch.Tensor,
        payload: CuroboPayload | None = None,
        world_metadata: dict[str, object] | None = None,
    ) -> ContextManager[MotionGenerator]:
        """Capture the declared observation and return a lazy owned query context.

        Args:
            context: Exact B1/env0 planning context and captured root.
            control_part: Explicit active group.
            full_qpos: Complete locked planning input in native joint order.
            payload: Explicit carried-body envelope, or no attachment.
            world_metadata: Optional caller-owned diagnostic map receiving the exact captured world IDs.

        Returns:
            A context manager yielding the standard motion facade. Observation
            preparation occurs at this call; backend resources open on entry.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaceInsideLanding:
    """Explicit native footprint binding for an ordinary inside release.

    Args:
        geometry: Cached native geometry for the exact bound payload and parent.
        subject_id: Payload identity used by the held state.
        container_id: Scene entity supplying the release frame.
        parent_id: Rigid reference identity supplying the measured floor frame.
        release_clearance: Authored release offset; this policy requires 0.12 m.

    The integration owns semantic route selection and exact native UID binding.
    This value contains no Task Program objects and certifies no physical effect.
    """

    geometry: NativePlanarPlacement
    subject_id: str
    container_id: str
    parent_id: str
    release_clearance: float

    def __post_init__(self) -> None:
        if type(self.geometry) is not NativePlanarPlacement:
            raise TypeError("Inside landing requires NativePlanarPlacement geometry.")
        for name in ("subject_id", "container_id", "parent_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact nonempty identity.")
        if self.release_clearance != 0.12:
            raise ValueError(
                "Inside world-yaw recovery requires 0.12 m release clearance."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaceSceneInputs:
    """Owned planning envelopes and explicit floor/support identities.

    Args:
        geometry: Scaled visual rigid envelopes and collision-input link meshes.
            Arrays/components are copied at this input boundary. Consumers borrow
            them read-only; their current poses always come from PlanningContext.
        floor_entity_id: Explicit rigid floor owner in the geometry mapping.
            No object name or task label selects this role.
        support_owner_id: Optional intended-contact owner excluded from free-space
            payload clearance. Release/support physical evidence is still required.
        inside_landing: Optional exact native footprint binding for declared yaw recovery.
    """

    geometry: SceneMeshSnapshot
    floor_entity_id: str
    support_owner_id: str | None = None
    inside_landing: PlaceInsideLanding | None = None

    def __post_init__(self) -> None:
        if (
            self.inside_landing is not None
            and type(self.inside_landing) is not PlaceInsideLanding
        ):
            raise TypeError("inside_landing must be PlaceInsideLanding or None.")
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
        owned = SceneMeshSnapshot(
            self.geometry,
            components=self.geometry.components,
            collision_roles=self.geometry.collision_roles,
            unavailable=self.geometry.unavailable,
        )
        object.__setattr__(self, "geometry", owned)

    def snapshot(self) -> PlaceSceneInputs:
        """Return independent arrays/components for a new ownership boundary."""
        return replace(self)


@runtime_checkable
class PlaceSceneProvider(Protocol):
    """Resolve declared planning inputs without exposing Task Program objects."""

    def prepare(
        self,
        request: ResolvedActionRequest[PlaceGoal, PlaceOptions],
        context: PlanningContext,
    ) -> PlaceSceneInputs:
        """Return owned geometry and explicit intended-contact identities.

        Args:
            request: Current resolved Place invocation and its declared policies.
            context: Current observations; providers must not read future state.

        Returns:
            Planning-only input envelopes. Unavailable required inputs raise,
            rather than replacing collision inputs with a different oracle.
        """
        ...
