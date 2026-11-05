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


"""Read-only Pick/Place planning inputs and explicit Motion service ports."""

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable, ContextManager
from dataclasses import dataclass, replace
import torch
from .pour_projection import PourGraspProjectionInputs
from embodichain.toolkits.graspkit.pg_grasp.preshape import PreshapeHandClearance
from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot
from embodichain.lab.sim.scene.planar_observation import NativePlanarPlacement

if TYPE_CHECKING:
    from .primitives.hand_over import HandOverGoal, HandOverOptions
    from embodichain.lab.sim.scene.native_hand_scene import NativeHandSceneQuery
    from embodichain.lab.sim.scene.robot_distal import RobotDistalMeshSnapshot
    from embodichain.lab.sim.evidence import RobotCommandObservation
    from embodichain.lab.sim.motion.mesh_collision import (
        RobotArticulationCollisionSnapshot,
    )
    from .invocation import ResolvedActionRequest
    from .primitives.pick_up import GraspGoal, PickUpOptions
    from .primitives.place import PlaceGoal, PlaceOptions
    from .state import PlanningContext
    from embodichain.lab.sim.motion.planners.utils import PlanResult
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from embodichain.lab.sim.motion.planners.curobo.payload import CuroboPayload
    from embodichain.lab.sim.motion.planners.curobo.observed import (
        CuroboWorldObservation,
    )
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.atomic_actions.source_native_contact import (
        NativeClosingQuery,
        NativeRigidRestScene,
    )
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
    from embodichain.compute.geometry.pouring import PourGraspProjection

__all__ = [
    "HandOverSceneProvider",
    "SlideHandClearanceProvider",
    "HeldStagingBinding",
    "HeldStagingProvider",
    "HeldArticulationStagingProvider",
    "RetainedCommandAuthority",
    "RetainedCommandProvider",
    "JointPrefixPlanner",
    "CuroboWorldObservationProvider",
    "ObservedCuroboQueryService",
    "PourGraspProjectionProvider",
    "PickClosingQueryProvider",
    "PickRigidSceneProvider",
    "PlaceInsideLanding",
    "PlaceInsideLandingProvider",
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


@runtime_checkable
class PourGraspProjectionProvider(Protocol):
    """Prepare a numeric candidate constraint without constructing a Pick plan.

    A Task Program implementation may privately resolve its exact program and
    occurrence declarations. Neither those objects nor callbacks that read future
    state may enter the returned numeric snapshot. Direct Atomic hosts may supply
    an equivalent provider without any Task Program dependency.
    """

    def prepare(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
        context: PlanningContext,
    ) -> PourGraspProjection | PourGraspProjectionInputs | None:
        """Resolve one owned, read-only B=1 projection for the current observation.

        Args:
            request: Exact bound Pick invocation, including immutable policies.
            context: Current observations and task state, borrowed read-only.

        Returns:
            Fresh numeric operands, optionally with raw trace/occurrence inputs,
            for the applicable bound occurrence, or None
            when this provider does not apply. A declared but unavailable scope
            must raise; it must not silently remove a required constraint.
        """
        ...


@runtime_checkable
class PickClosingQueryProvider(Protocol):
    """Prepare a fresh native query from one exact observed Pick request.

    Implementations adapt plain UID/target declarations to the Lab-owned
    native capture API. They replace static hand commands with this request's
    resolved open/grasp endpoints. An optional opening target must be grounded
    from the same observation and carry the query's exact NativeQueryScope.
    No program/compiler object enters the returned query.
    """

    def prepare(
        self,
        request: ResolvedActionRequest[GraspGoal, PickUpOptions],
        context: PlanningContext,
        *,
        hand_geometry_cfg: RobotHandGeometryCfg,
    ) -> NativeClosingQuery:
        """Return one owned B1/env0 query with an independent fresh query ID.

        Args:
            request: Exact invocation and its resolved endpoint commands.
            context: Borrowed same-timestamp robot and scene observation.
            hand_geometry_cfg: Owned calibration from the engine's only registry,
                already checked to use the source 65-sample closing grid.

        Returns:
            A fresh open query for the requested revision. The action closes it
            on success, rejection or exception; borrowed native entities remain
            alive. Missing required inputs raise instead of disabling screening.
        """
        ...


@runtime_checkable
class PickRigidSceneProvider(PickClosingQueryProvider, Protocol):
    """Optional native rigid/rest-scene capture capability for Pick revisions 8–9."""

    def prepare_rigid_scene(self, query: NativeClosingQuery) -> NativeRigidRestScene:
        """Capture and attach the query's same-scope native rigid rest scene.

        Args:
            query: Open native closing query from this provider's prepare call.

        Returns:
            The exact owned scene attached to query.rigid_scene_screen. Capture
            stays lazy, after input-mesh precontact checks pass. Closing the
            query releases owned FCL resources, never borrowed native entities.
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


@runtime_checkable
class PlaceInsideLandingProvider(Protocol):
    """Acquire exact native landing geometry only after ordinary Place fails."""

    def prepare_inside_landing(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PlaceSceneInputs:
        """Return scene inputs with native landing geometry without reading poses."""
        ...


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


@dataclass(frozen=True, slots=True, kw_only=True)
class HeldStagingBinding:
    """Host-validated rigid subject selected for one exact staging invocation.

    Args:
        rigid_entity_id: Exact held subject backed by a registered rigid object.
            The integration validates role and occurrence; no name inference.
        articulation_scene: Also capture the source robot/articulation-only scope,
            excluding rigids. The final command guard remains robot-only.
    """

    rigid_entity_id: str
    articulation_scene: bool = False

    def __post_init__(self) -> None:
        if type(self.articulation_scene) is not bool:
            raise TypeError("articulation_scene must be bool.")
        value = self.rigid_entity_id
        if type(value) is not str or not value or value != value.strip():
            raise ValueError("rigid_entity_id must be an exact nonempty identity.")


@runtime_checkable
class HeldStagingProvider(Protocol):
    """Separate static occurrence binding from late same-observation geometry."""

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed engine robot."""
        ...

    def binding(self, request: ResolvedActionRequest) -> HeldStagingBinding | None:
        """Return a declared rigid occurrence without reading native state.

        Args:
            request: Exact resolved occurrence and resource binding.

        Returns:
            Static source-compatible binding, or None when unselected.
        """
        ...

    def capture_distal_geometry(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotDistalMeshSnapshot:
        """Delegate late Scene capture under the complete original world stamp.

        Args:
            request: Exact active-arm binding for the eligible staging occurrence.
            context: The observed/projected planning context; native capture uses
                observed_robot when present, never the projected command start.

        Returns:
            Owned distal input geometry. The host validates complete source
            identity before/after capture and never falls back to a later world.
        """
        ...


@runtime_checkable
class HeldArticulationStagingProvider(HeldStagingProvider, Protocol):
    """Optional explicit source robot/articulation capture capability."""

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed engine robot."""
        ...

    def prepare_articulation_query(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotArticulationCollisionSnapshot:
        """Return a fresh owned robot/articulation snapshot, never a full scene query.

        The host binds the actual original observation and complete source stamp.
        Capture cannot substitute projected joints or read a later world on error.
        The Atomic planning scope closes the returned query.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class RetainedCommandAuthority:
    """Host-validated serial provenance for retaining an already applied target.

    Args:
        preset_id: Host's verified policy preset identity.
        predecessor_call_id: Exact prior acquisition/retention semantic identity.
        predecessor_invocation_id: Exact serial predecessor occurrence.
        entity_id: The same rigid payload identity.
        resource_id: The same ownership resource identity.
        attachment_monitor_id: Explicit verified acquisition monitor identity.
        attachment_monitor_revision: Its declared revision.
        retention_monitor_id: Explicit verified retention monitor identity.
        retention_monitor_revision: Its declared revision.

    The integration validates adjacency, resource identity and verified monitor
    declarations. This record does not prove physical attachment or authorize a
    new hand target.
    """

    preset_id: str
    predecessor_call_id: str
    predecessor_invocation_id: str
    entity_id: str
    resource_id: str
    attachment_monitor_id: str
    attachment_monitor_revision: str
    retention_monitor_id: str
    retention_monitor_revision: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact nonempty identity.")

    def metadata(self) -> dict[str, str | dict[str, str]]:
        """Return independent source-compatible provenance.

        Returns:
            Declared serial identities and monitor references. This mapping is
            not physical evidence and cannot authorize a changed hand target.
        """
        return {
            "preset_id": self.preset_id,
            "predecessor_call_id": self.predecessor_call_id,
            "predecessor_invocation_id": self.predecessor_invocation_id,
            "entity_id": self.entity_id,
            "resource_id": self.resource_id,
            "attachment_monitor": {
                "monitor_id": self.attachment_monitor_id,
                "revision": self.attachment_monitor_revision,
            },
            "retention_monitor": {
                "monitor_id": self.retention_monitor_id,
                "revision": self.retention_monitor_revision,
            },
        }


@runtime_checkable
class RetainedCommandProvider(Protocol):
    """Separate static authority from opt-in same-observation native reads."""

    @property
    def robot(self) -> Robot:
        """Return the exact engine robot; callers never rediscover a native owner."""
        ...

    def authority(
        self, request: ResolvedActionRequest
    ) -> RetainedCommandAuthority | None:
        """Return validated serial provenance without reading native state."""
        ...

    def capture(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotCommandObservation:
        """Capture original actual state and drive targets using the full stamp.

        Implementations use context.observed_robot for projected planning starts,
        never projected qpos as the expected native observation. Missing original
        observations or changed sources reject without a live-later fallback.
        """
        ...

    def verify_current(self, observation: RobotCommandObservation) -> None:
        """Reject any changed arrays or source stamp without refreshing values."""
        ...


@runtime_checkable
class SlideHandClearanceProvider(Protocol):
    """Supply explicit nominal and fraction-specific hand sweep geometry."""

    @property
    def robot(self) -> Robot:
        """Return the exact engine robot owning the hand geometry."""
        ...

    def nominal(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PreshapeHandClearance:
        """Return declaration-time cached nominal geometry for this occurrence."""
        ...

    def measure(
        self, request: ResolvedActionRequest, context: PlanningContext, fraction: float
    ) -> PreshapeHandClearance:
        """Capture one fractional sweep under the original native observation."""
        ...


@runtime_checkable
class HandOverSceneProvider(Protocol):
    """Prepare one owned native hand/scene query for an observed transfer."""

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed robot whose observed hand is captured."""
        ...

    def prepare(
        self,
        request: ResolvedActionRequest[HandOverGoal, HandOverOptions],
        context: PlanningContext,
    ) -> NativeHandSceneQuery:
        """Return fresh same-observation geometry; the Atomic caller closes it.

        Implementations resolve explicit native UIDs and receiver joints at the
        integration boundary. The returned Scene query contains no invocation,
        program or Task Engine objects. Missing source stamps or geometry reject
        instead of silently disabling receiver qualification.
        """
        ...
