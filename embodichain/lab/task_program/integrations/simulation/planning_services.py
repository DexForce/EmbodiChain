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

"""Closed declarations for shared geometry planning and observation services."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal
from embodichain.utils import configclass
from .place_scene import PlaceSceneBinding
from .coordinated_grasp import CoordinatedGraspServiceCfg

from .place_continuation import (
    PlaceContinuationBinding,
    HeldAlignmentContinuationBinding,
)
from .held_planning import HeldStagingInvocationBinding
from .pick_closing import PickClosingBinding
from .released_continuation import ReleasedHandContinuationBinding
from .pour_binding import PourProjectionBinding

from .slide_preshape import SlidePreshapeBinding
from .measured_rigid import MeasuredRigidInvocationBinding
from .drawer_planning import DrawerInvocationBinding

__all__ = [
    "SimulationPlanningServicesCfg",
    "ObservedCuroboServicesCfg",
    "CollisionSceneRosterCfg",
]


@configclass
class CollisionSceneRosterCfg:
    """One ordered physical world shared by geometry planning consumers.

    Attributes:
        rigid_objects: Ordered logical/native rigid identity pairs.
        articulation_links: Ordered logical/native-articulation/native-link triples.
    """

    rigid_objects: tuple[tuple[str, str], ...] = ()
    articulation_links: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        identities = []
        for field, width in (("rigid_objects", 2), ("articulation_links", 3)):
            rows = getattr(self, field)
            if not isinstance(rows, (tuple, list)):
                raise TypeError(f"{field} must be an ordered sequence.")
            owned = []
            for row in rows:
                if (
                    not isinstance(row, (tuple, list))
                    or len(row) != width
                    or any(
                        type(value) is not str or not value or value != value.strip()
                        for value in row
                    )
                ):
                    raise ValueError(
                        f"{field} entries require {width} exact identifiers."
                    )
                owned.append(tuple(row))
                identities.append(row[0])
            if len({row[1:] for row in owned}) != len(owned):
                raise ValueError(f"Duplicate native {field} source.")
            setattr(self, field, tuple(owned))
        if len(set(identities)) != len(identities):
            raise ValueError("Collision roster logical IDs must be unique.")

    def snapshot(self) -> CollisionSceneRosterCfg:
        """Own and validate ordered source declarations."""
        owned = deepcopy(self)
        owned.validate()
        owned.__post_init__()
        return owned


@configclass
class ObservedCuroboServicesCfg:
    """Backend settings consuming the shared collision roster.

    Attributes:
        cuda_device: Explicit planning device, or the backend default.
        max_voxel_count: Existing scene voxel capacity.
    """

    cuda_device: str | None = None
    max_voxel_count: int = 4_000_000

    def __post_init__(self) -> None:
        if type(self.max_voxel_count) is not int or self.max_voxel_count < 1:
            raise ValueError("max_voxel_count must be a positive integer.")
        if self.cuda_device is not None and (
            type(self.cuda_device) is not str
            or not self.cuda_device
            or self.cuda_device != self.cuda_device.strip()
        ):
            raise ValueError("cuda_device must be an exact device string or None.")

    def snapshot(self) -> ObservedCuroboServicesCfg:
        """Return a validated owned ordered declaration."""
        owned = deepcopy(self)
        owned.validate()
        owned.__post_init__()
        return owned


@configclass
class SimulationPlanningServicesCfg:
    """Explicit shared services; declarations never construct a simulator.

    Attributes:
        kind: Allowlisted simulation geometry service family.
        revision: Supported shared planning-service contract revision.
        free_motion_query: Optional B=1 discrete input-mesh collision service.
        native_handover_scene: Install explicit observed native receiver geometry.
        capture_coordinated_cargo: Capture declared rigid input meshes for the
            optional coordinated grasp preference; no poses or success verdicts.
        pour_projection_bindings: Exact serial Pick/stage/Pour/return/Place roles.

        coordinated_grasps: Explicit paired contact-sampling services; omitted
            leaves standalone generators and their defaults unchanged.
        capture_root_pose: Include the observed arena-local robot root frame.
        capture_native_revision: Capture an actual Default native-world revision.
        hold_qpos_source: Observed values or explicitly selected drive targets
            for unaddressed command columns.
    """

    kind: Literal["simulation_geometry"] = "simulation_geometry"
    revision: int = 1
    free_motion_query: Literal["none", "discrete_mesh"] = "none"
    collision_roster: CollisionSceneRosterCfg | None = None
    place_scene_bindings: tuple[PlaceSceneBinding, ...] = ()
    held_staging_bindings: tuple[HeldStagingInvocationBinding, ...] = ()
    pick_closing_bindings: tuple[PickClosingBinding, ...] = ()
    released_hand_continuations: tuple[ReleasedHandContinuationBinding, ...] = ()
    pour_projection_bindings: tuple[PourProjectionBinding, ...] = ()

    slide_preshape_bindings: tuple[SlidePreshapeBinding, ...] = ()

    place_continuation_bindings: tuple[PlaceContinuationBinding, ...] = ()

    drawer_bindings: tuple[DrawerInvocationBinding, ...] = ()
    measured_rigid_bindings: tuple[MeasuredRigidInvocationBinding, ...] = ()
    observed_curobo: ObservedCuroboServicesCfg | None = None
    native_handover_scene: bool = False
    capture_coordinated_cargo: bool = False
    coordinated_grasps: tuple[CoordinatedGraspServiceCfg, ...] = ()
    capture_root_pose: bool = False
    capture_native_revision: bool = False
    hold_qpos_source: Literal["observed", "drive_target"] = "observed"

    def __post_init__(self) -> None:
        if type(self.place_continuation_bindings) is not tuple or any(
            type(value)
            not in (PlaceContinuationBinding, HeldAlignmentContinuationBinding)
            for value in self.place_continuation_bindings
        ):
            raise TypeError(
                "Place continuation bindings must be an immutable typed tuple."
            )
        if len(
            {value.source_invocation_id for value in self.place_continuation_bindings}
        ) != len(self.place_continuation_bindings):
            raise ValueError("Duplicate Place continuation source occurrence.")
        if type(self.measured_rigid_bindings) is not tuple or any(
            type(value) is not MeasuredRigidInvocationBinding
            for value in self.measured_rigid_bindings
        ):
            raise TypeError("Measured bindings require an immutable typed tuple.")
        if len({value.invocation_id for value in self.measured_rigid_bindings}) != len(
            self.measured_rigid_bindings
        ):
            raise ValueError("Duplicate measured rigid occurrence.")
        if self.measured_rigid_bindings and (
            not self.capture_root_pose or not self.capture_native_revision
        ):
            raise ValueError(
                "Measured rigid services require root and native source capture."
            )
        if self.kind != "simulation_geometry":
            raise ValueError("Unknown simulation planning service kind.")
        if type(self.revision) is not int or self.revision != 1:
            raise ValueError("Unsupported simulation planning service revision.")
        if self.free_motion_query not in {"none", "discrete_mesh"}:
            raise ValueError("Unsupported free_motion_query declaration.")
        if (
            type(self.native_handover_scene) is not bool
            or type(self.capture_coordinated_cargo) is not bool
            or type(self.capture_root_pose) is not bool
            or type(self.capture_native_revision) is not bool
        ):
            raise TypeError(
                "Planning observation and geometry capture choices must be booleans."
            )
        if self.hold_qpos_source not in {"observed", "drive_target"}:
            raise ValueError("hold_qpos_source must be observed or drive_target.")

        if type(self.coordinated_grasps) is not tuple or any(
            type(value) is not CoordinatedGraspServiceCfg
            for value in self.coordinated_grasps
        ):
            raise TypeError("coordinated_grasps must be an immutable typed tuple.")
        self.coordinated_grasps = tuple(
            value.snapshot() for value in self.coordinated_grasps
        )
        targets = [
            name
            for value in self.coordinated_grasps
            for name in (value.left_grasp_target_id, value.right_grasp_target_id)
        ]
        if len(set(targets)) != len(targets):
            raise ValueError(
                "Configured coordinated grasp pairs cannot overlap targets."
            )
        if self.collision_roster is not None:
            if type(self.collision_roster) is not CollisionSceneRosterCfg:
                raise TypeError("collision_roster must be CollisionSceneRosterCfg.")
            self.collision_roster = self.collision_roster.snapshot()
        if type(self.pour_projection_bindings) is not tuple or any(
            type(value) is not PourProjectionBinding
            for value in self.pour_projection_bindings
        ):
            raise TypeError(
                "pour_projection_bindings must be an immutable typed tuple."
            )
        all_pour_ids = [
            identity
            for value in self.pour_projection_bindings
            for identity in value.invocation_ids
        ]
        if len(set(all_pour_ids)) != len(all_pour_ids):
            raise ValueError("Pour recipes cannot share execution occurrences.")
        if self.pour_projection_bindings and (
            not self.capture_root_pose or not self.capture_native_revision
        ):
            raise ValueError("Pour projection requires root and native-source capture.")
        if type(self.released_hand_continuations) is not tuple or any(
            type(v) is not ReleasedHandContinuationBinding
            for v in self.released_hand_continuations
        ):
            raise TypeError(
                "Released continuations require immutable typed declarations."
            )
        if self.released_hand_continuations and (
            self.free_motion_query != "discrete_mesh"
            or not self.capture_root_pose
            or not self.capture_native_revision
        ):
            raise ValueError(
                "Released continuation requires shared discrete collision, root and native-source services."
            )
        if type(self.pick_closing_bindings) is not tuple or any(
            type(value) is not PickClosingBinding
            for value in self.pick_closing_bindings
        ):
            raise TypeError("pick_closing_bindings must be an immutable typed tuple.")
        if len({value.invocation_id for value in self.pick_closing_bindings}) != len(
            self.pick_closing_bindings
        ):
            raise ValueError("Duplicate native Pick occurrence.")
        if self.pick_closing_bindings and (
            not self.capture_root_pose or not self.capture_native_revision
        ):
            raise ValueError("Native Pick requires root and native-source capture.")

        if type(self.drawer_bindings) is not tuple or any(
            type(value) is not DrawerInvocationBinding for value in self.drawer_bindings
        ):
            raise TypeError("drawer_bindings requires an immutable typed tuple.")
        if self.drawer_bindings and (
            not self.capture_native_revision or not self.capture_root_pose
        ):
            raise ValueError("Drawer planning requires native-source and root capture.")
        if (
            any(value.operation == "transport" for value in self.drawer_bindings)
            and self.observed_curobo is None
        ):
            raise ValueError("Drawer transport requires observed cuRobo queries.")
        if type(self.slide_preshape_bindings) is not tuple or any(
            type(value) is not SlidePreshapeBinding
            for value in self.slide_preshape_bindings
        ):
            raise TypeError(
                "slide_preshape_bindings requires an immutable typed tuple."
            )
        if len({value.invocation_id for value in self.slide_preshape_bindings}) != len(
            self.slide_preshape_bindings
        ):
            raise ValueError("Slide preshape occurrences must be unique.")
        if self.slide_preshape_bindings and not self.capture_native_revision:
            raise ValueError("Slide preshape requires native-source capture.")
        if type(self.held_staging_bindings) is not tuple or any(
            type(value) is not HeldStagingInvocationBinding
            for value in self.held_staging_bindings
        ):
            raise TypeError("held_staging_bindings must be an immutable typed tuple.")
        if len({value.invocation_id for value in self.held_staging_bindings}) != len(
            self.held_staging_bindings
        ):
            raise ValueError("Duplicate held staging occurrence.")
        if self.held_staging_bindings and (
            not self.capture_root_pose or not self.capture_native_revision
        ):
            raise ValueError("Held staging requires root and native-source capture.")
        if type(self.place_scene_bindings) is not tuple or any(
            type(value) is not PlaceSceneBinding for value in self.place_scene_bindings
        ):
            raise TypeError("place_scene_bindings must be an immutable typed tuple.")
        if len({value.invocation_id for value in self.place_scene_bindings}) != len(
            self.place_scene_bindings
        ):
            raise ValueError("Duplicate Place scene invocation identity.")
        if (
            self.observed_curobo is not None or self.place_scene_bindings
        ) and self.collision_roster is None:
            raise ValueError(
                "Geometry consumers require one declared collision_roster."
            )
        if self.observed_curobo is not None:
            if type(self.observed_curobo) is not ObservedCuroboServicesCfg:
                raise TypeError("observed_curobo must be ObservedCuroboServicesCfg.")
            self.observed_curobo = self.observed_curobo.snapshot()
            if not self.capture_root_pose or not self.capture_native_revision:
                raise ValueError(
                    "Observed cuRobo services require root and native-source capture."
                )

        if self.native_handover_scene and (
            self.collision_roster is None
            or not self.capture_root_pose
            or not self.capture_native_revision
        ):
            raise ValueError(
                "Native HandOver requires collision_roster, root and native revision capture."
            )

    def validate_num_envs(self, num_envs: object) -> None:
        """Retain the single-environment boundary of explicit geometry consumers."""
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("Planning services require a positive integer num_envs.")
        if num_envs != 1 and (
            self.free_motion_query == "discrete_mesh"
            or self.observed_curobo is not None
            or self.native_handover_scene
            or self.capture_coordinated_cargo
            or self.coordinated_grasps
            or self.place_scene_bindings
            or self.held_staging_bindings
            or self.pick_closing_bindings
            or self.released_hand_continuations
            or self.pour_projection_bindings
            or self.slide_preshape_bindings
            or self.place_continuation_bindings
            or self.drawer_bindings
            or self.measured_rigid_bindings
        ):
            raise ValueError("Declared geometry planning services require B=1.")

    def validate_physics(self, physics: object) -> None:
        """Reject native revision capture outside its declared Default backend."""
        if type(physics) is not str or physics not in {"default", "newton"}:
            raise ValueError(
                "Planning services require an explicit supported physics backend."
            )
        if physics != "default" and any(
            binding.inside_subject_id is not None
            for binding in self.place_scene_bindings
        ):
            raise ValueError("Inside yaw geometry requires Default physics.")
        if (
            self.capture_coordinated_cargo or self.coordinated_grasps
        ) and physics != "default":
            raise ValueError("Coordinated cargo geometry requires Default physics.")
        if self.capture_native_revision and physics != "default":
            raise ValueError(
                "Native contact revision capture requires Default physics."
            )

    def snapshot(self) -> SimulationPlanningServicesCfg:
        """Own and revalidate the complete declaration before registration/use."""
        self.validate()
        owned = deepcopy(self)
        owned.__post_init__()
        return owned


def _effective_planning_service_fields(
    value: SimulationPlanningServicesCfg,
    *,
    validate: bool = True,
) -> dict[str, object]:
    """Serialize effective choices while preserving pre-extension empty defaults."""
    payload = (value.snapshot() if validate else value).to_dict()
    for name in (
        "native_handover_scene",
        "capture_coordinated_cargo",
        "coordinated_grasps",
        "place_scene_bindings",
        "held_staging_bindings",
        "pick_closing_bindings",
        "released_hand_continuations",
        "pour_projection_bindings",
        "slide_preshape_bindings",
        "place_continuation_bindings",
        "drawer_bindings",
        "measured_rigid_bindings",
        "observed_curobo",
    ):
        if name == "native_handover_scene":
            if payload[name] is False:
                payload.pop(name)
            continue
        if (
            payload[name] is None
            or payload[name] is False
            or payload[name] == ()
            or payload[name] == []
        ):
            payload.pop(name)
    return payload
