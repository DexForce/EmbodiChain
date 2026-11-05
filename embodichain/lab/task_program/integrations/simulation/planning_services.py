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

__all__ = ["CollisionSceneRosterCfg", "SimulationPlanningServicesCfg"]


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
class SimulationPlanningServicesCfg:
    """Explicit shared geometry services, resolved only after host construction.

    Attributes:
        kind: Allowlisted service family.
        revision: Supported declaration contract revision.
        free_motion_query: Optional single-environment input-mesh collision service.
        collision_roster: Ordered physical owners used by receiver geometry.
        native_handover_scene: Enable the observed native receiver query factory.
        capture_root_pose: Capture the actual arena-local robot root.
        capture_native_revision: Capture an actual Default contact revision.
        hold_qpos_source: Observed state or explicitly selected drive targets.
    """

    kind: Literal["simulation_geometry"] = "simulation_geometry"
    revision: int = 1
    free_motion_query: Literal["none", "discrete_mesh"] = "none"
    collision_roster: CollisionSceneRosterCfg | None = None
    native_handover_scene: bool = False
    capture_root_pose: bool = False
    capture_native_revision: bool = False
    hold_qpos_source: Literal["observed", "drive_target"] = "observed"

    def __post_init__(self) -> None:
        if self.kind != "simulation_geometry":
            raise ValueError("Unknown simulation planning service kind.")
        if type(self.revision) is not int or self.revision != 1:
            raise ValueError("Unsupported simulation planning service revision.")
        if self.free_motion_query not in {"none", "discrete_mesh"}:
            raise ValueError("Unsupported free_motion_query declaration.")
        if any(
            type(value) is not bool
            for value in (
                self.native_handover_scene,
                self.capture_root_pose,
                self.capture_native_revision,
            )
        ):
            raise TypeError(
                "Planning observation and geometry capture choices must be booleans."
            )
        if self.hold_qpos_source not in {"observed", "drive_target"}:
            raise ValueError("hold_qpos_source must be observed or drive_target.")
        if self.collision_roster is not None:
            if type(self.collision_roster) is not CollisionSceneRosterCfg:
                raise TypeError("collision_roster must be CollisionSceneRosterCfg.")
            self.collision_roster = self.collision_roster.snapshot()
        if self.native_handover_scene and (
            self.collision_roster is None
            or not self.capture_root_pose
            or not self.capture_native_revision
        ):
            raise ValueError(
                "Native HandOver requires collision_roster, root and native revision capture."
            )

    def validate_num_envs(self, num_envs: object) -> None:
        """Reject batches unsupported by the selected live geometry services.

        Args:
            num_envs: Explicit positive host environment count.
        """
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("Planning services require a positive integer num_envs.")
        if num_envs != 1 and (
            self.free_motion_query == "discrete_mesh" or self.native_handover_scene
        ):
            raise ValueError("Declared geometry planning services require B=1.")

    def validate_physics(self, physics: object) -> None:
        """Reject native revision capture outside the Default backend.

        Args:
            physics: Explicit environment-owned backend name.
        """
        if type(physics) is not str or physics not in {"default", "newton"}:
            raise ValueError(
                "Planning services require an explicit supported physics backend."
            )
        if self.capture_native_revision and physics != "default":
            raise ValueError(
                "Native contact revision capture requires Default physics."
            )

    def snapshot(self) -> SimulationPlanningServicesCfg:
        """Own and revalidate the complete declaration before registration/use.

        Returns:
            A separately owned validated declaration.
        """
        self.validate()
        owned = deepcopy(self)
        owned.__post_init__()
        return owned
