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

"""Explicit ordinary planning service declarations; no live construction."""

from __future__ import annotations
from copy import deepcopy
from typing import Literal
from embodichain.utils import configclass
from .place_scene import PlaceSceneBinding

__all__ = [
    "CollisionSceneRosterCfg",
    "ObservedCuroboServicesCfg",
    "SimulationPlanningServicesCfg",
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
        if self.articulation_links:
            raise ValueError(
                "This planning service supports rigid collision sources only."
            )
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
        graph_planner_algorithm: Explicit cuRobo route initializer.
        enable_timeopt: Enable the backend's traversal-time optimizer.
        fit_graph_seed_to_bspline: Fit graph seeds before trajectory optimization.
    """

    cuda_device: str | None = None
    max_voxel_count: int = 4_000_000
    graph_planner_algorithm: Literal["rrt", "prm"] = "rrt"
    enable_timeopt: bool = True
    fit_graph_seed_to_bspline: bool = True

    use_cuda_graph: bool = False
    warmup_iterations: int = 0
    preserve_plan_samples: bool = True

    def __post_init__(self) -> None:
        if (
            self.use_cuda_graph is not False
            or type(self.warmup_iterations) is not int
            or self.warmup_iterations != 0
            or self.preserve_plan_samples is not True
        ):
            raise ValueError(
                "Observed prefix services require no CUDA graph, zero warmup and preserved plan samples."
            )
        if self.graph_planner_algorithm not in ("rrt", "prm"):
            raise ValueError("graph_planner_algorithm must be 'rrt' or 'prm'.")
        for name in ("enable_timeopt", "fit_graph_seed_to_bspline"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool.")
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
    """Opt-in ordinary B1 planning services without constructing live resources.

    Attributes:
        kind: Closed simulation geometry service family.
        revision: Supported declaration contract revision.
        free_motion_query: Optional discrete rigid free-scene qualification.
        collision_roster: Shared ordered physical source set.
        place_scene_bindings: Exact Place occurrence floor/support roles.
        observed_curobo: Optional observed joint-prefix query configuration.
        capture_root_pose: Capture the actual arena-local robot root transform.
        capture_native_revision: Capture actual Default native-world revisions.
        hold_qpos_source: Observed joints or explicitly selected drive targets.
    """

    kind: Literal["simulation_geometry"] = "simulation_geometry"
    revision: int = 1
    free_motion_query: Literal["none", "discrete_mesh"] = "none"
    collision_roster: CollisionSceneRosterCfg | None = None
    place_scene_bindings: tuple[PlaceSceneBinding, ...] = ()
    observed_curobo: ObservedCuroboServicesCfg | None = None
    capture_root_pose: bool = False
    capture_native_revision: bool = False
    hold_qpos_source: Literal["observed", "drive_target"] = "observed"

    def __post_init__(self) -> None:
        if (
            self.kind != "simulation_geometry"
            or type(self.revision) is not int
            or self.revision != 1
        ):
            raise ValueError(
                "Unsupported simulation planning service kind or revision."
            )
        if self.free_motion_query not in ("none", "discrete_mesh"):
            raise ValueError("Unknown free_motion_query.")
        if (
            type(self.capture_root_pose) is not bool
            or type(self.capture_native_revision) is not bool
        ):
            raise TypeError("Planning observation capture choices must be bool.")
        if self.hold_qpos_source not in ("observed", "drive_target"):
            raise ValueError("hold_qpos_source must be observed or drive_target.")
        if self.collision_roster is not None:
            if type(self.collision_roster) is not CollisionSceneRosterCfg:
                raise TypeError("collision_roster must be CollisionSceneRosterCfg.")
            self.collision_roster = self.collision_roster.snapshot()
        if type(self.place_scene_bindings) is not tuple or any(
            type(x) is not PlaceSceneBinding for x in self.place_scene_bindings
        ):
            raise TypeError("place_scene_bindings must be an immutable typed tuple.")
        if len({x.invocation_id for x in self.place_scene_bindings}) != len(
            self.place_scene_bindings
        ):
            raise ValueError("Duplicate Place scene invocation identity.")
        if (
            self.observed_curobo is not None or self.place_scene_bindings
        ) and self.collision_roster is None:
            raise ValueError(
                "Observed/Place geometry services require one collision_roster."
            )
        if self.observed_curobo is not None:
            if type(self.observed_curobo) is not ObservedCuroboServicesCfg:
                raise TypeError("observed_curobo must be ObservedCuroboServicesCfg.")
            self.observed_curobo = self.observed_curobo.snapshot()
            if not self.capture_root_pose or not self.capture_native_revision:
                raise ValueError(
                    "Observed cuRobo requires root and native-source capture."
                )
        if self.collision_roster is not None:
            owners = {x[0] for x in self.collision_roster.rigid_objects}
            if any(
                x.floor_entity_id not in owners
                or (x.support_owner_id is not None and x.support_owner_id not in owners)
                for x in self.place_scene_bindings
            ):
                raise ValueError(
                    "Place floor/support roles must select declared rigid geometry owners."
                )

    def validate_num_envs(self, num_envs: object) -> None:
        """Reject batched geometry consumers before live service assembly."""
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("Planning services require a positive integer num_envs.")
        if num_envs != 1 and (
            self.free_motion_query != "none"
            or self.place_scene_bindings
            or self.observed_curobo is not None
        ):
            raise ValueError("Declared geometry planning services require B=1.")

    def validate_physics(self, physics: object) -> None:
        """Reject native revision capture outside the Default backend."""
        if type(physics) is not str or physics not in {"default", "newton"}:
            raise ValueError(
                "Planning services require an explicit supported physics backend."
            )
        if self.capture_native_revision and physics != "default":
            raise ValueError(
                "Native contact revision capture requires Default physics."
            )

    def snapshot(self) -> SimulationPlanningServicesCfg:
        """Own and revalidate the declaration before registration or use."""
        self.validate()
        owned = deepcopy(self)
        owned.__post_init__()
        return owned
