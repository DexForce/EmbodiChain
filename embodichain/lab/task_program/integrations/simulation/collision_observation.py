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

"""Same-source deferred obstacle poses for declared observed motion queries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
import torch

from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.motion.planners.curobo.observed import CuroboWorldObservation
from .planning_services import CollisionSceneRosterCfg

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions import PlanningContext

__all__ = ["SimulationCollisionObservationProvider"]


class SimulationCollisionObservationProvider:
    """Capture an ordered collision roster only under its planning world's stamp.

    Args:
        simulation: Exact borrowed owner of the declared native entities.
        declaration: Ordered rigid and collision-only link selections.
        read_stamp: Source reader also used during joint/root/scene observation.

    The provider does not plan, step, read robot joints/root, or guess missing
    obstacle poses. A changed source or replaced owner requires a new observation.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        declaration: CollisionSceneRosterCfg,
        read_stamp: Callable[[], PhysicalObservationStamp],
    ) -> None:
        self._simulation = simulation
        self._declaration = declaration.snapshot()
        self._read_stamp = read_stamp
        self._rigid = {
            uid: simulation.get_rigid_object(uid)
            for _, uid in self._declaration.rigid_objects
        }
        self._articulations = {
            uid: simulation.get_articulation(uid)
            for _, uid, _ in self._declaration.articulation_links
        }
        if any(
            body is None
            for body in (*self._rigid.values(), *self._articulations.values())
        ):
            raise ValueError("Collision roster contains an unbound native source.")

    def capture(
        self, context: PlanningContext, *, excluded_entity_ids: tuple[str, ...] = ()
    ) -> CuroboWorldObservation:
        """Read every declared pose without mixing physical observation versions.

        Args:
            context: B1/env0 planning context with its captured native-world stamp.
            excluded_entity_ids: Explicit payload/source IDs omitted before any pose read.

        Returns:
            Owned ordered obstacle poses correlated with the supplied context.
        """
        stamp = context.world_source_stamp
        if (
            type(stamp) is not PhysicalObservationStamp
            or stamp.source_kind != "default.contacts"
        ):
            raise ValueError(
                "Observed collision poses require a qualified native-world stamp."
            )
        if context.batch_size != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("Observed collision pose capture supports only B1/env0.")
        if (
            context.scene.timestamp != context.robot.timestamp
            or self._read_stamp() != stamp
        ):
            raise ValueError(
                "Collision world no longer matches the planning observation."
            )
        declared_ids = {
            row[0]
            for row in (
                *self._declaration.rigid_objects,
                *self._declaration.articulation_links,
            )
        }
        if (
            type(excluded_entity_ids) is not tuple
            or len(set(excluded_entity_ids)) != len(excluded_entity_ids)
            or not set(excluded_entity_ids) <= declared_ids
        ):
            raise ValueError(
                "Collision exclusions must select unique declared source IDs."
            )
        excluded = set(excluded_entity_ids)
        poses = {}
        for logical_id, uid in self._declaration.rigid_objects:
            if logical_id in excluded:
                continue
            body = self._rigid[uid]
            if self._simulation.get_rigid_object(uid) is not body:
                raise ValueError("A collision-source owner was replaced.")
            poses[logical_id] = body.get_local_pose(to_matrix=True).detach().clone()
        for logical_id, uid, link_name in self._declaration.articulation_links:
            if logical_id in excluded:
                continue
            body = self._articulations[uid]
            if self._simulation.get_articulation(uid) is not body:
                raise ValueError("A collision-source owner was replaced.")
            poses[logical_id] = (
                body.get_link_pose(link_name, to_matrix=True).detach().clone()
            )
        if self._read_stamp() != stamp:
            raise ValueError("Collision source changed while capturing its poses.")
        for name, pose in poses.items():
            observed = context.scene.entities.get(name)
            expected = None if observed is None else observed.pose
            if expected is not None and expected.ndim == 2:
                expected = expected.unsqueeze(0)
            if expected is not None and not torch.equal(pose.to(expected), expected):
                raise ValueError(
                    f"Collision pose {name!r} differs from its planning observation."
                )
        return CuroboWorldObservation(
            timestamp=context.robot.timestamp,
            scene_version=context.scene.version,
            native_revision=stamp.revision,
            poses=poses,
        )
