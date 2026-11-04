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

"""Explicit source-compatible native Pick preparation, without registration patches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
import time
from typing import TYPE_CHECKING
import numpy as np
import torch

from embodichain.lab.sim.scene.native_hand import (
    NativeHandGeometry,
    capture_native_hand_geometry,
    capture_native_rigid_geometry,
    capture_native_rigid_obstacles,
)

from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
    ClosingContactPolicy,
    NativeQueryScope,
    NativeClosingQuery,
    NativeOpeningQuery,
    NativeRigidRestScene,
    _neutral_identity,
    _rigid_pose,
)
from embodichain.lab.sim.scene._collision_backends import _require_newton
from embodichain.lab.sim.scene._newton_queries import _ConvexCollection

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

__all__ = [
    "NativeOpeningTarget",
    "build_native_closing_query",
    "build_native_rigid_rest_scene",
]


@dataclass(frozen=True, slots=True, eq=False)
class NativeOpeningTarget:
    """Already-grounded support and object poses from the closing observation.

    Args:
        support_uid: Exact native rigid support UID selected by the caller.
        reference_pose: Support-to-world pose, shape (4, 4), copied to CPU float64.
        object_target_pose: Final placed-object-to-world pose, shape (4, 4), copied
            to CPU float64. Tensor inputs are detached without changing the input.
        scope: Must equal the closing query's fresh preparation scope.
        identity: Optional neutral diagnostic IDs; never runtime or program objects.
    """

    support_uid: str
    reference_pose: np.ndarray | torch.Tensor
    object_target_pose: np.ndarray | torch.Tensor
    scope: NativeQueryScope
    identity: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            type(self.support_uid) is not str
            or not self.support_uid
            or self.support_uid != self.support_uid.strip()
        ):
            raise ValueError("Opening requires an explicit native support UID.")
        if not isinstance(self.scope, NativeQueryScope):
            raise TypeError("Opening scope must be NativeQueryScope.")
        for name in ("reference_pose", "object_target_pose"):
            pose = _rigid_pose(getattr(self, name))
            pose.setflags(write=False)
            object.__setattr__(self, name, pose)
        object.__setattr__(
            self, "identity", MappingProxyType(_neutral_identity(self.identity))
        )


def _closing_policy(revision: int) -> ClosingContactPolicy:
    """Select the source Pick experiment rules before entering shared services."""
    if type(revision) is not int or revision not in range(1, 10):
        raise ValueError("Native closing revision must be between 1 and 9.")
    return ClosingContactPolicy(
        contact_halo=(
            "ignore" if revision == 1 else "reject" if revision < 4 else "report"
        ),
        reject_open_pad_contact=revision >= 5,
        depth_alternatives=revision >= 6,
        opening_check=revision >= 3,
        defer_source_proposals=revision >= 8,
        horizontal_pad_line=revision == 9,
    )


def _physical_identity(
    extra: Mapping[str, object] | None, physical: Mapping[str, object]
) -> dict[str, object]:
    result = _neutral_identity({} if extra is None else extra)
    if set(result) & set(physical):
        raise ValueError("Caller metadata cannot override captured native identity.")
    return {**result, **physical}


class _PreparedClosingQuery(NativeClosingQuery):
    """Own the Lab capture token outside the simulator-independent GraspKit query."""

    def __init__(
        self,
        geometry: NativeHandGeometry,
        *,
        scope: NativeQueryScope,
        policy: ClosingContactPolicy,
        revision: int,
        gravity_in_object: np.ndarray | None,
    ) -> None:
        self._geometry: NativeHandGeometry | None = geometry
        super().__init__(
            geometry.parts,
            geometry.target.meshes,
            geometry.target.rest_offset,
            scope=scope,
            policy=policy,
            collision_factory=_ConvexCollection,
            collision_metadata=_ConvexCollection.metadata(),
            report_metadata={"revision": revision},
            target_contact_offset=geometry.target.contact_offset,
            gravity_in_object=gravity_in_object,
        )

    def close(self) -> None:
        super().close()
        self._geometry = None


def build_native_closing_query(
    robot: Robot,
    simulation: SimulationManager,
    *,
    source_uid: str,
    hand_geometry_cfg: RobotHandGeometryCfg,
    open_qpos: torch.Tensor,
    grasp_qpos: torch.Tensor,
    observation_qpos: torch.Tensor,
    object_pose: torch.Tensor,
    scope: NativeQueryScope,
    revision: int,
    gravity_in_object: np.ndarray | None = None,
    opening: NativeOpeningTarget | None = None,
    identity: Mapping[str, object] | None = None,
) -> NativeClosingQuery:
    """Compose one explicitly selected native Pick qualification without simulation writes.

    Args:
        robot: Exact simulation-owned Robot.
        simulation: Supported B1 Default simulation.
        source_uid: Explicit registered native rigid target UID.
        hand_geometry_cfg: Declared topology, TCP and stroke sampling. Source-compatible
            B0/F1 preparation explicitly uses 65 samples, independent of surface budgets.
        open_qpos: Resolved open command in full hand-part order, shape (1, D).
        grasp_qpos: Resolved closed command in the same order and shape.
        observation_qpos: Same-observation full named robot state, shape (1, J).
        object_pose: Same-observation object-to-world pose, shape (1, 4, 4).
        scope: Fresh preparation identity, timestamp, scene version and env0.
        revision: Explicit source Pick policy selection, 1 through 9.
        gravity_in_object: Same-observation gravity, required only by revision 9.
        opening: Optional already-grounded opening target from this exact scope.
        identity: Neutral caller labels; captured physical identity keys are reserved.

    Returns:
        Fresh query owning collision resources and its capture lifetime. Close it
        explicitly or use its context manager; borrowed simulation entities are retained.
    """
    _require_newton()
    if not isinstance(scope, NativeQueryScope):
        raise TypeError("scope must be NativeQueryScope.")
    policy = _closing_policy(revision)
    if opening is not None and (
        not isinstance(opening, NativeOpeningTarget)
        or opening.scope != scope
        or not policy.opening_check
    ):
        raise ValueError(
            "Opening needs revision3-9 and the exact closing observation scope."
        )
    started = time.monotonic()
    geometry = capture_native_hand_geometry(
        robot,
        simulation,
        source_uid=source_uid,
        hand_geometry_cfg=hand_geometry_cfg,
        open_qpos=open_qpos,
        grasp_qpos=grasp_qpos,
        observation_qpos=observation_qpos,
        object_pose=object_pose,
        include_contact_offsets=policy.contact_halo != "ignore",
    )
    query = _PreparedClosingQuery(
        geometry,
        scope=scope,
        policy=policy,
        revision=revision,
        gravity_in_object=gravity_in_object,
    )
    try:
        query.identity = _physical_identity(identity, geometry.identity)
        query.identity.setdefault("object_id", source_uid)
        query.object_pose_at_capture = geometry.object_pose
        query.build_seconds = time.monotonic() - started
        if opening is not None:
            support = capture_native_rigid_geometry(
                simulation,
                geometry,
                opening.support_uid,
                include_contact_offsets=True,
            )
            if not support.meshes:
                raise ValueError(
                    "Native support has no enabled physical collision shapes."
                )
            query.opening_screen = NativeOpeningQuery(
                query,
                support.meshes,
                opening.reference_pose,
                opening.object_target_pose,
                support.rest_offset,
                support.contact_offset,
                scope=opening.scope,
                identity=_physical_identity(
                    opening.identity,
                    {
                        "reference_simulation_uid": opening.support_uid,
                        "native_reference_path": support.path,
                        "native_robot_path": geometry.identity["native_robot_path"],
                    },
                ),
            )
        return query
    except BaseException:
        query.close()
        raise


def build_native_rigid_rest_scene(
    simulation: SimulationManager,
    closing: NativeClosingQuery,
) -> NativeRigidRestScene:
    """Attach same-observation rigid clearance to one prepared native Pick query.

    Args:
        simulation: Exact simulation that owns the closing query's native source.
        closing: Open action preparation owning the captured hand stroke and scope.

    Returns:
        Owned rigid rest query, excluding only the captured source UID.

    Raises:
        ValueError: Source ownership changed or the query has no native capture.
        RuntimeError: The query is closed or already has an attached scene query.
    """
    _require_newton()
    closing._require_open()
    if closing.rigid_scene_screen is not None:
        raise RuntimeError("The closing query already owns its rigid scene capture.")
    if not isinstance(closing, _PreparedClosingQuery) or closing._geometry is None:
        raise ValueError("Rigid rest capture requires a prepared native hand owner.")
    targets = capture_native_rigid_obstacles(simulation, closing._geometry)
    result = NativeRigidRestScene(
        closing.parts,
        targets,
        scope=closing.scope,
        collision_factory=_ConvexCollection,
    )
    closing.rigid_scene_screen = result
    return result
