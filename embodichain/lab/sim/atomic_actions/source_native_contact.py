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

"""Explicit source-policy composition of shared native capture and GraspKit queries.

This module selects preserved explicit policies over a declared collision backend.
Newton remains the default; FCL is an explicit source-qualified choice. Closing,
opening and rigid-clearance algorithms are implemented only in GraspKit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import TYPE_CHECKING, Literal
import numpy as np
import torch
import trimesh
from embodichain.lab.sim.scene._collision_backends import _require_fcl, _require_newton
from embodichain.lab.sim.scene._fcl_queries import _FclCollection
from embodichain.lab.sim.scene._newton_queries import _ConvexCollection
from embodichain.lab.sim.scene.native_hand import (
    NativeHandGeometry,
    capture_native_hand_geometry,
    capture_native_rigid_geometry,
    capture_native_rigid_obstacles,
)
from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
    ClosingContactPolicy,
    NativeClosingQuery as _ClosingQuery,
    NativeContactPart,
    NativeOpeningQuery,
    NativeQueryScope,
    NativeRigidRestScene,
    _neutral_identity,
)
from .grasp_geometry import GraspOpeningTarget as NativeOpeningTarget

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

__all__ = [
    "NativeClosingQuery",
    "build_native_closing_query",
    "build_native_rigid_rest_scene",
]


def _source_policy(revision: int) -> ClosingContactPolicy:
    if type(revision) is not int or revision not in range(1, 10):
        raise ValueError("Unsupported Pick closing revision")
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


def _collision_factory(
    backend: Literal["newton", "fcl"],
) -> type[_ConvexCollection] | type[_FclCollection]:
    if type(backend) is not str or backend not in {"newton", "fcl"}:
        raise ValueError("collision_backend must be 'newton' or 'fcl'.")
    if backend == "fcl":
        _require_fcl()
        return _FclCollection
    _require_newton()
    return _ConvexCollection


class NativeClosingQuery(_ClosingQuery):
    """Bind the original explicit source policy to the sole GraspKit algorithm.

    ``revision`` is source policy provenance and never selects a Scene algorithm.
    Captured native owners are retained only for the open query lifetime.
    """

    def __init__(
        self,
        parts: Sequence[NativeContactPart],
        target_meshes: tuple[trimesh.Trimesh, ...],
        target_rest_offset: float,
        *,
        scope: NativeQueryScope,
        revision: int = 1,
        collision_backend: Literal["newton", "fcl"] = "newton",
        target_contact_offset: float | None = None,
        gravity_in_object: np.ndarray | None = None,
    ) -> None:
        """Compose source rules without changing the ordered geometry or thresholds.

        Args:
            parts: Captured enabled link collision meshes and closing FK.
            target_meshes: Exact target-local cooked collision meshes.
            target_rest_offset: Native target rest offset in metres.
            scope: Explicit preparation and observation identity.
            revision: Explicit preserved source policy, from one through nine.
            collision_backend: Explicit geometry backend. Missing dependencies
                and uncertain results fail without selecting another backend.
            target_contact_offset: Native contact offset for policies using halos.
            gravity_in_object: Object-frame gravity for horizontal pad-line checks.
        """
        factory = _collision_factory(collision_backend)
        self._collision_backend = collision_backend
        self._source_revision = revision
        self._geometry: NativeHandGeometry | None = None
        super().__init__(
            parts,
            target_meshes,
            target_rest_offset,
            scope=scope,
            policy=_source_policy(revision),
            collision_factory=factory,
            collision_metadata=factory.metadata(),
            report_metadata={"revision": revision},
            target_contact_offset=target_contact_offset,
            gravity_in_object=gravity_in_object,
        )

    @property
    def revision(self) -> int:
        """Return the immutable source policy provenance."""
        return self._source_revision

    @property
    def collision_backend(self) -> Literal["newton", "fcl"]:
        """Return the backend fixed for this query and its dependent screens."""
        return self._collision_backend

    def close(self) -> None:
        """Release collision resources and the borrowed native ownership token."""
        try:
            super().close()
        finally:
            self._geometry = None


def _physical_identity(
    extra: Mapping[str, object] | None, physical: Mapping[str, object]
) -> dict[str, object]:
    result = _neutral_identity({} if extra is None else extra)
    if set(result) & set(physical):
        raise ValueError("Caller metadata cannot override captured native identity.")
    return {**result, **physical}


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
    collision_backend: Literal["newton", "fcl"] = "newton",
    gravity_in_object: np.ndarray | None = None,
    opening: NativeOpeningTarget | None = None,
    identity: Mapping[str, object] | None = None,
) -> NativeClosingQuery:
    """Capture and assemble one source query without stepping or writing simulation.

    Args:
        robot: Exact simulation-owned Robot.
        simulation: The supported Default/B1 simulation.
        source_uid: Explicit registered native target identity.
        hand_geometry_cfg: Declared topology and stroke sampling. Source-qualified
            callers explicitly retain ``sample_count=65``.
        open_qpos: Current full hand-part open command, shape ``(1, D)``.
        grasp_qpos: Current close command in the same order.
        observation_qpos: Same-observation full named robot state.
        object_pose: Same-observation object pose, shape ``(1, 4, 4)``.
        scope: Fresh preparation and observation identity.
        revision: Explicit preserved source policy, from one through nine.
        collision_backend: Explicit geometry backend, fixed before capture.
            Omitted selects Newton; FCL never acts as an automatic fallback.
        gravity_in_object: Same-observation gravity for horizontal pad qualification.
        opening: Optional same-scope, already-grounded release target.
        identity: Neutral caller provenance; captured physical keys are reserved.

    Returns:
        Fresh owned query. Close it in a context manager or finally block.
    """
    _collision_factory(collision_backend)
    if not isinstance(scope, NativeQueryScope):
        raise TypeError("scope must be NativeQueryScope.")
    policy = _source_policy(revision)
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
    query = NativeClosingQuery(
        geometry.parts,
        geometry.target.meshes,
        geometry.target.rest_offset,
        scope=scope,
        revision=revision,
        collision_backend=collision_backend,
        target_contact_offset=geometry.target.contact_offset,
        gravity_in_object=gravity_in_object,
    )
    try:
        query._geometry = geometry
        query.identity = _physical_identity(identity, geometry.identity)
        query.identity.setdefault("object_id", source_uid)
        query.object_pose_at_capture = geometry.object_pose
        query.build_seconds = time.monotonic() - started
        if opening is not None:
            support = capture_native_rigid_geometry(
                simulation, geometry, opening.support_uid, include_contact_offsets=True
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
    simulation: SimulationManager, closing: NativeClosingQuery
) -> NativeRigidRestScene:
    """Attach ordered, same-arena captured obstacles using the explicit source backend.

    Args:
        simulation: Exact simulation that owns the closing query geometry.
        closing: Open source query with a captured native owner token.

    Returns:
        Owned rigid rest-clearance query excluding only the captured target.
    """
    if not isinstance(closing, NativeClosingQuery):
        raise TypeError("closing must be NativeClosingQuery.")
    closing._require_open()
    if closing.rigid_scene_screen is not None:
        raise RuntimeError("The closing query already owns its rigid scene capture.")
    if closing._geometry is None:
        raise ValueError("Rigid rest capture requires the exact native closing owner.")
    factory = _collision_factory(closing.collision_backend)
    result = NativeRigidRestScene(
        closing.parts,
        capture_native_rigid_obstacles(simulation, closing._geometry),
        scope=closing.scope,
        collision_factory=factory,
    )
    closing.rigid_scene_screen = result
    return result
