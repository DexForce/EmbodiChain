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

"""Observation-bound native hand clearance against finite bodies and planes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from types import TracebackType
from typing import Any, TYPE_CHECKING
import numpy as np
import torch
import trimesh
from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
    NativeQueryScope,
    _QueryLifetime,
    _neutral_identity,
)
from ._collision_backends import _require_fcl
from embodichain.compute.geometry.pose_checks import as_rigid_matrix as _pose
from embodichain.compute.geometry.mesh_bounds import input_mesh_bounds

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "NativeScenePlane",
    "NativeSceneBody",
    "NativeHandSceneQuery",
    "capture_native_hand_scene",
]


@dataclass(frozen=True, slots=True, eq=False)
class NativeScenePlane:
    """A body-local plane with unchanged unit normal and a point on its surface."""

    normal: np.ndarray
    point: np.ndarray

    def __post_init__(self) -> None:
        for name in ("normal", "point"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Native planes require finite 3D normal/point values.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isclose(np.linalg.norm(self.normal), 1.0, atol=1e-5, rtol=0):
            raise ValueError("Native plane normals must already be normalized.")


@dataclass(frozen=True, slots=True, eq=False)
class NativeSceneBody:
    """One captured native body, its meshes/planes, pose and original offsets.

    Meshes are borrowed read-only. Native capture copies cooked geometry once;
    manually supplied geometry does not establish native provenance. Pose/plane
    arrays are independently owned. Negative finite rest offsets are preserved
    when they do not exceed the nonnegative native contact offset.
    """

    name: str
    owner: str
    link: str | None
    meshes: tuple[trimesh.Trimesh, ...]
    pose: np.ndarray
    contact_offset: float
    rest_offset: float
    planes: tuple[NativeScenePlane, ...] = ()

    def __post_init__(self) -> None:
        for name in ("name", "owner", "link"):
            value = getattr(self, name)
            if name == "link" and value is None:
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    "Native scene body identities must be exact nonempty strings."
                )
        if (
            not math.isfinite(self.contact_offset)
            or not math.isfinite(self.rest_offset)
            or self.contact_offset < 0
            or self.rest_offset > self.contact_offset
        ):
            raise ValueError("Native scene offsets are unavailable.")
        meshes, planes = tuple(self.meshes), tuple(self.planes)
        if (
            (not meshes and not planes)
            or any(not isinstance(m, trimesh.Trimesh) for m in meshes)
            or any(type(p) is not NativeScenePlane for p in planes)
        ):
            raise ValueError("Native bodies require finite meshes or analytic planes.")
        pose = _pose(self.pose).copy()
        pose.setflags(write=False)
        object.__setattr__(self, "meshes", meshes)
        object.__setattr__(self, "planes", planes)
        object.__setattr__(self, "pose", pose)


def _manager(body: NativeSceneBody) -> trimesh.collision.CollisionManager:
    manager = trimesh.collision.CollisionManager()
    for index, mesh in enumerate(body.meshes):
        manager.add_object(str(index), mesh, transform=body.pose)
    return manager


def _bounds(body: NativeSceneBody, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return input_mesh_bounds(body.meshes, pose)


class NativeHandSceneQuery(_QueryLifetime):
    """Own FCL queries over one explicitly captured B1 hand/scene observation.

    Args:
        scope: Fresh query identity and common observation clock.
        robot_uid: Exact physical robot identity whose hand is represented.
        joint_names: Full named-joint FK order captured by the native reader.
        root_pose: Observed robot root in the scene arena frame.
        parts: Ordered finite moving hand bodies.
        obstacles: Ordered non-target native bodies; planes are retained.
        inventory: Plain capture diagnostics, never live handles.
        native_revision: Actual captured native source revision.
        read_native_revision: Reader for that exact native source, used before
            every inspection to reject stale geometry rather than read new poses.

    This is discrete geometry qualification, not physical effect evidence. The
    caller supplies planned link poses; this query owns neither IK nor commands.
    """

    def __init__(
        self,
        *,
        scope: NativeQueryScope,
        robot_uid: str,
        joint_names: Sequence[str],
        root_pose: np.ndarray,
        parts: Sequence[NativeSceneBody],
        obstacles: Sequence[NativeSceneBody],
        inventory: Sequence[Mapping[str, object]],
        native_revision: int,
        read_native_revision: Callable[[], int],
    ) -> None:
        self._start_scope(scope)
        if (
            type(robot_uid) is not str
            or not robot_uid
            or robot_uid != robot_uid.strip()
        ):
            raise ValueError("Native hand queries require an exact robot identity.")
        if (
            type(native_revision) is not int
            or native_revision < 0
            or not callable(read_native_revision)
        ):
            raise ValueError(
                "Native hand queries require an actual revision and its source reader."
            )
        self.robot_uid = robot_uid
        self.joint_names = tuple(joint_names)
        if (
            not self.joint_names
            or len(set(self.joint_names)) != len(self.joint_names)
            or any(type(n) is not str or not n for n in self.joint_names)
        ):
            raise ValueError("Native hand queries require exact unique joint order.")
        self.root = _pose(root_pose).copy()
        self.root.setflags(write=False)
        self.parts, self.obstacles = tuple(parts), tuple(obstacles)
        if (
            not self.parts
            or not self.obstacles
            or any(
                type(b) is not NativeSceneBody for b in (*self.parts, *self.obstacles)
            )
            or any(not b.meshes or b.planes or b.link is None for b in self.parts)
        ):
            raise ValueError(
                "Native hand queries require finite hand links and nonempty obstacle geometry."
            )
        self._link_names = tuple(part.link for part in self.parts)
        if len(set(self._link_names)) != len(self._link_names):
            raise ValueError("Native hand query link identities must be unique.")
        self._inventory = _neutral_identity({"inventory": list(inventory)})["inventory"]
        self.native_revision = native_revision
        self._read_native_revision = read_native_revision
        self.assert_current()
        _require_fcl()
        self.hand_managers = tuple(_manager(part) for part in self.parts)
        self.obstacle_managers = tuple(
            _manager(part) if part.meshes else None for part in self.obstacles
        )
        self.obstacle_bounds = tuple(
            _bounds(part, part.pose) if part.meshes else None for part in self.obstacles
        )

    @property
    def inventory(self) -> list[dict[str, Any]]:
        """Return independently owned plain native capture diagnostics."""
        return deepcopy(self._inventory)

    @property
    def link_names(self) -> tuple[str, ...]:
        """Return the exact ordered moving-link identities consumed by inspect."""
        return self._link_names

    def assert_current(self) -> None:
        """Reject a closed query or a different actual native observation revision."""
        self._require_open()
        revision = self._read_native_revision()
        if type(revision) is not int or revision != self.native_revision:
            raise ValueError("Native hand scene observation revision changed.")

    def link_poses(self, robot: Robot, qpos: torch.Tensor) -> np.ndarray:
        """Map a full named-joint path (F,D) to this capture's ordered arena links.

        Args:
            robot: Exact geometry model selected by the owning provider.
            qpos: Explicit finite full-joint samples; no live joint fallback.

        Returns:
            Arena transforms with shape (F,L,4,4), retaining named FK order and
            the source float64 projection used for collision queries.
        """
        self.assert_current()
        if robot.uid != self.robot_uid or tuple(robot.joint_names) != self.joint_names:
            raise ValueError("Native hand FK model identity differs from the capture.")
        if (
            qpos.ndim != 2
            or qpos.shape[1] != len(self.joint_names)
            or not len(qpos)
            or not torch.isfinite(qpos).all()
        ):
            raise ValueError("Native hand FK requires finite full named-joint samples.")
        fk = robot.compute_fk(
            qpos=qpos,
            link_names=list(self.link_names),
            qpos_joint_names=self.joint_names,
        )
        if (
            fk.shape != (len(qpos), len(self.parts), 4, 4)
            or not torch.isfinite(fk).all()
        ):
            raise ValueError("handover_scene_named_fk_unavailable")
        return self.root @ fk.detach().cpu().double().numpy()

    def inspect(self, arena_poses: np.ndarray) -> dict[str, Any]:
        """Check ordered link poses (F,L,4,4), retaining contact offsets and query order."""
        self.assert_current()
        arena_poses = np.asarray(arena_poses)
        if (
            arena_poses.ndim != 4
            or arena_poses.shape[1:] != (len(self.parts), 4, 4)
            or not len(arena_poses)
        ):
            raise ValueError("Native hand paths require nonempty (F,L,4,4) poses.")
        report = {
            "valid": True,
            "frames": len(arena_poses),
            "minimum_checked_distance_m": None,
            "fcl_queries": 0,
            "plane_queries": 0,
            "minimum_plane_signed_distance_m": None,
        }
        return self._inspect_hand_poses(arena_poses, report)

    def _inspect_hand_poses(
        self, arena_poses: np.ndarray, report: dict[str, Any]
    ) -> dict[str, Any]:
        minimum = None
        for frame_index, row in enumerate(arena_poses):
            for part_index, (part, arena_pose) in enumerate(
                zip(self.parts, row, strict=True)
            ):
                pose = _pose(arena_pose)
                manager = self.hand_managers[part_index]
                for shape in range(len(part.meshes)):
                    manager.set_transform(str(shape), pose)
                lower, upper = _bounds(part, pose)
                for obstacle, other, bounds in zip(
                    self.obstacles,
                    self.obstacle_managers,
                    self.obstacle_bounds,
                    strict=True,
                ):
                    threshold = part.contact_offset + obstacle.contact_offset
                    for plane in obstacle.planes:
                        normal = obstacle.pose[:3, :3] @ plane.normal
                        point = (
                            obstacle.pose[:3, :3] @ plane.point + obstacle.pose[:3, 3]
                        )
                        vertices = trimesh.transform_points(
                            np.vstack([mesh.vertices for mesh in part.meshes]), pose
                        )
                        signed_distance = float(((vertices - point) @ normal).min())
                        report["plane_queries"] += 1
                        previous = report["minimum_plane_signed_distance_m"]
                        report["minimum_plane_signed_distance_m"] = (
                            signed_distance
                            if previous is None
                            else min(previous, signed_distance)
                        )
                        if signed_distance <= threshold:
                            return {
                                **report,
                                "valid": False,
                                "frame": frame_index,
                                "receiver_link": part.link,
                                "obstacle_owner": obstacle.owner,
                                "obstacle_link": obstacle.link,
                                "obstacle_name": obstacle.name,
                                "minimum_signed_distance_m": signed_distance,
                                "native_pair_contact_offset_m": threshold,
                                "plane_normal_arena": normal.tolist(),
                                "plane_point_arena": point.tolist(),
                                "reason": "receiver_hand_non_target_plane_contact_envelope",
                            }
                    if other is None:
                        continue
                    gap = np.maximum(
                        np.maximum(bounds[0] - upper, lower - bounds[1]), 0
                    )
                    if np.linalg.norm(gap) > threshold:
                        continue
                    intersecting = bool(manager.in_collision_other(other))
                    distance = (
                        0.0
                        if intersecting
                        else float(manager.min_distance_other(other))
                    )
                    if not math.isfinite(distance) or distance < 0:
                        raise ValueError("handover_native_mesh_distance_unavailable")
                    report["fcl_queries"] += 1
                    minimum = distance if minimum is None else min(minimum, distance)
                    report["minimum_checked_distance_m"] = minimum
                    if distance <= threshold:
                        return {
                            **report,
                            "valid": False,
                            "frame": frame_index,
                            "receiver_link": part.link,
                            "obstacle_owner": obstacle.owner,
                            "obstacle_link": obstacle.link,
                            "obstacle_name": obstacle.name,
                            "minimum_distance_m": distance,
                            "native_pair_contact_offset_m": threshold,
                            "reason": "receiver_hand_non_target_contact_envelope",
                        }
        return report

    def close(self) -> None:
        """Release query-local FCL managers and geometry references, once."""
        if self.closed:
            return
        self._closed = True
        self.hand_managers = ()
        self.obstacle_managers = ()
        self.obstacle_bounds = ()
        self.parts = ()
        self.obstacles = ()
        self._read_native_revision = None

    def __enter__(self) -> NativeHandSceneQuery:
        """Enter this explicitly owned geometry lifetime."""
        self.assert_current()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close owned resources after success, rejection or caller failure."""
        self.close()


def capture_native_hand_scene(
    robot: Robot,
    simulation: SimulationManager,
    *,
    scope: NativeQueryScope,
    expected_qpos: torch.Tensor,
    root_pose: torch.Tensor,
    receiver_control_part: str,
    receiver_joint_ids: tuple[int, ...],
    required_owner_uids: tuple[str, ...],
    excluded_owner_uid: str,
    native_revision: int,
    read_native_revision: Callable[[], int],
) -> NativeHandSceneQuery:
    """Capture one same-revision Default B1 native hand and non-target scene.

    All physical owner identities, observed joints/root and the actual revision
    are explicit inputs. The reader preserves source body/link order, collision
    offsets, retained articulation scales and analytic planes, and verifies named
    hand FK before constructing FCL resources. It never substitutes later poses.
    """
    from ._native_hand_scene_capture import capture

    return capture(
        robot,
        simulation,
        scope=scope,
        expected_qpos=expected_qpos,
        root_pose=root_pose,
        receiver_control_part=receiver_control_part,
        receiver_joint_ids=receiver_joint_ids,
        required_owner_uids=required_owner_uids,
        excluded_owner_uid=excluded_owner_uid,
        native_revision=native_revision,
        read_native_revision=read_native_revision,
    )
