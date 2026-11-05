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

"""Same-observation native rest-offset scene queries for complete motion paths."""

from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET
import numpy as np
import torch
import trimesh
from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
    NativeQueryScope,
    NativeContactPart,
    _rigid_pose,
)
from embodichain.lab.sim.objects.backends.collision import _native_meshes

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "NativeRestCollisionPart",
    "NativeRestSceneBody",
    "NativeRestSceneQuery",
    "capture_native_rest_scene",
    "capture_native_arm_parts",
    "native_path_link_poses",
]


def _transform(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return vertices @ pose[:3, :3].T + pose[:3, 3]


@dataclass(frozen=True, slots=True)
class NativeRestCollisionPart:
    """One ordered moving collision part with its unchanged native rest offset.

    Native readers copy meshes once. Meshes are borrowed read-only by this DTO;
    manually supplied meshes provide controlled query inputs, not native assurance.
    """

    name: str
    meshes: tuple[trimesh.Trimesh, ...]
    rest_offset: float

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("Native motion parts require exact physical names.")
        if (
            type(self.meshes) is not tuple
            or not self.meshes
            or any(not isinstance(value, trimesh.Trimesh) for value in self.meshes)
        ):
            raise ValueError(
                "Native motion parts require their complete collision meshes."
            )
        if not math.isfinite(self.rest_offset) or self.rest_offset < 0:
            raise ValueError(
                "Native motion parts require finite nonnegative rest offsets."
            )


@dataclass(frozen=True, slots=True, eq=False)
class NativeRestSceneBody:
    """One captured native collision owner and its arena-local rigid pose."""

    part: NativeRestCollisionPart
    pose: np.ndarray

    def __post_init__(self) -> None:
        if type(self.part) is not NativeRestCollisionPart:
            raise TypeError("Native rest bodies require their typed collision part.")
        pose = _rigid_pose(self.pose)
        pose = np.frombuffer(pose.tobytes(), dtype=pose.dtype).reshape(pose.shape)
        object.__setattr__(self, "pose", pose)


class NativeRestSceneQuery:
    """Own an ordered same-revision rest-offset/FCL scene snapshot.

    This retains the original discrete query order and pair-rest acceptance.
    It does not test continuous motion, contact acquisition, task success or
    otherwise widen the source's finite-mesh shape coverage.
    """

    def __init__(
        self,
        bodies: tuple[NativeRestSceneBody, ...],
        *,
        scope: NativeQueryScope,
        validate_observation: Callable[[], None],
    ) -> None:
        """Own native obstacle query resources while borrowing source meshes.

        Args:
            bodies: Ordered native rigid bodies then ordered articulation links.
            scope: Exact current physical preparation identity.
            validate_observation: Reader retaining the same actual native revision,
                robot/root and simulation ownership throughout this query.
        """
        if type(bodies) is not tuple or any(
            type(value) is not NativeRestSceneBody for value in bodies
        ):
            raise TypeError("Native rest scene requires immutable typed body entries.")
        if not isinstance(scope, NativeQueryScope) or not callable(
            validate_observation
        ):
            raise TypeError(
                "Native rest scene requires explicit observation ownership."
            )
        names = [value.part.name for value in bodies]
        if len(names) != len(set(names)):
            raise ValueError("Native rest scene body identities are duplicated.")
        self.scope, self._guard, self._closed = scope, validate_observation, False
        self._bodies = bodies
        self._obstacles = []
        self.assert_current()
        try:
            for body in bodies:
                manager = trimesh.collision.CollisionManager()
                for index, mesh in enumerate(body.part.meshes):
                    manager.add_object(str(index), mesh, transform=body.pose)
                points = np.vstack(
                    [_transform(mesh.vertices, body.pose) for mesh in body.part.meshes]
                )
                self._obstacles.append(
                    (body, manager, np.stack((points.min(0), points.max(0))))
                )
            self.assert_current()
        except BaseException:
            self.close()
            raise

    @property
    def bodies(self) -> tuple[NativeRestSceneBody, ...]:
        """Return ordered owned body facts; meshes retain their read-only contract."""
        self.assert_current()
        return self._bodies

    @property
    def closed(self) -> bool:
        """Whether native query resources have been released."""
        return self._closed

    def assert_current(self) -> None:
        """Reject closed or stale native observation scope without refreshing it."""
        if self._closed:
            raise RuntimeError("Native rest scene query is closed.")
        self._guard()

    def inspect(
        self,
        parts: Sequence[NativeRestCollisionPart | NativeContactPart],
        poses: np.ndarray,
        source_id: str,
        *,
        source_contact_from: int = 0,
    ) -> dict[str, object]:
        """Query moving-part paths against all captured obstacles in source order.

        Args:
            parts: Complete ordered moving collision parts.
            poses: Explicit arena poses (F,L,4,4), matching part order.
            source_id: Native owner ignored from the declared source-contact frame.
            source_contact_from: First permitted-contact frame; zero excludes the
                source owner throughout. Earlier frames still inspect that owner.

        Returns:
            Original first violation and pair-rest threshold/query count, or
            discrete all-sample validity. A declared but unavailable shape/pose
            raises rather than becoming clearance.
        """
        self.assert_current()
        managers = []
        vertices = []
        for part in parts:
            if (
                not math.isfinite(part.rest_offset)
                or part.rest_offset < 0
                or not part.meshes
            ):
                raise ValueError(
                    "Native moving parts require finite rest offsets and enabled shapes."
                )
            manager = trimesh.collision.CollisionManager()
            for index, mesh in enumerate(part.meshes):
                manager.add_object(str(index), mesh)
            managers.append(manager)
            vertices.append(np.vstack([mesh.vertices for mesh in part.meshes]))
        if (
            not len(parts)
            or poses.shape != (len(poses), len(parts), 4, 4)
            or not np.isfinite(poses).all()
        ):
            raise ValueError("Native clearance requires complete finite part poses.")
        queries = 0
        for frame in range(len(poses)):
            for index, part in enumerate(parts):
                pose = poses[frame, index]
                cloud = _transform(vertices[index], pose)
                low, high = cloud.min(0), cloud.max(0)
                manager = managers[index]
                for key in range(len(part.meshes)):
                    manager.set_transform(str(key), pose)
                for obstacle, other, bounds in self._obstacles:
                    if obstacle.part.name == source_id and frame >= source_contact_from:
                        continue
                    threshold = part.rest_offset + obstacle.part.rest_offset
                    gap = np.maximum(0, np.maximum(bounds[0] - high, low - bounds[1]))
                    if np.linalg.norm(gap) > threshold + 0.002:
                        continue
                    distance = float(manager.min_distance_other(other))
                    queries += 1
                    if not math.isfinite(distance) or distance <= threshold:
                        self.assert_current()
                        return dict(
                            valid=False,
                            frame=frame,
                            link=part.name,
                            obstacle=obstacle.part.name,
                            distance_m=distance,
                            required_clearance_m=threshold,
                            queries=queries,
                        )
        self.assert_current()
        return dict(
            valid=True,
            frames=len(poses),
            queries=queries,
            scope="discrete_native_pair_rest_not_continuous_contact_or_task_success",
        )

    def close(self) -> None:
        """Release only owned FCL resources, never borrowed Robot/simulation objects."""
        self._closed = True
        self._obstacles.clear()
        self._bodies = ()

    def __enter__(self) -> NativeRestSceneQuery:
        self.assert_current()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def capture_native_rest_scene(
    simulation: SimulationManager,
    robot: Robot,
    *,
    scope: NativeQueryScope,
    validate_observation: Callable[[], None],
) -> NativeRestSceneQuery:
    """Capture the exact current Default/B1 finite native scene without stepping.

    Args:
        simulation: Exact current native physical owner.
        robot: Exact registered controlled robot selecting the native arena.
        scope: Current caller-owned physical preparation identity.
        validate_observation: Same-source guard called before/after native reads.

    Returns:
        Owned rest-offset/FCL query preserving original rigid/link capture order.
    """
    validate_observation()
    if (
        simulation.num_envs != 1
        or not simulation.is_default_backend
        or simulation.get_robot(robot.uid) is not robot
    ):
        raise ValueError(
            "Native motion scene requires its exact Default/B1 Robot owner."
        )
    spawn = simulation.spawn_result
    handles = [spawn.get(path) for path in spawn.get_paths()]
    robots = [handle for handle in handles if handle.get_name() == robot.uid]
    if len(robots) != 1:
        raise ValueError("Measured native robot identity is ambiguous.")
    native = robots[0]
    bodies = []

    def add(name: str, body: object, pose: np.ndarray) -> None:
        meshes = _native_meshes(body)
        if not meshes:
            return
        bodies.append(
            NativeRestSceneBody(
                NativeRestCollisionPart(name, meshes, float(body.get_rest_offset())),
                pose,
            )
        )

    for uid in simulation.get_rigid_object_uid_list():
        matches = [
            handle
            for handle in handles
            if handle.get_name() == uid and handle.arena_name == native.arena_name
        ]
        if len(matches) != 1:
            raise ValueError("Native rigid identity is unavailable or ambiguous.")
        pose = simulation.get_rigid_object(uid).get_local_pose(to_matrix=True)[0]
        add(
            uid,
            matches[0].native().get_physical_body(),
            pose.detach().cpu().double().numpy(),
        )
    for uid in simulation.get_articulation_uid_list():
        matches = [
            handle
            for handle in handles
            if handle.get_name() == uid and handle.arena_name == native.arena_name
        ]
        if len(matches) != 1:
            raise ValueError(
                "Native articulation identity is unavailable or ambiguous."
            )
        source = matches[0]
        for link in source.get_link_names():
            add(
                uid + ":" + link,
                source.render_articulation.get_physical_body(link),
                np.asarray(source.get_link_pose(link), dtype=np.float64),
            )
    validate_observation()
    selected_paths = tuple((path, spawn.get(path)) for path in spawn.get_paths())

    def guard() -> None:
        validate_observation()
        if (
            simulation.get_robot(robot.uid) is not robot
            or simulation.spawn_result is not spawn
        ):
            raise ValueError("Native motion scene owner changed after capture.")
        if tuple(spawn.get_paths()) != tuple(path for path, _ in selected_paths) or any(
            spawn.get(path) is not value for path, value in selected_paths
        ):
            raise ValueError(
                "Native motion scene handle inventory changed after capture."
            )

    return NativeRestSceneQuery(tuple(bodies), scope=scope, validate_observation=guard)


def capture_native_arm_parts(
    simulation: SimulationManager,
    robot: Robot,
    control_part: str,
    hand_parts: Sequence[NativeContactPart | NativeRestCollisionPart],
    *,
    validate_observation: Callable[[], None],
) -> tuple[NativeRestCollisionPart, ...]:
    """Read source arm collision topology excluding the separately captured hand.

    Args:
        simulation: Exact Default/B1 native owner.
        robot: Exact registered source robot and authored URDF.
        control_part: Explicit active arm solver/control part.
        hand_parts: Original captured hand names to exclude from arm geometry.
        validate_observation: Exact-source guard for the current preparation.

    Returns:
        Sorted enabled arm collision parts with original native rest offsets and
        observed FK verification. Missing topology rejects without a proxy.
    """
    validate_observation()
    if (
        simulation.num_envs != 1
        or not simulation.is_default_backend
        or simulation.get_robot(robot.uid) is not robot
    ):
        raise ValueError(
            "Native arm capture requires its exact Default/B1 Robot owner."
        )
    spawn = simulation.spawn_result
    handles = [spawn.get(path) for path in spawn.get_paths()]
    robots = [handle for handle in handles if handle.get_name() == robot.uid]
    if len(robots) != 1:
        raise ValueError("Measured native robot identity is ambiguous.")
    native = robots[0]
    document = ET.parse(robot.cfg.fpath).getroot()
    descendants = {robot.get_solver(control_part).root_link_name}
    while True:
        children = {
            joint.find("child").attrib["link"]
            for joint in document.findall("joint")
            if joint.find("parent").attrib["link"] in descendants
        }
        if children <= descendants:
            break
        descendants |= children
    names = descendants - {part.name for part in hand_parts}
    required = {
        link.attrib["name"]
        for link in document.findall("link")
        if link.attrib["name"] in names and link.find("collision") is not None
    }
    if not required <= set(native.get_link_names()):
        raise ValueError("Native arm collision-link inventory is incomplete.")
    parts = []
    root = robot.get_local_pose(to_matrix=True)[0].detach().cpu().numpy()
    for name in sorted(required):
        body = native.render_articulation.get_physical_body(name)
        if body is None:
            raise ValueError("Native arm collision body is unavailable.")
        meshes = _native_meshes(body)
        if not meshes:
            continue
        actual = (
            robot.compute_fk(
                qpos=robot.get_qpos(),
                link_names=[name],
                qpos_joint_names=robot.joint_names,
            )[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        if not np.allclose(
            root @ actual, native.get_link_pose(name), atol=2e-4, rtol=0
        ):
            raise ValueError("Native arm FK does not match observed frames.")
        parts.append(
            NativeRestCollisionPart(name, meshes, float(body.get_rest_offset()))
        )
    if not parts:
        raise ValueError("Native arm has no enabled collision geometry.")
    validate_observation()
    return tuple(parts)


def native_path_link_poses(
    robot: Robot,
    qpos: torch.Tensor,
    parts: Sequence[NativeRestCollisionPart | NativeContactPart],
) -> np.ndarray:
    """Adapt original full named-joint FK to arena-local ordered part poses.

    Args:
        robot: Existing exact controlled robot.
        qpos: Full finite emitted joint samples (F,D), in original order.
        parts: Explicit source physical link names, preserving query order.

    Returns:
        Arena-local float64 native query transforms, without stepping or changing
        robot state. Native revision ownership remains with the query's guard.
    """
    fk = (
        robot.compute_fk(
            qpos=qpos,
            link_names=[part.name for part in parts],
            qpos_joint_names=robot.joint_names,
        )
        .detach()
        .cpu()
        .double()
        .numpy()
    )
    root = robot.get_local_pose(to_matrix=True)[0].detach().cpu().double().numpy()
    return root[None, None] @ fk
