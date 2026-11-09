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

"""B=1 native collision-shape queries with explicit topology exclusions.

Robot geometry comes from materialized public collision descriptors; the URDF
supplies topology only. Public full-joint FK retains measured mimic states. Scene sources retain their declared
shape authority. Queries are discrete, not continuous or physical-effect evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import math
import re
import numpy as np
import torch

if TYPE_CHECKING:
    from types import TracebackType
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "MotionCollisionReport",
    "FreeMotionQuery",
    "SimulationFreeMotionQuery",
    "RobotMeshCollisionSnapshot",
    "robot_collision_query_scope",
    "inspect_robot_collision",
]


@dataclass(frozen=True, slots=True)
class MotionCollisionReport:
    """Result of checking actual commanded samples against collision inputs.

    Args:
        valid: Whether every sample passed the declared query.
        frame: First rejected sample, or None on success.
        kind: Scene, other-arm or self pair responsible for rejection.
        pairs: Ordered collision-object identities from that sample.
        frames: Checked sample count on success.
        scope: Whether the query includes the full rigid scene or robot-only inputs.
        backend: Actual collision implementation, when supplied by the query owner.
        geometry_source: Explicit captured geometry authority for this result.
    """

    valid: bool
    frame: int | None = None
    kind: Literal["scene", "other_arm", "self"] | None = None
    pairs: tuple[tuple[str, str], ...] = ()
    frames: int | None = None
    scope: Literal[
        "discrete_free_motion_meshes",
        "discrete_robot_meshes",
    ] = field(default="discrete_free_motion_meshes", kw_only=True)

    backend: str | None = field(default=None, kw_only=True)
    geometry_source: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.backend is not None and (
            type(self.backend) is not str
            or not self.backend
            or self.backend != self.backend.strip()
        ):
            raise ValueError("Collision backend identity must be a nonempty string.")
        if self.geometry_source is not None and (
            type(self.geometry_source) is not str
            or not self.geometry_source
            or self.geometry_source != self.geometry_source.strip()
        ):
            raise ValueError("Collision geometry source must be an exact identity.")
        if type(self.valid) is not bool:
            raise TypeError("Motion collision validity must be a boolean.")
        if self.scope not in {
            "discrete_free_motion_meshes",
            "discrete_robot_meshes",
        }:
            raise ValueError("Unknown discrete mesh query scope.")
        if self.valid:
            if self.frame is not None or self.kind is not None or self.pairs:
                raise ValueError("A successful query cannot contain a rejected pair.")
            if type(self.frames) is not int or self.frames < 0:
                raise ValueError(
                    "A successful query requires its checked sample count."
                )
        elif (
            type(self.frame) is not int
            or self.frame < 0
            or self.kind not in {"scene", "other_arm", "self"}
        ):
            raise ValueError("A rejected query requires a frame and collision kind.")
        object.__setattr__(self, "pairs", tuple(tuple(pair) for pair in self.pairs))

    def metadata(self) -> dict[str, Any]:
        """Return owned diagnostics preserving the discrete query scope.

        Returns:
            Serializable qualification fields; never physical-effect evidence.
        """
        extra = {}
        if self.backend is not None:
            extra["backend"] = self.backend
        if self.geometry_source is not None:
            extra["geometry_source"] = self.geometry_source
        if self.valid:
            return {
                "valid": True,
                "frames": self.frames,
                "scope": self.scope,
                **extra,
            }
        return {
            "scope": self.scope,
            "valid": False,
            "frame": self.frame,
            "kind": self.kind,
            "pairs": [list(pair) for pair in self.pairs],
            **extra,
        }


@runtime_checkable
class FreeMotionQuery(Protocol):
    """Read-only B=1 complete robot/scene query borrowed by planning services."""

    @property
    def robot(self) -> Robot:
        """Return the robot whose samples this query checks."""
        ...

    def inspect(
        self, positions: torch.Tensor, *, control_part: str
    ) -> MotionCollisionReport:
        """Inspect full-robot samples without changing observed simulation state.

        Args:
            positions: Exact finite full-robot samples, shape (T, D).
            control_part: Explicit declared motion control part.

        Returns:
            First rejected pair or successful discrete-mesh qualification.
        """
        ...


class SimulationFreeMotionQuery:
    """Capture a fresh read-only collision world for each B=1 query.

    Args:
        robot: Borrowed robot with an assembled URDF and explicit solver root/TCP.
        simulation: Borrowed scene owning the current rigid/articulation poses.

    Source exclusions are retained: URDF neighbours within two joints, the
    mechanism below the declared TCP, and the first moving link against its
    rigid mounting ancestors. No task or robot model names select exemptions.
    """

    def __init__(self, robot: Robot, simulation: SimulationManager) -> None:
        from embodichain.lab.sim.scene._collision_backends import _require_newton

        _require_newton()
        if simulation.num_envs != 1:
            raise ValueError("Discrete free-motion queries require B=1.")
        if (
            not simulation.is_default_backend
            or simulation.get_robot(robot.uid) is not robot
        ):
            raise ValueError(
                "Free-motion queries require their exact Default simulation Robot."
            )
        self._robot, self._simulation = robot, simulation

    @property
    def robot(self) -> Robot:
        """Return the borrowed robot whose full joint samples are inspected."""
        return self._robot

    def inspect(
        self, positions: torch.Tensor, *, control_part: str
    ) -> MotionCollisionReport:
        """Inspect each supplied sample in order using a fresh scene snapshot.

        Args:
            positions: Exact finite full-robot samples, shape (T, D).
            control_part: Solver/control group declaring the operated arm.

        Returns:
            First rejected collision pair, or qualification of every sample.
        """
        if (
            not isinstance(positions, torch.Tensor)
            or positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != self.robot.dof
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("Free-motion samples must be finite (T, robot.dof).")
        checker = _make_checker(
            self.robot, self._simulation, control_part, include_scene=True
        )
        try:
            checked = checker(positions)
        finally:
            close = getattr(checker, "close", None)
            if callable(close):
                close()
        return MotionCollisionReport(
            valid=checked["valid"],
            frame=checked.get("frame"),
            kind=checked.get("kind"),
            pairs=tuple(tuple(pair) for pair in checked.get("pairs", ())),
            frames=checked.get("frames"),
            backend="newton.CollisionPipeline",
            geometry_source="native_cooked_robot_and_typed_scene_shapes",
        )


class RobotMeshCollisionSnapshot:
    """Own one calibrated B=1 robot-only Newton query for a bounded planning scope.

    Observed qpos, root frame, model path and declared solver links must remain
    unchanged across queries. No rigid/articulation scene or payload is checked.
    Closing releases only this query's owned native collision resources.
    """

    def __init__(self, robot: Robot, *, control_part: str) -> None:
        """Capture the current native robot shapes without changing robot state.

        Args:
            robot: Borrowed B=1 robot with public cooked shapes and full named FK.
            control_part: Explicit solver/control group selecting the moving arm.
        """
        from embodichain.lab.sim.scene._collision_backends import _require_newton

        _require_newton()
        if robot.get_qpos().shape[0] != 1:
            raise ValueError("Robot mesh snapshots require B=1.")
        self._dof = robot.dof
        self._checker = _make_checker(robot, None, control_part, include_scene=False)

    def inspect(self, positions: torch.Tensor) -> MotionCollisionReport:
        """Check exact full-robot samples while the captured model remains current.

        Args:
            positions: Finite full-robot samples, shape (T, D).

        Returns:
            Robot-only qualification, retaining source pair and sample order.
        """
        if self._checker is None:
            raise RuntimeError("Robot mesh snapshot is closed.")
        if (
            not isinstance(positions, torch.Tensor)
            or positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != self._dof
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("Robot mesh samples must be finite (T, robot.dof).")
        checked = self._checker(positions)
        return MotionCollisionReport(
            valid=checked["valid"],
            frame=checked.get("frame"),
            kind=checked.get("kind"),
            pairs=tuple(tuple(pair) for pair in checked.get("pairs", ())),
            frames=checked.get("frames"),
            scope="discrete_robot_meshes",
            backend="newton.CollisionPipeline",
            geometry_source="native_cooked_robot_shapes",
        )

    def close(self) -> None:
        """Release owned query resources without mutating the borrowed robot."""
        close = getattr(self._checker, "close", None)
        if callable(close):
            close()
        self._checker = None

    def __enter__(self) -> RobotMeshCollisionSnapshot:
        if self._checker is None:
            raise RuntimeError("Robot mesh snapshot is closed.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _descendants(children: dict[str, set[str]], root: str) -> set[str]:
    result, pending = set(), [root]
    while pending:
        name = pending.pop()
        if name not in result:
            result.add(name)
            pending.extend(children.get(name, ()))
    return result


class _OwnedMotionChecker:
    """Own a native query while retaining the scoped FK and identity checks."""

    def __init__(
        self,
        inspect: Callable[[torch.Tensor], dict[str, Any]],
        close: Callable[[], None],
    ) -> None:
        self._inspect, self._close, self._closed = inspect, close, False

    def __call__(self, positions: torch.Tensor) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Motion collision query is closed.")
        return self._inspect(positions)

    def close(self) -> None:
        if not self._closed:
            self._close()
            self._closed = True


def _make_checker(
    robot: Robot,
    simulation: SimulationManager | None,
    control_part: str,
    *,
    include_scene: bool,
    include_rigids: bool = True,
) -> Callable[[torch.Tensor], dict[str, Any]]:
    """Own native robot shapes with topology exclusions and public full-joint FK."""
    import yourdfpy
    from dexsim.types import RigidBodyShape
    from embodichain.lab.sim.scene.feasibility.collision import (
        CollisionBody,
        NewtonCollisionChecker,
        UnsupportedFeasibilityError,
    )

    def body_for_shape(
        query_id: str, shape: CollisionShapeDesc, *, robot_shape: bool
    ) -> CollisionBody:
        options = dict(
            robot=robot_shape,
            local_pose=tuple(
                map(tuple, shape.local_pose.detach().cpu().double().numpy())
            ),
        )
        if shape.shape_type == RigidBodyShape.BOX:
            if shape.half_extents is None:
                raise UnsupportedFeasibilityError(
                    "Native box dimensions are unavailable."
                )
            return CollisionBody(
                query_id,
                query_id,
                "box",
                dimensions=tuple(
                    2 * shape.half_extents.detach().cpu().double().numpy()
                ),
                **options,
            )
        if shape.shape_type not in {RigidBodyShape.CONVEX, RigidBodyShape.MESH}:
            raise UnsupportedFeasibilityError(
                "Unsupported physical collision shape type."
            )
        if shape.vertices is None or shape.triangles is None:
            raise UnsupportedFeasibilityError(
                "Native collision topology is unavailable."
            )
        return CollisionBody(
            query_id,
            query_id,
            "convex" if shape.shape_type == RigidBodyShape.CONVEX else "mesh",
            vertices=tuple(map(tuple, shape.vertices.detach().cpu().double().numpy())),
            triangles=tuple(map(tuple, shape.triangles.detach().cpu().numpy())),
            **options,
        )

    model = yourdfpy.URDF.load(
        robot.cfg.fpath,
        load_meshes=False,
        build_scene_graph=False,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    world_bodies = []
    world_poses = {}
    world_names = {}
    if include_scene:
        from embodichain.lab.sim.scene._collision_query_inputs import (
            _capture_free_motion_sources,
        )

        for identity, snapshot in _capture_free_motion_sources(
            simulation, include_rigids=include_rigids
        ).items():
            shape = snapshot.shapes[0]
            query_id = f"scene:{len(world_bodies)}"
            world_bodies.append(body_for_shape(query_id, shape, robot_shape=False))
            world_names[query_id] = identity
            world_poses[query_id] = snapshot.pose[0].detach().cpu().numpy()

    links = set(model.link_map)
    children: dict[str, set[str]] = {}
    fixed_children: dict[str, set[str]] = {}
    fixed_parents: dict[str, str] = {}
    neighbors = {name: {name} for name in links}
    for joint in model.robot.joints:
        children.setdefault(joint.parent, set()).add(joint.child)
        neighbors[joint.parent].add(joint.child)
        neighbors[joint.child].add(joint.parent)
        if joint.type == "fixed":
            fixed_children.setdefault(joint.parent, set()).add(joint.child)
            fixed_parents[joint.child] = joint.parent
    solver = robot.get_solver(control_part)
    active_links = _descendants(children, solver.root_link_name)
    hand_links = _descendants(children, solver.end_link_name)
    # Virtual solver frames do not turn the shoulder's own rigid mounting body
    # into another arm. Do not exempt fixed siblings or later moving links.
    mount_ancestors = {solver.root_link_name}
    parent = solver.root_link_name
    while parent in fixed_parents:
        parent = fixed_parents[parent]
        mount_ancestors.add(parent)
    mount_branch = _descendants(fixed_children, solver.root_link_name)
    mounting_links = {
        joint.child
        for joint in model.robot.joints
        if joint.type != "fixed" and joint.parent in mount_branch
    }
    near = {name: set.union(*(neighbors[n] for n in neighbors[name])) for name in links}
    world_root = robot.get_local_pose(to_matrix=True)[0].cpu().numpy().copy()

    observed = robot.get_qpos()
    if (
        not isinstance(observed, torch.Tensor)
        or observed.shape != (1, robot.dof)
        or not torch.isfinite(observed).all()
    ):
        raise UnsupportedFeasibilityError(
            "Native robot collision queries require finite B=1 full qpos."
        )
    observed_qpos = observed[0].clone()
    model_path, joint_names = str(robot.cfg.fpath), tuple(robot.joint_names)
    solver_links = (solver.root_link_name, solver.end_link_name)
    # URDF remains the topology reference. Geometry is read only through
    # Robot's public materialized-link ports, never hullified from input files.
    geometry = []
    native_links = tuple(robot.link_names)
    required = {link.name for link in model.robot.links if link.collisions}
    if not required <= set(native_links):
        raise UnsupportedFeasibilityError(
            "Native robot collision-link inventory is incomplete."
        )
    for link in model.robot.links:
        owner = link.name
        if owner not in native_links:
            continue
        try:
            shapes = robot.get_link_collision_shapes(owner, env_id=0)
        except (AttributeError, RuntimeError) as error:
            raise UnsupportedFeasibilityError(
                "Native robot collision geometry is unavailable."
            ) from error
        if not isinstance(shapes, list):
            raise UnsupportedFeasibilityError(
                "Native robot shape inventory is unavailable."
            )
        if not shapes:
            if owner in required:
                raise UnsupportedFeasibilityError(
                    "Required native robot collision geometry is empty."
                )
            continue
        try:
            attrs = robot.get_link_physical_attr(
                link_names=[re.escape(owner)], env_ids=[0]
            )
        except (AttributeError, RuntimeError) as error:
            raise UnsupportedFeasibilityError(
                "Cooked robot collision snapshots require the Default physical-property port."
            ) from error
        if (
            len(attrs) != 1
            or type(attrs[0].enable_collision) is not bool
            or not attrs[0].enable_collision
        ):
            raise UnsupportedFeasibilityError(
                "Disabled or unknown native robot collision state does not certify clearance."
            )
        offsets = (float(attrs[0].rest_offset), float(attrs[0].contact_offset))
        if not all(math.isfinite(value) for value in offsets):
            raise UnsupportedFeasibilityError(
                "Native robot collision offsets are unavailable."
            )
        for index, shape in enumerate(shapes):
            if shape.shape_type not in {
                RigidBodyShape.BOX,
                RigidBodyShape.CONVEX,
                RigidBodyShape.MESH,
            }:
                raise UnsupportedFeasibilityError(
                    "Unsupported native robot collision shape type."
                )
            label = f"{owner}/{index}:{shape.name}"
            geometry.append((label, owner, shape))
    if any(
        name not in links and robot.get_link_collision_shapes(name, env_id=0)
        for name in native_links
    ):
        raise UnsupportedFeasibilityError(
            "Native collision owner has no declared topology/FK frame."
        )
    if not any(owner in active_links for _, owner, _ in geometry):
        raise UnsupportedFeasibilityError(
            "The operated arm has no enabled native collision geometry."
        )
    owners = tuple(dict.fromkeys(owner for _, owner, _ in geometry))
    fk_links = tuple(dict.fromkeys((*owners, solver.end_link_name)))
    owner_indices = {name: index for index, name in enumerate(fk_links)}

    def full_fk(positions: torch.Tensor) -> np.ndarray:
        # Full articulation FK consumes every measured/commanded joint column.
        # In particular, do not project measured mimic slaves onto their leaders.
        matrices = robot.compute_fk(
            qpos=positions,
            name=None,
            link_names=list(fk_links),
            qpos_joint_names=joint_names,
            to_matrix=True,
        )
        if (
            not isinstance(matrices, torch.Tensor)
            or matrices.shape != (len(positions), len(fk_links), 4, 4)
            or not torch.isfinite(matrices).all()
        ):
            raise ValueError(
                "Public full-joint FK requires finite (T, links, 4, 4) matrices."
            )
        return matrices.detach().cpu().numpy()

    # Native shapes and this public FK are link-local. Calibrate every owner,
    # including the original solver endpoint, then apply world_root exactly once.
    captured_fk = full_fk(observed_qpos[None])[0]
    for owner in fk_links:
        observed = robot.get_link_pose(owner, to_matrix=True)[0].cpu().numpy()
        predicted = world_root @ captured_fk[owner_indices[owner]]
        if not np.allclose(predicted, observed, atol=2e-4, rtol=0):
            raise ValueError(
                f"Recovery collision FK does not match the observed native link frame {owner!r}; "
                f"max_abs_error={float(np.max(np.abs(predicted - observed))):.9g}."
            )
    robot_names = {}
    bodies = []
    for index, (node, owner, shape) in enumerate(geometry):
        query_id = f"robot:{index}"
        robot_names[query_id] = (node, owner)
        bodies.append(body_for_shape(query_id, shape, robot_shape=True))
    bodies.extend(world_bodies)
    # Select the same topology and source scope before asking the native
    # pipeline. Exempt pairs are not numerical contacts or implicit name rules.
    pair_kinds = {}
    ignored = []
    for index, first in enumerate(bodies):
        a = first.collision_id
        for second in bodies[index + 1 :]:
            b = second.collision_id
            entry = None
            if a in robot_names and b in robot_names:
                a_node, a_owner = robot_names[a]
                b_node, b_owner = robot_names[b]
                a_active, b_active = a_owner in active_links, b_owner in active_links
                if a_active and b_active:
                    if b_owner not in near[a_owner] and not (
                        {a_owner, b_owner} <= hand_links
                    ):
                        entry = ("self", tuple(sorted((a_node, b_node))))
                elif a_active or b_active:
                    moving_node, moving_owner, fixed_node, fixed_owner = (
                        (a_node, a_owner, b_node, b_owner)
                        if a_active
                        else (b_node, b_owner, a_node, a_owner)
                    )
                    if not (
                        moving_owner in mounting_links
                        and fixed_owner in mount_ancestors
                    ):
                        entry = ("other_arm", (moving_node, fixed_node))
            elif a in robot_names or b in robot_names:
                moving, fixed = (a, b) if a in robot_names else (b, a)
                node, owner = robot_names[moving]
                if owner in active_links:
                    entry = ("scene", (node, world_names[fixed]))
            key = frozenset((a, b))
            if entry is None:
                ignored.append((a, b))
            else:
                pair_kinds[key] = entry
    native = NewtonCollisionChecker(
        bodies,
        ignored_pairs=ignored,
        # Actual offsets were checked above, but do not replace the original
        # zero-clearance planning predicate with a native rest/contact margin.
        clearance_m=0.0,
        device="cpu",
        use_cuda_graph=False,
    )

    def check(positions: torch.Tensor) -> dict[str, Any]:
        if not include_scene and (
            str(robot.cfg.fpath) != model_path
            or tuple(robot.joint_names) != joint_names
            or (
                robot.get_solver(control_part).root_link_name,
                robot.get_solver(control_part).end_link_name,
            )
            != solver_links
            or not torch.equal(robot.get_qpos()[0], observed_qpos)
            or not np.array_equal(
                robot.get_local_pose(to_matrix=True)[0].cpu().numpy(), world_root
            )
        ):
            raise ValueError(
                "Observed robot changed during the scoped collision check."
            )
        matrices = full_fk(positions) if len(positions) else ()
        for frame, local_poses in enumerate(matrices):
            poses = dict(world_poses)
            for query_id, (_, owner) in robot_names.items():
                poses[query_id] = np.asarray(
                    world_root @ local_poses[owner_indices[owner]], dtype=np.float32
                )
            contacts = native.contact_pairs(poses, all_pairs=True)
            grouped = {"scene": [], "other_arm": [], "self": []}
            for first, second in contacts:
                kind, pair = pair_kinds[frozenset((first, second))]
                grouped[kind].append(list(pair))
            for kind in ("scene", "other_arm", "self"):
                if grouped[kind]:
                    return {
                        "valid": False,
                        "frame": frame,
                        "kind": kind,
                        "pairs": sorted(grouped[kind]),
                    }
        return {
            "valid": True,
            "frames": len(positions),
            "scope": (
                "discrete_free_motion_meshes"
                if include_scene
                else "discrete_robot_meshes"
            ),
        }

    return _OwnedMotionChecker(check, native.close)


_ROBOT_QUERY_SCOPE: ContextVar[
    dict[tuple[int, str], RobotMeshCollisionSnapshot] | None
] = ContextVar("motion_robot_collision_query_scope", default=None)


@contextmanager
def robot_collision_query_scope() -> Iterator[None]:
    """Own robot-only snapshots for one bounded planning invocation.

    Nested scopes get independent snapshots and restore the outer scope even
    after cancellation. Each snapshot validates its frozen model, solver links,
    root and actual joint observation on every query; drift raises rather than
    refreshing the certificate. Scene and articulation queries are never cached.
    The caller must end this scope before advancing physics or changing episode.
    """
    queries: dict[tuple[int, str], RobotMeshCollisionSnapshot] = {}
    token = _ROBOT_QUERY_SCOPE.set(queries)
    try:
        yield
    finally:
        _ROBOT_QUERY_SCOPE.reset(token)
        for query in queries.values():
            query.close()
        queries.clear()


def inspect_robot_collision(
    robot: Robot, positions: torch.Tensor, *, control_part: str
) -> MotionCollisionReport:
    """Inspect robot-only samples, optionally borrowing the current owned scope.

    Args:
        robot: Exact robot identity whose captured model/root must remain current.
        positions: Finite full-robot samples with shape ``(T, D)``.
        control_part: Exact active arm, also part of the frozen solver identity.

    Returns:
        Discrete robot-only result. Without an explicit scope each call captures
        and closes its own snapshot, preserving the standalone default.
    """
    queries = _ROBOT_QUERY_SCOPE.get()
    if queries is None:
        with RobotMeshCollisionSnapshot(robot, control_part=control_part) as query:
            return query.inspect(positions)
    key = (id(robot), control_part)
    if key not in queries:
        queries[key] = RobotMeshCollisionSnapshot(robot, control_part=control_part)
    return queries[key].inspect(positions)
