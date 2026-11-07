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

"""B=1 discrete native Newton motion queries with explicit topology exclusions.

These input meshes are sampled planning proxies. They are not native-cooked
geometry, continuous collision detection or physical contact evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import math
import numpy as np
import torch

if TYPE_CHECKING:
    from types import TracebackType
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "MotionCollisionReport",
    "FreeMotionQuery",
    "PrecontactMotionQuery",
    "SimulationFreeMotionQuery",
    "RobotMeshCollisionSnapshot",
    "robot_collision_query_scope",
    "inspect_robot_collision",
    "RobotArticulationCollisionSnapshot",
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
        scope: Whether the query includes the full scene, robot-only inputs, or
            an explicit source-excluding precontact closure.
        excluded_rigid_uid: Exact source identity excluded only by precontact scope.
        backend: Actual collision implementation, when supplied by the query owner.
    """

    valid: bool
    frame: int | None = None
    kind: Literal["scene", "other_arm", "self"] | None = None
    pairs: tuple[tuple[str, str], ...] = ()
    frames: int | None = None
    scope: Literal[
        "discrete_free_motion_meshes",
        "discrete_robot_meshes",
        "discrete_precontact_meshes",
        "discrete_robot_articulation_meshes",
    ] = field(default="discrete_free_motion_meshes", kw_only=True)

    excluded_rigid_uid: str | None = field(default=None, kw_only=True)
    backend: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.backend is not None and (
            type(self.backend) is not str
            or not self.backend
            or self.backend != self.backend.strip()
        ):
            raise ValueError("Collision backend identity must be a nonempty string.")
        if (self.scope == "discrete_precontact_meshes") != (
            self.excluded_rigid_uid is not None
        ):
            raise ValueError(
                "Only precontact scope requires an excluded rigid identity."
            )
        if self.excluded_rigid_uid is not None and (
            type(self.excluded_rigid_uid) is not str
            or not self.excluded_rigid_uid
            or self.excluded_rigid_uid != self.excluded_rigid_uid.strip()
        ):
            raise ValueError(
                "Excluded rigid identity must be an exact nonempty string."
            )
        if type(self.valid) is not bool:
            raise TypeError("Motion collision validity must be a boolean.")
        if self.scope not in {
            "discrete_free_motion_meshes",
            "discrete_robot_meshes",
            "discrete_precontact_meshes",
            "discrete_robot_articulation_meshes",
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
        extra = (
            {}
            if self.excluded_rigid_uid is None
            else {"scope": self.scope, "excluded_rigid_uid": self.excluded_rigid_uid}
        )
        if self.backend is not None:
            extra["backend"] = self.backend
        if self.valid:
            return {
                "valid": True,
                "frames": self.frames,
                "scope": self.scope,
                **extra,
            }
        return {
            **(
                {"scope": self.scope}
                if self.scope == "discrete_robot_articulation_meshes"
                else {}
            ),
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


@runtime_checkable
class PrecontactMotionQuery(FreeMotionQuery, Protocol):
    """Optional fixed-arm, source-excluding input-mesh query capability."""

    def inspect_precontact_closure(
        self,
        positions: torch.Tensor,
        *,
        control_part: str,
        arm_joint_ids: tuple[int, ...],
        source_uid: str,
        contact_fraction: float,
        grid_samples: int,
    ) -> MotionCollisionReport:
        """Inspect only the declared nominal precontact grid with a fixed arm.

        Args:
            positions: Full close command samples (T, robot.dof).
            control_part: Explicit motion control group.
            arm_joint_ids: Exact ordered active arm columns.
            source_uid: One exact rigid source excluded from this query only.
            contact_fraction: Source-qualified nominal first-contact fraction.
            grid_samples: Original native closing sample count.

        Returns:
            A discrete_precontact_meshes report naming the excluded source.
            This is not native rest-offset or measured physical-grip evidence.
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
            positions.ndim != 2
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
        )

    def inspect_precontact_closure(
        self,
        positions: torch.Tensor,
        *,
        control_part: str,
        arm_joint_ids: tuple[int, ...],
        source_uid: str,
        contact_fraction: float,
        grid_samples: int,
    ) -> MotionCollisionReport:
        """Check the fixed-arm nominal prefix while native geometry owns source contact.

        Args:
            positions: Actual complete close commands, shape (T, robot.dof).
            control_part: Explicit operated solver/control group.
            arm_joint_ids: Its exact ordered joint columns, which must stay fixed.
            source_uid: One native rigid identity excluded from this query only.
            contact_fraction: First nominal pad-contact fraction in [0, 1].
            grid_samples: Original closing-stroke grid size, at least two.

        Returns:
            An explicitly source-excluding input-mesh certificate. Only nominal
            precontact samples are inspected; post-contact commands are not
            simulated penetration, and physical colliders are never disabled.
        """
        if (
            positions.ndim != 2
            or not len(positions)
            or positions.shape[1] != self.robot.dof
            or not torch.isfinite(positions).all()
        ):
            raise ValueError(
                "Closing scene samples must be finite nonempty (T, robot.dof)."
            )
        if tuple(self.robot.get_joint_ids(name=control_part)) != tuple(arm_joint_ids):
            raise ValueError(
                "Closing scene arm columns differ from the declared control part."
            )
        arm = positions[:, list(arm_joint_ids)]
        if not torch.allclose(arm, arm[:1].expand_as(arm), atol=1e-7, rtol=0):
            raise ValueError("Closing-only source contact requires a fixed arm.")
        if source_uid not in self._simulation.get_rigid_object_uid_list():
            raise ValueError("Closing scene requires the bound native rigid source.")
        if (
            not math.isfinite(contact_fraction)
            or not 0 <= contact_fraction <= 1
            or type(grid_samples) is not int
            or grid_samples < 2
        ):
            raise ValueError("Closing scene requires the native first-contact grid.")
        count = round(contact_fraction * (grid_samples - 1)) + 1
        fractions = torch.linspace(
            0, contact_fraction, count, device=positions.device, dtype=positions.dtype
        )
        samples = torch.lerp(positions[:1], positions[-1:], fractions[:, None])
        world = _SourceExcludedWorld(self._simulation, source_uid)
        checker = _make_checker(self.robot, world, control_part, include_scene=True)
        try:
            checked = checker(samples)
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
            scope="discrete_precontact_meshes",
            excluded_rigid_uid=source_uid,
            backend="newton.CollisionPipeline",
        )


@dataclass(frozen=True)
class _SourceExcludedWorld:
    world: Any
    source_uid: str

    def get_rigid_object_uid_list(self) -> list[str]:
        return [
            uid
            for uid in self.world.get_rigid_object_uid_list()
            if uid != self.source_uid
        ]

    def get_rigid_object(self, uid: str) -> Any:
        return self.world.get_rigid_object(uid)

    def get_articulation_uid_list(self) -> list[str]:
        return self.world.get_articulation_uid_list()

    def get_articulation(self, uid: str) -> Any:
        return self.world.get_articulation(uid)


class RobotMeshCollisionSnapshot:
    """Own one calibrated B=1 robot-only Newton query for a bounded planning scope.

    Observed qpos, root frame, model path and declared solver links must remain
    unchanged across queries. No rigid/articulation scene or payload is checked.
    Closing releases only this query's owned native collision resources.
    """

    def __init__(self, robot: Robot, *, control_part: str) -> None:
        """Capture the current robot proxy without changing robot state.

        Args:
            robot: Borrowed B=1 robot with complete assembled URDF geometry.
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
            positions.ndim != 2
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


class RobotArticulationCollisionSnapshot(RobotMeshCollisionSnapshot):
    """Own the source robot/articulation query, explicitly excluding rigid objects.

    This is a discrete planning constraint, not carried-object, rigid-scene or
    physical-contact evidence. It uses the same FK calibration and pair exclusions
    as the full free-motion query. The caller owns the unchanged source/pose
    lifetime; this snapshot never refreshes its articulation meshes.
    """

    def __init__(
        self, robot: Robot, simulation: SimulationManager, *, control_part: str
    ) -> None:
        """Capture complete robot and articulation meshes in one B1 scope.

        Args:
            robot: Exact borrowed robot with its assembled URDF and solver frames.
            simulation: Exact owner of the articulation collision sources/poses.
            control_part: Explicit active control group; names infer no exemptions.
        """
        from embodichain.lab.sim.scene._collision_backends import _require_newton

        _require_newton()
        if simulation.num_envs != 1 or robot.get_qpos().shape[0] != 1:
            raise ValueError("Robot/articulation mesh snapshots require B=1.")
        self._dof = robot.dof
        self._checker = _make_checker(
            robot, simulation, control_part, include_scene=True, include_rigids=False
        )

    def inspect(self, positions: torch.Tensor) -> MotionCollisionReport:
        """Check each full-robot sample against robot and captured articulation meshes."""
        return replace(
            super().inspect(positions), scope="discrete_robot_articulation_meshes"
        )

    def __enter__(self) -> RobotArticulationCollisionSnapshot:
        super().__enter__()
        return self


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
    """Own one read-only FK-calibrated geometry snapshot."""
    import trimesh
    import yourdfpy
    from embodichain.lab.sim.scene.feasibility.collision import (
        CollisionBody,
        NewtonCollisionChecker,
    )

    model = yourdfpy.URDF.load(
        robot.cfg.fpath,
        load_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
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
            if shape.vertices is not None and shape.triangles is not None:
                body = CollisionBody(
                    query_id,
                    query_id,
                    "mesh",
                    vertices=tuple(map(tuple, shape.vertices.detach().cpu().numpy())),
                    triangles=tuple(map(tuple, shape.triangles.detach().cpu().numpy())),
                )
            elif shape.half_extents is not None:
                # This service qualifies input triangle proxies, not cooked
                # physical solids. Retain the original box surface topology
                # on the same native mesh route as the authored URDF inputs.
                mesh = trimesh.creation.box(
                    extents=2 * shape.half_extents.detach().cpu().numpy()
                )
                body = CollisionBody(
                    query_id,
                    query_id,
                    "mesh",
                    vertices=tuple(map(tuple, mesh.vertices)),
                    triangles=tuple(map(tuple, mesh.faces)),
                )
            else:
                raise ValueError(
                    "Unsupported physical collision shape for free motion."
                )
            world_bodies.append(body)
            world_names[query_id] = identity
            world_poses[query_id] = (
                snapshot.pose[0].detach().cpu().numpy()
                @ shape.local_pose.detach().cpu().numpy()
            )

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

    def configure(q: torch.Tensor) -> None:
        model.update_cfg(
            {
                name: float(value)
                for name, value in zip(robot.joint_names, q)
                if name in model.joint_map
            }
        )

    observed_qpos = robot.get_qpos()[0].clone()
    model_path, joint_names = str(robot.cfg.fpath), tuple(robot.joint_names)
    solver_links = (solver.root_link_name, solver.end_link_name)
    configure(observed_qpos)
    observed = (
        robot.get_link_pose(solver.end_link_name, to_matrix=True)[0].cpu().numpy()
    )
    if not np.allclose(
        world_root @ model.get_transform(solver.end_link_name),
        observed,
        atol=2e-4,
        rtol=0,
    ):
        raise ValueError(
            "Recovery collision FK does not match the observed robot frame."
        )
    graph = model.collision_scene.graph
    geometry = []
    for node in graph.nodes_geometry:
        owner = node
        while owner not in links:
            owner = graph.transforms.parents[owner]
        _, mesh_name = graph[node]
        geometry.append((node, owner, model.collision_scene.geometry[mesh_name]))
    if not any(owner in active_links for _, owner, _ in geometry):
        raise ValueError("The operated arm has no collision geometry.")
    robot_names = {}
    bodies = []
    for index, (node, owner, mesh) in enumerate(geometry):
        query_id = f"robot:{index}"
        robot_names[query_id] = (node, owner)
        bodies.append(
            CollisionBody(
                query_id,
                query_id,
                "mesh",
                vertices=tuple(map(tuple, mesh.vertices)),
                triangles=tuple(map(tuple, mesh.faces)),
                robot=True,
            )
        )
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
        clearance_m=0.0,
        device="cpu",
        use_cuda_graph=False,
        allow_triangle_meshes=True,
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
        for frame, q in enumerate(positions):
            configure(q)
            poses = dict(world_poses)
            for query_id, (node, _) in robot_names.items():
                pose, _ = graph[node]
                poses[query_id] = np.asarray(world_root @ pose, dtype=np.float32)
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
