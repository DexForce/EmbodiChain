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

"""B=1 discrete URDF/Newton motion queries with explicit topology exclusions.

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
from threading import RLock
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

    def __post_init__(self) -> None:
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
    """Recapture B=1 world inputs while owning reusable Newton geometry.

    Args:
        robot: Borrowed robot with an assembled URDF and explicit solver root/TCP.
            Its device owns the isolated query; CUDA queries use graph replay.
        simulation: Borrowed scene owning the current rigid/articulation poses.

    Source exclusions are retained: URDF neighbours within two joints, the
    mechanism below the declared TCP, and the first moving link against its
    rigid mounting ancestors. No task or robot model names select exemptions.
    Each call reloads model/geometry declarations and observed poses; equal
    immutable geometry/filter declarations can reuse native buffers. Calls and
    close are serialized. Use the context manager or close to release resources.
    """

    def __init__(self, robot: Robot, simulation: SimulationManager) -> None:
        from embodichain.lab.sim.scene._collision_backends import _require_newton

        _require_newton()
        if simulation.num_envs != 1:
            raise ValueError("Discrete free-motion queries require B=1.")
        self._robot, self._simulation = robot, simulation
        self._lock = RLock()
        self._checker: _OwnedMotionChecker | None = None
        self._closed = False

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
        checked = self._inspect(positions, self._simulation, control_part)
        return MotionCollisionReport(
            valid=checked["valid"],
            frame=checked.get("frame"),
            kind=checked.get("kind"),
            pairs=tuple(tuple(pair) for pair in checked.get("pairs", ())),
            frames=checked.get("frames"),
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
        checked = self._inspect(samples, world, control_part)
        return MotionCollisionReport(
            valid=checked["valid"],
            frame=checked.get("frame"),
            kind=checked.get("kind"),
            pairs=tuple(tuple(pair) for pair in checked.get("pairs", ())),
            frames=checked.get("frames"),
            scope="discrete_precontact_meshes",
            excluded_rigid_uid=source_uid,
        )

    def _inspect(
        self, positions: torch.Tensor, world: Any, control_part: str
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Free-motion collision query is closed.")
            # Recapture FK, poses and shapes on every inspection. Only an exactly
            # equal immutable geometry/filter declaration can reuse native buffers.
            self._checker = _make_checker(
                self.robot,
                world,
                control_part,
                include_scene=True,
                reuse=self._checker,
            )
            return self._checker(positions)

    def close(self) -> None:
        """Release this query's cached native geometry; repeated close is harmless."""
        with self._lock:
            if self._checker is not None:
                self._checker.close()
                self._checker = None
            self._closed = True

    def __enter__(self) -> SimulationFreeMotionQuery:
        if self._closed:
            raise RuntimeError("Free-motion collision query is closed.")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_checker", None) is not None:
            self.close()


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
    Closing releases only this query's owned Newton resources.
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
        )

    def close(self) -> None:
        """Release owned query resources without mutating the borrowed robot."""
        if self._checker is not None:
            self._checker.close()
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
    """Close isolated native resources even when a discrete inspection raises."""

    def __init__(self, check: Callable, query: Any, signature: tuple) -> None:
        self._check, self._query = check, query
        self._signature = signature

    def transfer(self, signature: tuple) -> Any:
        if signature != self._signature:
            return None
        query, self._query = self._query, None
        return query

    def __call__(self, positions: torch.Tensor) -> dict[str, Any]:
        if self._check is None:
            raise RuntimeError("Motion collision query is closed.")
        return self._check(positions)

    def close(self) -> None:
        if self._query is not None:
            self._query.close()
        self._check = self._query = None

    def __enter__(self) -> _OwnedMotionChecker:
        if self._query is None:
            raise RuntimeError("Motion collision query is closed.")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _make_checker(
    robot: Robot,
    simulation: SimulationManager | None,
    control_part: str,
    *,
    include_scene: bool,
    include_rigids: bool = True,
    reuse: _OwnedMotionChecker | None = None,
) -> _OwnedMotionChecker:
    """Bind FK and pair policy to the existing isolated Newton pipeline."""
    import trimesh
    import yourdfpy
    from embodichain.lab.sim.scene.feasibility import (
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
    world_root = robot.get_local_pose(to_matrix=True)[0].detach().cpu().numpy().copy()
    observed_qpos = robot.get_qpos()[0].detach().cpu().clone()
    model_path, joint_names = str(robot.cfg.fpath), tuple(robot.joint_names)
    solver_links = (solver.root_link_name, solver.end_link_name)
    passive_ids = [
        index
        for index, name in enumerate(joint_names)
        if name in model.joint_map and model.joint_map[name].child not in active_links
    ]

    def configure(q: torch.Tensor) -> None:
        model.update_cfg(
            {
                name: float(value)
                for name, value in zip(joint_names, q, strict=True)
                if name in model.joint_map
            }
        )

    configure(observed_qpos)
    observed = (
        robot.get_link_pose(solver.end_link_name, to_matrix=True)[0]
        .detach()
        .cpu()
        .numpy()
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

    bodies, info, static_poses = [], {}, {}

    def add_mesh(
        identity: str, frame: str, mesh: trimesh.Trimesh, *, moving: bool
    ) -> None:
        # A convex input retains the same hull. Nonconvex inputs keep their
        # original triangles and enter Newton's explicit triangle route.
        kind = "convex" if mesh.is_convex and mesh.is_volume else "mesh"
        bodies.append(
            CollisionBody(
                identity,
                frame,
                kind,
                vertices=tuple(map(tuple, np.asarray(mesh.vertices))),
                triangles=tuple(map(tuple, np.asarray(mesh.faces))),
                robot=moving,
            )
        )

    for node, owner, mesh in geometry:
        identity = "robot:" + node
        add_mesh(identity, "robot_frame:" + node, mesh, moving=owner in active_links)
        info[identity] = ("active" if owner in active_links else "passive", node, owner)
    if include_scene:
        from embodichain.lab.sim.scene._collision_query_inputs import (
            _capture_free_motion_sources,
        )

        for name, snapshot in _capture_free_motion_sources(
            simulation, include_rigids=include_rigids
        ).items():
            shape = snapshot.shapes[0]
            if shape.vertices is not None and shape.triangles is not None:
                mesh = trimesh.Trimesh(
                    vertices=shape.vertices.cpu().numpy(),
                    faces=shape.triangles.cpu().numpy(),
                    process=False,
                )
            elif shape.half_extents is not None:
                mesh = trimesh.creation.box(
                    extents=2 * shape.half_extents.cpu().numpy()
                )
            else:
                raise ValueError(
                    "Unsupported physical collision shape for free motion."
                )
            identity, frame = "scene:" + name, "scene_frame:" + name
            add_mesh(identity, frame, mesh, moving=False)
            info[identity] = ("scene", name, name)
            static_poses[frame] = (
                snapshot.pose[0].cpu().numpy() @ shape.local_pose.cpu().numpy()
            )

    ignored = []
    for index, a in enumerate(bodies):
        ga, _, oa = info[a.collision_id]
        for b in bodies[index + 1 :]:
            gb, _, ob = info[b.collision_id]
            skip = "active" not in (ga, gb)
            if ga == gb == "active":
                skip = ob in near[oa] or {oa, ob} <= hand_links
            elif {ga, gb} == {"active", "passive"}:
                moving, fixed = (oa, ob) if ga == "active" else (ob, oa)
                skip = moving in mounting_links and fixed in mount_ancestors
            if skip:
                ignored.append((a.collision_id, b.collision_id))
    signature = (tuple(bodies), tuple(ignored), str(robot.device))
    query = None if reuse is None else reuse.transfer(signature)
    if query is None:
        query = NewtonCollisionChecker(
            bodies,
            ignored_pairs=ignored,
            allow_triangle_meshes=True,
            device=str(robot.device),
            use_cuda_graph=str(robot.device).startswith("cuda"),
        )
    if reuse is not None:
        reuse.close()

    def check(positions: torch.Tensor) -> dict[str, Any]:
        if (
            positions.ndim != 2
            or positions.shape[1] != len(joint_names)
            or not len(positions)
            or not torch.isfinite(positions).all()
        ):
            raise ValueError(
                "Motion collision samples must be finite nonempty (T, robot.dof)."
            )
        positions = positions.detach().cpu()
        if passive_ids and not torch.allclose(
            positions[:, passive_ids],
            observed_qpos[passive_ids][None].expand(len(positions), -1),
            atol=1e-7,
            rtol=0,
        ):
            raise ValueError(
                "Single-arm collision queries require passive joints to remain at their observed positions."
            )
        if not include_scene and (
            str(robot.cfg.fpath) != model_path
            or tuple(robot.joint_names) != joint_names
            or (
                robot.get_solver(control_part).root_link_name,
                robot.get_solver(control_part).end_link_name,
            )
            != solver_links
            or not torch.equal(robot.get_qpos()[0].detach().cpu(), observed_qpos)
            or not np.array_equal(
                robot.get_local_pose(to_matrix=True)[0].detach().cpu().numpy(),
                world_root,
            )
        ):
            raise ValueError(
                "Observed robot changed during the scoped collision check."
            )
        for frame, q in enumerate(positions):
            configure(q)
            poses = dict(static_poses)
            for node, _, _ in geometry:
                pose, _ = graph[node]
                poses["robot_frame:" + node] = world_root @ pose
            result, _ = query.check(poses, all_pairs=True)
            if result.status != "passed":
                groups = {"scene": [], "other_arm": [], "self": []}
                for a, b in query.last_pairs:
                    ia, ib = info[a], info[b]
                    if ia[0] != "active":
                        ia, ib = ib, ia
                    kind = (
                        "scene"
                        if ib[0] == "scene"
                        else "other_arm" if ib[0] == "passive" else "self"
                    )
                    pair = (ia[1], ib[1])
                    groups[kind].append(tuple(sorted(pair)) if kind == "self" else pair)
                for kind, pairs in groups.items():
                    if pairs:
                        return {
                            "valid": False,
                            "frame": frame,
                            "kind": kind,
                            "pairs": sorted(map(list, pairs)),
                        }
                raise RuntimeError(
                    "Newton rejected a collision query without qualified pair evidence."
                )
        return {
            "valid": True,
            "frames": len(positions),
            "scope": (
                "discrete_free_motion_meshes"
                if include_scene
                else "discrete_robot_meshes"
            ),
        }

    return _OwnedMotionChecker(check, query, signature)


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
