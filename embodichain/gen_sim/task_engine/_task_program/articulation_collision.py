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

"""Discrete mesh clearance for GenSim articulation free-space recovery only."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np
import torch

from .drawer_geometry import collision_meshes

__all__: list[str] = []

_ROBOT_CHECKERS: ContextVar[dict[tuple[int, str], Callable] | None] = ContextVar(
    "gen_sim_robot_collision_checkers", default=None
)


@contextmanager
def robot_collision_scope() -> Iterator[None]:
    """Reuse only invocation-owned robot proxies; scene checks stay fresh."""
    token = _ROBOT_CHECKERS.set({})
    try:
        yield
    finally:
        _ROBOT_CHECKERS.reset(token)


def _descendants(children: dict[str, set[str]], root: str) -> set[str]:
    result, pending = set(), [root]
    while pending:
        name = pending.pop()
        if name not in result:
            result.add(name)
            pending.extend(children.get(name, ()))
    return result


def check_free_motion(
    robot: Any, simulation: Any, positions: torch.Tensor, control_part: str
) -> dict[str, Any]:
    """Screen every command sample against a fresh, read-only scene snapshot.

    Only the operated arm's approach or empty-hand withdrawal is checked here.
    Intended hand/handle contact during sliding is not a free-space certificate.
    URDF neighbors within two joints and intra-gripper mechanism pairs are
    excluded from self tests. The first moving link may meet its rigid mounting
    ancestors across fixed frames; scene and other-arm contacts remain checked.
    """
    return _check_motion(robot, simulation, positions, control_part, include_scene=True)


def check_robot_motion(
    robot: Any, positions: torch.Tensor, control_part: str
) -> dict[str, Any]:
    """Screen robot proxies only; do not certify held-object or scene clearance."""
    return _check_motion(robot, None, positions, control_part, include_scene=False)


def _check_motion(
    robot: Any,
    simulation: Any,
    positions: torch.Tensor,
    control_part: str,
    *,
    include_scene: bool,
) -> dict[str, Any]:
    """Keep both certificates on the same URDF/FK and pair-exclusion rules."""
    if positions.ndim != 2 or not torch.isfinite(positions).all():
        raise ValueError("Recovery collision samples must be finite (T, robot.dof).")
    cache = None if include_scene else _ROBOT_CHECKERS.get()
    key = (id(robot), control_part)
    if cache is not None and key in cache:
        checker = cache[key]
    else:
        checker = _make_checker(
            robot, simulation, control_part, include_scene=include_scene
        )
        if cache is not None:
            cache[key] = checker
    return checker(positions)


def _make_checker(
    robot: Any, simulation: Any, control_part: str, *, include_scene: bool
) -> Callable[[torch.Tensor], dict[str, Any]]:
    """Own one read-only FK-calibrated geometry snapshot."""
    import trimesh
    import yourdfpy

    model = yourdfpy.URDF.load(
        robot.cfg.fpath,
        load_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    world = trimesh.collision.CollisionManager()
    for uid in simulation.get_rigid_object_uid_list() if include_scene else ():
        obj = simulation.get_rigid_object(uid)
        pose = obj.get_local_pose(to_matrix=True)[0].detach().cpu().numpy()
        for index, shape in enumerate(obj.get_collision_shapes()):
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
                    "Unsupported physical collision shape for E6 recovery."
                )
            world.add_object(
                f"{uid}:{index}",
                mesh,
                transform=pose @ shape.local_pose.cpu().numpy(),
            )
    for uid in simulation.get_articulation_uid_list() if include_scene else ():
        art = simulation.get_articulation(uid)
        meshes = collision_meshes(
            {"fpath": art.cfg.fpath, "body_scale": art.cfg.body_scale}
        )
        for (link, name), mesh in meshes.items():
            pose = art.get_link_pose(link, to_matrix=True)[0].cpu().numpy()
            world.add_object(f"{uid}:{name}", mesh, transform=pose)

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
    owners = {node: owner for node, owner, _ in geometry}
    active = trimesh.collision.CollisionManager()
    passive = trimesh.collision.CollisionManager()
    for node, owner, mesh in geometry:
        pose, _ = graph[node]
        manager = active if owner in active_links else passive
        manager.add_object(node, mesh, transform=world_root @ pose)

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
            # Geometry is immutable for this checker; only FK transforms change.
            for node, owner, _ in geometry:
                pose, _ = graph[node]
                manager = active if owner in active_links else passive
                manager.set_transform(
                    node, np.asarray(world_root @ pose, dtype=np.float32)
                )
            obstacles = (
                (("scene", world), ("other_arm", passive))
                if include_scene
                else (("other_arm", passive),)
            )
            for kind, other in obstacles:
                hit, pairs = active.in_collision_other(other, return_names=True)
                if hit and kind == "other_arm":
                    pairs = {
                        (a, b)
                        for a, b in pairs
                        if not (
                            owners[a] in mounting_links and owners[b] in mount_ancestors
                        )
                    }
                    hit = bool(pairs)
                if hit:
                    return {
                        "valid": False,
                        "frame": frame,
                        "kind": kind,
                        "pairs": sorted(map(list, pairs)),
                    }
            _, pairs = active.in_collision_internal(return_names=True)
            invalid = [
                [a, b]
                for a, b in pairs
                if owners[b] not in near[owners[a]]
                and not ({owners[a], owners[b]} <= hand_links)
            ]
            if invalid:
                return {
                    "valid": False,
                    "frame": frame,
                    "kind": "self",
                    "pairs": sorted(invalid),
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

    return check
