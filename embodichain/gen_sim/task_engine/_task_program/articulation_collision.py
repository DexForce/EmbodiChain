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

from typing import Any

import numpy as np
import torch

from .drawer_geometry import collision_meshes

__all__: list[str] = []


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
    excluded from self tests, never from scene or other-arm tests.
    """
    import trimesh
    import yourdfpy

    if positions.ndim != 2 or not torch.isfinite(positions).all():
        raise ValueError("Recovery collision samples must be finite (T, robot.dof).")
    model = yourdfpy.URDF.load(
        robot.cfg.fpath,
        load_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    world = trimesh.collision.CollisionManager()
    for uid in simulation.get_rigid_object_uid_list():
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
    for uid in simulation.get_articulation_uid_list():
        art = simulation.get_articulation(uid)
        meshes = collision_meshes(
            {"fpath": art.cfg.fpath, "body_scale": art.cfg.body_scale}
        )
        for (link, name), mesh in meshes.items():
            pose = art.get_link_pose(link, to_matrix=True)[0].cpu().numpy()
            world.add_object(f"{uid}:{name}", mesh, transform=pose)

    links = set(model.link_map)
    children: dict[str, set[str]] = {}
    neighbors = {name: {name} for name in links}
    for joint in model.robot.joints:
        children.setdefault(joint.parent, set()).add(joint.child)
        neighbors[joint.parent].add(joint.child)
        neighbors[joint.child].add(joint.parent)
    solver = robot.get_solver(control_part)
    active_links = _descendants(children, solver.root_link_name)
    hand_links = _descendants(children, solver.end_link_name)
    near = {name: set.union(*(neighbors[n] for n in neighbors[name])) for name in links}
    world_root = robot.get_local_pose(to_matrix=True)[0].cpu().numpy()

    def configure(q: torch.Tensor) -> None:
        model.update_cfg(
            {
                name: float(value)
                for name, value in zip(robot.joint_names, q)
                if name in model.joint_map
            }
        )

    configure(robot.get_qpos()[0])
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
    for frame, q in enumerate(positions):
        configure(q)
        active = trimesh.collision.CollisionManager()
        passive = trimesh.collision.CollisionManager()
        for node, owner, mesh in geometry:
            pose, _ = graph[node]
            manager = active if owner in active_links else passive
            manager.add_object(node, mesh, transform=world_root @ pose)
        for kind, other in (("scene", world), ("other_arm", passive)):
            hit, pairs = active.in_collision_other(other, return_names=True)
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
        "scope": "discrete_free_motion_meshes",
    }
