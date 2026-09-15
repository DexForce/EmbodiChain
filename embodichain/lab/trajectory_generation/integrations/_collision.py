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

"""Conservative CPU collision geometry for the bounded PickUp integration."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import numpy as np
import torch

__all__ = []


class _FullStateCollisionWorld:
    """URDF collision convex hulls and explicitly supported physical cuboids.

    Never substitute render meshes for collision geometry. One convex hull per
    link conservatively covers all its URDF collision shapes, including their
    local transforms. Unsupported scene shapes/scales fail at construction.
    """

    def __init__(self, sim, robot):
        import fcl
        import trimesh
        from yourdfpy import URDF
        from embodichain.lab.sim.shapes import CubeCfg

        if tuple(robot.cfg.body_scale) != (1.0, 1.0, 1.0):
            raise ValueError("PickUp collision validation requires an unscaled URDF")
        self.fcl, self.robot = fcl, robot
        urdf = URDF.load(
            robot.cfg.fpath,
            build_scene_graph=False,
            load_meshes=False,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
        )
        scene = urdf.collision_scene
        meshes = defaultdict(list)
        for node in scene.graph.nodes_geometry:
            parent = scene.graph.transforms.parents[node]
            if parent not in urdf.link_map:
                raise ValueError(
                    "Collision geometry is not directly owned by a URDF link"
                )
            transform, name = scene.graph.get(frame_to=node, frame_from=parent)
            mesh = scene.geometry[name].copy()
            mesh.apply_transform(transform)
            meshes[parent].append(mesh)
        expected = {name for name, link in urdf.link_map.items() if link.collisions}
        if not expected or set(meshes) != expected:
            raise ValueError("Every URDF collision link must have loaded geometry")
        self.links = tuple(meshes)
        neighbors = defaultdict(set)
        fixed_links = {urdf.base_link}
        while True:
            previous = len(fixed_links)
            fixed_links.update(
                j.child
                for j in urdf.joint_map.values()
                if j.type == "fixed" and j.parent in fixed_links
            )
            if previous == len(fixed_links):
                break
        self.fixed_links = frozenset(fixed_links)
        for joint in urdf.joint_map.values():
            neighbors[joint.parent].add(joint.child)
            neighbors[joint.child].add(joint.parent)
        # Match the existing cuRobo profile's two-hop structural exclusions.
        self.adjacent_pairs = set()
        for link in self.links:
            nearby = neighbors[link] | set().union(
                *(neighbors[n] for n in neighbors[link])
            )
            self.adjacent_pairs.update(
                frozenset((link, other)) for other in nearby if other != link
            )
        self.objects = {}
        self.bounds = {}
        for link, parts in meshes.items():
            hull = trimesh.util.concatenate(parts).convex_hull
            faces = np.c_[np.full(len(hull.faces), 3), hull.faces].astype(np.int32)
            geometry = fcl.Convex(
                hull.vertices.astype(np.float64), len(faces), faces.ravel()
            )
            self.objects[link] = fcl.CollisionObject(geometry)
            self.bounds[link] = hull.bounds
        self.entity_ids = tuple(sim.get_rigid_object_uid_list())
        if set(self.links) & set(self.entity_ids) or "__ground__" in self.entity_ids:
            raise ValueError("Collision link and object names must be distinct")
        for uid in self.entity_ids:
            obj = sim.get_rigid_object(uid)
            if not isinstance(obj.cfg.shape, CubeCfg):
                raise ValueError(
                    "PickUp collision validation currently supports cuboid rigid objects only"
                )
            scale = obj.get_body_scale().detach().cpu().numpy()
            if not np.allclose(scale, scale[:1], atol=0, rtol=0):
                raise ValueError(
                    "Collision shapes must have identical dimensions in every row"
                )
            size = np.asarray(obj.cfg.shape.size, dtype=float) * scale[0]
            self._box(uid, size)
        # SimulationManager's implicit floor, in each arena's local frame.
        self._box("__ground__", np.array([1000.0, 1000.0, 100.0]))
        self.ground_pose = np.eye(4)
        self.ground_pose[2, 3] = -50.001
        self.self_pairs = tuple(combinations(self.links, 2))
        self.world_pairs = tuple(
            (link, uid)
            for link in self.links
            for uid in (*self.entity_ids, "__ground__")
        )

    def _box(self, uid, size):
        if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("Cuboid dimensions must be three positive finite values")
        self.objects[uid] = self.fcl.CollisionObject(self.fcl.Box(*size))
        self.bounds[uid] = np.stack([-size / 2, size / 2])

    def link_poses(self, positions, root_pose, extra_links=()):
        from embodichain.lab.sim.objects.articulation import Articulation

        local = Articulation.compute_fk(
            self.robot,
            positions.to(self.robot.device),
            link_names=tuple(dict.fromkeys((*self.links, *extra_links))),
            qpos_joint_names=tuple(self.robot.joint_names),
            to_dict=True,
        )
        root = root_pose.detach().cpu().double().numpy()
        return {
            name: root @ local[name].get_matrix().detach().cpu().double().numpy()
            for name in dict.fromkeys((*self.links, *extra_links))
        }

    def collisions(self, link_poses, entity_poses, target_id, allowed):
        poses = {**link_poses, **entity_poses, "__ground__": self.ground_pose}
        aabbs = {}
        for name, pose in poses.items():
            if name not in self.objects:
                raise ValueError(f"Missing collision shape: {name}")
            self.objects[name].setTransform(
                self.fcl.Transform(pose[:3, :3], pose[:3, 3])
            )
            low, high = self.bounds[name]
            center = pose[:3, :3] @ ((low + high) / 2) + pose[:3, 3]
            radius = np.abs(pose[:3, :3]) @ ((high - low) / 2)
            aabbs[name] = (center - radius, center + radius)
        pairs = (
            *self.self_pairs,
            *self.world_pairs,
            *(
                (target_id, uid)
                for uid in (*self.entity_ids, "__ground__")
                if uid != target_id
            ),
        )
        for first, second in pairs:
            if allowed(first, second):
                continue
            al, ah = aabbs[first]
            bl, bh = aabbs[second]
            if (al > bh).any() or (bl > ah).any():
                continue
            result = self.fcl.CollisionResult()
            if self.fcl.collide(
                self.objects[first],
                self.objects[second],
                self.fcl.CollisionRequest(),
                result,
            ):
                return f"{first} / {second}"
        return None
