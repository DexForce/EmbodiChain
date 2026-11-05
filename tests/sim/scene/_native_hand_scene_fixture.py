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

from __future__ import annotations
import numpy as np
import trimesh


def clearance_case(body_type, plane_type, mode):
    hand = body_type(
        "robot/finger",
        "robot",
        "finger",
        (trimesh.creation.box((0.02, 0.02, 0.02)),),
        np.eye(4),
        0.01,
        -0.001,
    )
    plane = plane_type(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.02]))
    plane_body = body_type("floor", "floor", None, (), np.eye(4), 0.02, 0.0, (plane,))
    obstacle_pose = np.eye(4)
    obstacle_pose[0, 3] = 0.2
    obstacle = body_type(
        "wall",
        "wall",
        None,
        (trimesh.creation.box((0.02, 0.02, 0.02)),),
        obstacle_pose,
        0.02,
        0.0,
    )
    poses = np.eye(4)[None, None].copy()
    poses[0, 0, 2, 3] = 0.1
    if mode.startswith("plane"):
        obstacles = (plane_body,)
        if mode == "plane_envelope":
            poses[0, 0, 2, 3] = 0.06
    elif mode == "mixed":
        obstacles = (
            body_type(
                "mixed", "mixed", None, obstacle.meshes, np.eye(4), 0.02, 0.0, (plane,)
            ),
        )
        poses[0, 0, 2, 3] = 0.06
    else:
        obstacles = (obstacle,)
        poses[0, 0, 2, 3] = 0
        if mode == "mesh_envelope":
            poses[0, 0, 0, 3] = 0.155
        elif mode == "mesh_intersection":
            poses[0, 0, 0, 3] = 0.19
        elif mode == "later_frame":
            poses = np.repeat(poses, 2, axis=0)
            poses[1, 0, 0, 3] = 0.19
        elif mode == "ordered_obstacles":
            obstacles = (
                body_type(
                    "first", "first", None, obstacle.meshes, obstacle_pose, 0.02, 0.0
                ),
                obstacle,
            )
            poses[0, 0, 0, 3] = 0.19
    return (hand,), obstacles, poses


class Plane:
    def __init__(self):
        self.normal = np.array([0.0, 0.0, 2.0])
        self.distance = 0.1
        self.local_pose = np.eye(4)
        self.local_pose[2, 3] = -0.3


def capture_case(directory):
    from types import SimpleNamespace as NS
    from ._native_capture_fixture import setup_case, Box, Body, Rigid

    robot, simulation, _, _, _ = setup_case(directory)
    robot.control_parts = robot.cfg.control_parts
    simulation.native_robot.bodies["mount"] = None
    simulation.native_robot.get_world_pose = lambda: robot.pose[0].numpy()
    simulation.spawn_result.get_arena = lambda name: NS(
        get_root_node=lambda: NS(get_world_pose=lambda: np.eye(4))
    )
    for item in simulation.rigids.values():
        item.get_world_pose = lambda item=item: item.pose[0].numpy()

    def excluded_read(*args, **kwargs):
        raise AssertionError("The excluded target pose/body must never be read.")

    simulation.rigids["source"].get_world_pose = excluded_read
    simulation.rigids["source"].native = excluded_read
    bodies = {
        "empty": None,
        "hidden": Body((Box((0.05, 0.06, 0.07)),), rest=-0.001, contact=0.01),
        "disabled": Body((), enabled=False),
    }
    cabinet = NS(
        path="/arena0/cabinet",
        desc=NS(body_scale=np.array([2.0, 3.0, 4.0])),
        get_name=lambda: "cabinet",
        get_link_names=lambda: list(bodies),
        render_articulation=NS(get_physical_body=bodies.get),
        get_link_desc=lambda name: NS(
            collisions=() if name == "empty" else (object(),)
        ),
    )
    pose = np.eye(4)
    pose[:3, 3] = [0.5, 1.0, 0.1]
    cabinet.get_link_pose = lambda name: pose.copy()
    simulation.handles[cabinet.path] = cabinet
    plane = Rigid(
        "plane", Body((Plane(),), rest=-0.001, contact=0.005), (0.0, 0.0, -1.0)
    )
    plane.get_world_pose = lambda: plane.pose[0].numpy()
    simulation.handles[plane.path] = plane
    return robot, simulation


def capture_record(query, robot):
    def record(body):
        return {
            "name": body.name,
            "owner": body.owner,
            "link": body.link,
            "pose": body.pose.tolist(),
            "contact": body.contact_offset,
            "rest": body.rest_offset,
            "meshes": [
                {"vertices": m.vertices.tolist(), "faces": m.faces.tolist()}
                for m in body.meshes
            ],
            "planes": [
                {"normal": p.normal.tolist(), "point": p.point.tolist()}
                for p in body.planes
            ],
        }

    return {
        "root": query.root.tolist(),
        "root_dtype": str(query.root.dtype),
        "parts": [record(p) for p in query.parts],
        "obstacles": [record(p) for p in query.obstacles],
        "inventory": query.inventory,
        "fk_calls": robot.calls,
        "qpos_reads": robot.reads,
    }
