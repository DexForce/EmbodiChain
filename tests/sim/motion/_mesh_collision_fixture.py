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

"""Fixed geometry fixture shared by direct motion and Atomic invocation tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import torch

from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc, RigidBodyShape


def mesh_query_fixture():
    robot = Mock()
    robot.uid = "robot"
    robot.device, robot.dof, robot.num_instances = torch.device("cpu"), 5, 1
    robot.cfg.fpath = str(Path(__file__).parent / "data" / "free_motion_fixture.urdf")
    robot.joint_names = ["arm_slide", "arm_fold", "jaw_a", "jaw_b", "passive_slide"]
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_joint_ids.side_effect = lambda name=None: {
        None: [0, 1, 2, 3, 4],
        "arm": [0, 1],
        "hand": [2, 3],
    }[name]
    robot.get_qpos.return_value = torch.zeros(1, 5)
    robot.get_qvel.return_value = torch.zeros(1, 5)
    robot.get_qpos_limits.side_effect = (
        lambda name=None: torch.tensor([[[-1.0, 1.0]]])
        .expand(1, len(robot.get_joint_ids(name=name)), 2)
        .clone()
    )
    robot.get_qvel_limits.side_effect = lambda name=None: torch.full(
        (1, len(robot.get_joint_ids(name=name))), 20.0
    )
    robot.get_solver.return_value = NS(root_link_name="mount", end_link_name="tool")
    robot.get_local_pose.return_value = torch.eye(4)[None]
    tool = torch.eye(4)[None]
    tool[:, :3, 3] = torch.tensor([0.55, 0, 0.3])
    robot.get_link_pose.return_value = tool

    def full_fk(*, qpos, name, link_names, qpos_joint_names, to_matrix):
        assert name is None and to_matrix
        assert tuple(qpos_joint_names) == tuple(robot.joint_names)
        result = torch.eye(4).repeat(len(qpos), len(link_names), 1, 1)
        for row, q in enumerate(qpos):
            slide, fold, jaw_a, jaw_b, passive = q
            # Independently specified prismatic fixture FK. Each jaw consumes
            # its actual supplied state, even when a source declares a mimic.
            origins = {
                "base": (0.0, 0.0, 0.0),
                "mount": (0.0, 0.0, 0.3),
                "shoulder": (slide, 0.0, 0.3),
                "middle": (0.3 + slide + fold, 0.0, 0.3),
                "wrist": (0.4 + slide + fold, 0.0, 0.3),
                "tool": (0.55 + slide + fold, 0.0, 0.3),
                "pad_a": (0.59 + slide + fold, 0.02 + jaw_a, 0.3),
                "pad_b": (0.59 + slide + fold, -0.02 - jaw_b, 0.3),
                "passive": (1.2, passive, 0.3),
            }
            for column, link in enumerate(link_names):
                result[row, column, :3, 3] = torch.tensor(origins[link])
        return result

    robot.compute_fk.side_effect = full_fk

    def link_pose(name, **kwargs):
        if name == "tool":
            return robot.get_link_pose.return_value.clone()
        local = full_fk(
            qpos=robot.get_qpos(),
            name=None,
            link_names=[name],
            qpos_joint_names=robot.joint_names,
            to_matrix=True,
        )[:, 0]
        return robot.get_local_pose(to_matrix=True) @ local

    robot.get_link_pose.side_effect = link_pose
    # Materialized public shape ports retain the same authored analytic boxes;
    # no source geometry, dimensions, local origin or collision part is changed.
    document = ET.parse(robot.cfg.fpath).getroot()
    robot.link_names = tuple(link.attrib["name"] for link in document.findall("link"))
    shapes = {}
    index = 0
    for link in document.findall("link"):
        values = []
        for collision in link.findall("collision"):
            size = torch.tensor(
                [
                    float(value)
                    for value in collision.find("geometry/box").attrib["size"].split()
                ]
            )
            local = torch.eye(4)
            origin = collision.find("origin")
            if origin is not None:
                local[:3, 3] = torch.tensor(
                    [
                        float(value)
                        for value in origin.attrib.get("xyz", "0 0 0").split()
                    ]
                )
            values.append(
                CollisionShapeDesc(
                    f"geometry_{index}",
                    RigidBodyShape.BOX,
                    local,
                    half_extents=size / 2,
                )
            )
            index += 1
        shapes[link.attrib["name"]] = values
    robot.get_link_collision_shapes.side_effect = lambda name, env_id=0: shapes[name]
    robot.get_link_physical_attr.side_effect = lambda **kwargs: [
        NS(enable_collision=True, rest_offset=0.0, contact_offset=0.0)
    ]
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = [], [], []
    shape = CollisionShapeDesc(
        "obstacle",
        RigidBodyShape.BOX,
        torch.eye(4),
        half_extents=torch.tensor([0.04, 0.04, 0.04]),
    )
    obstacle_pose = torch.eye(4)[None]
    obstacle_pose[:, :3, 3] = torch.tensor([0.85, 0, 0.3])
    obstacle = NS(
        get_local_pose=lambda **kw: obstacle_pose.clone(),
        get_collision_shapes=lambda env_id=0: [shape],
    )
    physical = NS(
        is_collision_enable=lambda: True,
        get_rest_offset=lambda: 0.0,
        get_contact_offset=lambda: 0.0,
        get_shape_count=lambda: len(obstacle.get_collision_shapes()),
    )
    handle = NS(
        get_name=lambda: "obstacle",
        native=lambda: NS(get_physical_body=lambda: physical),
    )
    simulation = NS(
        spawn_result=NS(get_paths=lambda: ("obstacle",), get=lambda path: handle),
        num_envs=1,
        is_default_backend=True,
        get_rigid_object_uid_list=lambda: ["obstacle"],
        get_rigid_object=lambda uid: obstacle,
        get_articulation_uid_list=lambda: [],
        get_robot=lambda uid: robot if uid == robot.uid else None,
    )
    return robot, simulation
