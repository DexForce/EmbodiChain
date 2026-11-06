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

"""Deterministic observed robot/geometry inputs for source Pour comparisons."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch
import trimesh

from embodichain.lab.sim.atomic_actions import (
    AxisAlignAffordance,
    EntityState,
    HeldObjectState,
    ObjectSemantics,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.utils.math import pose_inv


def pour_fixture(directory: Path):
    robot = Mock()
    robot.device, robot.dof, robot.num_instances = torch.device("cpu"), 5, 1
    robot.control_parts = {"left_arm": object(), "hand": object()}
    robot.joint_names = ["arm_slide", "arm_fold", "jaw_a", "jaw_b", "passive_slide"]
    robot.get_joint_ids.side_effect = lambda name=None: {
        None: [0, 1, 2, 3, 4],
        "left_arm": [0, 1],
        "hand": [2, 3],
    }[name]
    robot.cfg.control_parts = {
        "left_arm": robot.joint_names[:2],
        "hand": robot.joint_names[2:4],
    }
    robot.cfg.solver_cfg = {
        "left_arm": NS(
            root_link_name="mount", end_link_name="tool", tcp=torch.eye(4).tolist()
        )
    }
    template = Path(__file__).parents[1] / "motion/data/free_motion_fixture.urdf"
    path = directory / "pour_fixture.urdf"
    path.write_text(
        template.read_text()
        .replace("pad_a", "left_inner_finger_pad")
        .replace("pad_b", "left_right_inner_finger_pad")
    )
    robot.cfg.fpath = str(path)
    robot.link_names = [
        "base",
        "mount",
        "shoulder",
        "middle",
        "wrist",
        "tool",
        "left_inner_finger_pad",
        "left_right_inner_finger_pad",
        "passive",
    ]
    robot.get_qpos.return_value = torch.zeros(1, 5)
    robot.get_qvel.return_value = torch.zeros(1, 5)
    robot.get_qvel_limits.side_effect = lambda name=None: torch.full(
        (1, len(robot.get_joint_ids(name=name))), 20.0
    )
    robot.get_qpos_limits.side_effect = (
        lambda name=None: torch.tensor([[[-1.0, 1.0]]])
        .expand(1, len(robot.get_joint_ids(name=name)), 2)
        .clone()
    )
    robot.get_solver.return_value = robot.cfg.solver_cfg["left_arm"]
    robot.get_local_pose.return_value = torch.eye(4)[None]
    tool = torch.eye(4)[None]
    tool[:, :3, 3] = torch.tensor([0.55, 0.0, 0.3])
    pads = {}
    for name, y in [
        ("left_inner_finger_pad", 0.02),
        ("left_right_inner_finger_pad", -0.02),
    ]:
        pose = tool.clone()
        pose[:, 0, 3] += 0.04
        pose[:, 1, 3] += y
        pads[name] = pose
    robot.get_link_pose.side_effect = lambda name, **kw: pads.get(name, tool).clone()
    robot.compute_fk.return_value = tool.clone()
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = [], [], []
    mesh = trimesh.creation.box(extents=[0.06, 0.08, 0.13])
    mesh.apply_translation([0, 0, 0.015])
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    receiver_vertices = torch.tensor(
        list(product([-0.15, 0.15], [-0.12, 0.12], [-0.02, 0.04])), dtype=torch.float32
    )
    semantics = ObjectSemantics(
        AxisAlignAffordance(
            mesh_vertices=vertices,
            mesh_triangles=torch.tensor(mesh.faces),
            internal_axis=torch.tensor([0.0, 0.0, 1.0]),
        ),
        {},
        "object",
    )
    relation = torch.eye(4)[None]
    relation[:, :3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    relation[:, :3, 3] = torch.tensor([0.015, -0.02, 0.03])
    source_pose = tool @ pose_inv(relation)
    observed_source = source_pose.clone()
    observed_source[:, 0, 3] += 0.002
    receiver_pose = torch.eye(4)[None]
    receiver_pose[:, :3, 3] = torch.tensor([0.8, 0.2, 0.15])
    held = HeldObjectState(semantics, object_to_eef=relation, grasp_xpos=tool)
    context = PlanningContext(
        robot=RobotObservation(
            timestamp=3.5, qpos=robot.get_qpos(), qvel=torch.zeros(1, 5)
        ),
        task=TaskState(1, "cpu", held_objects={"left_arm": held}),
        scene=SceneSnapshot(
            timestamp=3.5,
            version=4,
            entities={
                "object": EntityState(pose=observed_source),
                "receiver": EntityState(pose=receiver_pose),
            },
        ),
        env_ids=torch.tensor([0]),
        control_dt=0.04,
    )
    returned = torch.eye(4)
    returned[:3, 3] = torch.tensor([0.1, 0.2, 0.3])
    return robot, context, vertices, receiver_vertices, returned


def hand_cfg():
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

    return RobotHandGeometryCfg(
        motion_part="left_arm",
        hand_part="hand",
        hand_root_link="tool",
        pad_link_names=("left_inner_finger_pad", "left_right_inner_finger_pad"),
        open_qpos=(0.0, 0.0),
        closed_qpos=(0.1, 0.1),
        sample_count=5,
        body_max_edge=0.03,
        pad_max_edge=0.02,
        max_subdivisions=2,
    )
