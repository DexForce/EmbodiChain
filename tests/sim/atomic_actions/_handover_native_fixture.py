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

"""Controlled six-coordinate pose solvers with real rigid FK and FCL geometry."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import random
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from embodichain.lab.sim.atomic_actions import (
    HeldObjectState,
    TaskState,
    RobotObservation,
    SceneSnapshot,
    EntityState,
    HandOverGoal,
)
from ._ordinary_pick_fixture import ordinary_fixture, OrdinaryCandidates


def native_transfer_fixture(directory, mode):
    gen, old, semantics, _, _ = ordinary_fixture(directory)
    del gen._generate_once
    robot = gen.robot
    robot.uid = "robot"
    robot.dof = 14
    robot.joint_names = [
        prefix + axis
        for prefix in ("s_", "d_")
        for axis in ("x", "y", "z", "rx", "ry", "rz")
    ] + ["sh", "dh"]
    ids = {
        None: list(range(14)),
        "source_arm": list(range(6)),
        "destination_arm": list(range(6, 12)),
        "source_hand": [12],
        "destination_hand": [13],
    }
    robot.control_parts = {name: object() for name in ids if name is not None}
    robot.get_joint_ids.side_effect = lambda name=None: ids[name]
    qpos = torch.tensor(
        [[-0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.01, 0.04]]
    )
    robot.get_qpos.side_effect = lambda name=None: qpos[:, ids[name]].clone()
    robot.get_qvel.side_effect = lambda name=None: torch.zeros_like(qpos[:, ids[name]])
    robot.get_qpos_limits.side_effect = (
        lambda name=None, **kw: torch.tensor([[[-4.0, 4.0]]])
        .expand(1, len(ids[name]), 2)
        .clone()
    )
    robot.get_qvel_limits.side_effect = lambda name=None, **kw: torch.full(
        (1, len(kw.get("joint_ids", ids[name]))), 100.0
    )
    robot.cfg.solver_cfg = {
        name: SimpleNamespace(
            root_link_name=name + "_root",
            end_link_name=name + "_tcp",
            class_type="PytorchSolver",
            pos_eps=1e-5,
            rot_eps=1e-5,
            is_only_position_constraint=False,
        )
        for name in ("source_arm", "destination_arm")
    }

    def root_pose(link_name, **kwargs):
        value = torch.eye(4)[None]
        value[0, 0, 3] = -0.5 if link_name == "source_arm_root" else 0.5
        return value

    robot.get_link_pose.side_effect = root_pose
    solver = SimpleNamespace(
        lower_qpos_limits=torch.full((6,), -4.0),
        upper_qpos_limits=torch.full((6,), 4.0),
        get_iteration_params=lambda: {"pos_eps": 1e-5, "rot_eps": 1e-5},
    )
    robot.get_solver.side_effect = lambda name: solver
    path = directory / "six_coordinate.urdf"
    path.write_text(
        '<robot name="fixture">'
        + "".join(
            f'<joint name="{name}" type="prismatic"><limit velocity="100"/></joint>'
            for name in robot.joint_names
        )
        + "</robot>"
    )
    robot.cfg.fpath = str(path)

    def pose_from_joints(values):
        result = torch.eye(4).expand(*values.shape[:-1], 4, 4).clone()
        rotations = Rotation.from_rotvec(
            values[..., 3:6].reshape(-1, 3).double().numpy()
        ).as_matrix()
        result[..., :3, :3] = torch.tensor(rotations, dtype=values.dtype).reshape(
            *values.shape[:-1], 3, 3
        )
        result[..., :3, 3] = values[..., :3]
        return result

    fk_calls = []

    def fk(qpos, **kwargs):
        if "link_names" in kwargs:
            fk_calls.append(
                {
                    "joints": qpos.tolist(),
                    "links": kwargs["link_names"],
                    "order": list(kwargs["qpos_joint_names"]),
                }
            )
            return (
                pose_from_joints(qpos[:, 6:12])[:, None]
                .expand(-1, len(kwargs["link_names"]), -1, -1)
                .clone()
            )
        return pose_from_joints(qpos)

    robot.compute_fk.side_effect = fk
    robot.compute_batch_fk.side_effect = fk
    grasps = []
    ik = []

    class Generator(OrdinaryCandidates):
        def get_valid_grasp_poses(
            self,
            *,
            obj_poses,
            approach_direction,
            obj_longest_axis,
            is_positive_part,
            **kwargs,
        ):
            grasps.append(
                {
                    "poses": obj_poses.tolist(),
                    "direction": approach_direction.tolist(),
                    "axis": obj_longest_axis.tolist(),
                    "positive": is_positive_part.tolist(),
                }
            )
            torch.rand(())
            np.random.rand()
            random.random()
            result = obj_poses.clone()
            result[:, :3, 3] += (
                0.05
                * obj_longest_axis
                * torch.where(is_positive_part, 1.0, -1.0)[:, None]
            )
            reference = approach_direction.new_tensor([0.0, 1.0, 0.0]).expand_as(
                approach_direction
            )
            x = torch.linalg.cross(reference, approach_direction)
            x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            result[:, :3, 0] = x
            result[:, :3, 1] = torch.linalg.cross(approach_direction, x)
            result[:, :3, 2] = approach_direction
            return [(value[None], torch.zeros(1)) for value in result]

    def solve(pose, joint_seed, name, **kwargs):
        endpoint = "env_ids" in kwargs
        ik.append(
            {
                "part": name,
                "pose": pose.tolist(),
                "seed": joint_seed.tolist(),
                "endpoint": endpoint,
            }
        )
        torch.rand(())
        np.random.rand()
        random.random()
        rotations = Rotation.from_matrix(pose[:, :3, :3].double().numpy()).as_rotvec()
        result = torch.cat(
            (pose[:, :3, 3], torch.tensor(rotations, dtype=pose.dtype)), 1
        )
        if mode == "bad_command" and name == "destination_arm" and not endpoint:
            result[:, 1] += 0.1
        return (
            torch.tensor(
                [
                    not (
                        endpoint
                        and (
                            mode == "reject"
                            or (
                                mode == "late_reject"
                                and sum(call["endpoint"] for call in ik) % 3 == 0
                            )
                        )
                    )
                ]
            ),
            result,
        )

    robot.compute_ik.side_effect = solve
    object_pose = pose_from_joints(qpos[:, :6])
    held = HeldObjectState(semantics, torch.eye(4)[None], object_pose)
    context = replace(
        old,
        robot=RobotObservation(
            0.0, qpos, torch.zeros_like(qpos), root_pose=torch.eye(4)[None]
        ),
        task=TaskState(1, "cpu", held_objects={"source_arm": held}),
        scene=SceneSnapshot(0.0, 0, {"object": EntityState(object_pose)}),
    )
    target = object_pose.clone()
    target[:, 2, 3] = 0.5
    return (
        gen,
        context,
        HandOverGoal(semantics, target_pose=target),
        Generator(),
        grasps,
        ik,
        fk_calls,
    )


def transfer_scene(body_type, mode):
    part = body_type(
        "robot/receiver_palm",
        "robot",
        "receiver_palm",
        (trimesh.creation.box((0.02, 0.02, 0.02)),),
        np.eye(4),
        0.002,
        0.0,
    )
    pose = np.eye(4)
    pose[:3, 3] = [
        0.0,
        2.0 if mode == "clear" else (0.1 if mode == "bad_command" else 0.0),
        0.45,
    ]
    obstacle = body_type(
        "wall",
        "wall",
        None,
        (trimesh.creation.box((0.04, 0.04, 0.04)),),
        pose,
        0.005,
        0.0,
    )
    return (
        (part,),
        (obstacle,),
        [
            {
                "owner": "wall",
                "link": None,
                "status": "controlled_fixture",
                "shapes": 1,
                "plane_shapes": 0,
            }
        ],
    )
