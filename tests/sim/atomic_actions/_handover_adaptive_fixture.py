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

"""Two analytic translation arms and original ordered receiving-grasp inputs."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import random
import numpy as np
import torch
from embodichain.lab.sim.atomic_actions import (
    HeldObjectState,
    TaskState,
    RobotObservation,
    SceneSnapshot,
    EntityState,
    HandOverGoal,
)
from ._ordinary_pick_fixture import ordinary_fixture, OrdinaryCandidates


def handover_fixture(directory, mode):
    gen, old, semantics, _, _ = ordinary_fixture(directory)
    del gen._generate_once
    robot = gen.robot
    robot.dof = 8
    robot.joint_names = ["sx", "sy", "sz", "dx", "dy", "dz", "sh", "dh"]
    robot.control_parts = {
        name: object()
        for name in ("source_arm", "destination_arm", "source_hand", "destination_hand")
    }
    ids = {
        None: list(range(8)),
        "source_arm": [0, 1, 2],
        "destination_arm": [3, 4, 5],
        "source_hand": [6],
        "destination_hand": [7],
    }
    robot.get_joint_ids.side_effect = lambda name=None: ids[name]
    qpos = torch.tensor([[-0.2, 0.0, 0.4, 0.2, 0.0, 0.4, 0.01, 0.04]])
    robot.get_qpos.side_effect = lambda name=None: qpos[:, ids[name]].clone()
    robot.get_qvel.side_effect = lambda name=None: torch.zeros_like(qpos[:, ids[name]])
    robot.get_qpos_limits.side_effect = (
        lambda name=None, **kw: torch.tensor([[[-2.0, 2.0]]])
        .expand(1, len(ids[name]), 2)
        .clone()
    )
    robot.get_qvel_limits.side_effect = lambda name=None, **kw: torch.ones(
        1, len(kw.get("joint_ids", ids[name]))
    )
    robot.cfg.solver_cfg = {
        name: SimpleNamespace(root_link_name=name + "_root")
        for name in ("source_arm", "destination_arm")
    }

    def root_pose(link_name, **kw):
        value = torch.eye(4)[None]
        value[0, 1, 3] = -0.5 if link_name == "source_arm_root" else 0.5
        return value

    robot.get_link_pose.side_effect = root_pose
    path = directory / "handover.urdf"
    path.write_text(
        '<robot name="fixture">'
        + "".join(
            f'<joint name="{name}" type="prismatic"><limit velocity="1"/></joint>'
            for name in robot.joint_names
        )
        + "</robot>"
    )
    robot.cfg.fpath = str(path)
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
                    "axis": (
                        None if obj_longest_axis is None else obj_longest_axis.tolist()
                    ),
                    "positive": is_positive_part.tolist(),
                    "supported_region": "axis_region_selector" in kwargs,
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
            x = torch.tensor([1.0, 0.0, 0.0]).expand_as(approach_direction)
            result[:, :3, 0] = x
            result[:, :3, 1] = torch.linalg.cross(approach_direction, x)
            result[:, :3, 2] = approach_direction
            return [(value[None], torch.zeros(1)) for value in result]

    def solve(pose, joint_seed, name, **kw):
        ik.append({"part": name, "pose": pose.tolist(), "seed": joint_seed.tolist()})
        torch.rand(())
        np.random.rand()
        random.random()
        valid = mode in {"nominal", "exit"} or (mode == "retry" and len(grasps) > 1)
        return torch.tensor([valid]), pose[:, :3, 3].clone()

    robot.compute_ik.side_effect = solve
    object_pose = robot.compute_fk(qpos=qpos[:, :3], name="source_arm", to_matrix=True)
    held = HeldObjectState(semantics, torch.eye(4)[None], object_pose)
    context = replace(
        old,
        robot=RobotObservation(0.0, qpos, torch.zeros_like(qpos)),
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
    )


def plain_metadata(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {
            key.removeprefix("gen_sim_"): plain_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [plain_metadata(item) for item in value]
    return value
