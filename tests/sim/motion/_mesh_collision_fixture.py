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

import torch

from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc, RigidBodyShape


def mesh_query_fixture():
    robot = Mock()
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
        get_collision_shapes=lambda: [shape],
    )
    simulation = NS(
        num_envs=1,
        get_rigid_object_uid_list=lambda: ["obstacle"],
        get_rigid_object=lambda uid: obstacle,
        get_articulation_uid_list=lambda: [],
    )
    return robot, simulation
