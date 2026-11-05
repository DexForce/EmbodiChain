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

"""Verified dual attachments and bounded synchronized motion inputs."""

from __future__ import annotations
from dataclasses import replace
import torch
from embodichain.lab.sim.atomic_actions import (
    CoordinatedPickGoal,
    HeldObjectState,
    TaskState,
    SceneSnapshot,
    EntityState,
    RobotObservation,
)
from ._handover_native_fixture import native_transfer_fixture


def coordinated_fixture(directory, mode):
    gen, context, old_goal, grasp, grasps, ik, fk = native_transfer_fixture(
        directory, "clear"
    )
    import random
    import numpy as np
    from scipy.spatial.transform import Rotation

    gen.robot.get_solver("source_arm").supports_continuous_batch_ik = (
        mode == "continuous"
    )

    def batch(pose, joint_seed, name, **kwargs):
        ik.append(
            {
                "batch": True,
                "part": name,
                "pose": pose.tolist(),
                "seed": joint_seed.tolist(),
                "continuous": kwargs.get("continuous"),
            }
        )
        torch.rand(())
        np.random.rand()
        random.random()
        rv = Rotation.from_matrix(
            pose[..., :3, :3].reshape(-1, 3, 3).double().numpy()
        ).as_rotvec()
        rv = torch.tensor(rv, dtype=pose.dtype).reshape(*pose.shape[:-2], 3)
        return torch.ones(pose.shape[:-2], dtype=torch.bool), torch.cat(
            (pose[..., :3, 3], rv), -1
        )

    gen.robot.compute_batch_ik.side_effect = batch
    semantics = old_goal.semantics
    initial = torch.eye(4)[None]
    initial[:, 2, 3] = 0.4
    target = initial.clone()
    target[:, 1, 3] = 0.1
    target[:, 2, 3] = 0.55
    target[:, :3, :3] = torch.tensor(
        [[0.9950042, -0.09983342, 0.0], [0.09983342, 0.9950042, 0.0], [0.0, 0.0, 1.0]]
    )
    left = torch.eye(4)[None]
    left[:, 0, 3] = -0.3
    right = torch.eye(4)[None]
    right[:, 0, 3] = 0.3
    holders = {
        "source_arm": HeldObjectState(semantics, left, initial @ left),
        "destination_arm": HeldObjectState(semantics, right, initial @ right),
    }
    if mode == "missing":
        holders.pop("destination_arm")
    context = replace(
        context,
        task=TaskState(1, "cpu", held_objects=holders),
        scene=SceneSnapshot(0.0, 0, {"object": EntityState(initial)}),
    )
    if mode == "partial":
        qpos = context.robot.qpos.repeat(2, 1)
        gen.robot.num_instances = 2
        gen.robot.get_qvel_limits.side_effect = lambda name=None, **kwargs: torch.full(
            (2, len(kwargs.get("joint_ids", gen.robot.get_joint_ids(name)))), 100.0
        )
        gen.robot.get_qpos.side_effect = lambda name=None: qpos[
            :, gen.robot.get_joint_ids(name)
        ].clone()
        holders = {
            key: HeldObjectState(
                value.semantics,
                value.object_to_eef.repeat(2, 1, 1),
                value.grasp_xpos.repeat(2, 1, 1),
                env_mask=torch.tensor([True, False]),
            )
            for key, value in holders.items()
        }
        context = replace(
            context,
            robot=RobotObservation(
                0.0,
                qpos,
                torch.zeros_like(qpos),
                root_pose=torch.eye(4).repeat(2, 1, 1),
            ),
            task=TaskState(2, "cpu", held_objects=holders),
            env_ids=torch.tensor([0, 1]),
            scene=SceneSnapshot(
                0.0, 0, {"object": EntityState(initial.repeat(2, 1, 1))}
            ),
        )
        target = target.repeat(2, 1, 1)
    return gen, context, CoordinatedPickGoal(semantics, target), grasp, grasps, ik
