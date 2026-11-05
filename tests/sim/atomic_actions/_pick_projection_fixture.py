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

"""Small candidate/IK oracles shared by frozen and direct Pick projection tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from embodichain.lab.sim.atomic_actions import TaskState
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.toolkits.graspkit import (
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)
from ._pour_fixture import pour_fixture


class ProjectionCandidates(ParallelJawGraspPoseGenerator):
    def __init__(self):
        super().__init__(ParallelJawGripperModelCfg(model_id="projection_fixture"))

    def get_valid_grasp_poses(self, *, obj_poses, **kwargs):
        relative = torch.eye(4).repeat(2, 1, 1)
        relative[1, :3, :3] = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
        return [(pose[None] @ relative, torch.tensor([1.0, 0.0])) for pose in obj_poses]

    def get_best_grasp_poses(self, **kwargs):
        raise AssertionError("Candidate projection must retain the sampled pool.")

    def get_dual_arm_valid_grasp_poses(self, **kwargs):
        raise AssertionError("Single-hand projection cannot request a paired grasp.")


def pick_fixture(directory: Path):
    robot, context, vertices, receiving, returned = pour_fixture(directory)
    semantics = context.get_held_object("left_arm").semantics
    context = replace(context, task=TaskState(1, "cpu"))
    ik_calls, motion_calls = [], []

    def solve(*, pose, joint_seed, name):
        result = joint_seed + torch.rand(()) * 0.001
        ik_calls.append(
            {
                "targets": pose.tolist(),
                "seed": joint_seed.tolist(),
                "solution": result.tolist(),
            }
        )
        return torch.ones(pose.shape[:2], dtype=torch.bool), result

    robot.compute_batch_ik.side_effect = solve
    tool = robot.compute_fk.return_value
    robot.compute_batch_fk.side_effect = (
        lambda qpos, **kw: tool[:, None]
        .expand(qpos.shape[0], qpos.shape[1], 4, 4)
        .clone()
    )

    def generate(targets, *, options):
        count = options.sample_count
        motion_calls.append(
            {
                "ik_before_motion": len(ik_calls),
                "targets": [state.xpos.tolist() for state in targets],
                "sample_count": count,
            }
        )
        positions = torch.linspace(0.0, 0.01, count).reshape(1, -1, 1).repeat(1, 1, 2)
        dt = torch.full((1, count), 0.04)
        dt[:, 0] = 0
        return PlanResult(
            success=torch.tensor([True]),
            positions=positions,
            dt=dt,
            waypoint_indices=torch.tensor([[2, 4, count - 1]]),
        )

    generator = object.__new__(MotionGenerator)
    generator.robot, generator.device = robot, robot.device
    generator.planner = NS(cfg=NS(planner_type="oracle"), collision_world_info=None)
    generator._generate_once = generate
    return (
        generator,
        context,
        semantics,
        vertices,
        receiving,
        returned,
        ik_calls,
        motion_calls,
    )
