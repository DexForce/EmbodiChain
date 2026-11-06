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

"""One declared fixed Pick with deterministic IK and collision facts."""

from __future__ import annotations
from dataclasses import replace
import torch
from ._ordinary_pick_fixture import ordinary_fixture


def fixed_fixture(directory, mode):
    gen, context, sem, calls, ik = ordinary_fixture(directory)
    gen.robot.get_solver("arm").supports_continuous_batch_ik = False
    if mode == "ik_failed":
        batch = gen.robot.compute_batch_ik.side_effect
        single = gen.robot.compute_ik.side_effect

        def fail_batch(*a, **kw):
            valid, qpos = batch(*a, **kw)
            return torch.zeros_like(valid), qpos

        def fail_single(*a, **kw):
            valid, qpos = single(*a, **kw)
            return torch.zeros_like(valid), qpos

        gen.robot.compute_batch_ik.side_effect = fail_batch
        gen.robot.compute_ik.side_effect = fail_single
    fixed = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    fixed[2, 3] = 0.01
    if mode == "alignment":
        fixed[:3, :3] = torch.eye(3)
    robot_checks = []
    scene_checks = []

    def robot_query(robot, positions, part):
        result = {
            "valid": True,
            "frames": len(positions),
            "scope": "discrete_robot_meshes",
        }
        robot_checks.append(result)
        return result

    def scene_query(robot, simulation, positions, part):
        result = (
            {
                "valid": True,
                "frames": len(positions),
                "scope": "discrete_free_motion_meshes",
            }
            if mode != "scene_failed"
            else {
                "valid": False,
                "frame": 0,
                "kind": "scene",
                "pairs": [["hand", "obstacle"]],
            }
        )
        scene_checks.append(result)
        return result

    return (
        gen,
        context,
        sem,
        fixed,
        calls,
        ik,
        robot_query,
        scene_query,
        robot_checks,
        scene_checks,
    )
