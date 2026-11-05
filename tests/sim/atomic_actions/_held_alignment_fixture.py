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

"""Controlled six-coordinate FK/IK, heading paths and robot collision oracle."""

from __future__ import annotations
from dataclasses import replace
import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from embodichain.lab.sim.atomic_actions import HeldObjectPoseGoal
from embodichain.lab.sim.motion.planners.utils import PlanResult
from ._handover_native_fixture import native_transfer_fixture


def alignment_fixture(directory, mode):
    gen, context, goal, grasp, grasps, ik, fk = native_transfer_fixture(
        directory, "clear"
    )
    robot = gen.robot
    robot.get_solver("source_arm").supports_continuous_batch_ik = False
    target = torch.eye(4)[None].repeat(1, 3, 1, 1)
    target[:, :, :3, 3] = torch.tensor(
        [[-0.3, 0.0, 0.5], [-0.1, 0.1, 0.5], [-0.1, 0.1, 0.4]]
    )
    held_goal = HeldObjectPoseGoal(
        target[:, -1] if mode == "lift_fallback" else target, world_yaw_free=True
    )
    calls, checks = [], []
    if mode == "raw_failed":
        original = robot.compute_ik.side_effect

        def failed_ik(*args, **kwargs):
            valid, positions = original(*args, **kwargs)
            return torch.zeros_like(valid), positions

        robot.compute_ik.side_effect = failed_ik

    def generate(targets, options):
        calls.append(
            {
                "targets": [t.xpos.tolist() for t in targets],
                "start": options.start_qpos.tolist(),
                "count": options.sample_count,
                "draws": [
                    float(torch.rand(())),
                    float(np.random.rand()),
                    random.random(),
                ],
            }
        )
        poses = torch.stack([t.xpos for t in targets], 1)
        rv = torch.tensor(
            Rotation.from_matrix(poses[0, :, :3, :3].double().numpy()).as_rotvec(),
            dtype=poses.dtype,
        )[None]
        q = torch.cat((poses[:, :, :3, 3], rv), -1)
        points = torch.cat((options.start_qpos[:, None], q), 1)
        visits = (
            torch.linspace(0, options.sample_count - 1, len(targets) + 1).round().long()
        )
        path = options.start_qpos.new_empty(1, options.sample_count, 6)
        for i in range(len(targets)):
            lo, hi = int(visits[i]), int(visits[i + 1])
            path[:, lo : hi + 1] = torch.lerp(
                points[:, i : i + 1],
                points[:, i + 1 : i + 2],
                torch.linspace(0, 1, hi - lo + 1)[None, :, None],
            )
        if mode in {"fallback", "lift_fallback", "initial_collision"} or (
            mode == "later" and len(calls) == 1
        ):
            path[:, options.sample_count // 2, 0] = 1.3
        dt = path.new_full(path.shape[:2], options.interpolation_dt)
        dt[:, 0] = 0.0
        return PlanResult(
            success=torch.tensor(
                [
                    mode not in {"raw_failed", "unsupported"}
                    and (mode != "denser" or options.sample_count >= 260)
                ]
            ),
            positions=path,
            dt=dt,
        )

    def check(positions):
        checks.append(positions.tolist())
        # Collision inspection may use libraries which draw randomness; the
        # owning scope must restore these reads without rewinding plan draws.
        torch.rand(())
        np.random.rand()
        random.random()
        if mode == "unavailable":
            raise ValueError("controlled robot geometry unavailable")
        bad = bool((positions[:, 0] > 1.0).any()) or (
            mode == "initial_collision" and bool((positions[:, 12] > 0.005).any())
        )
        return (
            {"valid": False, "frame": 0, "kind": "self", "pairs": [["arm", "torso"]]}
            if bad
            else {
                "valid": True,
                "frames": len(positions),
                "scope": "discrete_robot_meshes",
            }
        )

    return gen, context, held_goal, generate, check, calls, checks, ik
