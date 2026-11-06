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

"""Controlled staging ports; native URDF capture is independently source-tested."""

from __future__ import annotations
from dataclasses import replace
import numpy as np
import torch
from ._held_alignment_fixture import alignment_fixture


def staging_fixture(directory, mode):
    gen, context, goal, original_generate, check, calls, checks, ik = alignment_fixture(
        directory, "raw_failed"
    )
    goal = replace(goal, object_target_pose=goal.object_target_pose[:, [0, 2]])
    robot = gen.robot

    def generate(targets, options):
        result = original_generate(targets, options)
        path = result.positions.clone()
        bad = mode != "original_safe" and (
            float(targets[0].xpos[0, 0, 3]) >= -0.300001
            or mode == "all_reject"
            or (mode == "second" and float(targets[0].xpos[0, 1, 3]) > -1e-7)
        )
        if bad:
            path[:, options.sample_count // 2, 0] = 1.3
        return replace(result, success=torch.tensor([True]), positions=path)

    tcp = {}
    for part in ("source_arm", "destination_arm"):
        tcp[part] = robot.compute_fk(
            context.robot.qpos[:, robot.get_joint_ids(part)], name=part, to_matrix=True
        )[0].numpy()
    roots = {part: pose.copy() for part, pose in tcp.items()}
    roots["destination_arm"][1, 3] = 0.3
    points = {
        part: pose[None, :3, 3] + np.array([[-0.1, -0.1, -0.04], [0.1, 0.1, 0.04]])
        for part, pose in tcp.items()
    }
    snapshot = dict(
        active="source_arm",
        passive="destination_arm",
        tcp=tcp,
        roots=roots,
        points=points,
        hand_owner_fk_error={part: 0.0 for part in tcp},
    )
    observed = context.robot.qpos.clone()
    velocity = context.robot.qvel.clone()
    desired = observed.clone()
    desired[:, 12] = 0.0
    desired_velocity = torch.zeros_like(desired)
    if mode == "target_changed":
        desired[:, 12] = 0.02
    if mode == "velocity":
        velocity[:, 12] = 0.2
    context = replace(context, robot=replace(context.robot, qvel=velocity))
    reads = []
    target_reads = 0

    def qpos(name=None, **kwargs):
        nonlocal target_reads
        target = kwargs.get("target", False)
        if "target" in kwargs:
            reads.append(["position", target])
        if target:
            target_reads += 1
            if mode == "during_check" and target_reads >= 2:
                desired[:, 12] = 0.02
        value = desired if target else observed
        return value[:, robot.get_joint_ids(name)].clone()

    def qvel(name=None, **kwargs):
        target = kwargs.get("target", False)
        if "target" in kwargs:
            reads.append(["velocity", target])
        value = desired_velocity if target else velocity
        return value[:, robot.get_joint_ids(name)].clone()

    robot.get_qpos.side_effect = qpos
    robot.get_qvel.side_effect = qvel
    if mode in {
        "retained",
        "target_changed",
        "during_check",
        "velocity",
        "authority_missing",
    }:

        def limits(name=None, **kwargs):
            ids = kwargs.get("joint_ids", robot.get_joint_ids(name))
            result = torch.full((1, len(ids)), 100.0)
            if 12 in ids:
                result[:, list(ids).index(12)] = 0.1
            return result

        robot.get_qvel_limits.side_effect = limits
    return gen, context, goal, generate, check, calls, checks, ik, snapshot, reads
