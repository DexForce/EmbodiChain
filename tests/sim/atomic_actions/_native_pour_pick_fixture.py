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

"""Native Pour Pick ports with original observed pad line and return recipe."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import trimesh
from embodichain.lab.sim.atomic_actions import EntityState
from ._ordinary_pick_fixture import ordinary_fixture, native_end_candidates


def native_pour_fixture(directory: Path, mode: str):
    gen, context, sem, calls, ik = ordinary_fixture(directory)
    robot = gen.robot
    if mode == "audit_limit":
        initial = context.robot.qpos.clone()
        initial[:, :2] = torch.tensor([0.2, -0.2])
        context = replace(context, robot=replace(context.robot, qpos=initial))
        robot.get_qpos.return_value = initial
    get_ids = robot.get_joint_ids.side_effect
    robot.get_joint_ids.side_effect = lambda name=None: get_ids(
        "arm" if name == "left_arm" else name
    )
    robot.control_parts = {"left_arm": object(), "hand": object()}
    robot.cfg.control_parts = {"left_arm": ["x", "y", "z"], "hand": ["h0", "h1"]}
    robot.cfg.solver_cfg = {
        "left_arm": NS(end_link_name="arm_tip", tcp=torch.eye(4).numpy())
    }
    names = ("left_inner_finger_pad", "left_right_inner_finger_pad")
    robot.link_names = ["base", "x_link", "y_link", "arm_tip", "hand_root", *names]
    text = ['<robot name="native_pour">'] + [
        f'<link name="{name}"/>' for name in robot.link_names
    ]
    for joint, parent, child, axis in (
        ("x", "base", "x_link", "1 0 0"),
        ("y", "x_link", "y_link", "0 1 0"),
        ("z", "y_link", "arm_tip", "0 0 1"),
        ("h0", "hand_root", names[0], "1 0 0"),
        ("h1", "hand_root", names[1], "-1 0 0"),
    ):
        text.append(
            f'<joint name="{joint}" type="prismatic"><parent link="{parent}"/><child link="{child}"/><axis xyz="{axis}"/><limit velocity="1" lower="-2" upper="2" effort="10"/></joint>'
        )
    text.append(
        '<joint name="hand_mount" type="fixed"><parent link="arm_tip"/><child link="hand_root"/></joint>'
    )
    text.append("</robot>")
    Path(robot.cfg.fpath).write_text("".join(text))

    def link(name, **kwargs):
        pose = robot.compute_fk(
            context.robot.qpos[:, :3], name="left_arm", to_matrix=True
        )
        local = torch.tensor(
            [0.0, 0.0, 0.02] if mode == "degenerate" else [0.02, 0.0, 0.0]
        )
        pose[:, :3, 3] += (pose[:, :3, :3] @ local[:, None]).squeeze(-1) * (
            1 if name == names[1] else -1
        )
        return pose

    robot.get_link_pose.side_effect = link
    receiver = torch.eye(4)[None]
    receiver[:, :3, 3] = torch.tensor([0.5, 0.0, 0.15])
    context = replace(
        context,
        scene=replace(
            context.scene,
            entities={**context.scene.entities, "receiver": EntityState(receiver)},
        ),
    )
    receiving = torch.tensor(
        trimesh.creation.box((0.3, 0.3, 0.1)).vertices, dtype=torch.float32
    )
    returned = context.scene.entities["object"].pose[0].clone()
    original = robot.compute_batch_ik.side_effect

    def solve(*args, **kwargs):
        valid, positions = original(*args, **kwargs)
        if mode in {"downstream_fail", "audit_limit"} and len(ik) % 8 == 0:
            valid = torch.zeros_like(valid)
        return valid, positions

    robot.compute_batch_ik.side_effect = solve
    if mode == "path_fail":
        original_generate = gen._generate_once

        def failed(targets, *, options):
            result = original_generate(targets, options=options)
            return replace(result, success=torch.zeros_like(result.success))

        gen._generate_once = failed
    return gen, context, sem, receiving, returned, native_end_candidates(), calls, ik
