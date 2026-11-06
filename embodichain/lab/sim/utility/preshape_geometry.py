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

"""Actual robot hand sweep adaptation for qualified prismatic grasp filtering."""

from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING
import math
import torch
from embodichain.toolkits.graspkit.pg_grasp.preshape import PreshapeHandClearance
from .grasp_geometry import RobotHandGeometryCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
__all__ = ["capture_preshape_hand_clearance"]


def capture_preshape_hand_clearance(
    robot: Robot,
    *,
    cfg: RobotHandGeometryCfg,
    expected_qpos: torch.Tensor,
    table_z: float,
    preshape_fraction: float = 0.85,
) -> PreshapeHandClearance:
    """Capture the source seven-state closing sweep using explicit hand calibration.

    Args:
        robot: Exact runtime robot owning complete model/joint geometry.
        cfg: Existing explicit motion/hand/TCP/pad and command calibration.
        expected_qpos: Actual B1 observation, never a projected command start.
        table_z: Measured table top for this declaration's support.
        preshape_fraction: Selected closure fraction in the qualified source band.

    Returns:
        Numeric whole-hand points, usable aperture and retreat reserve. This is
        input geometry for candidate rejection, not a native contact certificate.
    """
    cfg = deepcopy(cfg)
    cfg.validate_values()
    if (
        type(preshape_fraction) not in (int, float)
        or not math.isfinite(preshape_fraction)
        or not 0.75 <= preshape_fraction <= 0.85
        or not math.isfinite(table_z)
    ):
        raise ValueError(
            "Preshape capture requires source-band fraction and finite table height."
        )
    if (
        expected_qpos.ndim != 2
        or expected_qpos.shape[0] != 1
        or not torch.equal(robot.get_qpos(), expected_qpos)
    ):
        raise ValueError("Preshape capture requires the exact actual B1 observation.")
    motion_part, hand_part = cfg.motion_part, cfg.hand_part
    commands = {"open": cfg.open_qpos, "grasp": cfg.closed_qpos}
    joint_names = robot.cfg.control_parts[hand_part]
    chains = {link: robot.get_parent_joint_chain(link) for link in robot.link_names}
    master = next(
        joint
        for chain in chains.values()
        for joint in chain
        if joint.name == joint_names[0]
    )
    root = master.parent_link_name
    if root != cfg.hand_root_link:
        raise ValueError("Preshape hand root differs from its explicit calibration.")
    links = [
        link
        for link, chain in chains.items()
        if link == root or any(joint.parent_link_name == root for joint in chain)
    ]
    solver = robot.cfg.solver_cfg[motion_part]
    states = expected_qpos[0:1].repeat(7, 1)
    ids = [robot.joint_names.index(name) for name in joint_names]
    opened = states.new_tensor(commands["open"])
    closed = states.new_tensor(commands["grasp"])
    fraction = preshape_fraction
    fractions = torch.linspace(fraction, 1.0, 7, device=states.device)
    states[:, ids] = opened + fractions[:, None] * (closed - opened)
    fk = robot.compute_fk(
        qpos=states,
        link_names=links + [solver.end_link_name],
        qpos_joint_names=robot.joint_names,
    )
    tools = fk[:, -1] @ torch.as_tensor(solver.tcp, dtype=fk.dtype, device=fk.device)
    inverse_tools = torch.linalg.inv(tools)
    cloud, pads = [], []
    for index, link in enumerate(links):
        vertices, _ = robot.get_link_vert_face(link)
        if not vertices.numel():
            continue
        relative = inverse_tools @ fk[:, index]
        points = (
            vertices.to(relative)[None] @ relative[:, :3, :3].transpose(-1, -2)
            + relative[:, None, :3, 3]
        )
        cloud.append(points.reshape(-1, 3))
        if link in cfg.pad_link_names:
            pads.append(points)
    if len(pads) != 2:
        raise ValueError(
            "E6 clearance requires the calibrated Robotiq pair of finger pads."
        )
    negative, positive = sorted(pads, key=lambda points: float(points[0, :, 0].mean()))
    gap = positive[0, :, 0].amin() - negative[0, :, 0].amax()
    pad_end = torch.minimum(
        negative[:, :, 2].amax(dim=1), positive[:, :, 2].amax(dim=1)
    ).amin()
    if not torch.equal(robot.get_qpos(), expected_qpos):
        raise ValueError("Preshape robot observation changed during capture.")
    return PreshapeHandClearance(
        torch.cat(cloud), float(gap) - 0.004, float(pad_end) - 0.004, table_z
    )
