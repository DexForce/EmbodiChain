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

"""Actual-hand clearance for the GenSim dual-Franka/Robotiq E6 deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__: list[str] = []

PRESHAPE_FRACTION = 0.85
RELEASE_RETREAT_DISTANCE = 0.04
ARM_STIFFNESS = 50000.0
PASSIVE_FRICTION = 0.01


@dataclass(frozen=True, slots=True)
class HandClearance:
    """Sampled closing sweep in TCP coordinates, not a collision-free certificate."""

    points: torch.Tensor
    aperture: float
    max_retreat: float
    table_z: float

    def filter(
        self, poses: torch.Tensor, widths: torch.Tensor, costs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep sampler ranking; reject aperture/table violations without fallback."""
        result = poses.clone()
        if self.max_retreat < 0.020:
            return result, widths, torch.full_like(costs, torch.inf)
        accepted = torch.zeros_like(costs, dtype=torch.bool)
        finite = torch.isfinite(costs) & torch.isfinite(widths)
        finite &= torch.isfinite(poses).all(dim=-1).all(dim=-1)
        finite &= (widths >= 0.0) & (widths <= self.aperture)
        points = self.points.to(poses)
        # Reserve pad depth for contact rather than moving arbitrarily far out.
        # Seven mm includes the five-mm geometric margin and tracking allowance.
        for retreat in torch.arange(0.020, self.max_retreat + 1.0e-7, 0.005):
            shifted = poses.clone()
            shifted[:, :3, 3] -= float(retreat) * poses[:, :3, 2]
            min_z = (points @ shifted[:, 2, :3].T).amin(dim=0) + shifted[:, 2, 3]
            clear = finite & ~accepted & (min_z >= self.table_z + 0.007)
            result[clear] = shifted[clear]
            accepted |= clear
        return result, widths, torch.where(accepted, costs, torch.inf)


def build_hand_clearance(
    robot: Any,
    *,
    motion_part: str,
    hand_part: str,
    commands: Any,
    table_z: float,
) -> HandClearance:
    """Measure the whole hand subtree and Robotiq pads through public FK APIs."""
    joint_names = robot.cfg.control_parts[hand_part]
    chains = {link: robot.get_parent_joint_chain(link) for link in robot.link_names}
    master = next(
        joint
        for chain in chains.values()
        for joint in chain
        if joint.name == joint_names[0]
    )
    root = master.parent_link_name
    links = [
        link
        for link, chain in chains.items()
        if link == root or any(joint.parent_link_name == root for joint in chain)
    ]
    solver = robot.cfg.solver_cfg[motion_part]
    states = robot.get_qpos()[0:1].repeat(7, 1)
    ids = [robot.joint_names.index(name) for name in joint_names]
    opened = states.new_tensor(commands["open"])
    closed = states.new_tensor(commands["grasp"])
    fractions = torch.linspace(PRESHAPE_FRACTION, 1.0, 7, device=states.device)
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
        if link.endswith("finger_pad"):
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
    return HandClearance(
        torch.cat(cloud), float(gap) - 0.004, float(pad_end) - 0.004, table_z
    )


def profile_clearances(
    registration: Any, robot: Any, table: Any
) -> dict[str, HandClearance]:
    """Resolve arm/hand pairs from the profile rather than inferring their names."""
    pose = table.get_local_pose(to_matrix=True)[0]
    vertices = table.get_vertices(scale=True)[0]
    table_z = float((vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2].max())
    profile = registration.robot_profile_binding
    commands = {preset.preset_id: preset for preset in profile.command_presets}
    result = {}
    for resource in profile.resources:
        endpoints = {endpoint.endpoint_id: endpoint for endpoint in resource.endpoints}
        if "motion" not in endpoints or "grasp" not in endpoints:
            continue
        hand, motion = endpoints["grasp"], endpoints["motion"]
        result[hand.control_part] = build_hand_clearance(
            robot,
            motion_part=motion.control_part,
            hand_part=hand.control_part,
            commands=commands[hand.command_preset].commands,
            table_z=table_z,
        )
    return result
