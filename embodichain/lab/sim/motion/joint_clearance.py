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

"""Bounded empty-arm joint offset selection from declared model geometry."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from .mesh_collision import FreeMotionQuery, MotionCollisionReport

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
__all__ = ["select_clearance_joint_offset"]


def select_clearance_joint_offset(
    robot: Robot,
    observed_qpos: torch.Tensor,
    *,
    active_control_part: str,
    joint_name: str,
    maximum_offset: float,
    protected_link_name: str,
    query: FreeMotionQuery,
) -> float:
    """Rank the original six offsets by FK separation then screen 101 samples.

    Args:
        robot: Borrowed exact robot with assembled model and active solver.
        observed_qpos: Full measured B1 joint positions; no live state is read.
        active_control_part: Explicit bound motion control part.
        joint_name: Exact model joint whose offset is varied.
        maximum_offset: Source requested offset magnitude in radians.
        protected_link_name: Explicit protected model link from the declared resource.
        query: Shared robot/full-scene free-motion query.

    Returns:
        First collision-qualified offset in stable descending FK separation order.
        No IK, execution or physical effect certificate is produced.
    """
    import yourdfpy

    if observed_qpos.ndim != 2 or observed_qpos.shape[0] != 1:
        raise ValueError("Idle offset selection requires B1 joint observations.")
    model = yourdfpy.URDF.load(robot.cfg.fpath, load_meshes=False)
    start = observed_qpos[0].detach().cpu()
    joint = robot.joint_names.index(joint_name)
    active = robot.get_solver(active_control_part).end_link_name
    limit = model.joint_map[joint_name].limit
    proposals = []
    for factor in (-1.0, -2 / 3, -1 / 3, 1 / 3, 2 / 3, 1.0):
        offset = abs(maximum_offset) * factor
        goal = start.clone()
        goal[joint] += offset
        if not limit.lower <= float(goal[joint]) <= limit.upper:
            continue
        model.update_cfg(
            {
                name: float(value)
                for name, value in zip(robot.joint_names, goal)
                if name in model.joint_map
            }
        )
        a, b = (
            model.get_transform(active)[:3, 3],
            model.get_transform(protected_link_name)[:3, 3],
        )
        separation = float(((a - b) ** 2).sum())
        proposals.append((separation, offset, goal))
    for _, offset, goal in sorted(proposals, key=lambda item: item[0], reverse=True):
        sweep = start[None] + torch.linspace(0, 1, 101)[:, None] * (goal - start)[None]
        report = query.inspect(sweep, control_part=active_control_part)
        if (
            not isinstance(report, MotionCollisionReport)
            or report.scope != "discrete_free_motion_meshes"
        ):
            raise ValueError(
                "Idle search requires complete robot/scene collision scope."
            )
        if report.valid and report.frames == len(sweep):
            return offset
    raise ValueError("No bounded empty-arm clearance proposal passed swept geometry.")
