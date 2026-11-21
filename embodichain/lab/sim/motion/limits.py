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

"""Strict authored/runtime velocity limits for fixed-grid motion."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["authored_joint_velocity_limits"]


def authored_joint_velocity_limits(
    robot: Robot, control_part: str | None
) -> torch.Tensor:
    """Intersect runtime limits with every controlled joint's authored URDF cap.

    Args:
        robot: Robot exposing named control joints, runtime limits and its URDF.
        control_part: Controlled part, or all joints when ``None``.

    Returns:
        Per-environment limits in runtime dtype/device, shape ``(B, D)``.

    Raises:
        ValueError: If any controlled joint lacks a positive finite authored cap,
            or runtime limits do not match the named joints.
    """
    declared = {}
    for joint in ET.parse(robot.cfg.fpath).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "velocity" in limit.attrib:
            declared[joint.attrib["name"]] = float(limit.attrib["velocity"])
    names = [robot.joint_names[int(i)] for i in robot.get_joint_ids(name=control_part)]
    if any(
        name not in declared or not math.isfinite(declared[name]) or declared[name] <= 0
        for name in names
    ):
        raise ValueError(
            "Every controlled joint requires a finite positive URDF velocity limit."
        )
    runtime = robot.get_qvel_limits(name=control_part)
    if runtime.ndim != 2 or runtime.shape[1] != len(names):
        raise ValueError(
            "Runtime velocity limits must match the named control-part joints."
        )
    return torch.minimum(
        runtime, runtime.new_tensor([declared[name] for name in names])[None]
    )
