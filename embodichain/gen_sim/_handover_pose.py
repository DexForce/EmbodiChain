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

"""Derive a default hand-over delivery pose from the two bound arm roots."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

__all__: list[str] = []


def _arm_root_positions(
    simulation: dict[str, Any], profile: dict[str, Any], resources: Sequence[str]
) -> np.ndarray:
    """Return the world positions of the arm root links of ``resources``.

    Each resource's motion endpoint names an arm control part; that arm's
    solver names its root link. The link is located in the robot's initial
    world pose through fixed joints only, so the result does not depend on a
    joint configuration.

    Args:
        simulation: Resolved robot declaration of the embodiment.
        profile: Embodiment skill profile declaring the resources.
        resources: Resource IDs of the arms.

    Returns:
        One world position per resource, shape ``(len(resources), 3)``.

    Raises:
        ValueError: A resource has no arm solver root link, or that link is
            not rigidly attached to the robot base.
    """
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.toolkits.graspkit.pg_grasp._urdf_geometry import (
        _origin_transform,
    )

    control_parts = {
        resource["resource_id"]: endpoint["control_part"]
        for resource in profile["resources"]
        for endpoint in resource.get("endpoints", [])
        if endpoint["endpoint_id"] == "motion"
    }
    declaration = deepcopy(simulation)
    robot_type = declaration.pop("class_type", None) or declaration.pop(
        "robot_type", None
    )
    cfg = (
        RobotCfg.from_dict(declaration)
        if robot_type is None
        else resolve_robot_cfg(robot_type, declaration)
    )
    path = cfg.fpath or deepcopy(cfg.urdf_cfg).assemble_urdf()
    document = ET.parse(Path(path)).getroot()
    children = {joint.find("child").get("link") for joint in document.findall("joint")}
    bases = [
        link.get("name")
        for link in document.findall("link")
        if link.get("name") not in children
    ]
    if len(bases) != 1:
        raise ValueError("Arm root positions require a robot with one base link")
    # Breadth-first over fixed joints: base-frame transform of every link that
    # is rigidly attached to the base.
    edges: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in document.findall("joint"):
        if joint.get("type") == "fixed":
            edges.setdefault(joint.find("parent").get("link"), []).append(
                (
                    joint.find("child").get("link"),
                    _origin_transform(joint.find("origin")),
                )
            )
    fixed = {bases[0]: np.eye(4)}
    pending = deque(fixed)
    while pending:
        parent = pending.popleft()
        for child, transform in edges.get(parent, ()):
            fixed[child] = fixed[parent] @ transform
            pending.append(child)
    if cfg.init_local_pose is None:
        # ObjectBaseCfg reads ``init_rot`` as extrinsic x-y-z degrees.
        world = np.eye(4)
        world[:3, :3] = Rotation.from_euler(
            "xyz", cfg.init_rot, degrees=True
        ).as_matrix()
        world[:3, 3] = cfg.init_pos
    else:
        world = np.asarray(cfg.init_local_pose, dtype=float)
    positions = []
    for resource in resources:
        solver = (cfg.solver_cfg or {}).get(control_parts.get(resource))
        root = getattr(solver, "root_link_name", None)
        if root not in fixed:
            raise ValueError(
                f"Resource {resource!r} has no arm root link rigidly attached to "
                "the robot base; configure its hand-over pose provider instead"
            )
        positions.append((world @ fixed[root])[:3, 3])
    return np.asarray(positions)


def _arm_root_midpoint_provider(
    simulation: dict[str, Any], profile: dict[str, Any], resources: Sequence[str]
) -> dict[str, Any]:
    """Return a configured hand-over pose provider at the arm-root midpoint.

    GenSim hand-overs leave the receiving arm holding the object, so HandOver
    plans the exchange itself: it moves the lifted object to the midpoint of
    the two arm roots along their separation and takes from this pose only a
    lower bound on the exchange height. At the arm roots' own height, an object
    lifted above them is exchanged at its lifted height.

    Args:
        simulation: Resolved robot declaration of the embodiment.
        profile: Embodiment skill profile declaring the resources.
        resources: Resource IDs of the two arms that exchange the object.

    Returns:
        A ``handover_pose_providers`` entry.

    Raises:
        ValueError: The hand-over does not involve exactly two arms, or their
            root links cannot be located.
    """
    if len(resources) != 2:
        raise ValueError(
            "An arm-root hand-over pose requires exactly two arms, got "
            f"{sorted(resources)}"
        )
    midpoint = _arm_root_positions(simulation, profile, resources).mean(axis=0)
    return {
        "kind": "configured_pose",
        # Adding 0.0 turns a rounded negative zero into zero.
        "final_position": [round(float(value), 6) + 0.0 for value in midpoint],
        "final_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
