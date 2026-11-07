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

"""Explicit URDF/open-state geometry for the mounted example grippers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np

from embodichain.data import get_data_path
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
from embodichain.toolkits.graspkit.pg_grasp import FixedGripperCollisionGeometry
from embodichain.toolkits.graspkit.pg_grasp.model_geometry import (
    get_parallel_jaw_collision_geometry,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RobotCfg


def geometry_from_asset(
    asset: str,
    *,
    root_from_tcp: np.ndarray,
    root_from_grasp: np.ndarray,
    joint_positions: dict[str, float],
    root_link: str | None = None,
) -> FixedGripperCollisionGeometry:
    """Use every collision link in an explicitly opened gripper subtree.

    Both transforms are expressed in the source gripper root. The canonical
    contact frame has X along jaw opening and Z along approach. ``root_link``
    is required for a hand embedded in a full robot asset.
    """
    path = get_data_path(asset)
    document = ET.parse(path).getroot()
    links = {node.attrib["name"]: node for node in document.findall("link")}
    joints = document.findall("joint")
    if root_link is None:
        children = {joint.find("child").attrib["link"] for joint in joints}
        roots = links.keys() - children
        if len(roots) != 1:
            raise ValueError(f"Expected one source gripper root in {path}.")
        root_link = next(iter(roots))
    descendants = {root_link}
    while True:
        extended = descendants | {
            joint.find("child").attrib["link"]
            for joint in joints
            if joint.find("parent").attrib["link"] in descendants
        }
        if extended == descendants:
            break
        descendants = extended
    collision_links = tuple(
        name
        for name, link in links.items()
        if name in descendants and link.find("collision") is not None
    )
    return FixedGripperCollisionGeometry.from_urdf(
        path,
        root_link=root_link,
        tcp_link=root_link,
        collision_links=collision_links,
        joint_positions=joint_positions,
        link_from_tcp=root_from_tcp,
        grasp_from_tcp=np.linalg.inv(root_from_grasp) @ root_from_tcp,
        point_spacing=0.012,
    )


def mounted_gripper_geometry(
    robot_cfg: RobotCfg, *, arm_control_part: str = "arm"
) -> tuple[ParallelJawGripperModelCfg, FixedGripperCollisionGeometry]:
    """Bind the known mounted PGI, PGC or Robotiq fixture to its actual TCP.

    Unknown hardware must supply its own explicit geometry. The identity is
    read from the mounted URDF, never guessed from the robot or joint count.
    """
    component_name = (
        "hand"
        if arm_control_part == "arm"
        else arm_control_part.removesuffix("_arm") + "_hand"
    )
    component = robot_cfg.urdf_cfg.components[component_name]
    asset = component["urdf_path"]
    root_from_tcp = np.linalg.inv(np.asarray(component["transform"])) @ np.asarray(
        robot_cfg.solver_cfg[arm_control_part].tcp
    )
    name = Path(asset).stem.lower()
    if name == "dh_pgi_140_80":
        return (
            ParallelJawGripperModelCfg(
                model_id="dh_pgi_140_80",
                min_opening_width=0.003,
                max_opening_width=0.1,
            ),
            get_parallel_jaw_collision_geometry(
                "dh_pgi_140_80", root_from_tcp=root_from_tcp
            ),
        )
    root_from_grasp = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    if name == "robotiq_arg2f_140":
        model = ParallelJawGripperModelCfg(
            model_id="robotiq_2f_140",
            min_opening_width=0.005,
            max_opening_width=0.14,
        )
        root_from_grasp[2, 3] = 0.21
    elif name == "dh_pgc_140_50":
        model = ParallelJawGripperModelCfg(
            model_id=name,
            min_opening_width=0.002,
            max_opening_width=0.05,
        )
        root_from_grasp[2, 3] = 0.12
    elif name == "dh_pgc_140_50_m":
        # The modified long-finger fixture declares its contact TCP explicitly
        # in each caller. Preserve those task calibrations and aperture bounds;
        # collision surfaces still come only from this distinct source asset.
        model = ParallelJawGripperModelCfg(
            model_id=name,
            min_opening_width=0.003,
            max_opening_width=0.088,
        )
        root_from_grasp = root_from_tcp.copy()
    else:
        raise ValueError(f"No declared open-state geometry for mounted URDF {asset!r}.")
    # These two example fixtures explicitly command zero to open every active
    # hand joint; dependent mimic values are resolved by the URDF loader.
    document = ET.parse(get_data_path(asset)).getroot()
    opened = {
        joint.attrib["name"]: 0.0
        for joint in document.findall("joint")
        if joint.attrib["type"] != "fixed" and joint.find("mimic") is None
    }
    return model, geometry_from_asset(
        asset,
        root_from_tcp=root_from_tcp,
        root_from_grasp=root_from_grasp,
        joint_positions=opened,
    )


__all__ = ["geometry_from_asset", "mounted_gripper_geometry"]
