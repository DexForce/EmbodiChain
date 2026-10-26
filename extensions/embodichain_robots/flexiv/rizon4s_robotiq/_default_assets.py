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
"""Compose the registered Rizon4S and 2F85 assets without mesh conversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from embodichain.data import get_data_path
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.lab.sim.cfg import URDFCfg

__all__: list[str] = []


def prepare_default_assets() -> str:
    """Preserve source visuals, collisions, inertials, names and mimic relations.

    The default gripper mounts directly on the arm's flange. The engineering
    TCP is 145 mm along its base z axis, with a 90-degree yaw so its x-opening
    direction is the task's TCP y direction. This is not hardware calibration.
    """
    arm = Path(get_data_path("FlexivRizon4S/rizon4s.urdf"))
    hand = Path(get_data_path("Robotiq2F85/Robotiq2F85.urdf"))
    identity = {
        "converter_version": 1,
        "arm_sha256": hashlib.sha256(arm.read_bytes()).hexdigest(),
        "hand_sha256": hashlib.sha256(hand.read_bytes()).hexdigest(),
        "mount": "identity at flange",
        "tcp_xyz": [0, 0, 0.145],
        "tcp_rpy": [0, 0, 1.5707963267948966],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = (
        Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
        / "assembled"
        / f"rizon4s_2f85_default_{key}"
    )
    path = directory / f"{directory.name}.urdf"
    if path.is_file() and (directory / "fidelity.json").is_file():
        return str(path)
    cfg = URDFCfg(
        components=[
            {"component_type": "arm", "urdf_path": str(arm)},
            {"component_type": "hand", "urdf_path": str(hand)},
        ],
        base_link_name="assembled_base",
        fname=directory.name,
        fpath_prefix=str(directory.parent),
        use_signature_check=False,
    )
    cfg.assemble_urdf()
    tree = ET.parse(path)
    robot = tree.getroot()
    ET.SubElement(robot, "link", name="gripper_tcp")
    joint = ET.SubElement(robot, "joint", name="gripper_tcp_fixed", type="fixed")
    ET.SubElement(joint, "parent", link="robotiq_85_base_link")
    ET.SubElement(joint, "child", link="gripper_tcp")
    ET.SubElement(joint, "origin", xyz="0 0 0.145", rpy="0 0 1.5707963267948966")
    ET.indent(robot)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    (directory / "fidelity.json").write_text(
        json.dumps(
            {
                **identity,
                "asset_source": "default",
                "arm_asset": "FlexivRizon4S/rizon4s.urdf",
                "gripper_asset": "Robotiq2F85/Robotiq2F85.urdf",
                "source_visuals_and_materials_preserved": True,
                "source_collisions_and_inertials_preserved": True,
                "source_mimic_tree_preserved": True,
                "exact_source_dynamics": False,
                "paper_dynamics_identified": False,
                "qualification": "Default registered URDF composition; mimic tree is not a calibrated closed-loop gripper model or a validated policy transfer.",
            },
            indent=2,
        )
        + "\n"
    )
    return str(path)
