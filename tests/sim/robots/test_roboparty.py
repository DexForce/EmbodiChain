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

"""Verify RoboParty catalog defaults against the pinned upstream models."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from embodichain.robot_registry import resolve_robot_cfg
from embodichain.extensions.robots._catalog import ROBOT_CONFIGS

pytestmark = pytest.mark.no_sim


@pytest.mark.slow
@pytest.mark.parametrize(
    "robot_id,variant,root_link,independent_dof,mimic_count",
    [
        ("RoboPartyRPO", None, "base_link", 23, 0),
        ("RoboPartyRP1", None, "base_link", 28, 0),
        ("RoboPartyRP1", "24dof", "base_link", 24, 0),
        ("RoboPartyRP1", "hand", "base_link", 40, 10),
        ("RoboPartyRP1", "hand_fixed", "base_link", 28, 0),
        ("RoboPartyRPHand", "left", "lh_base_link", 6, 5),
        ("RoboPartyRPHand", "right", "rh_base_link", 6, 5),
    ],
)
def test_pinned_description_matches_robot_preset(
    robot_id: str,
    variant: str | None,
    root_link: str,
    independent_dof: int,
    mimic_count: int,
) -> None:
    """Check the public path through the real asset cache and config loader."""
    cfg = resolve_robot_cfg(robot_id, variant=variant)
    cfg.validate()
    urdf = Path(cfg.fpath)
    source = ET.parse(urdf)
    joints = source.findall("joint")
    moving = [joint for joint in joints if joint.attrib["type"] != "fixed"]
    mimics = {
        joint.attrib["name"]: mimic
        for joint in moving
        if (mimic := joint.find("mimic")) is not None
    }
    independent = {joint.attrib["name"] for joint in moving} - mimics.keys()
    controlled = {joint for joints in cfg.control_parts.values() for joint in joints}
    assert controlled == independent
    assert len(independent) == independent_dof
    assert len(mimics) == mimic_count

    spec = ROBOT_CONFIGS[robot_id]
    if variant is not None:
        spec = spec["variants"][variant]
    assert spec["dof"] == independent_dof
    assert spec["control_parts"] == {
        part: len(names) for part, names in cfg.control_parts.items()
    }
    assert spec.get("mimic_joints", {}) == {
        joint: mimic.attrib["joint"] for joint, mimic in mimics.items()
    }

    children = {joint.find("child").attrib["link"] for joint in joints}
    assert {link.attrib["name"] for link in source.findall("link")} - children == {
        root_link
    }
    for mesh in source.findall(".//mesh"):
        filename = (urdf.parent / mesh.attrib["filename"]).resolve()
        assert filename.is_relative_to(urdf.parent)
        assert filename.is_file()
    assert cfg.root_props.fixed_base == (robot_id == "RoboPartyRPHand")
    assert not cfg.build_pk_chain
    assert cfg.build_pk_serial_chain() == {}

    assert len(cfg.init_qpos) == len(moving)
    qpos = {joint.attrib["name"]: value for joint, value in zip(moving, cfg.init_qpos)}
    for joint in moving:
        limit = joint.find("limit")
        assert (
            float(limit.attrib["lower"])
            <= qpos[joint.attrib["name"]]
            <= float(limit.attrib["upper"])
        )
    for joint, mimic in mimics.items():
        assert qpos[joint] == pytest.approx(
            float(mimic.attrib["multiplier"]) * qpos[mimic.attrib["joint"]]
            + float(mimic.attrib["offset"])
        )
    body_joint_names = cfg.control_parts.get("locomotion", ())
    for joint in moving:
        name = joint.attrib["name"]
        if name in body_joint_names:
            limit = joint.find("limit")
            assert cfg.joint_drive_props.max_effort[name] == float(
                limit.attrib["effort"]
            )
            assert cfg.joint_drive_props.max_velocity[name] == float(
                limit.attrib["velocity"]
            )

    # A serialized preset preserves the selected body/side and works without
    # re-resolving the default source.
    override = cfg.to_dict()
    assert type(cfg).from_dict(override).to_dict() == override


def test_local_urdf_limits_survive_a_partial_drive_override(tmp_path: Path) -> None:
    """Apply a regex override while retaining unaffected source limits."""
    urdf = tmp_path / "rpo.urdf"
    urdf.write_text(
        """<robot name="local_rpo">
  <link name="base"/>
  <link name="left"/>
  <link name="right"/>
  <joint name="left_joint" type="revolute">
    <parent link="base"/><child link="left"/>
    <limit lower="-1" upper="1" effort="60" velocity="10"/>
  </joint>
  <joint name="right_joint" type="revolute">
    <parent link="base"/><child link="right"/>
    <limit lower="-1" upper="1" effort="70" velocity="12"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )
    cfg = resolve_robot_cfg(
        "RoboPartyRPO",
        {"fpath": str(urdf), "joint_drive_props": {"max_velocity": {"left_.*": 5.0}}},
    )
    assert cfg.joint_drive_props.max_velocity == {"left_.*": 5.0, "right_joint": 12.0}
    assert cfg.joint_drive_props.max_effort == {"left_joint": 60.0, "right_joint": 70.0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-m", "slow or not slow"]))
