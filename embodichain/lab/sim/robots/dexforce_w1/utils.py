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
import xml.etree.ElementTree as ET

import numpy as np

from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1Type,
    DexforceW1ArmSide,
    DexforceW1Version,
    DexforceW1HandBrand,
    DexforceW1HandVersion,
    parse_w1_arm_side,
    parse_w1_hand_version,
    parse_w1_version,
)
from embodichain.lab.sim.robots.dexforce_w1.hand_specs import (
    get_default_w1_hand_version,
    get_w1_hand_spec,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import URDFCfg
from embodichain.lab.sim.robots.dexforce_w1.specs import (
    get_w1_version_spec,
)

__all__ = [
    "ChassisManager",
    "TorsoManager",
    "HeadManager",
    "ArmManager",
    "HandManager",
    "EyesManager",
    "build_dexforce_w1_assembly_urdf_cfg",
]


class ChassisManager:
    def get_urdf(self, version=DexforceW1Version.V021):
        return get_data_path(
            get_w1_version_spec(version).component_urdf(DexforceW1Type.CHASSIS)
        )

    def get_config(self, version=DexforceW1Version.V021):
        return {
            "urdf_path": self.get_urdf(version),
            "joint_names": [],
            "end_link_name": "base_link",
            "root_link_name": "base_link",
        }


class TorsoManager:
    def __init__(self):
        self.joint_names = ["ANKLE", "KNEE", "BUTTOCK", "WAIST"]

    def get_urdf(self, version=DexforceW1Version.V021):
        return get_data_path(
            get_w1_version_spec(version).component_urdf(DexforceW1Type.TORSO)
        )

    def get_config(self, version=DexforceW1Version.V021):
        return {
            "urdf_path": self.get_urdf(version),
            "joint_names": self.joint_names,
            "end_link_name": "waist",
            "root_link_name": "base_link",
        }


class HeadManager:
    def __init__(self):
        self.joint_names = ["NECK1", "NECK2"]

    def get_urdf(self, version=DexforceW1Version.V021):
        return get_data_path(
            get_w1_version_spec(version).component_urdf(DexforceW1Type.HEAD)
        )

    def get_config(self, version=DexforceW1Version.V021):
        return {
            "urdf_path": self.get_urdf(version),
            "joint_names": self.joint_names,
            "end_link_name": "neck2",
            "root_link_name": "neck1",
        }


class EyesManager:
    def get_urdf(self, version=DexforceW1Version.V021):
        return get_data_path(
            get_w1_version_spec(version).component_urdf(DexforceW1Type.EYES)
        )

    def get_config(self, version=DexforceW1Version.V021):
        return {
            "urdf_path": self.get_urdf(version),
            "joint_names": [],
            "end_link_name": "eyes",
            "root_link_name": "base_link",
        }


class ArmManager:
    def get_urdf(self, side, version=DexforceW1Version.V021):
        spec = get_w1_version_spec(version)
        return get_data_path(spec.component_urdf(self.get_component_type(side)))

    @staticmethod
    def get_component_type(side):
        return (
            DexforceW1Type.LEFT_ARM
            if side == DexforceW1ArmSide.LEFT
            else DexforceW1Type.RIGHT_ARM
        )

    def get_config(self, side, version=DexforceW1Version.V021):
        prefix = "LEFT" if side == DexforceW1ArmSide.LEFT else "RIGHT"
        return {
            "urdf_path": self.get_urdf(side, version),
            "joint_names": [f"{prefix}_J{i}" for i in range(1, 8)],
            "end_link_name": f"{prefix.lower()}_ee",
            "root_link_name": f"{prefix.lower()}_arm_base",
        }


class HandManager:
    def get_config(
        self,
        brand: DexforceW1HandBrand,
        side: DexforceW1ArmSide,
        version: DexforceW1HandVersion | None = None,
    ):
        version = version or get_default_w1_hand_version(brand)
        side_spec = get_w1_hand_spec(brand, version).for_side(side)
        return {
            "urdf_path": get_data_path(side_spec.urdf_path),
            "joint_names": list(side_spec.joint_names),
            "end_link_name": side_spec.end_link_name,
            "root_link_name": side_spec.root_link_name,
        }

    def get_urdf(
        self,
        brand: DexforceW1HandBrand,
        side: DexforceW1ArmSide,
        version: DexforceW1HandVersion | None = None,
    ):
        version = version or get_default_w1_hand_version(brand)
        side_spec = get_w1_hand_spec(brand, version).for_side(side)
        return get_data_path(side_spec.urdf_path)

    def get_attach_xpos(
        self,
        brand: DexforceW1HandBrand,
        arm_side: DexforceW1ArmSide = DexforceW1ArmSide.LEFT,
        version: DexforceW1HandVersion | None = None,
    ):
        version = version or get_default_w1_hand_version(brand)
        side_spec = get_w1_hand_spec(brand, version).for_side(arm_side)
        return np.asarray(side_spec.attach_xpos, dtype=float).copy()


eyes_manager = EyesManager()
chassis_manager = ChassisManager()
torso_manager = TorsoManager()
head_manager = HeadManager()
arm_manager = ArmManager()
hand_manager = HandManager()


def build_dexforce_w1_assembly_urdf_cfg(
    version: DexforceW1Version = DexforceW1Version.V021,
    fname: str | None = None,
    hand_types: dict[DexforceW1ArmSide, DexforceW1HandBrand] | None = None,
    hand_versions: dict[DexforceW1ArmSide, DexforceW1HandVersion] | None = None,
    hand_attach_xposes: dict[DexforceW1ArmSide, np.ndarray] | None = None,
    include_hand: bool = True,
) -> URDFCfg:
    """
    Assemble DexforceW1 robot urdf configuration.

    Args:
        version: W1 version used by every robot component.
        fname: Output configuration name. Defaults to the version assembly name.
        hand_types: Dict specifying hand brand (DexforceW1HandBrand) for each arm side. Default None, which uses the default brand.
        hand_versions: Hand asset version for each side. Defaults to hand V021,
            independently of the W1 robot version.
        hand_attach_xposes: Dict specifying hand attachment pose for each arm side. Default None, which uses the default attachment pose.
        include_hand: Whether to include hand. Default True.

    Returns:
        URDFCfg: Assembled URDF configuration.
    """

    version = parse_w1_version(version)
    hand_versions = {
        parse_w1_arm_side(side): parse_w1_hand_version(hand_version)
        for side, hand_version in (hand_versions or {}).items()
    }

    if fname is None:
        fname = get_w1_version_spec(version).assembly_name

    components = [
        {
            "component_type": "chassis",
            "urdf_path": chassis_manager.get_urdf(version),
        },
        {
            "component_type": "torso",
            "urdf_path": torso_manager.get_urdf(version),
        },
        {
            "component_type": "head",
            "urdf_path": head_manager.get_urdf(version),
        },
    ]

    sensors = []

    head_spec = get_w1_version_spec(version)
    head_contains_eyes = head_spec.head_contains_eyes
    if not head_contains_eyes:
        # TODO: Support user-defined eye transforms
        attach_xpos = head_spec.eyes_xpos()

        joint_xml = """
        <joint name="EYES" type="fixed">
            <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
            <parent link="neck2"/>
            <child link="eyes"/>
        </joint>
        """

        link_xml = """
        <link name="eyes">
        <inertial>
            <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
        </inertial>
        </link>
        """

        joint_elem = ET.fromstring(joint_xml)
        link_elem = ET.fromstring(link_xml)

        sensors.append(
            {
                "sensor_name": "eyes",
                "sensor_source": ([link_elem], [joint_elem]),  # eyes_manager.get_urdf()
                "parent_component": "head",
                "parent_link": "neck2",
                "transform": attach_xpos,
                "sensor_type": "camera",
            }
        )
    for arm_side in DexforceW1ArmSide:
        camera_spec = get_w1_version_spec(version)
        attach_xpos = camera_spec.wrist_camera_xpos(arm_side)

        joint_xml = f"""
        <joint name="{arm_side.value.lower()}_wrist_camera" type="fixed">
            <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
            <parent link="{arm_side.value}_ee"/>
            <child link="{arm_side.value.lower()}_wrist_camera"/>
        </joint>
        """

        link_xml = f"""
        <link name="{arm_side.value.lower()}_wrist_camera">
        <inertial>
            <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
        </inertial>
        </link>
        """

        joint_elem = ET.fromstring(joint_xml)
        link_elem = ET.fromstring(link_xml)
        sensors.append(
            {
                "sensor_name": f"{arm_side.value.lower()}_wrist_camera",
                "sensor_source": ([link_elem], [joint_elem]),
                "parent_component": f"{arm_side.value}_arm",
                "parent_link": f"{arm_side.value}_ee",
                "transform": attach_xpos,
                "sensor_type": "camera",
            }
        )

    for arm_side in DexforceW1ArmSide:
        arm_cfg = arm_manager.get_config(arm_side, version)
        components.append(
            {
                "component_type": f"{arm_side.value}_arm",
                "urdf_path": arm_cfg["urdf_path"],
            }
        )

    if include_hand:
        for arm_side in DexforceW1ArmSide:
            # hand_brand: DexforceW1HandBrand
            hand_brand = (hand_types or {}).get(
                arm_side, DexforceW1HandBrand.BRAINCO_HAND
            )
            hand_version = hand_versions.get(
                arm_side, get_default_w1_hand_version(hand_brand)
            )
            urdf_path = hand_manager.get_urdf(hand_brand, arm_side, hand_version)

            custom_attach_xpos = (hand_attach_xposes or {}).get(arm_side)
            if custom_attach_xpos is None:
                hand_attach_xpos = hand_manager.get_attach_xpos(
                    hand_brand, arm_side, hand_version
                )
                arm_spec = get_w1_version_spec(version)
                attach_xpos = arm_spec.compose_eef_attach_xpos(
                    arm_side, hand_attach_xpos
                )
            else:
                arm_spec = get_w1_version_spec(version)
                attach_xpos = arm_spec.compose_eef_attach_xpos(
                    arm_side, custom_attach_xpos
                )
            components.append(
                {
                    "component_type": f"{arm_side.value}_hand",
                    "urdf_path": urdf_path,
                    "transform": attach_xpos,
                }
            )
    # W1 exposes stable uppercase joint names and lowercase link names
    # independently of the casing used by each source component URDF.
    return URDFCfg(
        components=components,
        sensors=sensors,
        fname=fname,
        name_case={"joint": "upper", "link": "lower"},
    )


def build_dexforce_w1_control_parts(
    version: DexforceW1Version,
    hand_types: dict[DexforceW1ArmSide, DexforceW1HandBrand] | None,
    hand_versions: dict[DexforceW1ArmSide, DexforceW1HandVersion] | None,
    include_hand: bool,
) -> dict[str, list[str]]:
    """Build control-part joint lists for a complete dual-arm W1."""
    version = parse_w1_version(version)
    hand_versions = {
        parse_w1_arm_side(side): parse_w1_hand_version(hand_version)
        for side, hand_version in (hand_versions or {}).items()
    }

    arm_joints = {}
    for arm_side in DexforceW1ArmSide:
        arm_joints[arm_side] = arm_manager.get_config(arm_side, version)["joint_names"]

    torso_joints = torso_manager.get_config(version)["joint_names"]
    head_joints = head_manager.get_config(version)["joint_names"]

    hand_joints = {}
    if include_hand:
        for arm_side in DexforceW1ArmSide:
            hand_brand = (hand_types or {}).get(
                arm_side, DexforceW1HandBrand.BRAINCO_HAND
            )
            hand_version = hand_versions.get(
                arm_side, get_default_w1_hand_version(hand_brand)
            )
            hand_joints[arm_side] = hand_manager.get_config(
                hand_brand, arm_side, hand_version
            )["joint_names"]

    left_arm_joints = arm_joints.get(DexforceW1ArmSide.LEFT, [])
    right_arm_joints = arm_joints.get(DexforceW1ArmSide.RIGHT, [])
    control_parts = {}
    control_parts["torso"] = torso_joints
    control_parts["head"] = head_joints
    control_parts["left_arm"] = left_arm_joints
    control_parts["right_arm"] = right_arm_joints
    control_parts["dual_arm"] = left_arm_joints + right_arm_joints
    if DexforceW1ArmSide.LEFT in hand_joints:
        control_parts["left_eef"] = hand_joints[DexforceW1ArmSide.LEFT]
    if DexforceW1ArmSide.RIGHT in hand_joints:
        control_parts["right_eef"] = hand_joints[DexforceW1ArmSide.RIGHT]
    control_parts["full_body"] = (
        torso_joints + head_joints + left_arm_joints + right_arm_joints
    )
    return control_parts
