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
import numpy as np
from typing import List, Dict

from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmKind,
    DexforceW1Type,
    DexforceW1ArmSide,
    DexforceW1Version,
    DexforceW1HandBrand,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.solvers import SolverCfg
from embodichain.lab.sim.cfg import URDFCfg
from embodichain.lab.sim.robots.dexforce_w1.specs import (
    get_w1_version_spec,
    normalize_component_versions,
)

__all__ = [
    "ChassisManager",
    "TorsoManager",
    "HeadManager",
    "ArmManager",
    "HandManager",
    "EyesManager",
    "build_dexforce_w1_assembly_urdf_cfg",
    "build_dexforce_w1_solver_cfg",
    "build_dexforce_w1_cfg",
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
    def get_urdf(self, kind, side, version=DexforceW1Version.V021):
        spec = get_w1_version_spec(version)
        spec.validate_arm_kind(kind)
        return get_data_path(spec.component_urdf(self.get_component_type(kind, side)))

    @staticmethod
    def get_component_type(kind, side):
        if kind == DexforceW1ArmKind.ANTHROPOMORPHIC:
            return (
                DexforceW1Type.LEFT_ARM1
                if side == DexforceW1ArmSide.LEFT
                else DexforceW1Type.RIGHT_ARM1
            )
        return (
            DexforceW1Type.LEFT_ARM2
            if side == DexforceW1ArmSide.LEFT
            else DexforceW1Type.RIGHT_ARM2
        )

    def get_config(self, kind, side, version=DexforceW1Version.V021):
        prefix = "LEFT" if side == DexforceW1ArmSide.LEFT else "RIGHT"
        return {
            "urdf_path": self.get_urdf(kind, side, version),
            "joint_names": [f"{prefix}_J{i}" for i in range(1, 8)],
            "end_link_name": f"{prefix.lower()}_ee",
            "root_link_name": f"{prefix.lower()}_arm_base",
        }


class HandManager:
    def __init__(self):
        self._urdf_rel_paths = {
            (
                DexforceW1HandBrand.BRAINCO_HAND,
                DexforceW1ArmSide.LEFT,
                DexforceW1Version.V021,
            ): "BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf",
            (
                DexforceW1HandBrand.BRAINCO_HAND,
                DexforceW1ArmSide.RIGHT,
                DexforceW1Version.V021,
            ): "BrainCoHandRevo1/BrainCoRightHand/BrainCoRightHand.urdf",
            (
                DexforceW1HandBrand.DH_PGC_GRIPPER,
                DexforceW1ArmSide.LEFT,
                DexforceW1Version.V021,
            ): "DH_PGC_140_50/DH_PGC_140_50.urdf",
            (
                DexforceW1HandBrand.DH_PGC_GRIPPER,
                DexforceW1ArmSide.RIGHT,
                DexforceW1Version.V021,
            ): "DH_PGC_140_50/DH_PGC_140_50.urdf",
            (
                DexforceW1HandBrand.DH_PGC_GRIPPER_M,
                DexforceW1ArmSide.LEFT,
                DexforceW1Version.V021,
            ): "DH_PGC_140_50_M/DH_PGC_140_50_M.urdf",
            (
                DexforceW1HandBrand.DH_PGC_GRIPPER_M,
                DexforceW1ArmSide.RIGHT,
                DexforceW1Version.V021,
            ): "DH_PGC_140_50_M/DH_PGC_140_50_M.urdf",
        }
        for version in DexforceW1Version:
            if version == DexforceW1Version.V021:
                continue
            for brand in DexforceW1HandBrand:
                for side in DexforceW1ArmSide:
                    self._urdf_rel_paths[(brand, side, version)] = self._urdf_rel_paths[
                        (brand, side, DexforceW1Version.V021)
                    ]

    def get_config(
        self,
        brand: DexforceW1HandBrand,
        side: DexforceW1ArmSide,
        version: DexforceW1Version = DexforceW1Version.V021,
    ):
        prefix = "LEFT" if side == DexforceW1ArmSide.LEFT else "RIGHT"
        if brand == DexforceW1HandBrand.BRAINCO_HAND:
            if side == DexforceW1ArmSide.LEFT:
                base_link_name = f"{prefix.lower()}_hand_base"
                root_link_name = f"{prefix.lower()}_thumb_dist"
                joint_names = [
                    f"{prefix}_HAND_THUMB1",  # Left thumb flexion
                    f"{prefix}_HAND_THUMB2",  # Left thumb abduction/adduction
                    f"{prefix}_HAND_INDEX",  # Left index finger flexion
                    f"{prefix}_HAND_MIDDLE",  # Left middle finger flexion
                    f"{prefix}_HAND_RING",  # Left ring finger flexion
                    f"{prefix}_HAND_PINKY",  # Left pinky finger flexion
                ]
            else:
                base_link_name = f"{prefix.lower()}_hand_base"
                root_link_name = f"{prefix.lower()}_thumb_dist"
                joint_names = [
                    f"{prefix}_HAND_THUMB1",  # Right thumb flexion
                    f"{prefix}_HAND_THUMB2",  # Right thumb abduction/adduction
                    f"{prefix}_HAND_INDEX",  # Right index finger flexion
                    f"{prefix}_HAND_MIDDLE",  # Right middle finger flexion
                    f"{prefix}_HAND_RING",  # Right ring finger flexion
                    f"{prefix}_HAND_PINKY",  # Right pinky finger flexion
                ]
        elif brand == DexforceW1HandBrand.DH_PGC_GRIPPER:
            base_link_name = f"{prefix.lower()}_base_link_1"
            root_link_name = f"{prefix.lower()}_finger2_link"
            joint_names = [f"{prefix}_FINGER1_JOINT", f"{prefix}_FINGER2_JOINT"]
        elif brand == DexforceW1HandBrand.DH_PGC_GRIPPER_M:
            base_link_name = f"{prefix.lower()}_base_link_1"
            root_link_name = f"{prefix.lower()}_finger2"
            joint_names = [f"{prefix}_FINGER1", f"{prefix}_FINGER2"]
        else:
            raise ValueError(f"Unknown hand brand: {brand}")

        return {
            "urdf_path": self.get_urdf(brand, side, version),
            "joint_names": joint_names,
            "end_link_name": base_link_name,
            "root_link_name": root_link_name,
        }

    def get_urdf(
        self,
        brand: DexforceW1HandBrand,
        side: DexforceW1ArmSide,
        version: DexforceW1Version = DexforceW1Version.V021,
    ):
        return get_data_path(self._urdf_rel_paths[(brand, side, version)])

    def get_attach_xpos(
        self,
        brand: DexforceW1HandBrand,
        arm_kind: DexforceW1ArmKind = DexforceW1ArmKind.INDUSTRIAL,
        arm_side: DexforceW1ArmSide = DexforceW1ArmSide.LEFT,
        version: DexforceW1Version = DexforceW1Version.V021,
    ):
        return get_w1_version_spec(version).hand_attach_xpos(brand, arm_kind, arm_side)


eyes_manager = EyesManager()
chassis_manager = ChassisManager()
torso_manager = TorsoManager()
head_manager = HeadManager()
arm_manager = ArmManager()
hand_manager = HandManager()


def build_dexforce_w1_assembly_urdf_cfg(
    arm_kind: DexforceW1ArmKind,
    version: DexforceW1Version = DexforceW1Version.V021,
    arm_sides: List[DexforceW1ArmSide] = [
        DexforceW1ArmSide.LEFT,
        DexforceW1ArmSide.RIGHT,
    ],
    fname: str | None = None,
    hand_types: dict[DexforceW1ArmSide, DexforceW1HandBrand] | None = None,
    hand_versions: dict[DexforceW1ArmSide, DexforceW1Version] | None = None,
    hand_attach_xposes: dict[DexforceW1ArmSide, np.ndarray] | None = None,
    include_chassis: bool = True,
    include_torso: bool = True,
    include_head: bool = True,
    include_hand: bool = True,
    include_eyes: bool = True,
    include_wrist_cameras: bool = True,
    component_versions: dict[DexforceW1Type, DexforceW1Version] | None = None,
) -> URDFCfg:
    """
    Assemble DexforceW1 robot urdf configuration.

    Args:
        arm_kind: Arm type (anthropomorphic or industrial).
        arm_sides: List of arm sides to include (left/right). Default both sides.
        version: Default W1 revision for every component.
        fname: Output configuration name. Defaults to the version assembly name.
        hand_types: Dict specifying hand brand (DexforceW1HandBrand) for each arm side. Default None, which uses the default brand.
        hand_versions: Dict specifying hand version for each arm side. Default None, which uses the default version.
        hand_attach_xposes: Dict specifying hand attachment pose for each arm side. Default None, which uses the default attachment pose.
        include_chassis: Whether to include chassis. Default True.
        include_torso: Whether to include torso. Default True.
        include_head: Whether to include head. Default True.
        include_hand: Whether to include hand. Default True.
        include_wrist_cameras: Whether to include wrist cameras. Default True.
        component_versions: Optional per-component overrides. Components not
            listed here use ``version``.

    Returns:
        URDFCfg: Assembled URDF configuration.
    """

    if not isinstance(version, DexforceW1Version):
        version = DexforceW1Version(version)
    component_versions = normalize_component_versions(component_versions)

    def get_version(component_type):
        return component_versions.get(component_type, version)

    if fname is None:
        fname = get_w1_version_spec(version).assembly_name

    components = []
    if include_chassis:
        components.append(
            {
                "component_type": "chassis",
                "urdf_path": chassis_manager.get_urdf(
                    get_version(DexforceW1Type.CHASSIS)
                ),
            }
        )
    if include_torso:
        components.append(
            {
                "component_type": "torso",
                "urdf_path": torso_manager.get_urdf(get_version(DexforceW1Type.TORSO)),
            }
        )
    if include_head:
        components.append(
            {
                "component_type": "head",
                "urdf_path": head_manager.get_urdf(get_version(DexforceW1Type.HEAD)),
            }
        )

    sensors = []

    head_version = get_version(DexforceW1Type.HEAD)
    head_spec = get_w1_version_spec(head_version)
    head_contains_eyes = head_spec.head_contains_eyes
    if include_head and include_eyes and not head_contains_eyes:
        # TODO: Support user-defined eye transforms
        import xml.etree.ElementTree as ET

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
    if include_wrist_cameras:
        for arm_side in arm_sides:
            # TODO: Support user-defined eye transforms
            import xml.etree.ElementTree as ET

            arm_type = arm_manager.get_component_type(arm_kind, arm_side)
            camera_spec = get_w1_version_spec(get_version(arm_type))
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

    for arm_side in arm_sides:
        arm_type = arm_manager.get_component_type(arm_kind, arm_side)
        arm_version = get_version(arm_type)
        arm_cfg = arm_manager.get_config(arm_kind, arm_side, arm_version)
        components.append(
            {
                "component_type": f"{arm_side.value}_arm",
                "urdf_path": arm_cfg["urdf_path"],
            }
        )

    if include_hand:
        for arm_side in arm_sides:
            # hand_brand: DexforceW1HandBrand
            hand_brand = (hand_types or {}).get(
                arm_side, DexforceW1HandBrand.BRAINCO_HAND
            )
            hand_version = (hand_versions or {}).get(
                arm_side,
                get_version(
                    DexforceW1Type.LEFT_HAND
                    if arm_side == DexforceW1ArmSide.LEFT
                    else DexforceW1Type.RIGHT_HAND
                ),
            )
            urdf_path = hand_manager.get_urdf(hand_brand, arm_side, hand_version)

            attach_xpos = (hand_attach_xposes or {}).get(
                arm_side,
                hand_manager.get_attach_xpos(
                    hand_brand,
                    arm_kind,
                    arm_side,
                    get_version(arm_manager.get_component_type(arm_kind, arm_side)),
                ),
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


def build_dexforce_w1_solver_cfg(
    arm_kind: DexforceW1ArmKind,
    version: DexforceW1Version = DexforceW1Version.V021,
    arm_sides: List[DexforceW1ArmSide] = [
        DexforceW1ArmSide.LEFT,
        DexforceW1ArmSide.RIGHT,
    ],
    component_versions: dict[DexforceW1Type, DexforceW1Version] | None = None,
    urdf_cfg: URDFCfg | None = None,
) -> Dict[str, SolverCfg]:
    """
    Build DexforceW1 solver configuration dict.

    Args:
        arm_kind: Arm type.
        version: Default W1 revision.
        arm_sides: Included arm sides. Optional, default both sides.
        component_versions: Optional per-component version overrides.
        urdf_cfg: Optional, URDFCfg object from build_dexforce_w1_assembly_urdf_cfg.

    Returns:
        Dict[str, SolverCfg]: solver config keyed by control part name
        (e.g. ``"left_arm"``, ``"full_body"``).
    """

    if not isinstance(version, DexforceW1Version):
        version = DexforceW1Version(version)
    component_versions = normalize_component_versions(component_versions)

    solver_cfg = {}

    for arm_side in arm_sides:
        arm_type = arm_manager.get_component_type(arm_kind, arm_side)
        arm_version = component_versions.get(arm_type, version)
        arm_cfg = arm_manager.get_config(arm_kind, arm_side, arm_version)
        # Use control_parts-aligned key (e.g. "left_arm") so init_solver
        # can match this entry to the corresponding control part.
        solver_key = f"{arm_side.value}_arm"
        solver_cfg[solver_key] = SolverCfg.from_dict(
            {
                "class_type": "PytorchSolver",
                "urdf_path": arm_cfg["urdf_path"],
                "joint_names": arm_cfg["joint_names"],
                "end_link_name": arm_cfg["end_link_name"],
                "root_link_name": arm_cfg["root_link_name"],
                "tcp": get_w1_version_spec(arm_version).tcp(arm_kind, arm_side),
            }
        )

    if urdf_cfg is not None and urdf_cfg.fpath:
        full_body_urdf_path = urdf_cfg.fpath
    else:
        full_body_urdf_path = get_data_path(
            get_w1_version_spec(version).full_robot_urdf(arm_kind)
        )

    solver_cfg[DexforceW1Type.FULL_BODY.value] = SolverCfg.from_dict(
        {
            "class_type": "PytorchSolver",
            "urdf_path": full_body_urdf_path,
            "joint_names": [
                "ANKLE",
                "KNEE",
                "BUTTOCK",
                "WAIST",
                "NECK1",
                "NECK2",
                "LEFT_J1",
                "LEFT_J2",
                "LEFT_J3",
                "LEFT_J4",
                "LEFT_J5",
                "LEFT_J6",
                "LEFT_J7",
                "RIGHT_J1",
                "RIGHT_J2",
                "RIGHT_J3",
                "RIGHT_J4",
                "RIGHT_J5",
                "RIGHT_J6",
                "RIGHT_J7",
            ],
            "end_link_name": "right_ee",
            "root_link_name": "base_link",
        }
    )
    return solver_cfg


def build_dexforce_w1_cfg(
    arm_kind: DexforceW1ArmKind,
    version: DexforceW1Version = DexforceW1Version.V021,
    arm_sides: List[DexforceW1ArmSide] = [
        DexforceW1ArmSide.LEFT,
        DexforceW1ArmSide.RIGHT,
    ],
    hand_types: dict[DexforceW1ArmSide, DexforceW1HandBrand] | None = None,
    hand_versions: dict[DexforceW1ArmSide, DexforceW1Version] | None = None,
    hand_attach_xposes: dict[DexforceW1ArmSide, np.ndarray] | None = None,
    include_chassis: bool = True,
    include_torso: bool = True,
    include_head: bool = True,
    include_hand: bool = True,
    include_eyes: bool = True,
    include_wrist_cameras: bool = True,
    component_versions: dict[DexforceW1Type, DexforceW1Version] | None = None,
    solver_cfg: dict[DexforceW1Type, SolverCfg] | None = None,
) -> "DexforceW1Cfg":
    """
    Build DexforceW1 robot configuration object.

    Args:
        arm_kind: Arm type (anthropomorphic or industrial).
        version: Default W1 revision.
        arm_sides: List of arm sides to include (left/right). Default both sides.
        hand_types: Dict specifying hand brand (DexforceW1HandBrand) for each arm side. Default None, which uses the default brand.
        hand_versions: Dict specifying hand version for each arm side. Default None, which uses the default version.
        hand_attach_xposes: Dict specifying hand attachment pose for each arm side. Default None, which uses the default attachment pose.
        include_chassis: Whether to include chassis. Optional, default True.
        include_torso: Whether to include torso. Optional, default True.
        include_head: Whether to include head. Optional, default True.
        include_hand: Whether to include hand. Optional, default True.
        include_eyes: Whether to include eyes when they are not already part of
            the selected head asset.
        include_wrist_cameras: Whether to include wrist cameras.
        component_versions: Dict specifying version for each robot component.
        solver_cfg: Optional, pre-defined solver configuration dict.

    Returns:
        DexforceW1Cfg: Robot configuration object.
    """
    if not isinstance(version, DexforceW1Version):
        version = DexforceW1Version(version)
    component_versions = normalize_component_versions(component_versions)

    urdf_cfg = build_dexforce_w1_assembly_urdf_cfg(
        arm_kind=arm_kind,
        version=version,
        arm_sides=arm_sides,
        hand_types=hand_types,
        hand_versions=hand_versions,
        hand_attach_xposes=hand_attach_xposes,
        include_chassis=include_chassis,
        include_torso=include_torso,
        include_head=include_head,
        include_hand=include_hand,
        include_eyes=include_eyes,
        include_wrist_cameras=include_wrist_cameras,
        component_versions=component_versions,
    )

    left_arm_joints = []
    right_arm_joints = []
    for arm_side in arm_sides:
        arm_type = arm_manager.get_component_type(arm_kind, arm_side)
        arm_version = component_versions.get(arm_type, version)
        arm_cfg = arm_manager.get_config(arm_kind, arm_side, arm_version)
        if arm_side == DexforceW1ArmSide.LEFT:
            left_arm_joints = arm_cfg["joint_names"]
        elif arm_side == DexforceW1ArmSide.RIGHT:
            right_arm_joints = arm_cfg["joint_names"]

    torso_joints = []
    head_joints = []
    left_hand_joints = []
    right_hand_joints = []

    if include_torso:
        torso_joints = torso_manager.get_config(
            component_versions.get(DexforceW1Type.TORSO, version)
        )["joint_names"]
    if include_head:
        head_joints = head_manager.get_config(
            component_versions.get(DexforceW1Type.HEAD, version)
        )["joint_names"]
    if include_hand:
        if DexforceW1ArmSide.LEFT in arm_sides:
            left_hand_brand = (hand_types or {}).get(
                DexforceW1ArmSide.LEFT, DexforceW1HandBrand.BRAINCO_HAND
            )
            left_hand_version = (hand_versions or {}).get(
                DexforceW1ArmSide.LEFT,
                component_versions.get(DexforceW1Type.LEFT_HAND, version),
            )
            left_hand_cfg = hand_manager.get_config(
                left_hand_brand, DexforceW1ArmSide.LEFT, left_hand_version
            )
            left_hand_joints = left_hand_cfg["joint_names"]
        if DexforceW1ArmSide.RIGHT in arm_sides:
            right_hand_brand = (hand_types or {}).get(
                DexforceW1ArmSide.RIGHT, DexforceW1HandBrand.BRAINCO_HAND
            )
            right_hand_version = (hand_versions or {}).get(
                DexforceW1ArmSide.RIGHT,
                component_versions.get(DexforceW1Type.RIGHT_HAND, version),
            )
            right_hand_cfg = hand_manager.get_config(
                right_hand_brand, DexforceW1ArmSide.RIGHT, right_hand_version
            )
            right_hand_joints = right_hand_cfg["joint_names"]

    control_parts = {}

    if torso_joints:
        control_parts["torso"] = torso_joints
    if head_joints:
        control_parts["head"] = head_joints
    if left_arm_joints:
        control_parts["left_arm"] = left_arm_joints
    if right_arm_joints:
        control_parts["right_arm"] = right_arm_joints
    if left_arm_joints and right_arm_joints:
        control_parts["dual_arm"] = left_arm_joints + right_arm_joints
    if left_hand_joints:
        control_parts["left_eef"] = left_hand_joints
    if right_hand_joints:
        control_parts["right_eef"] = right_hand_joints

    if torso_joints and head_joints and left_arm_joints and right_arm_joints:
        control_parts["full_body"] = (
            torso_joints + head_joints + left_arm_joints + right_arm_joints
        )

    from embodichain.lab.sim.robots.dexforce_w1.cfg import DexforceW1Cfg

    cfg = DexforceW1Cfg()
    cfg.version = version
    cfg.arm_kind = arm_kind
    cfg.component_versions = component_versions
    cfg.urdf_cfg = urdf_cfg
    cfg.control_parts = control_parts

    if solver_cfg is not None:
        cfg.solver_cfg = solver_cfg
    else:
        cfg.solver_cfg = build_dexforce_w1_solver_cfg(
            arm_kind=arm_kind,
            version=version,
            arm_sides=arm_sides,
            component_versions=component_versions,
            urdf_cfg=urdf_cfg,
        )

    return cfg
