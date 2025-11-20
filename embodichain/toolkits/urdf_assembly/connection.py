# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from scipy.spatial.transform import Rotation as R

from embodichain.toolkits.urdf_assembly.logging_utils import (
    URDFAssemblyLogger,
)

__all__ = ["URDFConnectionManager"]


class URDFConnectionManager:
    r"""
    Responsible for managing connection rules between components and sensor attachments.
    """

    def __init__(self, base_link_name: str):
        r"""Initialize the URDFConnectionManager.

        Args:
            base_link_name (str): The name of the base link to which the chassis or other components may be attached.
        """
        self.base_link_name = base_link_name
        self.logger = URDFAssemblyLogger.get_logger("connection_manager")

    def add_connections(
        self,
        joints: list,
        base_points: dict,
        parent_attach_points: dict,
        connection_rules: list,
        component_transforms: dict = None,
    ):
        r"""Add connection joints between robot components according to the specified rules.

        Args:
            joints (list): A list to collect joint elements.
            base_points (dict): A mapping from component names to their child connection link names.
            parent_attach_points (dict): A mapping from component names to their parent connection link names.
            connection_rules (list): A list of (parent, child) tuples specifying connection relationships.
            component_transforms (dict): Optional mapping from component names to their transform matrices.
        """
        chassis_component = "chassis"
        component_transforms = component_transforms or {}

        existing_joint_names = {
            joint.get("name") for joint in joints if hasattr(joint, "get")
        }

        # chassis is always attached to base_link (no transform applied to this connection)
        if chassis_component in base_points:
            chassis_first_link = base_points[chassis_component]
            joint_name = f"BASE_LINK_TO_{chassis_component.upper()}_CONNECTOR"
            if joint_name not in existing_joint_names:
                joint = ET.Element("joint", name=joint_name, type="fixed")
                ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
                ET.SubElement(joint, "parent", link=self.base_link_name)
                ET.SubElement(joint, "child", link=chassis_first_link)
                joints.append(joint)
                existing_joint_names.add(joint_name)
                self.logger.info(
                    f"[{chassis_component.capitalize()}] connected to [base_link] via ({chassis_first_link})"
                )
        else:
            # If no chassis, connect components directly to base_link with their transforms
            self.logger.info(
                "No chassis found, connecting components directly to base_link"
            )

            # Find components that don't have parents in connection_rules
            components_with_parents = {child for parent, child in connection_rules}
            orphan_components = [
                comp
                for comp in base_points.keys()
                if comp not in components_with_parents
            ]

            for comp in orphan_components:
                comp_first_link = base_points[comp]
                joint_name = f"BASE_TO_{comp.upper()}_CONNECTOR"

                if joint_name not in existing_joint_names:
                    joint = ET.Element("joint", name=joint_name, type="fixed")

                    # Apply transform to this specific connection if the component has one
                    if comp in component_transforms:
                        transform = component_transforms[comp]
                        xyz = transform[:3, 3]  # Extract translation
                        rotation = R.from_matrix(transform[:3, :3])
                        rpy = rotation.as_euler("xyz")

                        ET.SubElement(
                            joint,
                            "origin",
                            xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}",
                            rpy=f"{rpy[0]} {rpy[1]} {rpy[2]}",
                        )
                        self.logger.info(
                            f"Applied transform to base connection {comp}: xyz={xyz}, rpy={rpy}"
                        )
                    else:
                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

                    ET.SubElement(joint, "parent", link=self.base_link_name)
                    ET.SubElement(joint, "child", link=comp_first_link)
                    joints.append(joint)
                    existing_joint_names.add(joint_name)

                    self.logger.info(
                        f"[{comp.capitalize()}] connected to [base_link] via ({comp_first_link})"
                    )

        # Process other connection relationships
        for parent, child in connection_rules:
            if parent in parent_attach_points and child in base_points:
                parent_connect_link = parent_attach_points[parent].lower()
                child_connect_link = base_points[child].lower()

                self.logger.info(
                    f"Connecting [{parent}]-({parent_connect_link}) to [{child}]-({child_connect_link})"
                )

                # Create a unique joint name
                base_joint_name = f"{parent.upper()}_TO_{child.upper()}_CONNECTOR"
                if base_joint_name not in existing_joint_names:
                    joint = ET.Element("joint", name=base_joint_name, type="fixed")

                    # Apply transform to this specific connection if the child component has one
                    if child in component_transforms:
                        transform = component_transforms[child]
                        xyz = transform[:3, 3]  # Extract translation
                        rotation = R.from_matrix(transform[:3, :3])
                        rpy = rotation.as_euler("xyz")

                        ET.SubElement(
                            joint,
                            "origin",
                            xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}",
                            rpy=f"{rpy[0]} {rpy[1]} {rpy[2]}",
                        )
                        self.logger.info(
                            f"Applied transform to connection {parent} -> {child}: xyz={xyz}, rpy={rpy}"
                        )
                    else:
                        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

                    ET.SubElement(joint, "parent", link=parent_connect_link)
                    ET.SubElement(joint, "child", link=child_connect_link)
                    joints.append(joint)
                    existing_joint_names.add(base_joint_name)
                else:
                    self.logger.warning(
                        f"Duplicate connection rule: {parent} -> {child}"
                    )
            else:
                self.logger.error(f"Invalid connection rule: {parent} -> {child}")

    def add_sensor_attachments(
        self, joints: list, attach_dict: dict, base_points: dict
    ):
        r"""Attach sensors to the robot by creating fixed joints."""
        for sensor_name, attach in attach_dict.items():
            sensor_urdf = ET.parse(attach.sensor_urdf).getroot()

            # Add sensor links and joints to the main lists
            for link in sensor_urdf.findall("link"):
                # Ensure sensor link names are lowercase
                link.set("name", link.get("name").lower())
                joints.append(link)  # This should be added to links list instead

            for joint in sensor_urdf.findall("joint"):
                # Ensure sensor joint names are uppercase and link references are lowercase
                joint.set("name", joint.get("name").upper())
                parent_elem = joint.find("parent")
                child_elem = joint.find("child")
                if parent_elem is not None:
                    parent_elem.set("link", parent_elem.get("link").lower())
                if child_elem is not None:
                    child_elem.set("link", child_elem.get("link").lower())
                joints.append(joint)

            parent_link = base_points.get(
                attach.parent_component, attach.parent_component
            ).lower()  # Ensure lowercase

            # Create connection joint with uppercase name
            joint_name = (
                f"{attach.parent_component.upper()}_TO_{sensor_name.upper()}_CONNECTOR"
            )
            joint = ET.Element("joint", name=joint_name, type="fixed")
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(joint, "parent", link=parent_link)
            ET.SubElement(
                joint, "child", link=sensor_urdf.find("link").get("name").lower()
            )
            joints.append(joint)
