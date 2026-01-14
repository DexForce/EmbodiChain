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

import os
import copy
import traceback
import numpy as np
from dataclasses import dataclass
import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Union, Tuple

from embodichain.toolkits.urdf_assembly.logging_utils import (
    URDFAssemblyLogger,
)
from embodichain.toolkits.urdf_assembly.mesh import URDFMeshManager

__all__ = ["SensorRegistry", "SensorAttachment", "URDFSensorManager"]


class SensorRegistry:
    """Registry for storing and retrieving SensorAttachment objects."""

    def __init__(self):
        self._sensors = {}

    def add(self, sensor_name: str, sensor_obj):
        self._sensors[sensor_name] = sensor_obj

    def get(self, sensor_name: str):
        return self._sensors.get(sensor_name)

    def all(self):
        return self._sensors

    def remove(self, sensor_name: str):
        if sensor_name in self._sensors:
            del self._sensors[sensor_name]


@dataclass
class SensorAttachment:
    r"""Represents a sensor attachment configuration to a robot component.

    This dataclass defines how a sensor should be attached to a specific component
    and link within the robot assembly, including optional spatial transformation
    to position the sensor correctly relative to the attachment point.
    """

    sensor_urdf: str  # Path to the sensor's URDF file
    parent_component: str  # Name of the component to which the sensor is attached
    parent_link: str  # Specific link name within the parent component for attachment
    transform: np.ndarray | None = (
        None  # 4x4 transformation matrix for sensor positioning, or None
    )
    sensor_type: str | None = None  # Sensor type field, or None


class URDFSensorManager:
    r"""Responsible for loading, processing, and managing sensor attachments."""

    def __init__(self, mesh_manager: URDFMeshManager):
        r"""Initialize the URDFSensorManager.

        Args:
            mesh_manager (URDFMeshManager): Manager for handling mesh files.
        """
        self.mesh_manager = mesh_manager
        self.logger = URDFAssemblyLogger.get_logger("sensor_manager")
        self.attached_sensors = {}  # Maps sensor_name to processed sensor data

    def attach_sensor(
        self,
        sensor_name: str,
        sensor_source: Union[
            str, ET.Element, Dict, Tuple[List[ET.Element], List[ET.Element]]
        ],
        parent_component: str,
        parent_link: str,
        transform: np.ndarray | None = None,
        sensor_type: str | None = None,
        extract_links: list[str] | None = None,
        extract_joints: list[str] | None = None,
    ) -> bool:
        r"""Attach a sensor to a specific component and link with multiple input format support.

        Args:
            sensor_name (str): Unique identifier for the sensor attachment.
            sensor_source: Sensor definition source (multiple formats supported).
            parent_component (str): Target component name for sensor attachment.
            parent_link (str): Specific link within parent component for attachment.
            transform (np.ndarray | None): Optional 4x4 homogeneous transformation matrix.
            sensor_type (str | None): Sensor type classification.
            extract_links (list[str] | None): Specific link names to extract from URDF.
            extract_joints (list[str] | None): Specific joint names to extract from URDF.

        Returns:
            bool: True if sensor attachment successful, False on failure.
        """
        try:
            # Phase 1: Input validation
            if not self._validate_sensor_params(
                sensor_name, sensor_source, parent_component, parent_link, transform
            ):
                return False

            # Phase 2: Process sensor source based on input type
            sensor_elements = self._process_sensor_source(
                sensor_source, extract_links, extract_joints, sensor_name
            )

            if not sensor_elements:
                self.logger.error("Failed to process sensor source")
                return False

            sensor_links, sensor_joints, sensor_urdf_path = sensor_elements

            # Phase 3: Validate sensor elements
            if not self._validate_sensor_elements(sensor_links, sensor_joints):
                return False

            # Phase 4: Process and rename sensor elements to avoid conflicts
            processed_elements = self._process_and_rename_sensor_elements(
                sensor_links, sensor_joints, sensor_name
            )

            if not processed_elements:
                self.logger.error("Failed to process sensor elements")
                return False

            processed_links, processed_joints = processed_elements

            # Phase 5: Create sensor attachment data (compatible with existing SensorAttachment)
            sensor_attachment = SensorAttachment(
                sensor_urdf=sensor_urdf_path,
                parent_component=parent_component,
                parent_link=parent_link,
                transform=transform,
            )

            # Store processed sensor data
            self.attached_sensors[sensor_name] = {
                "attachment": sensor_attachment,
                "links": processed_links,
                "joints": processed_joints,
                "sensor_type": sensor_type,
            }

            self.logger.debug(
                f"Successfully attached sensor [{sensor_name}] "
                f"({sensor_type or 'unspecified'}) with {len(processed_links)} links "
                f"and {len(processed_joints)} joints to component ({parent_component}) "
                f"at link ({parent_link})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Sensor attachment failed for [{sensor_name}]: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    def _validate_sensor_params(
        self,
        sensor_name: str,
        sensor_source,
        parent_component: str,
        parent_link: str,
        transform: np.ndarray | None,
    ) -> bool:
        r"""Validate input parameters for sensor attachment.

        Args:
            sensor_name: Sensor identifier to validate
            sensor_source: Sensor source to validate
            parent_component: Parent component name to validate
            parent_link: Parent link name to validate
            transform: Transformation matrix to validate

        Returns:
            bool: True if all parameters are valid, False otherwise
        """
        # Validate sensor name
        if not sensor_name or not isinstance(sensor_name, str):
            self.logger.error("Sensor name must be a non-empty string")
            return False

        if not sensor_name.replace("_", "").replace("-", "").isalnum():
            self.logger.error(
                "Sensor name must contain only alphanumeric characters, underscores, and hyphens"
            )
            return False

        # Validate sensor source
        if sensor_source is None:
            self.logger.error("Sensor source cannot be None")
            return False

        # Validate parent component and link
        if not parent_component or not isinstance(parent_component, str):
            self.logger.error("Parent component must be a non-empty string")
            return False

        if not parent_link or not isinstance(parent_link, str):
            self.logger.error("Parent link must be a non-empty string")
            return False

        # Validate transformation matrix if provided
        if transform is not None:
            if not isinstance(transform, np.ndarray):
                self.logger.error("Transform must be a numpy array")
                return False

            if transform.shape != (4, 4):
                self.logger.error(
                    f"Transform must be 4x4 matrix, got shape {transform.shape}"
                )
                return False

            if not self._is_valid_homogeneous_transform(transform):
                self.logger.error(
                    "Transform is not a valid homogeneous transformation matrix"
                )
                return False

        return True

    def _process_sensor_source(
        self,
        sensor_source,
        extract_links: list[str] | None,
        extract_joints: list[str] | None,
        sensor_name: str,
    ) -> tuple[list[ET.Element], list[ET.Element], str] | None:
        r"""Process sensor source based on input type and extract relevant elements.

        Args:
            sensor_source: Input sensor source in various formats
            extract_links: Optional list of specific link names to extract
            extract_joints: Optional list of specific joint names to extract
            sensor_name: Sensor name for path generation

        Returns:
            Optional tuple of (links, joints, urdf_path) or None on failure
        """
        try:
            if isinstance(sensor_source, str):
                # Handle URDF file path
                return self._process_urdf_file_source(
                    sensor_source, extract_links, extract_joints
                )

            elif isinstance(sensor_source, ET.Element):
                # Handle pre-loaded URDF element
                return self._process_urdf_element_source(
                    sensor_source, extract_links, extract_joints, sensor_name
                )

            elif isinstance(sensor_source, dict):
                # Handle configuration dictionary
                return self._process_config_dict_source(sensor_source, sensor_name)

            elif isinstance(sensor_source, tuple) and len(sensor_source) == 2:
                # Handle direct (links, joints) tuple
                return self._process_element_tuple_source(sensor_source, sensor_name)

            else:
                self.logger.error(
                    f"Unsupported sensor source type: {type(sensor_source)}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Error processing sensor source: {str(e)}")
            return None

    def _process_urdf_file_source(
        self,
        file_path: str,
        extract_links: list[str] | None,
        extract_joints: list[str] | None,
    ) -> tuple[list[ET.Element], list[ET.Element], str] | None:
        r"""Process URDF file source and extract specified elements.

        Args:
            file_path: Path to URDF file
            extract_links: Optional list of link names to extract
            extract_joints: Optional list of joint names to extract

        Returns:
            Tuple of (links, joints, file_path) or None on failure
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Sensor URDF file not found: {file_path}")
            return None

        try:
            urdf_element = ET.parse(file_path).getroot()
            links, joints = self._extract_elements_from_urdf(
                urdf_element, extract_links, extract_joints
            )
            return links, joints, file_path

        except ET.ParseError as e:
            self.logger.error(f"Failed to parse URDF file {file_path}: {str(e)}")
            return None

    def _process_urdf_element_source(
        self,
        urdf_element: ET.Element,
        extract_links: list[str] | None,
        extract_joints: list[str] | None,
        sensor_name: str,
    ) -> tuple[list[ET.Element], list[ET.Element], str]:
        r"""Process pre-loaded URDF element source.

        Args:
            urdf_element: Pre-loaded URDF root element
            extract_links: Optional list of link names to extract
            extract_joints: Optional list of joint names to extract
            sensor_name: Sensor name for path generation

        Returns:
            Tuple of (links, joints, generated_path)
        """
        links, joints = self._extract_elements_from_urdf(
            urdf_element, extract_links, extract_joints
        )
        generated_path = f"<inline_urdf_{sensor_name}>"
        return links, joints, generated_path

    def _process_config_dict_source(
        self, config: Dict, sensor_name: str
    ) -> tuple[list[ET.Element], list[ET.Element], str]:
        r"""Process configuration dictionary source and create URDF elements.

        Args:
            config: Configuration dictionary for sensor creation
            sensor_name: Sensor name for element generation

        Returns:
            Tuple of (links, joints, generated_path)
        """
        urdf_element = self._create_sensor_from_config(config, sensor_name)
        links = urdf_element.findall("link")
        joints = urdf_element.findall("joint")
        generated_path = f"<generated_from_config_{sensor_name}>"
        return links, joints, generated_path

    def _process_element_tuple_source(
        self, element_tuple: tuple, sensor_name: str
    ) -> tuple[list[ET.Element], list[ET.Element], str] | None:
        r"""Process direct element tuple source.

        Args:
            element_tuple: Tuple containing (links_list, joints_list)
            sensor_name: Sensor name for path generation

        Returns:
            Tuple of (links, joints, generated_path) or None on failure
        """
        links_list, joints_list = element_tuple

        if not isinstance(links_list, list) or not isinstance(joints_list, list):
            self.logger.error(
                "Element tuple must contain (List[ET.Element], List[ET.Element])"
            )
            return None

        # Validate that all elements are actually ET.Element instances
        for i, link in enumerate(links_list):
            if not isinstance(link, ET.Element):
                self.logger.error(f"Links list item {i} is not an ET.Element")
                return None

        for i, joint in enumerate(joints_list):
            if not isinstance(joint, ET.Element):
                self.logger.error(f"Joints list item {i} is not an ET.Element")
                return None

        generated_path = f"<inline_elements_{sensor_name}>"
        return links_list, joints_list, generated_path

    def _extract_elements_from_urdf(
        self,
        urdf_element: ET.Element,
        extract_links: list[str] | None = None,
        extract_joints: list[str] | None = None,
    ) -> tuple[list[ET.Element], list[ET.Element]]:
        r"""Extract specified links and joints from URDF element.

        Args:
            urdf_element: URDF root element to extract from
            extract_links: Optional list of specific link names to extract
            extract_joints: Optional list of specific joint names to extract

        Returns:
            Tuple of (extracted_links, extracted_joints)
        """
        # Extract links
        all_links = urdf_element.findall("link")
        if extract_links:
            links = []
            for link_name in extract_links:
                link = urdf_element.find(f".//link[@name='{link_name}']")
                if link is not None:
                    links.append(link)
                    self.logger.debug(f"Extracted link: {link_name}")
                else:
                    self.logger.warning(f"Link '{link_name}' not found in URDF")
        else:
            links = all_links
            self.logger.debug(f"Extracted all {len(links)} links from URDF")

        # Extract joints
        all_joints = urdf_element.findall("joint")
        if extract_joints:
            joints = []
            for joint_name in extract_joints:
                joint = urdf_element.find(f".//joint[@name='{joint_name}']")
                if joint is not None:
                    joints.append(joint)
                    self.logger.debug(f"Extracted joint: {joint_name}")
                else:
                    self.logger.warning(f"Joint '{joint_name}' not found in URDF")
        else:
            joints = all_joints
            self.logger.debug(f"Extracted all {len(joints)} joints from URDF")

        return links, joints

    def _validate_sensor_elements(
        self, sensor_links: list[ET.Element], sensor_joints: list[ET.Element]
    ) -> bool:
        r"""Validate extracted sensor elements for completeness and consistency.

        Args:
            sensor_links: List of sensor link elements
            sensor_joints: List of sensor joint elements

        Returns:
            bool: True if elements are valid, False otherwise
        """
        if not sensor_links:
            self.logger.error("No links found in sensor definition")
            return False

        # Validate link elements
        for i, link in enumerate(sensor_links):
            if not isinstance(link, ET.Element):
                self.logger.error(f"Invalid link element at index {i}")
                return False

            link_name = link.get("name")
            if not link_name:
                self.logger.error(f"Link at index {i} has no name attribute")
                return False

        # Validate joint elements
        for i, joint in enumerate(sensor_joints):
            if not isinstance(joint, ET.Element):
                self.logger.error(f"Invalid joint element at index {i}")
                return False

            joint_name = joint.get("name")
            if not joint_name:
                self.logger.error(f"Joint at index {i} has no name attribute")
                return False

        self.logger.debug(
            f"Validated {len(sensor_links)} links and {len(sensor_joints)} joints"
        )
        return True

    def _is_valid_homogeneous_transform(self, transform: np.ndarray) -> bool:
        """
        Validate that a 4x4 matrix is a plausible homogeneous transformation matrix.
        Only warn if not strictly valid, but still return True.

        Args:
            transform: 4x4 transformation matrix to validate

        Returns:
            bool: Always True, but warns if not strictly valid
        """
        try:
            # Check shape
            if transform.shape != (4, 4):
                self.logger.warning("Transform matrix is not 4x4.")
                return False

            # Check bottom row
            expected_bottom_row = np.array([0, 0, 0, 1])
            if not np.allclose(transform[3, :], expected_bottom_row, atol=1e-6):
                self.logger.warning("Transform bottom row is not [0, 0, 0, 1].")

            # Check rotation matrix orthogonality
            rotation_matrix = transform[:3, :3]
            should_be_identity = np.dot(rotation_matrix, rotation_matrix.T)
            if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
                self.logger.warning("Rotation part of transform is not orthogonal.")

            # Check determinant
            if not np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-6):
                self.logger.warning("Rotation matrix determinant is not close to 1.")

            # Always return True, just warn
            return True

        except Exception as e:
            self.logger.warning(f"Transform validation exception: {e}")
            return True

    def _process_and_rename_sensor_elements(
        self,
        sensor_links: list[ET.Element],
        sensor_joints: list[ET.Element],
        sensor_name: str,
    ) -> tuple[list[ET.Element], list[ET.Element]] | None:
        r"""Process and rename sensor link and joint elements to avoid name conflicts.

        Args:
            sensor_links (List[ET.Element]): List of sensor link XML elements.
            sensor_joints (List[ET.Element]): List of sensor joint XML elements.
            sensor_name (str): The sensor's name, used as a prefix.

        Returns:
            Optional[Tuple[List[ET.Element], List[ET.Element]]]: Tuple of processed (links, joints),
            or None if processing fails.
        """
        try:
            processed_links = []
            processed_joints = []
            sensor_prefix = f"{sensor_name}_"
            sensor_name_lower = sensor_name.lower()
            link_name_mapping = {}

            # Process links: add prefix if needed and build mapping
            for link in sensor_links:
                original_name = link.get("name")
                # If the name already contains the sensor name (case-insensitive), do not add prefix
                if sensor_name_lower in original_name.lower():
                    new_name = original_name
                else:
                    new_name = f"{sensor_prefix}{original_name}"
                link_name_mapping[original_name] = new_name
                new_link = copy.deepcopy(link)
                new_link.set("name", new_name)
                processed_links.append(new_link)

            # Process joints: add prefix if needed and update parent/child references
            for joint in sensor_joints:
                original_name = joint.get("name")
                if sensor_name_lower in original_name.lower():
                    new_name = original_name
                else:
                    new_name = f"{sensor_prefix}{original_name}"
                new_joint = copy.deepcopy(joint)
                new_joint.set("name", new_name)

                # Update parent link reference
                parent_elem = new_joint.find("parent")
                if parent_elem is not None:
                    parent_link_name = parent_elem.get("link")
                    parent_elem.set(
                        "link",
                        link_name_mapping.get(parent_link_name, parent_link_name),
                    )
                # Update child link reference
                child_elem = new_joint.find("child")
                if child_elem is not None:
                    child_link_name = child_elem.get("link")
                    child_elem.set(
                        "link", link_name_mapping.get(child_link_name, child_link_name)
                    )

                processed_joints.append(new_joint)

            return processed_links, processed_joints
        except Exception as e:
            self.logger.error(f"Failed to process sensor elements: {str(e)}")
            return None

    def _create_sensor_from_config(self, config: Dict, sensor_name: str) -> ET.Element:
        r"""Create sensor URDF element from configuration dictionary.

        Args:
            config: Configuration dictionary containing sensor specifications
            sensor_name: Name for the generated sensor

        Returns:
            ET.Element: Root element of generated sensor URDF
        """
        # Create root robot element
        robot = ET.Element("robot", name=f"sensor_{sensor_name}")

        # Create main sensor link
        link_name = config.get("link_name", f"{sensor_name}_link")
        link = ET.SubElement(robot, "link", name=link_name)

        # Add visual element if specified
        if "visual" in config:
            visual_config = config["visual"]
            visual = ET.SubElement(link, "visual")

            # Add origin if specified
            if "origin" in visual_config:
                origin_data = visual_config["origin"]
                ET.SubElement(
                    visual,
                    "origin",
                    xyz=origin_data.get("xyz", "0 0 0"),
                    rpy=origin_data.get("rpy", "0 0 0"),
                )

            # Add geometry
            geometry = ET.SubElement(visual, "geometry")
            geom_type = visual_config.get("type", "box")

            if geom_type == "box":
                size = visual_config.get("size", "0.1 0.1 0.1")
                ET.SubElement(geometry, "box", size=size)

            elif geom_type == "cylinder":
                radius = str(visual_config.get("radius", 0.05))
                length = str(visual_config.get("length", 0.1))
                ET.SubElement(geometry, "cylinder", radius=radius, length=length)

            elif geom_type == "sphere":
                radius = str(visual_config.get("radius", 0.05))
                ET.SubElement(geometry, "sphere", radius=radius)

            elif geom_type == "mesh":
                filename = visual_config.get("filename", "")
                if filename:
                    mesh_elem = ET.SubElement(geometry, "mesh", filename=filename)
                    if "scale" in visual_config:
                        mesh_elem.set("scale", visual_config["scale"])

            # Add material/color if specified
            if "color" in visual_config:
                material = ET.SubElement(
                    visual, "material", name=f"{sensor_name}_material"
                )
                ET.SubElement(material, "color", rgba=visual_config["color"])

        # Add collision element if specified
        if "collision" in config:
            collision_config = config["collision"]
            collision = ET.SubElement(link, "collision")

            # Add origin if specified
            if "origin" in collision_config:
                origin_data = collision_config["origin"]
                ET.SubElement(
                    collision,
                    "origin",
                    xyz=origin_data.get("xyz", "0 0 0"),
                    rpy=origin_data.get("rpy", "0 0 0"),
                )

            # Add geometry (similar to visual)
            geometry = ET.SubElement(collision, "geometry")
            geom_type = collision_config.get("type", "box")

            if geom_type == "box":
                size = collision_config.get("size", "0.1 0.1 0.1")
                ET.SubElement(geometry, "box", size=size)

            elif geom_type == "cylinder":
                radius = str(collision_config.get("radius", 0.05))
                length = str(collision_config.get("length", 0.1))
                ET.SubElement(geometry, "cylinder", radius=radius, length=length)

            elif geom_type == "sphere":
                radius = str(collision_config.get("radius", 0.05))
                ET.SubElement(geometry, "sphere", radius=radius)

        # Add inertial properties if specified
        if "inertial" in config:
            inertial_config = config["inertial"]
            inertial = ET.SubElement(link, "inertial")

            # Add origin if specified
            if "origin" in inertial_config:
                origin_data = inertial_config["origin"]
                ET.SubElement(
                    inertial,
                    "origin",
                    xyz=origin_data.get("xyz", "0 0 0"),
                    rpy=origin_data.get("rpy", "0 0 0"),
                )

            # Add mass
            mass_value = str(inertial_config.get("mass", 0.1))
            ET.SubElement(inertial, "mass", value=mass_value)

            # Add inertia tensor
            inertia_elem = ET.SubElement(inertial, "inertia")
            inertia_properties = {
                "ixx": "ixx",
                "iyy": "iyy",
                "izz": "izz",
                "ixy": "ixy",
                "ixz": "ixz",
                "iyz": "iyz",
            }

            for attr, config_key in inertia_properties.items():
                value = str(inertial_config.get(config_key, 0.0))
                inertia_elem.set(attr, value)

        # Add any additional joints if specified in config
        if "joints" in config:
            for joint_config in config["joints"]:
                joint = ET.SubElement(
                    robot,
                    "joint",
                    name=joint_config.get("name", f"{sensor_name}_joint"),
                    type=joint_config.get("type", "fixed"),
                )

                # Add origin
                if "origin" in joint_config:
                    origin_data = joint_config["origin"]
                    ET.SubElement(
                        joint,
                        "origin",
                        xyz=origin_data.get("xyz", "0 0 0"),
                        rpy=origin_data.get("rpy", "0 0 0"),
                    )

                # Add parent and child links
                if "parent" in joint_config:
                    ET.SubElement(joint, "parent", link=joint_config["parent"])
                if "child" in joint_config:
                    ET.SubElement(joint, "child", link=joint_config["child"])

                # Add axis for revolute/prismatic joints
                if (
                    joint_config.get("type") in ["revolute", "prismatic"]
                    and "axis" in joint_config
                ):
                    ET.SubElement(joint, "axis", xyz=joint_config["axis"])

                # Add limits for revolute/prismatic joints
                if "limits" in joint_config:
                    limits_data = joint_config["limits"]
                    limit_elem = ET.SubElement(joint, "limit")
                    for attr in ["lower", "upper", "effort", "velocity"]:
                        if attr in limits_data:
                            limit_elem.set(attr, str(limits_data[attr]))

        self.logger.debug(f"Generated sensor URDF from config for '{sensor_name}'")
        return robot

    def process_sensor_attachments(
        self,
        links: list,
        joints: list,
        base_points: dict,
        existing_link_names: set,
        existing_joint_names: set,
    ):
        r"""Process all attached sensors by adding their link and joint elements to the robot.

        Args:
            links (list): Global list to collect sensor link elements.
            joints (list): Global list to collect sensor joint elements.
            base_points (dict): Mapping from component names to their base link names.
            existing_link_names (set): Set of existing link names to avoid conflicts.
            existing_joint_names (set): Set of existing joint names to avoid conflicts.
        """
        for sensor_name, sensor_data in self.attached_sensors.items():
            try:
                attachment = sensor_data["attachment"]
                sensor_links = sensor_data["links"]
                sensor_joints = sensor_data["joints"]
                sensor_type = sensor_data.get("sensor_type", "unknown")

                self.logger.debug(
                    f"Processing sensor attachment: {sensor_name} ({sensor_type})"
                )

                # Process sensor links: ensure names are lowercase and prefixed
                for link in sensor_links:
                    link_name = link.get("name")
                    if link_name:
                        # Get original and sensor type strings
                        original_name = link_name.lower()
                        sensor_type_str = (
                            str(sensor_type).lower() if sensor_type else ""
                        )
                        # Add prefix only if not already present
                        if sensor_type_str and sensor_type_str not in original_name:
                            formatted_name = f"{original_name}_{sensor_type_str}"
                        else:
                            formatted_name = original_name

                        # Ensure unique link names
                        unique_name = formatted_name
                        count = 1
                        while unique_name in existing_link_names:
                            unique_name = f"{formatted_name}_{count}"
                            self.logger.warning(
                                f"Link name '{unique_name}' already exists. Trying a new name '{unique_name}' with suffix: '{count}'"
                            )
                            formatted_name = unique_name
                            count += 1

                        link.set("name", formatted_name)

                        # Track link names and add to global list
                        existing_link_names.add(formatted_name)
                        links.append(link)

                        # Process meshes for this sensor link
                        self._process_sensor_meshes(
                            link, attachment.sensor_urdf, sensor_name
                        )

                        self.logger.debug(f"Added sensor link: {formatted_name}")

                # Process sensor joints: ensure names are UPPERCASE and follow PARENT_TO_CHILD format
                for joint in sensor_joints:
                    joint_name = joint.get("name")
                    if joint_name:
                        parent_elem = joint.find("parent")
                        child_elem = joint.find("child")
                        parent_link = (
                            parent_elem.get("link").lower()
                            if parent_elem is not None
                            else ""
                        )
                        child_link = (
                            child_elem.get("link").lower()
                            if child_elem is not None
                            else ""
                        )

                        # Format joint name as PARENT_TO_CHILD in uppercase
                        formatted_name = f"{parent_link}_to_{child_link}".upper()
                        joint.set("name", formatted_name)

                        # Ensure parent/child link references are lowercase
                        if parent_elem is not None:
                            parent_elem.set("link", parent_link)
                        if child_elem is not None:
                            child_elem.set("link", child_link)

                        if attachment.transform is not None:
                            transform = attachment.transform
                            xyz = transform[:3, 3]
                            rotation = R.from_matrix(transform[:3, :3])
                            rpy = rotation.as_euler("xyz")

                            origin_elem = joint.find("origin")
                            if origin_elem is None:
                                origin_elem = ET.SubElement(joint, "origin")
                            origin_elem.set("xyz", f"{xyz[0]} {xyz[1]} {xyz[2]}")
                            origin_elem.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")

                            self.logger.info(
                                f"Applied transform to sensor joint {joint.get('name')}: xyz={xyz}, rpy={rpy}"
                            )

                        existing_joint_names.add(formatted_name)
                        joints.append(joint)
                        self.logger.debug(f"Added sensor joint: {formatted_name}")

            except Exception as e:
                self.logger.error(
                    f"Failed to process sensor attachment {sensor_name}: {str(e)}"
                )
                self.logger.debug(f"Traceback: {traceback.format_exc()}")

    def _process_sensor_meshes(
        self, link: ET.Element, base_urdf_path: str, sensor_name: str
    ):
        r"""Process visual and collision meshes for a sensor link.

        Args:
            link (ET.Element): The URDF link element to process.
            base_urdf_path (str): The base path for the URDF files.
            sensor_name (str): The name of the sensor being processed.
        """
        try:
            # Process visual meshes
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        filename = mesh.get("filename")
                        if filename is not None:
                            self.logger.debug(
                                f"Processing sensor visual mesh: {filename}"
                            )
                            new_mesh_filename = self.mesh_manager.copy_and_modify_mesh_file(
                                base_urdf_path,
                                filename,
                                "Visual",
                                f"sensor_{sensor_name}",  # Use sensor prefix for organization
                            )
                            self.logger.debug(
                                f"Updated sensor visual mesh filename: {new_mesh_filename}"
                            )
                            mesh.set("filename", new_mesh_filename)

            # Process collision meshes
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        filename = mesh.get("filename")
                        if filename is not None:
                            self.logger.debug(
                                f"Processing sensor collision mesh: {filename}"
                            )
                            new_mesh_filename = self.mesh_manager.copy_and_modify_mesh_file(
                                base_urdf_path,
                                filename,
                                "Collision",
                                f"sensor_{sensor_name}",  # Use sensor prefix for organization
                            )
                            self.logger.debug(
                                f"Updated sensor collision mesh filename: {new_mesh_filename}"
                            )
                            mesh.set("filename", new_mesh_filename)

        except Exception as e:
            self.logger.error(f"Failed to process meshes for sensor {sensor_name}: {e}")

    def get_attached_sensors(self) -> Dict:
        r"""Get all attached sensors with processed data."""
        return self.attached_sensors

    def convert_to_legacy_format(self) -> Dict:
        r"""Convert processed sensors to legacy attach_dict format for compatibility."""
        legacy_dict = {}
        for sensor_name, sensor_data in self.attached_sensors.items():
            legacy_dict[sensor_name] = sensor_data["attachment"]
        return legacy_dict
