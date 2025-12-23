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
import time
import logging
import numpy as np
from pathlib import Path
from functools import wraps
import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple

from embodichain.toolkits.urdf_assembly.logging_utils import (
    setup_urdf_logging,
)
from embodichain.toolkits.urdf_assembly.signature import (
    URDFAssemblySignatureManager,
)
from embodichain.toolkits.urdf_assembly.component import (
    URDFComponent,
    ComponentRegistry,
    URDFComponentManager,
)
from embodichain.toolkits.urdf_assembly.sensor import (
    SensorAttachment,
    SensorRegistry,
    URDFSensorManager,
)
from embodichain.toolkits.urdf_assembly.connection import (
    URDFConnectionManager,
)
from embodichain.toolkits.urdf_assembly.mesh import URDFMeshManager
from embodichain.toolkits.urdf_assembly.file_writer import (
    URDFFileWriter,
)
from embodichain.toolkits.urdf_assembly.utils import (
    ensure_directory_exists,
)

__all__ = ["URDFAssemblyManager"]


def performance_monitor(func):
    r"""Performance monitoring decorator for tracking function execution time"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            self.logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise

    return wrapper


class URDFAssemblyManager:
    r"""
    A class to manage the assembly of URDF files and their components.
    """

    # Supported wheel types for chassis components
    SUPPORTED_WHEEL_TYPES = [
        "omni",
        "differential",
        "tracked",
    ]

    # Supported robot component types
    SUPPORTED_COMPONENTS = [
        "chassis",
        "legs",
        "torso",
        "head",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "arm",
        "hand",
    ]

    # Supported sensor types for attachment
    SUPPORTED_SENSORS = [
        "camera",
        "lidar",
        "imu",
        "gps",
        "force",
    ]

    # Supported mesh file formats
    SUPPORTED_MESH_TYPES = [
        "stl",
        "obj",
        "ply",
        "dae",
        "glb",
    ]

    def __init__(
        self,
        component_registry: ComponentRegistry = None,
        sensor_registry: SensorRegistry = None,
        mesh_manager: URDFMeshManager = None,
        component_manager: "URDFComponentManager" = None,
        sensor_manager: "URDFSensorManager" = None,
    ):
        self.logger = setup_urdf_logging()

        # Use registries for components and sensors
        self.component_registry = component_registry or ComponentRegistry()
        self.sensor_registry = sensor_registry or SensorRegistry()

        # Initialize mesh manager
        self.mesh_manager = mesh_manager or URDFMeshManager(output_dir=".")

        # Initialize managers for components and sensors
        self.component_manager = component_manager or URDFComponentManager(
            self.mesh_manager
        )
        self.sensor_manager = sensor_manager or URDFSensorManager(self.mesh_manager)

        # Processing order for components with their name prefixes
        # Tuple format: (component_name, prefix)
        self.component_order = [
            ("chassis", None),
            ("legs", None),
            ("torso", None),
            ("head", None),
            ("left_arm", "left_"),
            ("right_arm", "right_"),
            ("left_hand", "left_"),
            ("right_hand", "right_"),
            ("arm", None),
            ("hand", None),
        ]

        # Attachment position indices for component connections.
        # This dictionary defines which link of each component should be used as the connection point
        # when attaching to other components:
        #   0   : use the first link in the component's URDF (typically for child connections)
        #   -1  : use the last link in the component's URDF (typically for parent connections)
        # For example, 'chassis': 0 means the first link of the chassis is used for child attachments;
        # 'torso': -1 means the last link of the torso is used for child attachments, etc.
        self.attach_positions = {
            "chassis": 0,  # Use first link of chassis for child connections
            "legs": -1,  # Use last link of legs for child connections
            "torso": -1,  # Use last link of torso for child connections
            "head": 0,  # Use first link of head for connections
            "left_arm": -1,  # Use last link of left_arm for hand attachment
            "right_arm": -1,  # Use last link of right_arm for hand attachment
            "left_hand": 0,  # Use first link of left_hand for connections
            "right_hand": 0,  # Use first link of right_hand for connections
            "arm": -1,  # Use last link of arm for hand attachment
            "hand": 0,  # Use first link of hand for connections
        }

        # Connection rules defining parent-child relationships between components
        self.connection_rules = [
            ("chassis", "legs"),
            ("legs", "torso"),
            ("chassis", "torso"),
            ("chassis", "left_arm"),
            ("chassis", "right_arm"),
            ("chassis", "arm"),
            ("torso", "head"),
            ("torso", "left_arm"),
            ("torso", "right_arm"),
            ("torso", "arm"),
            ("left_arm", "left_hand"),
            ("right_arm", "right_hand"),
            ("arm", "hand"),
        ]

        # Configure logging
        logging.basicConfig(level=logging.INFO)

        # Name of the base link for the robot
        self.base_link_name = "base_link"

        # Initialize the URDF file writer for output formatting
        self.file_writer = URDFFileWriter()

        # Initialize signature manager instead of cache manager
        self.signature_manager = URDFAssemblySignatureManager()

    def add_component(
        self,
        component_type: str,
        urdf_path: Union[str, Path],
        transform: np.ndarray | None = None,
        **params,
    ) -> bool:
        r"""Add a URDF component to the component registry.

        This method creates a URDFComponent object and registers it in the component registry.

        Args:
            component_type (str): The type/name of the component (e.g., 'chassis', 'head').
            urdf_path (str or Path): Path to the URDF file for this component.
            transform (np.ndarray, optional): 4x4 transformation matrix for positioning the component.
            **params: Additional component-specific parameters (e.g., wheel_type for chassis).

        Returns:
            bool: True if component added successfully, False otherwise.
        """
        try:
            if not isinstance(component_type, str):
                raise ValueError("component_type must be a string")
            if not isinstance(urdf_path, (str, Path)):
                raise ValueError("urdf_path must be a string or Path")
            if component_type not in self.SUPPORTED_COMPONENTS:
                raise ValueError(
                    f"Unsupported component_type: {component_type}. "
                    f"Supported types: {self.SUPPORTED_COMPONENTS}"
                )

            component = URDFComponent(
                urdf_path=urdf_path, params=params, transform=transform
            )
            self.component_registry.add(component_type, component)
            self.logger.info(
                f"Added component: [{component_type}], URDF: ({urdf_path})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to add component [{component_type}]: {e}")
            return False

    def attach_sensor(
        self,
        sensor_name: str,
        sensor_source,
        parent_component: str,
        parent_link: str,
        transform: np.ndarray | None = None,
        **kwargs,
    ) -> bool:
        r"""Attach a sensor to a specific component and link, and register it in the sensor registry.

        This method creates a SensorAttachment object and registers it in the sensor registry.

        Args:
            sensor_name (str): Unique name for the sensor (e.g., 'camera').
            sensor_source (str or ET.Element): Path to the sensor's URDF file or an XML element.
            parent_component (str): Name of the component to which the sensor is attached.
            parent_link (str): Name of the link within the parent component for attachment.
            **kwargs: Additional keyword arguments (e.g., transform, sensor_type).

        Returns:
            bool: True if sensor attached successfully, False otherwise.
        """
        try:
            sensor = SensorAttachment(
                sensor_urdf=sensor_source,
                parent_component=parent_component,
                parent_link=parent_link,
                transform=transform,
                **kwargs,
            )
            self.sensor_registry.add(sensor_name, sensor)
            urdf_info = (
                f"\n\tURDF: ({sensor.sensor_urdf})"
                if sensor.sensor_urdf
                else ", URDF: [N/A]"
            )
            self.logger.info(
                f"Attached sensor: [{sensor_name}] "
                f"to [{parent_component}] at link [{parent_link}]{urdf_info}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to attach sensor [{sensor_name}]: {e}")
            return False

    def get_component(self, component_type: str):
        r"""Retrieve a component from the registry by its type/name.

        Args:
            component_type (str): The type/name of the component to retrieve.

        Returns:
            URDFComponent or None: The registered component object, or None if not found.
        """
        return self.component_registry.get(component_type)

    def get_attached_sensors(self):
        r"""Get all attached sensors from the sensor registry.

        Returns:
            dict: A dictionary mapping sensor names to SensorAttachment objects.
        """
        return self.sensor_registry.all()

    def _load_urdf(self, urdf_path: str) -> ET.Element | None:
        r"""Load a URDF file and return its root element.

        Args:
            urdf_path (str): Path to the URDF file.

        Returns:
            ET.Element: The root element of the parsed URDF XML.
        """
        try:
            tree = ET.parse(urdf_path)
            return tree.getroot()
        except Exception as e:
            self.logger.error(f"Failed to load URDF {urdf_path}: {e}")
            return None

    def _apply_transformation(
        self, urdf: ET.Element, transform: np.ndarray, link_name: str
    ):
        r"""Applies a transformation matrix to the 'xyz' attributes of the origins of the specified link and its first joint in the URDF.

        Args:
            urdf (ET.Element): The root element of the URDF to transform.
            transform (np.ndarray): A 4x4 transformation matrix to apply.
            link_name (str): The name of the link to apply the transformation to.
        """
        # Now handle the first joint connected to this link
        for joint in urdf.findall("joint"):
            origin = joint.find("origin")
            if origin is not None:
                # Check if the joint connects to the specified link
                child_link = joint.find("child").get("link")
                if child_link == link_name:
                    xyz = np.array([float(val) for val in origin.get("xyz").split()])
                    transformed_xyz = np.dot(transform[:3, :3], xyz) + transform[:3, 3]
                    origin.set("xyz", " ".join(map(str, transformed_xyz)))

                    # Apply transformation to rpy
                    if "rpy" in origin.attrib:
                        rpy = np.array(
                            [float(val) for val in origin.get("rpy").split()]
                        )
                        rotation = R.from_euler("xyz", rpy)
                        transformed_rotation = (
                            R.from_matrix(transform[:3, :3]) * rotation
                        )
                        transformed_rpy = transformed_rotation.as_euler("xyz")
                        origin.set("rpy", " ".join(map(str, transformed_rpy)))
                    elif "quat" in origin.attrib:
                        quat = np.array(
                            [float(val) for val in origin.get("quat").split()]
                        )
                        rotation = R.from_euler("xyz", quat)
                        transformed_rotation = (
                            R.from_matrix(transform[:3, :3]) * rotation
                        )
                        transformed_rpy = transformed_rotation.as_euler("xyz")
                        origin.set("rpy", " ".join(map(str, transformed_rpy)))

                    break  # Stop after processing the first joint

    def _create_base_link(self) -> ET.Element:
        r"""Creates a base link and returns it.

        Returns:
            ET.Element: The base link element.
        """
        base_link = ET.Element("link", name=self.base_link_name)

        return base_link

    def _validate_urdf_file(self, urdf_path: str) -> bool:
        r"""Validate URDF file integrity and format compliance

        Args:
            urdf_path (str): Path to the URDF file to validate

        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(urdf_path):
                self.logger.error(f"URDF file not found: {urdf_path}")
                return False

            # Check file size to ensure it's not empty
            if os.path.getsize(urdf_path) == 0:
                self.logger.error(f"URDF file is empty: {urdf_path}")
                return False

            # Attempt to parse XML structure
            root = ET.parse(urdf_path).getroot()
            if root.tag != "robot":
                self.logger.error(f"Invalid URDF root element: {root.tag}")
                return False

            # Check for presence of basic link elements
            if not root.findall("link"):
                self.logger.error(f"No links found in URDF: {urdf_path}")
                return False

            # Check robot name attribute
            robot_name = root.get("name")
            if not robot_name:
                self.logger.warning(f"URDF robot has no name attribute: {urdf_path}")

            self.logger.debug(f"URDF file validation passed: {urdf_path}")
            return True

        except ET.ParseError as e:
            self.logger.error(f"XML parse error in {urdf_path}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Validation error for {urdf_path}: {e}")
            return False

    def _generate_connection_rules(self) -> list:
        r"""Dynamically generate connection rules based on available components.

        Returns:
            list: A list of (parent, child) tuples specifying connection relationships.
        """
        connection_rules = []

        # Filter components that exist in urdf_dict
        existing_components = [
            comp
            for comp in self.SUPPORTED_COMPONENTS
            if self.component_registry.get(comp)
        ]

        self.logger.debug(f"Existing components: {existing_components}")

        # Define explicit connection rules - only meaningful relationships
        # Rule 1: chassis connects to torso (if both exist)
        if "chassis" in existing_components and "legs" in existing_components:
            connection_rules.append(("chassis", "legs"))
            if "torso" in existing_components:
                connection_rules.append(("legs", "torso"))
        elif "chassis" in existing_components and "torso" in existing_components:
            # If there are no legs, chassis connects directly to torso
            connection_rules.append(("chassis", "torso"))

        # Rule 2: torso connects to head (if both exist)
        if "torso" in existing_components and "head" in existing_components:
            connection_rules.append(("torso", "head"))

        # Rule 3: torso connects to arms (if they exist)
        if "torso" in existing_components:
            if "left_arm" in existing_components:
                connection_rules.append(("torso", "left_arm"))
            if "right_arm" in existing_components:
                connection_rules.append(("torso", "right_arm"))
            if "arm" in existing_components:
                connection_rules.append(("torso", "arm"))

        # Rule 4: arms connect to hands (if both exist)
        if "left_arm" in existing_components and "left_hand" in existing_components:
            connection_rules.append(("left_arm", "left_hand"))
        if "right_arm" in existing_components and "right_hand" in existing_components:
            connection_rules.append(("right_arm", "right_hand"))

        # Rule 5: single arm connects to hand
        if "arm" in existing_components and "hand" in existing_components:
            connection_rules.append(("arm", "hand"))

        # Rule 6: If no torso, chassis can directly connect to head and arms
        if "chassis" in existing_components and "torso" not in existing_components:
            if "head" in existing_components:
                connection_rules.append(("chassis", "head"))
            if "left_arm" in existing_components:
                connection_rules.append(("chassis", "left_arm"))
            if "right_arm" in existing_components:
                connection_rules.append(("chassis", "right_arm"))
            # Connect single arm directly to chassis (no torso scenario)
            if "arm" in existing_components:
                connection_rules.append(("chassis", "arm"))

        connection_rules = list(set(connection_rules))

        self.logger.info(
            f"Generated connection rules: {connection_rules}, total {len(connection_rules)} rules"
        )

        return connection_rules

    def _find_end_link(
        self, component: str, base_points: dict, joints: list
    ) -> Union[str, None]:
        """Find the end link of a component by traversing the joint chain downward.

        Args:
            component (str): Component name to find the end link for.
            base_points (dict): Mapping from component names to their base link names.
            joints (list): List of joint elements to traverse.

        Returns:
            Union[str, None]: Name of the end link, or None if component not found.
        """
        current_link = base_points.get(component)
        if not current_link:
            return None

        visited_links = set()  # Prevent infinite loops in joint chains
        while True:
            visited_links.add(current_link)
            found = False
            for joint in joints:
                if hasattr(joint, "find"):  # Ensure it's an XML element, not a comment
                    parent = joint.find("parent")
                    child = joint.find("child")
                    if parent is not None and parent.get("link") == current_link:
                        if child is not None:
                            next_link = child.get("link")
                            if next_link not in visited_links:  # Avoid revisiting links
                                current_link = next_link
                                found = True
                                break
            if not found:
                break  # No further links found in the chain
        return current_link

    @performance_monitor
    def merge_urdfs(
        self,
        output_path: str = "./assembly_robot.urdf",
        use_signature_check: bool = True,
    ) -> ET.Element:
        """Merge URDF files according to single base link, connection point naming,
        and type compatibility matrix rules.

        Args:
            output_path (str): Path where the merged URDF file will be saved.
            use_signature_check (bool): Whether to check signatures to avoid redundant processing.

        Returns:
            ET.Element: The root element of the merged URDF.
        """
        output_path = os.path.abspath(output_path)
        assembly_signature = None

        # Log components to be merged
        available_components = [
            comp
            for comp, obj in self.component_registry.all().items()
            if obj is not None
        ]
        self.logger.info(f"🔧 Preparing to merge components: {available_components}")

        for comp in available_components:
            comp_obj = self.component_registry.get(comp)
            self.logger.info(f"  [{comp}]: {comp_obj.urdf_path}")
            if comp_obj.params:
                self.logger.debug(f"    Parameters: {comp_obj.params}")
            if comp_obj.transform is not None:
                self.logger.debug(f"    Transform: applied")

        if use_signature_check:
            # Calculate current assembly signature
            assembly_signature = self.signature_manager.calculate_assembly_signature(
                self.component_registry.all(), output_path
            )

            self.logger.info(f"Current assembly signature: [{assembly_signature}]")
            self.logger.debug(f"Target output path: ({output_path})")

            # Check if assembly is up-to-date
            if self.signature_manager.is_assembly_up_to_date(
                assembly_signature, output_path
            ):
                self.logger.info(
                    f"✅ URDF assembly is up-to-date: ({output_path}), skipping rebuild."
                )
                return ET.parse(output_path).getroot()
            else:
                self.logger.info(
                    "Assembly configuration has changed or file doesn't exist, rebuilding..."
                )

        # Perform normal assembly process
        self.logger.info("🔄 Building new URDF assembly...")

        # 1. Generate standard header with module information
        module_names = [
            os.path.splitext(os.path.basename(obj.urdf_path))[0]
            for comp, obj in self.component_registry.all().items()
            if obj
        ]

        robot_name = os.path.splitext(os.path.basename(output_path))[0]
        merged_urdf = ET.Element("robot", name=robot_name)

        # 2. Create single base link for the entire robot
        base_link = ET.Element("link", name=self.base_link_name)
        # Store links and joints separately for proper ordering
        links = [base_link]
        joints = []

        # Mapping tables for component processing
        name_mapping = {}  # Maps (component, original_name) to new_name
        base_points = {}  # Maps component to its base connection link
        parent_attach_points = {}  # Maps component to its parent connection link

        # Initialize managers for mesh handling and component processing
        output_dir = os.path.dirname(output_path) or "."
        ensure_directory_exists(output_dir, self.logger)
        mesh_manager = URDFMeshManager(output_dir)
        mesh_manager.ensure_dirs()
        component_manager = URDFComponentManager(mesh_manager)
        connection_manager = URDFConnectionManager(self.base_link_name)

        # Initialize sensor manager with mesh_manager
        sensor_manager = URDFSensorManager(mesh_manager)

        # Process any pending enhanced sensors
        if hasattr(self, "_pending_sensors"):
            for sensor_name, params in self._pending_sensors.items():
                success = sensor_manager.attach_sensor(
                    sensor_name=sensor_name, **params
                )
                if success:
                    # Sync to legacy attach_dict for backward compatibility
                    self.attach_dict.update(sensor_manager.convert_to_legacy_format())

        # 3. Process all components in defined order
        connection_rules = self._generate_connection_rules()

        # Collect component transforms for connection joints
        component_transforms = {}
        for comp, comp_obj in self.component_registry.all().items():
            if comp_obj and comp_obj.transform is not None:
                component_transforms[comp] = comp_obj.transform

        for comp, prefix in self.component_order:
            comp_obj = self.component_registry.get(comp)
            if not comp_obj:
                continue

            # Add section comments using file writer
            self.file_writer.add_section_comments(links, comp, "Links")
            self.file_writer.add_section_comments(joints, comp, "Joints")

            # Parse component URDF to analyze its structure
            urdf_root = ET.parse(comp_obj.urdf_path).getroot()

            # Determine parent component and attachment point for current component
            parent_component = None
            parent_attach_link = None

            # Find parent component based on connection rules
            for parent, child in connection_rules:
                if child == comp and parent in base_points:
                    parent_component = parent
                    # Use base connection point for chassis
                    if parent == "chassis":
                        parent_attach_link = base_points[parent]
                    else:
                        # For other components, find their end link
                        parent_attach_link = self._find_end_link(
                            parent, base_points, joints
                        )
                    break

            if parent_component and parent_attach_link:
                self.logger.debug(
                    f"Component [{comp}] will connect to parent [{parent_component}] at link: ({parent_attach_link})"
                )
            else:
                self.logger.debug(
                    f"Component [{comp}] has no parent component (likely chassis or standalone)"
                )

            # Process the component using the component manager
            component_manager.process_component(
                comp, prefix, comp_obj, name_mapping, base_points, links, joints
            )

            # Determine attachment positions for current component
            original_links = urdf_root.findall("link")

            if original_links:
                # Set base connection point (always first link for child connections)
                first_original_name = original_links[0].get("name")
                first_mapped_name = name_mapping.get(
                    (comp, first_original_name), first_original_name
                )
                base_points[comp] = first_mapped_name

                self.logger.debug(
                    f"Set base_points[{comp}] = ({first_mapped_name}) .first link for child connection, original: ({first_original_name})"
                )

                # Set parent connection point based on attach_positions configuration
                index = self.attach_positions.get(comp, 0)
                try:
                    if 0 <= index < len(original_links):
                        original_attach_name = original_links[index].get("name")
                    elif index == -1 and original_links:
                        original_attach_name = original_links[-1].get("name")
                    else:
                        original_attach_name = (
                            original_links[0].get("name") if original_links else None
                        )

                    # Find mapped name for the attachment point
                    mapped_attach_name = name_mapping.get(
                        (comp, original_attach_name), original_attach_name
                    )
                    parent_attach_points[comp] = mapped_attach_name

                    self.logger.debug(
                        f"Set parent_attach_points[{comp}] = ({mapped_attach_name}). Index {index} for parent connection, original: ({original_attach_name})"
                    )
                except IndexError:
                    # Fall back to first link if index is out of range
                    parent_attach_points[comp] = first_mapped_name
                    self.logger.warning(
                        f"Index {index} out of range for component {comp}, using first link: {first_mapped_name}"
                    )

            # Add section end comments using file writer
            self.file_writer.add_section_end_comments(links, comp, "Links")
            self.file_writer.add_section_end_comments(joints, comp, "Joints")

        # 4. Create connection joints between components using transforms
        connection_manager.add_connections(
            joints,
            base_points,
            parent_attach_points,
            connection_rules,
            component_transforms,
        )

        # Track existing names for sensor processing
        existing_link_names = {
            link.get("name").lower() for link in links if link.get("name")
        }
        existing_joint_names = {
            joint.get("name").upper() for joint in joints if joint.get("name")
        }

        # 5. Process sensor attachments using the new sensor manager
        for sensor_name, sensor_attach in self.sensor_registry.all().items():
            sensor_manager.attach_sensor(
                sensor_name=sensor_name,
                sensor_source=sensor_attach.sensor_urdf,
                parent_component=sensor_attach.parent_component,
                parent_link=sensor_attach.parent_link,
                transform=sensor_attach.transform,
            )

        sensor_manager.process_sensor_attachments(
            links, joints, base_points, existing_link_names, existing_joint_names
        )

        # 6. Add all links and joints to merged URDF in proper order
        for link in links:
            merged_urdf.append(link)
        for joint in joints:
            merged_urdf.append(joint)

        # 7. Write the final URDF file with proper formatting, header and signature
        if use_signature_check and assembly_signature:
            self.file_writer.write_urdf(
                merged_urdf, output_path, module_names, assembly_signature
            )
            self.logger.info(
                f"✅ URDF assembly written with signature: {assembly_signature}"
            )
        else:
            self.file_writer.write_urdf(merged_urdf, output_path, module_names)
            self.logger.info("✅ URDF assembly written without signature.")
        return merged_urdf
