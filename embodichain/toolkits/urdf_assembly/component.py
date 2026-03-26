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

import traceback
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Dict

from embodichain.toolkits.urdf_assembly.logging_utils import (
    URDFAssemblyLogger,
)
from embodichain.toolkits.urdf_assembly.mesh import URDFMeshManager


__all__ = ["ComponentRegistry", "URDFComponent", "URDFComponentManager"]


class ComponentRegistry:
    r"""Registry for storing and retrieving URDFComponent objects."""

    def __init__(self):
        self._components = {}

    def add(self, component_type: str, component_obj):
        self._components[component_type] = component_obj

    def get(self, component_type: str):
        return self._components.get(component_type)

    def all(self):
        return self._components

    def remove(self, component_type: str):
        if component_type in self._components:
            del self._components[component_type]


@dataclass
class URDFComponent:
    r"""Represents a URDF component with its configuration and transformation.

    This dataclass encapsulates all the information needed to process and integrate
    a URDF component into the robot assembly, including file path, attachment
    configuration, parameters, and optional spatial transformation.
    """

    urdf_path: str  # Path to the URDF file for this component
    default_attach_link: str = (
        "base_link"  # Default link name for attachment (usually the first link)
    )
    params: Dict = None  # Component-specific parameters (e.g., wheel_type for chassis)
    transform: np.ndarray | None = (
        None  # Optional 4x4 transformation matrix for positioning
    )

    def __post_init__(self):
        # Convert path to Path object for better path handling
        if isinstance(self.urdf_path, str):
            self.urdf_path = Path(self.urdf_path)

        # Validate transformation matrix dimensions and type
        if self.transform is not None:
            if not isinstance(self.transform, np.ndarray) or self.transform.shape != (
                4,
                4,
            ):
                raise ValueError("Transform must be a 4x4 numpy array")


class URDFComponentManager:
    """Responsible for loading, renaming, and processing meshes for a single component.

    This manager normalizes link and joint names according to a configurable
    case policy so that the overall assembly naming scheme can be controlled
    centrally (e.g. all links lowercase, all joints uppercase).
    """

    def __init__(
        self,
        mesh_manager: URDFMeshManager,
        name_case: dict[str, str] | None = None,
    ):
        """Create a component manager.

        Args:
            mesh_manager (URDFMeshManager): Mesh manager used for copying and
                rewriting mesh references.
            name_case (dict[str, str] | None): Optional mapping controlling
                how joint and link names are normalized. Supported keys are
                ``"joint"`` and ``"link"`` with values ``"upper``,
                ``"lower"`` or ``"none"``. When omitted, joints are
                uppercased and links are lowercased (the previous default
                behavior).
        """

        self.mesh_manager = mesh_manager
        self.logger = URDFAssemblyLogger.get_logger("component_manager")

        # By default, mimic the legacy behavior: links are lowercase, joints
        # are uppercase. This should stay in sync conceptually with
        # URDFConnectionManager.
        self._name_case: dict[str, str] = {
            "joint": "upper",
            "link": "lower",
        }
        if name_case is not None:
            for key, mode in name_case.items():
                if key in self._name_case and mode in {"upper", "lower", "none"}:
                    self._name_case[key] = mode
                else:
                    self.logger.warning(
                        "Ignoring invalid name_case entry %r=%r (allowed keys: 'joint', 'link'; "
                        "allowed modes: 'upper', 'lower', 'none')",
                        key,
                        mode,
                    )

    @property
    def name_case(self) -> dict[str, str]:
        """Get the current name case policy.

        Returns:
            dict[str, str]: A mapping of ``"joint"`` / ``"link"`` to their
            case modes.
        """
        return self._name_case

    @name_case.setter
    def name_case(self, new_name_case: dict[str, str]) -> None:
        """Replace the name case policy.

        Invalid entries are silently ignored (logged as warnings) to stay
        consistent with the constructor behaviour. This intentionally differs
        from :class:`URDFAssemblyManager`, which raises ``ValueError`` for
        bad inputs, because this low-level manager is designed to be lenient
        so it can be used in contexts where the caller cannot easily validate
        inputs upfront.

        Args:
            new_name_case (dict[str, str]): Mapping of ``"joint"`` / ``"link"``
                to ``"upper"``, ``"lower"``, or ``"none"``.
        """
        updated: dict[str, str] = {}
        for key, mode in new_name_case.items():
            if key in {"joint", "link"} and mode in {"upper", "lower", "none"}:
                updated[key] = mode
            else:
                self.logger.warning(
                    "Ignoring invalid name_case entry %r=%r (allowed keys: 'joint', 'link'; "
                    "allowed modes: 'upper', 'lower', 'none')",
                    key,
                    mode,
                )
        self._name_case.update(updated)

    def _apply_case(self, kind: str, name: str | None) -> str | None:
        """Normalize a name according to the configured case policy.

        Args:
            kind (str): One of ``"joint"`` or ``"link"``.
            name (str | None): The original name.

        Returns:
            str | None: The normalized name, or the original value if
            ``kind`` is unknown or mode is ``"none"``.
        """

        if name is None:
            return None

        mode = self._name_case.get(kind, "none")
        if mode == "lower":
            return name.lower()
        if mode == "upper":
            return name.upper()
        return name

    def process_component(
        self,
        comp: str,
        prefix: str,
        comp_obj,
        name_mapping: dict,
        base_points: dict,
        links: list,
        joints: list,
    ):
        r"""Process a single URDF component by renaming elements and handling meshes.

        Args:
            comp (str): Component name (e.g., 'chassis', 'left_arm', 'hand').
            prefix (str): Prefix to add to component elements for uniqueness (e.g., 'left_').
                        None means no prefix will be applied.
            comp_obj: URDFComponent object containing the component's URDF path and parameters.
            name_mapping (dict): Dictionary mapping (component, original_name) tuples to new names.
                            Used for resolving cross-references between components.
            base_points (dict): Dictionary mapping component names to their base connection link names.
                            Used for establishing parent-child relationships.
            links (list): Global list to collect all processed link elements from all components.
            joints (list): Global list to collect all processed joint elements from all components.
        """

        try:
            urdf_root = ET.parse(comp_obj.urdf_path).getroot()

            # Safe way to get link and joint names, handling None values
            global_link_names = {
                self._apply_case("link", link.get("name"))
                for link in links
                if link.get("name") is not None
            }
            global_joint_names = {
                self._apply_case("joint", joint.get("name"))
                for joint in joints
                if joint.get("name") is not None
            }

            first_link_flag = True
            joint_name_mapping = {}

            # Process links first
            for link in urdf_root.findall("link"):
                orig_name = link.get("name")
                if orig_name is None:
                    self.logger.warning(
                        f"Found link with no name in component {comp}, skipping"
                    )
                    continue

                # Generate unique name
                if prefix:
                    new_name = self._generate_unique_name(
                        orig_name, prefix, global_link_names
                    )
                else:
                    # For components without prefix, ensure names are unique
                    normalized_orig = self._apply_case("link", orig_name)
                    if normalized_orig in global_link_names:
                        new_name = self._apply_case("link", f"{comp}_{orig_name}")
                    else:
                        new_name = normalized_orig

                global_link_names.add(new_name)

                # Set first link as base point
                if first_link_flag:
                    base_points[comp] = new_name
                    first_link_flag = False

                # Update link name mapping and set link name according to policy
                name_mapping[(comp, orig_name)] = new_name
                link.set("name", new_name)
                links.append(link)

                self._process_meshes(link, comp_obj.urdf_path, comp)

            # Process joints: Build mapping table AND process all at once
            joints_to_process = []

            # First collect all joints and build complete mapping
            for joint in urdf_root.findall("joint"):
                orig_joint_name = joint.get("name")
                if orig_joint_name is None:
                    continue

                new_joint_name = self._apply_case(
                    "joint",
                    self._generate_unique_name(
                        orig_joint_name, prefix, global_joint_names
                    ),
                )
                global_joint_names.add(new_joint_name)

                # Build the complete mapping table
                joint_name_mapping[orig_joint_name] = new_joint_name
                joints_to_process.append((joint, orig_joint_name, new_joint_name))

            self.logger.debug(f"Joint name mapping for [{comp}]: {joint_name_mapping}")

            # Now process all joints with complete mapping available
            for joint, orig_joint_name, new_joint_name in joints_to_process:
                # Set the new joint name
                joint.set("name", new_joint_name)

                # Update parent and child links with case normalization - with None checks
                parent_elem = joint.find("parent")
                child_elem = joint.find("child")

                if parent_elem is not None:
                    parent = parent_elem.get("link")
                    if parent is not None:
                        new_parent_name = self._apply_case(
                            "link", name_mapping.get((comp, parent), parent)
                        )
                        parent_elem.set("link", new_parent_name)
                    else:
                        self.logger.warning(
                            f"Found parent element with no link attribute in joint {orig_joint_name}"
                        )

                if child_elem is not None:
                    child = child_elem.get("link")
                    if child is not None:
                        new_child_name = self._apply_case(
                            "link", name_mapping.get((comp, child), child)
                        )
                        child_elem.set("link", new_child_name)
                    else:
                        self.logger.warning(
                            f"Found child element with no link attribute in joint {orig_joint_name}"
                        )

                # Process mimic joint references using the complete mapping table
                mimic_elem = joint.find("mimic")
                if mimic_elem is not None:
                    mimic_joint = mimic_elem.get("joint")
                    if mimic_joint is not None:
                        self.logger.info(
                            f"Processing mimic joint reference: ({mimic_joint}) in joint ({orig_joint_name})"
                        )
                        # Look up the corresponding new joint name in the mapping table
                        new_mimic_joint = joint_name_mapping.get(mimic_joint)
                        if new_mimic_joint:
                            # Update the mimic element to reference the renamed joint
                            mimic_elem.set("joint", new_mimic_joint)
                            self.logger.info(
                                f"✓ Updated mimic joint reference: ({mimic_joint}) -> ({new_mimic_joint})"
                            )
                        else:
                            self.logger.warning(
                                f"✗ Could not find mapping for mimic joint: ({mimic_joint})"
                            )
                            self.logger.warning(
                                f"Available mappings: {list(joint_name_mapping.keys())}"
                            )

                joints.append(joint)

            self.logger.debug(
                f"Processed component: [{comp}], links: {len(urdf_root.findall('link'))}, joints: {len(urdf_root.findall('joint'))}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to process component [{comp}]: {e}", exc_info=True
            )
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _generate_unique_name(
        self, orig_name: str, prefix: str, existing_names: set
    ) -> str:
        r"""Generate a unique name by adding a prefix and ensuring no conflicts.

        Args:
            orig_name (str): The original name to modify.
            prefix (str): The prefix to add to the name.
            existing_names (set): A set of existing names to check for conflicts.

        Returns:
            str: A unique name derived from the original name.
        """
        if orig_name is None:
            orig_name = "unnamed"

        # For uniqueness checks we always operate on a normalized form that is
        # consistent with the link case policy. This keeps collisions and
        # generated names aligned with how names are written back to the URDF.
        base_name = orig_name
        if prefix and not orig_name.lower().startswith(prefix.lower()):
            base_name = f"{prefix}{orig_name}"

        new_name = base_name

        # Ensure the new name is unique
        if new_name in existing_names:
            counter = 1
            unique_name = f"{new_name}_{counter}"
            while unique_name in existing_names:
                counter += 1
                unique_name = f"{new_name}_{counter}"
            new_name = unique_name

        return new_name

    def _process_meshes(self, link: ET.Element, base_urdf_path: str, comp_name: str):
        r"""Process visual and collision meshes for a link.

        Args:
            link (ET.Element): The URDF link element to process.
            base_urdf_path (str): The base path for the URDF files.
            comp_name (str): The name of the component being processed.
        """
        try:
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        filename = mesh.get("filename")
                        if filename is not None:
                            self.logger.debug(f"Processing visual mesh: {filename}")
                            new_mesh_filename = (
                                self.mesh_manager.copy_and_modify_mesh_file(
                                    base_urdf_path,
                                    filename,
                                    "Visual",
                                    comp_name,
                                )
                            )
                            self.logger.debug(
                                f"Updated visual mesh filename: {new_mesh_filename}"
                            )
                            mesh.set("filename", new_mesh_filename)

            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        filename = mesh.get("filename")
                        if filename is not None:
                            self.logger.debug(f"Processing collision mesh: {filename}")
                            new_mesh_filename = (
                                self.mesh_manager.copy_and_modify_mesh_file(
                                    base_urdf_path,
                                    filename,
                                    "Collision",
                                    comp_name,
                                )
                            )
                            self.logger.debug(
                                f"Updated collision mesh filename: {new_mesh_filename}"
                            )
                            mesh.set("filename", new_mesh_filename)
        except Exception as e:
            self.logger.error(
                f"Failed to process meshes for component {comp_name}: {e}"
            )
