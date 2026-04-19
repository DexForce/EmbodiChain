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

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from scipy.spatial.transform import Rotation as R

from embodichain.toolkits.urdf_assembly.logging_utils import URDFAssemblyLogger
from embodichain.toolkits.urdf_assembly.name_normalizer import NameNormalizer

__all__ = ["URDFConnectionManager"]


class URDFConnectionManager:
    r"""Responsible for managing connection rules between components and sensor attachments."""

    _DEFAULT_ORIGIN = {"xyz": "0 0 0", "rpy": "0 0 0"}

    def __init__(self, base_link_name: str, name_case: dict[str, str] | None = None):
        """Initialize the URDFConnectionManager.

        Args:
            base_link_name: The name of the base link to which the chassis or other
                components may be attached.
            name_case: Optional mapping controlling how joint and link names are
                normalized. Supported keys are ``"joint"`` and ``"link"`` with
                values ``"upper"``, ``"lower"`` or ``"none"``.

                When omitted, joints are uppercased and links are lowercased (the
                previous default behavior).
        """
        self.base_link_name = base_link_name
        self.logger = URDFAssemblyLogger.get_logger("connection_manager")
        self.name_normalizer = NameNormalizer(name_case)

    def _apply_case(self, kind: str, name: str | None) -> str | None:
        """Normalize a name using the NameNormalizer."""
        return self.name_normalizer.normalize(kind, name)

    @staticmethod
    def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
        """Read attribute from object or key from dict."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _format_scalar(value: Any) -> str:
        """Format scalar values for URDF attribute strings."""
        try:
            f = float(value)
        except Exception:
            return "0"

        # Keep strings stable and compact (avoid long repr / numpy scalars).
        s = f"{f:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _format_vec3(self, vec3: Any) -> str:
        """Format a 3D vector as URDF 'x y z' string."""
        try:
            x, y, z = vec3[0], vec3[1], vec3[2]
        except Exception:
            return "0 0 0"
        return f"{self._format_scalar(x)} {self._format_scalar(y)} {self._format_scalar(z)}"

    def _origin_kwargs_from_transform(self, transform: Any | None) -> dict[str, str]:
        """Convert a 4x4 transform matrix to URDF origin attributes."""
        if transform is None:
            return dict(self._DEFAULT_ORIGIN)

        try:
            xyz = transform[:3, 3]
            rotation = R.from_matrix(transform[:3, :3])
            rpy = rotation.as_euler("xyz")
        except Exception as exc:
            self.logger.warning(f"Invalid transform, fallback to identity: {exc}")
            return dict(self._DEFAULT_ORIGIN)

        return {"xyz": self._format_vec3(xyz), "rpy": self._format_vec3(rpy)}

    @staticmethod
    def _make_unique(base: str, existing: set[str]) -> str:
        """Make a unique name by appending suffixes when needed."""
        if base not in existing:
            return base
        idx = 1
        while f"{base}_{idx}" in existing:
            idx += 1
        return f"{base}_{idx}"

    def _collect_existing_joint_names(self, joints: list) -> set[str]:
        names: set[str] = set()
        for joint in joints:
            if not hasattr(joint, "get"):
                continue
            raw = joint.get("name")
            if not raw:
                continue
            normalized = self._apply_case("joint", raw)
            if normalized:
                names.add(normalized)
        return names

    def _append_fixed_joint(
        self,
        joints: list,
        existing_joint_names: set[str],
        joint_name: str,
        parent_link: str,
        child_link: str,
        origin_kwargs: dict[str, str] | None = None,
    ) -> None:
        """Append a fixed joint if it doesn't already exist."""
        normalized_joint_name = self._apply_case("joint", joint_name)
        if not normalized_joint_name:
            self.logger.error(f"Empty joint name for joint_name={joint_name!r}")
            return

        if normalized_joint_name in existing_joint_names:
            self.logger.warning(f"Duplicate joint: {normalized_joint_name}")
            return

        joint = ET.Element("joint", name=normalized_joint_name, type="fixed")
        ET.SubElement(joint, "origin", **(origin_kwargs or dict(self._DEFAULT_ORIGIN)))
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=child_link)

        joints.append(joint)
        existing_joint_names.add(normalized_joint_name)

    def _normalize_link_or_none(self, link_name: str | None) -> str | None:
        if not link_name:
            return None
        return self._apply_case("link", link_name)

    def _connect_chassis_to_base(
        self,
        joints: list,
        base_points: dict,
        existing_joint_names: set[str],
        chassis_component: str,
    ) -> bool:
        if chassis_component not in base_points:
            return False

        chassis_first_link = self._normalize_link_or_none(
            base_points.get(chassis_component)
        )
        if not chassis_first_link:
            self.logger.error("Invalid chassis base link (None)")
            return True

        self._append_fixed_joint(
            joints=joints,
            existing_joint_names=existing_joint_names,
            joint_name=f"BASE_LINK_TO_{chassis_component}_CONNECTOR",
            parent_link=self.base_link_name,
            child_link=chassis_first_link,
        )
        self.logger.info(
            f"[{chassis_component.capitalize()}] connected to [base_link] via ({chassis_first_link})"
        )
        return True

    def _connect_orphan_components_to_base(
        self,
        joints: list,
        base_points: dict,
        connection_rules: list,
        component_transforms: dict,
        existing_joint_names: set[str],
    ) -> None:
        # Find components that don't have parents in connection_rules
        components_with_parents = {child for parent, child in connection_rules}
        orphan_components = [
            comp for comp in base_points.keys() if comp not in components_with_parents
        ]

        for comp in orphan_components:
            comp_first_link = self._normalize_link_or_none(base_points.get(comp))
            if not comp_first_link:
                self.logger.error(f"Invalid base link for component [{comp}]")
                continue

            origin_kwargs = self._origin_kwargs_from_transform(
                component_transforms.get(comp)
            )
            if comp in component_transforms:
                self.logger.info(
                    f"Applied transform to base connection {comp}: {origin_kwargs}"
                )

            self._append_fixed_joint(
                joints=joints,
                existing_joint_names=existing_joint_names,
                joint_name=f"BASE_TO_{comp}_CONNECTOR",
                parent_link=self.base_link_name,
                child_link=comp_first_link,
                origin_kwargs=origin_kwargs,
            )

            self.logger.info(
                f"[{comp.capitalize()}] connected to [base_link] via ({comp_first_link})"
            )

    def _connect_component_pair(
        self,
        joints: list,
        base_points: dict,
        parent_attach_points: dict,
        parent: str,
        child: str,
        component_transforms: dict,
        existing_joint_names: set[str],
    ) -> None:
        if parent not in parent_attach_points or child not in base_points:
            self.logger.error(f"Invalid connection rule: {parent} -> {child}")
            return

        parent_connect_link = self._normalize_link_or_none(
            parent_attach_points.get(parent)
        )
        child_connect_link = self._normalize_link_or_none(base_points.get(child))

        if not parent_connect_link or not child_connect_link:
            self.logger.error(
                f"Invalid link in connection: {parent} ({parent_connect_link}) -> {child} ({child_connect_link})"
            )
            return

        self.logger.info(
            f"Connecting [{parent}]-({parent_connect_link}) to [{child}]-({child_connect_link})"
        )

        origin_kwargs = self._origin_kwargs_from_transform(
            component_transforms.get(child)
        )
        if child in component_transforms:
            self.logger.info(
                f"Applied transform to connection {parent} -> {child}: {origin_kwargs}"
            )

        self._append_fixed_joint(
            joints=joints,
            existing_joint_names=existing_joint_names,
            joint_name=self._apply_case("joint", f"{parent}_TO_{child}_CONNECTOR"),
            parent_link=parent_connect_link,
            child_link=child_connect_link,
            origin_kwargs=origin_kwargs,
        )

    def add_connections(
        self,
        joints: list,
        base_points: dict,
        parent_attach_points: dict,
        connection_rules: list,
        component_transforms: dict | None = None,
    ) -> None:
        r"""Add connection joints between robot components according to the specified rules.

        Args:
            joints: A list to collect joint elements.
            base_points: Mapping from component names to their child connection link names.
            parent_attach_points: Mapping from component names to their parent connection link names.
            connection_rules: A list of (parent, child) tuples specifying connection relationships.
            component_transforms: Optional mapping from component names to their 4x4 transform matrices.
        """
        chassis_component = "chassis"
        component_transforms = component_transforms or {}

        existing_joint_names = self._collect_existing_joint_names(joints)

        # chassis is always attached to base_link (no transform applied to this connection)
        if not self._connect_chassis_to_base(
            joints=joints,
            base_points=base_points,
            existing_joint_names=existing_joint_names,
            chassis_component=chassis_component,
        ):
            # If no chassis, connect components directly to base_link with their transforms
            self.logger.info(
                "No chassis found, connecting components directly to base_link"
            )
            self._connect_orphan_components_to_base(
                joints=joints,
                base_points=base_points,
                connection_rules=connection_rules,
                component_transforms=component_transforms,
                existing_joint_names=existing_joint_names,
            )

        # Process other connection relationships
        for parent, child in connection_rules:
            self._connect_component_pair(
                joints=joints,
                base_points=base_points,
                parent_attach_points=parent_attach_points,
                parent=parent,
                child=child,
                component_transforms=component_transforms,
                existing_joint_names=existing_joint_names,
            )

    def add_sensor_attachments(
        self, links: list, joints: list, attach_dict: dict, base_points: dict
    ) -> None:
        r"""Attach sensors by adding their URDF links/joints and creating a fixed connector.

        .. attention::
            This is a legacy helper kept for backward compatibility. Newer code paths
            use :class:`URDFSensorManager`.

        Args:
            links: Global list to collect sensor link elements.
            joints: Global list to collect sensor joint elements.
            attach_dict: Mapping from sensor names to attachment configs.
            base_points: Mapping from component names to their base link names.
        """
        existing_link_names = {
            self._apply_case("link", link.get("name"))
            for link in links
            if hasattr(link, "get") and link.get("name")
        }
        existing_link_names.discard(None)

        existing_joint_names = self._collect_existing_joint_names(joints)

        for sensor_name, attach in attach_dict.items():
            sensor_urdf_path = self._get_attr(attach, "sensor_urdf")
            if not sensor_urdf_path:
                self.logger.error(f"Sensor [{sensor_name}] has no sensor_urdf")
                continue

            try:
                sensor_urdf = ET.parse(sensor_urdf_path).getroot()
            except Exception as exc:
                self.logger.error(
                    f"Failed to parse sensor URDF for [{sensor_name}]: {exc}"
                )
                continue

            link_name_map: dict[str, str] = {}
            processed_link_names: list[str] = []

            # Add sensor links to the links list (ensure lowercase + uniqueness)
            for link in sensor_urdf.findall("link"):
                raw_name = link.get("name")
                if not raw_name:
                    continue

                normalized_raw = self._apply_case("link", raw_name)
                if not normalized_raw:
                    continue

                base_name = normalized_raw
                sensor_suffix = str(sensor_name).lower()
                if sensor_suffix and sensor_suffix not in base_name:
                    base_name = f"{base_name}_{sensor_suffix}"

                unique_name = self._make_unique(base_name, existing_link_names)
                link.set("name", unique_name)

                link_name_map[normalized_raw] = unique_name
                processed_link_names.append(unique_name)
                existing_link_names.add(unique_name)
                links.append(link)

            # Add sensor joints to the joints list (ensure uppercase + update link references)
            for joint in sensor_urdf.findall("joint"):
                raw_joint_name = joint.get("name") or "sensor_joint"

                normalized_joint_name = self._apply_case(
                    "joint", f"{sensor_name}_{raw_joint_name}"
                )
                if not normalized_joint_name:
                    continue

                normalized_joint_name = self._make_unique(
                    normalized_joint_name, existing_joint_names
                )
                joint.set("name", normalized_joint_name)

                parent_elem = joint.find("parent")
                child_elem = joint.find("child")

                if parent_elem is not None:
                    raw_parent = parent_elem.get("link")
                    normalized_parent = self._apply_case("link", raw_parent)
                    if normalized_parent and normalized_parent in link_name_map:
                        parent_elem.set("link", link_name_map[normalized_parent])
                    elif normalized_parent:
                        parent_elem.set("link", normalized_parent)

                if child_elem is not None:
                    raw_child = child_elem.get("link")
                    normalized_child = self._apply_case("link", raw_child)
                    if normalized_child and normalized_child in link_name_map:
                        child_elem.set("link", link_name_map[normalized_child])
                    elif normalized_child:
                        child_elem.set("link", normalized_child)

                joints.append(joint)
                existing_joint_names.add(normalized_joint_name)

            if not processed_link_names:
                self.logger.error(f"Sensor [{sensor_name}] has no <link> elements")
                continue

            # Determine parent link: prefer explicit parent_link if provided.
            parent_component = self._get_attr(attach, "parent_component")
            raw_parent_link = self._get_attr(attach, "parent_link")
            if raw_parent_link:
                parent_link = self._apply_case("link", raw_parent_link)
            else:
                parent_link = self._apply_case(
                    "link",
                    base_points.get(parent_component, parent_component),
                )

            if not parent_link:
                self.logger.error(
                    f"Invalid parent link for sensor [{sensor_name}] on component [{parent_component}]"
                )
                continue

            # Create connector joint (apply transform if provided by attachment).
            origin_kwargs = self._origin_kwargs_from_transform(
                self._get_attr(attach, "transform")
            )

            connector_joint_name = self._make_unique(
                self._apply_case(
                    "joint", f"{parent_component}_TO_{sensor_name}_CONNECTOR"
                )
                or self._apply_case(
                    "joint", f"{parent_component}_TO_{sensor_name}_CONNECTOR".upper()
                ),
                existing_joint_names,
            )

            self._append_fixed_joint(
                joints=joints,
                existing_joint_names=existing_joint_names,
                joint_name=connector_joint_name,
                parent_link=parent_link,
                child_link=processed_link_names[0],
                origin_kwargs=origin_kwargs,
            )
