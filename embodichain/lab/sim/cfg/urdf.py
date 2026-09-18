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

"""URDF assembly configuration."""

from __future__ import annotations

from dataclasses import field
import os
from typing import Any, Dict, List

import numpy as np

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.utils import configclass, logger


def _get_data_path(path: str) -> str:
    """Resolve data through the public facade for monkeypatch compatibility."""
    from . import get_data_path

    return get_data_path(path)


@configclass
class URDFCfg:
    """Standalone configuration class for URDF assembly."""

    components: Dict[str, Dict[str, str | Dict | np.ndarray]] = field(
        default_factory=dict
    )
    """Dictionary of robot components to be assembled."""

    sensors: Dict[str, Dict[str, str | np.ndarray]] = field(default_factory=dict)
    """Dictionary of sensors to be attached to the robot."""

    use_signature_check: bool = True
    """Whether to use signature check when merging URDFs."""

    base_link_name: str = "base_link"
    """Name of the base link in the assembled robot."""

    fpath: str | None = None
    """Full output file path for the assembled URDF. If specified, overrides fname and fpath_prefix."""

    fname: str | None = None
    """Name used for output file and directory. If not specified, auto-generated from component names."""

    fpath_prefix: str = EMBODICHAIN_DEFAULT_DATA_ROOT + "/assembled"
    """Output directory prefix for the assembled URDF file."""

    component_prefix: List[tuple[str, str | None]] = field(
        default_factory=lambda: [
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
    )
    """Component name prefixes used during URDF assembly.

    Preferred form is a list of ``(component_name, prefix)`` tuples. For
    convenience, a mapping ``{component_name: prefix}`` is also accepted when
    constructing :class:`URDFCfg` and will be normalized internally.
    """

    name_case: dict[str, str] = field(
        default_factory=lambda: {
            "joint": "original",
            "link": "original",
        }
    )
    """Case normalization policy applied to joint/link names during URDF assembly.

    Supported values per key are ``"upper"``, ``"lower"`` or ``"original"``
    (legacy alias ``"none"``). The default preserves source URDF casing.
    """

    def __init__(
        self,
        components: list[dict[str, str | np.ndarray]] | None = None,
        sensors: dict[str, dict[str, str | np.ndarray]] | None = None,
        fpath: str | None = None,
        fname: str | None = None,
        fpath_prefix: str = EMBODICHAIN_DEFAULT_DATA_ROOT + "/assembled",
        use_signature_check: bool = True,
        base_link_name: str = "base_link",
        component_prefix: list[tuple[str, str | None]] | None = None,
        name_case: dict[str, str] | None = None,
    ):
        """
        Initialize URDFCfg with optional list of components and output path settings.

        Args:
            components (list[dict[str, str | np.ndarray]] | None): List of component configurations. Each dict should contain:
                - 'component_type' (str): The type/name of the component (e.g., 'chassis', 'arm', 'hand').
                - 'urdf_path' (str): Path to the component's URDF file.
                - 'transform' (np.ndarray | None): 4x4 transformation matrix (optional).
                - Additional params can be included as extra keys.
            sensors (dict[str, dict[str, str | np.ndarray]] | None): Sensor configurations for the robot.
            fpath (str | None): Full output file path for the assembled URDF. If specified, overrides fname and fpath_prefix.
            fname (str | None): Name used for output file and directory. If not specified, auto-generated from component names.
            fpath_prefix (str): Output directory prefix for the assembled URDF file.
            use_signature_check (bool): Whether to use signature check when merging URDFs.
            base_link_name (str): Name of the base link in the assembled robot.
            component_prefix (list[tuple[str, str | None]] | None): Optional
                list of (component_type, prefix) pairs to override default
                component name prefixes.
        """
        self.components = {}
        self.sensors = sensors or {}
        self.fpath = fpath
        self.use_signature_check = use_signature_check
        self.base_link_name = base_link_name
        self.fname = fname
        self.fpath_prefix = fpath_prefix

        # Initialize component prefixes (patch-style mapping per component type)
        if component_prefix is None:
            # Use the same default as the dataclass field
            self.component_prefix = [
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
        elif isinstance(component_prefix, dict):
            # Allow dict-style config: {"left_hand": "l_", ...}
            self.component_prefix = list(component_prefix.items())
        else:
            # Assume caller provided a list of (component_name, prefix) tuples
            self.component_prefix = component_prefix

        if name_case is None:
            self.name_case = {
                "joint": "original",
                "link": "original",
            }
        else:
            self.name_case = name_case

        # Auto-add components if provided
        if components:
            for comp_config in components:
                if not isinstance(comp_config, dict):
                    logger.log_error(
                        f"Component configuration must be a dict, got {type(comp_config)}"
                    )
                    continue

                # Extract required fields
                component_type = comp_config.get("component_type")
                urdf_path = comp_config.get("urdf_path")

                if not component_type or not urdf_path:
                    logger.log_error(
                        f"Component configuration must contain 'component_type' and 'urdf_path', got {comp_config}"
                    )
                    continue

                # Extract optional fields
                transform = comp_config.get("transform", np.eye(4))

                # Extract additional params (exclude known keys)
                params = {
                    k: v
                    for k, v in comp_config.items()
                    if k not in ["component_type", "urdf_path", "transform"]
                }

                # Add the component
                self.add_component(component_type, urdf_path, transform, **params)

        if sensors is not None:
            # Accept both list and dict; serialization round-trips an empty
            # dict when no sensors are configured (the field default).
            if isinstance(sensors, dict) and not sensors:
                self.sensors = []
            elif not isinstance(sensors, (list, dict)):
                logger.log_error(
                    f"sensors must be a list of dicts or a dict, got {type(sensors)}"
                )
                self.sensors = []
            elif isinstance(sensors, dict):
                # dict keyed by sensor_name -> config
                self.sensors = list(sensors.values())
            else:
                # Optionally check each sensor dict
                valid_sensors = []
                for sensor_config in sensors:
                    if not isinstance(sensor_config, dict):
                        logger.log_error(
                            f"Sensor configuration must be a dict, got {type(sensor_config)}"
                        )
                        continue
                    sensor_name = sensor_config.get("sensor_name")
                    if not sensor_name:
                        logger.log_error(
                            f"Sensor configuration must contain 'sensor_name', got {sensor_config}"
                        )
                        continue
                    valid_sensors.append(sensor_config)
                self.sensors = valid_sensors

    def set_urdf(self, urdf_path: str) -> "URDFCfg":
        """Directly specify a single URDF file for the robot, compatible with the single-URDF robot case.

        Args:
            urdf_path (str): Path to the robot's URDF file.

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        self.components.clear()
        urdf_file = os.path.splitext(os.path.basename(urdf_path))[0]
        self.components[urdf_file] = {
            "urdf_path": urdf_path,
            "transform": None,
            "params": {},
        }
        self.fpath = urdf_path
        return self

    def add_component(
        self,
        component_type: str,
        urdf_path: str,
        transform: np.ndarray | None = None,
        **params,
    ) -> URDFCfg:
        """Add a robot component to the assembly configuration.

        Args:
            component_type (str): The type/name of the component. Should be one of SUPPORTED_COMPONENTS
                (e.g., 'chassis', 'torso', 'head', 'left_arm', 'right_hand', 'arm', 'hand', etc.).
            urdf_path (str): Path to the component's URDF file.
            transform (np.ndarray | None): 4x4 transformation matrix for the component in the robot frame (default: None).
            **params: Additional keyword parameters for the component (e.g., color, material, etc.).

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        if urdf_path:
            if not os.path.exists(urdf_path):
                urdf_path_candidate = _get_data_path(urdf_path)
                if os.path.exists(urdf_path_candidate):
                    urdf_path = urdf_path_candidate
                else:
                    logger.log_error(f"URDF path '{urdf_path}' does not exist.")
                    raise FileNotFoundError(f"URDF path '{urdf_path}' does not exist.")

        if transform is None:
            transform = np.eye(4)

        self.components[component_type] = {
            "urdf_path": urdf_path,
            "transform": np.array(transform),
            "params": params,
        }

        if self.fname:
            self.fpath = f"{self.fpath_prefix}/{self.fname}/{self.fname}.urdf"
        else:
            # Update output_path to use all component urdf file names joined by underscores as directory
            if len(self.components) == 1:
                # Only one component, use its urdf file name
                urdf_file = os.path.splitext(os.path.basename(urdf_path))[0]
                name = urdf_file
            else:
                # Multiple components, join all urdf file names
                urdf_files = [
                    os.path.splitext(os.path.basename(v["urdf_path"]))[0]
                    for v in self.components.values()
                ]
                name = "_".join(urdf_files)
            self.fpath = f"{self.fpath_prefix}/{name}/{name}.urdf"

        return self

    def add_sensor(self, sensor_name: str, **sensor_config) -> URDFCfg:
        """Add a sensor to the robot configuration.

        Args:
            sensor_name (str): The name of the sensor.
            **sensor_config: Additional configuration parameters for the sensor.

        Returns:
            URDFCfg: Returns self to allow method chaining.
        """
        self.sensors.append({"sensor_name": sensor_name, **sensor_config})
        return self

    def assemble_urdf(self) -> str:
        """Assemble URDF files for the robot based on the configuration.

        Returns:
            str: The path to the resulting (possibly merged) URDF file.
        """
        components = list(self.components.items())
        # If there is only one component, return its URDF path directly.
        if len(components) == 1:
            _, comp_config = components[0]
            return comp_config["urdf_path"]

        from embodichain.toolkits.urdf_assembly import URDFAssemblyManager

        # If there are multiple components, merge them into a single URDF file.
        manager = URDFAssemblyManager()
        manager.base_link_name = self.base_link_name

        if self.component_prefix is None:
            self.component_prefix = [
                ("left_arm", "left_"),
                ("right_arm", "right_"),
                ("left_hand", "left_"),
                ("right_hand", "right_"),
            ]
        if isinstance(self.component_prefix, dict):
            self.component_prefix = list(self.component_prefix.items())
        # Forward configured component prefixes to the assembly manager
        manager.component_prefix = self.component_prefix

        if self.name_case is not None:
            manager.name_case = self.name_case

        for comp_type, comp_config in components:
            params = comp_config.get("params", {})
            success = manager.add_component(
                comp_type,
                comp_config["urdf_path"],
                comp_config.get("transform"),
                **params,
            )
            if not success:
                logger.log_error(
                    f"Failed to add component '{comp_type}' with config: {comp_config}"
                )

        for sensor in self.sensors:
            manager.attach_sensor(
                sensor_name=sensor.get("sensor_name"),
                sensor_source=sensor.get("sensor_source"),
                parent_component=sensor.get("parent_component"),
                parent_link=sensor.get("parent_link"),
                sensor_type=sensor.get("sensor_type"),
                **{
                    k: v
                    for k, v in sensor.items()
                    if k
                    not in [
                        "sensor_name",
                        "sensor_source",
                        "parent_component",
                        "parent_link",
                        "sensor_type",
                    ]
                },
            )

        try:
            # Merge all added components into a single URDF file at the specified output path.
            merged_urdf_xml = manager.merge_urdfs(self.fpath, self.use_signature_check)
        except Exception as e:
            logger.log_error(f"URDF merge failed: {e}")

        return self.fpath

    @classmethod
    def from_dict(cls, init_dict: Dict) -> "URDFCfg":
        if isinstance(init_dict, cls):
            return init_dict
        components = init_dict.get("components", None)
        if isinstance(components, dict):
            components = [{"component_type": k, **v} for k, v in components.items()]
        sensors = init_dict.get("sensors", None)
        fpath = init_dict.get("fpath", None)
        use_signature_check = init_dict.get("use_signature_check", True)
        base_link_name = init_dict.get("base_link_name", "base_link")
        component_prefix = init_dict.get("component_prefix", None)
        name_case = init_dict.get("name_case", None)
        return cls(
            components=components,
            sensors=sensors,
            fpath=fpath,
            use_signature_check=use_signature_check,
            base_link_name=base_link_name,
            component_prefix=component_prefix,
            name_case=name_case,
        )
