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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Union

from embodichain.lab.devices.device import Device
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot


class DeviceController:
    """Controller that bridges input devices (VR, keyboard, etc.) with robot control.

    This controller is agnostic to the environment and can be used in both
    gym environments and pure simulation contexts. It handles:
    - Mapping device input to robot joint commands
    - Joint limit enforcement
    - Filtering and smoothing (if not handled by device)
    - Multi-device support

    Example:
        # In gym environment
        controller = DeviceController(robot=env.robot, device=vr_device)
        action = controller.get_action()
        env.step(action)

        # In pure simulation
        controller = DeviceController(robot=sim_robot, device=vr_device)
        qpos = controller.get_action()
        sim_robot.set_qpos(qpos)
    """

    def __init__(
        self,
        robot: Robot,
        device: Optional[Device] = None,
        device_name: str = "default",
    ):
        """Initialize Device Controller.

        Args:
            robot: Robot instance to control.
            device: Input device (VR, keyboard, etc.). Can be None initially.
            device_name: Name identifier for this device.
        """
        self.robot = robot
        self._devices: Dict[str, Device] = {}
        self._active_device_name: Optional[str] = None

        if device is not None:
            self.add_device(device, device_name, set_active=True)

        # Joint mapping state: maps device joint name -> robot joint index
        self._joint_mapping: Dict[str, int] = {}
        self._mapping_initialized = False
        self._binary_control_keys = set()  # Device keys that need binary control

        logger.log_info(f"Device Controller initialized for robot: {robot.uid}")
        logger.log_info(f"  Robot has {len(robot.joint_names)} joints")

    def _get_gripper_joints(self) -> List[str]:
        """Get gripper joint names from control_parts."""
        if self.robot.control_parts:
            gripper_joints = []
            for part_name, joint_names in self.robot.control_parts.items():
                if "eef" in part_name.lower():
                    gripper_joints.extend(joint_names)
            return gripper_joints
        return []

    def add_device(
        self, device: Device, device_name: str, set_active: bool = False
    ) -> None:
        """Add a new input device.

        Args:
            device: Device instance to add.
            device_name: Name identifier for the device.
            set_active: Whether to set this as the active device.
        """
        self._devices[device_name] = device

        if set_active or self._active_device_name is None:
            self._active_device_name = device_name

        logger.log_info(f"Added device '{device_name}' (Active: {set_active})")

    def remove_device(self, device_name: str) -> None:
        """Remove a device.

        Args:
            device_name: Name of the device to remove.
        """
        if device_name in self._devices:
            del self._devices[device_name]
            logger.log_info(f"Removed device '{device_name}'")

            if self._active_device_name == device_name:
                self._active_device_name = (
                    list(self._devices.keys())[0] if self._devices else None
                )

    def set_active_device(self, device_name: str) -> None:
        """Set the active input device.

        Args:
            device_name: Name of the device to activate.
        """
        if device_name not in self._devices:
            logger.log_error(f"Device '{device_name}' not found")
            return

        self._active_device_name = device_name
        logger.log_info(f"Active device set to '{device_name}'")

    def get_action(
        self, device_name: Optional[str] = None, as_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, float], None]:
        """Get robot action from device input.

        Args:
            device_name: Name of device to query. If None, uses active device.
            as_dict: Whether to return action as dict (joint_name -> value).

        Returns:
            Robot action tensor (shape: [num_envs, num_joints]) or dict,
            or None if no valid data available.
        """
        # Get device
        device_name = device_name or self._active_device_name
        if device_name is None or device_name not in self._devices:
            return None

        device = self._devices[device_name]

        # Get device state
        state = device.get_controller_state()
        device_data = state.get("filtered_data") or state.get("raw_data")

        if device_data is None:
            return None

        # Map device data to robot action
        return self._map_device_to_robot(device_data, as_dict=as_dict)

    def _map_device_to_robot(
        self, device_data: Dict[str, float], as_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, float], None]:
        """Map device input to robot action.

        Args:
            device_data: Device joint data (joint_name -> value).
            as_dict: Whether to return as dict instead of tensor.

        Returns:
            Robot action or None if mapping failed.
        """
        try:
            robot_joint_names = self.robot.joint_names

            if not self._mapping_initialized:
                self._build_joint_mapping(device_data, robot_joint_names)

            # Get robot joint limits
            qpos_limits = self.robot.body_data.qpos_limits[0]  # [num_joints, 2]

            # Extract joint values based on mapping
            joint_values = []
            joint_indices = []
            mapped_joints = {}

            gripper_threshold = 0.99

            for device_joint, robot_indices in self._joint_mapping.items():
                if device_joint in device_data:
                    vr_value = device_data[device_joint]

                    # Binary control for grippers, direct value for other joints
                    if device_joint in self._binary_control_keys:
                        idx = robot_indices[0]
                        joint_min, joint_max = (
                            qpos_limits[idx, 0].item(),
                            qpos_limits[idx, 1].item(),
                        )
                        robot_value = (
                            joint_min if vr_value >= gripper_threshold else joint_max
                        )

                        # Log state change
                        state = "CLOSE" if vr_value >= gripper_threshold else "OPEN"
                        if not hasattr(self, "_last_gripper_state"):
                            self._last_gripper_state = {}
                        if self._last_gripper_state.get(device_joint) != state:
                            logger.log_info(
                                f"{device_joint}: VR={vr_value:.3f} -> {state} (robot={robot_value:.3f}, limit=[{joint_min:.3f}, {joint_max:.3f}])"
                            )
                            self._last_gripper_state[device_joint] = state
                    else:
                        robot_value = vr_value

                    # One device value may correspond to multiple robot joints (e.g., two fingers of gripper)
                    for robot_idx in robot_indices:
                        joint_values.append(robot_value)
                        joint_indices.append(robot_idx)
                        mapped_joints[robot_joint_names[robot_idx]] = robot_value

            if len(joint_indices) == 0:
                return None

            # Convert to tensor and create full action
            device_tensor = torch.tensor(
                joint_values, dtype=torch.float32, device=self.robot.device
            )
            indices_tensor = torch.tensor(
                joint_indices, dtype=torch.long, device=self.robot.device
            )

            # Get current robot qpos
            current_qpos = self.robot.get_qpos()  # [num_envs, num_joints]

            # Create action by updating controlled joints
            action = current_qpos.clone()
            action[:, indices_tensor] = device_tensor.unsqueeze(0)

            # Enforce joint limits
            action = self._enforce_joint_limits(action)

            return action

        except Exception as e:
            logger.log_error(f"Error mapping device data to robot action: {e}")
            import traceback

            logger.log_error(traceback.format_exc())
            return None

    def _build_joint_mapping(
        self, device_data: Dict[str, float], robot_joint_names: List[str]
    ) -> None:
        """Build mapping from device joints to robot joints."""
        # Store: device_joint_name -> [robot_joint_indices]
        self._joint_mapping = {}

        logger.log_info("=" * 60)
        logger.log_info("Building joint mapping...")
        logger.log_info(
            f"VR device data keys ({len(device_data)} total): {sorted(device_data.keys())}"
        )
        logger.log_info(
            f"Robot joint names ({len(robot_joint_names)} total): {robot_joint_names}"
        )
        logger.log_info(f"Robot control_parts: {self.robot.control_parts}")

        # Direct name matching
        for robot_idx, robot_joint in enumerate(robot_joint_names):
            if robot_joint in device_data:
                self._joint_mapping.setdefault(robot_joint, []).append(robot_idx)

        # Special gripper mapping: LEFT_GRIPPER/RIGHT_GRIPPER -> multiple finger joints
        # Only for parallel grippers (< 5 joints), not dexterous hands (>= 5 joints)
        if self.robot.control_parts:
            for part_name, joint_names in self.robot.control_parts.items():
                if "eef" in part_name.lower() and len(joint_names) < 5:
                    device_key = (
                        f"{'LEFT' if 'left' in part_name.lower() else 'RIGHT'}_GRIPPER"
                    )
                    if (
                        device_key in device_data
                        and device_key not in robot_joint_names
                    ):
                        indices = [
                            robot_joint_names.index(j)
                            for j in joint_names
                            if j in robot_joint_names
                        ]
                        if indices:
                            self._joint_mapping[device_key] = indices
                            self._binary_control_keys.add(device_key)
                            logger.log_info(
                                f"  Gripper mapping: '{device_key}' -> {joint_names} (indices: {indices})"
                            )

        self._mapping_initialized = True

        total_mappings = sum(len(v) for v in self._joint_mapping.values())
        logger.log_info(
            f"Joint mapping initialized: {len(self._joint_mapping)} device joints mapped to {total_mappings} robot joints"
        )
        logger.log_info(f"  Device joints: {list(self._joint_mapping.keys())}")
        logger.log_info("=" * 60)

    def _enforce_joint_limits(self, action: torch.Tensor) -> torch.Tensor:
        """Enforce robot joint limits on action.

        Args:
            action: Action tensor [num_envs, num_joints].

        Returns:
            Clamped action tensor.
        """
        qpos_limits = self.robot.body_data.qpos_limits[0]  # [num_joints, 2]
        return torch.clamp(action, qpos_limits[:, 0], qpos_limits[:, 1])

    def reset(self) -> None:
        """Reset controller state."""
        # Reset all devices
        for device in self._devices.values():
            if hasattr(device, "reset"):
                device.reset()

        # Reset mapping
        self._joint_mapping = {}
        self._mapping_initialized = False
        self._binary_control_keys.clear()

        logger.log_info("Device Controller reset")

    def get_device_info(self, device_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a device.

        Args:
            device_name: Name of device. If None, uses active device.

        Returns:
            Device information dict.
        """
        device_name = device_name or self._active_device_name
        if device_name is None or device_name not in self._devices:
            return {"error": "No device available"}

        device = self._devices[device_name]
        state = device.get_controller_state()

        return {
            "device_name": device_name,
            "is_active": device_name == self._active_device_name,
            "mapped_joints": list(self._joint_mapping.keys()),
            "device_state": state,
        }

    def get_all_devices(self) -> List[str]:
        """Get list of all registered device names."""
        return list(self._devices.keys())

    @property
    def active_device(self) -> Optional[Device]:
        """Get the currently active device."""
        if self._active_device_name is None:
            return None
        return self._devices.get(self._active_device_name)
