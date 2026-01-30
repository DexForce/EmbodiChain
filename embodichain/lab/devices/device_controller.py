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

        # Priority 1: Use device-specific mapping if available
        action = device.map_to_robot(self.robot, device_data)

        # Priority 2: Use generic mapping as fallback
        if action is None:
            action = self._generic_mapping(device_data)

        return action

    def _generic_mapping(self, device_data: Dict[str, float]) -> Optional[torch.Tensor]:
        """Generic device-to-robot mapping (simple direct name matching).

        This is a fallback for devices that don't implement custom mapping.
        Only supports direct joint name matching (e.g., keyboard, simple controllers).

        Args:
            device_data: Device joint data (joint_name -> value).

        Returns:
            Robot action tensor or None if no joints matched.
        """
        try:
            robot_joint_names = self.robot.joint_names
            qpos_limits = self.robot.body_data.qpos_limits[0]

            joint_values = []
            joint_indices = []

            # Direct name matching only
            for robot_idx, robot_joint in enumerate(robot_joint_names):
                if robot_joint in device_data:
                    joint_values.append(device_data[robot_joint])
                    joint_indices.append(robot_idx)

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
            current_qpos = self.robot.get_qpos()

            # Create action by updating controlled joints
            action = current_qpos.clone()
            action[:, indices_tensor] = device_tensor.unsqueeze(0)

            # Enforce joint limits
            action = torch.clamp(action, qpos_limits[:, 0], qpos_limits[:, 1])

            return action

        except Exception as e:
            logger.log_error(f"Error in generic device mapping: {e}")
            return None

    def reset(self) -> None:
        """Reset controller state."""
        # Reset all devices
        for device in self._devices.values():
            if hasattr(device, "reset"):
                device.reset()

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
