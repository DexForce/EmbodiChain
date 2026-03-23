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

import dexsim
import math
import torch
import uuid
import numpy as np
import warp as wp

from typing import Union, Tuple, Sequence, List, Optional, Dict
from tensordict import TensorDict

from embodichain.lab.sim.sensors import BaseSensor, SensorCfg
from embodichain.utils import logger, configclass
from embodichain.utils.warp.kernels import scatter_contact_data


@configclass
class ContactSensorCfg(SensorCfg):
    """Configuration class for contact sensors.

    This class defines the configuration for contact sensors that detect
    collisions between rigid bodies and articulation links.
    """

    rigid_uid_list: List[str] = []
    """rigid body contact filter configs"""

    articulation_cfg_list: List[ArticulationContactFilterCfg] = []
    """articulation link contact filter configs"""

    filter_need_both_actor: bool = True
    """Whether to filter contact only when both actors are in the filter list."""

    max_contacts_per_env: int = 64
    """Maximum number of contacts per environment the sensor can handle."""

    sensor_type: str = "ContactSensor"


@configclass
class ArticulationContactFilterCfg:
    """Configuration for filtering contacts from an articulation's links.

    This class defines which articulation and which links to monitor
    for contact events.
    """

    articulation_uid: str = ""
    """Articulation unique identifier."""

    link_name_list: List[str] = []
    """link names in the articulation whose contacts need to be filtered."""

    @classmethod
    def from_dict(
        cls, init_dict: Dict[str, Union[str, List[str]]]
    ) -> "ArticulationContactFilterCfg":
        """Initialize the configuration from a dictionary.

        Args:
            init_dict: Dictionary containing configuration parameters.

        Returns:
            ArticulationContactFilterCfg: The initialized configuration.
        """
        cfg = cls()
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.log_warning(f"Key '{key}' not found in {cls.__name__}.")
        return cfg


class ContactSensor(BaseSensor):
    """Sensor to get contacts from rigid body and articulation links."""

    SUPPORTED_DATA_TYPES = [
        "position",
        "normal",
        "friction",
        "impulse",
        "distance",
        "user_ids",
        "is_valid",
    ]

    def __init__(
        self, config: ContactSensorCfg, device: torch.device = torch.device("cpu")
    ) -> None:
        from embodichain.lab.sim import SimulationManager

        self._sim = SimulationManager.get_instance()
        """simulation manager reference"""

        self.item_user_ids: torch.Tensor | None = None
        """Dexsim userid of the contact filter items."""

        self.item_env_ids: torch.Tensor | None = None
        """Environment ids of the contact filter items."""

        self.item_user_env_ids_map: torch.Tensor | None = None
        """Map from dexsim userid to environment id."""

        self._visualizer: Optional[dexsim.models.PointCloud] = None
        """contact point visualizer. Default to None"""
        self.device = device
        self.cfg = config

        self._num_contacts_per_env: torch.Tensor | None = None
        """Number of contacts per environment."""

        super().__init__(config, device)

    @property
    def max_total_contacts(self) -> int:
        """Get the maximum total number of contacts across all environments.

        Returns:
            int: Maximum total number of contacts.
        """
        return self.cfg.max_contacts_per_env * self.num_instances

    @property
    def total_current_contacts(self) -> int:
        """Get the current total number of contacts across all environments.

        Note:
            This method returns the total number of contacts detected in the most recent update.

        Returns:
            int: Total number of contacts.
        """
        return self._num_contacts_per_env.sum().item()

    def _precompute_filter_ids(self, config: ContactSensorCfg):
        self.item_user_ids = torch.tensor([], dtype=torch.int32, device=self.device)
        self.item_env_ids = torch.tensor([], dtype=torch.int32, device=self.device)
        self.item_user_env_ids_map = torch.tensor(
            [], dtype=torch.int32, device=self.device
        )
        for rigid_uid in config.rigid_uid_list:
            rigid_object = self._sim.get_rigid_object(rigid_uid)
            if rigid_object is None:
                logger.log_warning(
                    f"Rigid body with uid '{rigid_uid}' not found in simulation."
                )
                continue
            self.item_user_ids = torch.cat(
                (self.item_user_ids, rigid_object.get_user_ids())
            )
            env_ids = torch.tensor(
                rigid_object._all_indices, dtype=torch.int32, device=self.device
            )
            self.item_env_ids = torch.cat((self.item_env_ids, env_ids))

        for articulation_cfg in config.articulation_cfg_list:
            articulation = self._sim.get_articulation(articulation_cfg.articulation_uid)
            if articulation is None:
                articulation = self._sim.get_robot(articulation_cfg.articulation_uid)
            if articulation is None:
                logger.log_warning(
                    f"Articulation with uid '{articulation_cfg.articulation_uid}' not found in simulation."
                )
                continue
            all_link_names = articulation.link_names
            link_names = (
                all_link_names
                if len(articulation_cfg.link_name_list) == 0
                else articulation_cfg.link_name_list
            )
            for link_name in link_names:
                if link_name not in all_link_names:
                    logger.log_warning(
                        f"Link {link_name} not found in articulation {articulation_cfg.uid}."
                    )
                    continue
                link_user_ids = articulation.get_user_ids(link_name).reshape(-1)
                self.item_user_ids = torch.cat((self.item_user_ids, link_user_ids))
                env_ids = torch.tensor(
                    articulation._all_indices, dtype=torch.int32, device=self.device
                )
                self.item_env_ids = torch.cat((self.item_env_ids, env_ids))
        # build user_id to env_id map
        max_user_id = int(self.item_user_ids.max().item())
        self.item_user_env_ids_map = torch.full(
            size=(max_user_id + 1,),
            fill_value=-1,
            dtype=self.item_user_ids.dtype,
            device=self.device,
        )
        self.item_user_env_ids_map[self.item_user_ids] = self.item_env_ids

    def _build_sensor_from_config(self, config: ContactSensorCfg, device: torch.device):
        self._precompute_filter_ids(config)
        self._world: dexsim.World = dexsim.default_world()
        self._ps = self._world.get_physics_scene()
        world_config = dexsim.get_world_config()
        self.is_use_gpu_physics = device.type == "cuda" and world_config.enable_gpu_sim
        if self.is_use_gpu_physics:
            self.contact_data_buffer = torch.zeros(
                self.max_total_contacts,
                11,
                dtype=torch.float32,
                device=device,
            )
            self.contact_user_ids_buffer = torch.zeros(
                self.max_total_contacts,
                2,
                dtype=torch.int32,
                device=device,
            )
        else:
            self._ps.enable_contact_data_update_on_cpu(True)

        num_envs = self.num_instances
        self._num_contacts_per_env = torch.zeros(
            num_envs, dtype=torch.int32, device=device
        )

        # TODO: We may pre-allocate the data buffer for contact data.
        self._data_buffer = TensorDict(
            {
                "position": torch.zeros(
                    (num_envs, config.max_contacts_per_env, 3), device=device
                ),
                "normal": torch.zeros(
                    (num_envs, config.max_contacts_per_env, 3), device=device
                ),
                "friction": torch.zeros(
                    (num_envs, config.max_contacts_per_env, 3), device=device
                ),
                "impulse": torch.zeros(
                    (num_envs, config.max_contacts_per_env), device=device
                ),
                "distance": torch.zeros(
                    (num_envs, config.max_contacts_per_env), device=device
                ),
                "user_ids": torch.zeros(
                    (num_envs, config.max_contacts_per_env, 2),
                    dtype=torch.int32,
                    device=device,
                ),
                "is_valid": torch.zeros(
                    (num_envs, config.max_contacts_per_env),
                    dtype=torch.bool,
                    device=device,
                ),
            },
            batch_size=[num_envs, config.max_contacts_per_env],
            device=device,
        )
        """
            position: [num_envs, num_contacts, 3] tensor, contact position in arena frame
            normal: [num_envs, num_contacts, 3] tensor, contact normal
            friction: [num_envs, num_contacts, 3] tensor, contact friction. Currently this value is not accurate.
            impulse: [num_envs, num_contacts] tensor, contact impulse
            distance: [num_envs, num_contacts] tensor, contact distance
            user_ids: [num_envs, num_contacts, 2] of int, contact user ids
                , use rigid_object.get_user_id() and find which object it belongs to.
            is_valid: [num_envs, num_contacts] bool tensor, indicating which contacts are valid
        """

    def update(self, **kwargs) -> None:
        """Update the sensor state based on the current simulation state.

        This method is called periodically to ensure the sensor data is up-to-date.

        Args:
            **kwargs: Additional keyword arguments for sensor update.
        """

        self._num_contacts_per_env.zero_()

        if not self.is_use_gpu_physics:
            contact_data_np, body_user_indices_np = self._ps.get_cpu_contact_buffer()
            n_contact = contact_data_np.shape[0]
            contact_data = torch.tensor(
                contact_data_np, dtype=torch.float32, device=self.device
            )
            body_user_indices = torch.tensor(
                body_user_indices_np, dtype=torch.int32, device=self.device
            )
        else:
            n_contact = self._ps.gpu_fetch_contact_data(
                self.contact_data_buffer, self.contact_user_ids_buffer
            )
            contact_data = self.contact_data_buffer[:n_contact]
            body_user_indices = self.contact_user_ids_buffer[:n_contact]

        if n_contact == 0:
            return

        filter0_mask = torch.isin(body_user_indices[:, 0], self.item_user_ids)
        filter1_mask = torch.isin(body_user_indices[:, 1], self.item_user_ids)
        if self.cfg.filter_need_both_actor:
            filter_mask = torch.logical_and(filter0_mask, filter1_mask)
        else:
            filter_mask = torch.logical_or(filter0_mask, filter1_mask)

        if not filter_mask.any():
            return

        filtered_contact_data = contact_data[filter_mask]
        filtered_user_ids = body_user_indices[filter_mask]

        # Get environment IDs for the filtered contacts
        filtered_env_ids = self.item_user_env_ids_map[filtered_user_ids[:, 0]]

        # Subtract arena offsets from contact positions
        contact_offsets = self._sim.arena_offsets[filtered_env_ids]
        filtered_contact_data[:, 0:3] = (
            filtered_contact_data[:, 0:3] - contact_offsets
        )  # minus arean offsets

        # Reset is_valid buffer
        self._data_buffer["is_valid"][:] = False

        num_contacts = len(filtered_contact_data)
        device = str(self.device)
        wp.launch(
            kernel=scatter_contact_data,
            dim=num_contacts,
            inputs=[
                wp.from_torch(filtered_contact_data),
                wp.from_torch(filtered_user_ids),
                wp.from_torch(filtered_env_ids),
                wp.from_torch(self._num_contacts_per_env),
                self.cfg.max_contacts_per_env,
            ],
            outputs=[
                wp.from_torch(self._data_buffer["position"]),
                wp.from_torch(self._data_buffer["normal"]),
                wp.from_torch(self._data_buffer["friction"]),
                wp.from_torch(self._data_buffer["impulse"]),
                wp.from_torch(self._data_buffer["distance"]),
                wp.from_torch(self._data_buffer["user_ids"]),
                wp.from_torch(self._data_buffer["is_valid"]),
            ],
            device="cuda:0" if device == "cuda" else device,
        )

    def get_arena_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Not used.

        Args:
            to_matrix: If True, return the pose as a 4x4 transformation matrix.

        Returns:
            A tensor representing the pose of the sensor in the arena frame.
        """
        logger.log_error("`get_arena_pose` for contact sensor is not implemented yet.")
        return None

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get the local pose of the camera.

        Args:
            to_matrix (bool): If True, return the pose as a 4x4 matrix. If False, return as a quaternion.

        Returns:
            torch.Tensor: The local pose of the camera.
        """
        logger.log_error("`get_local_pose` for contact sensor is not implemented yet.")

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set the local pose of the camera.

        Note: The pose should be in the OpenGL coordinate system, which means the Y is up and Z is forward.

        Args:
            pose (torch.Tensor): The local pose to set, should be a 4x4 transformation matrix.
            env_ids (Sequence[int] | None): The environment IDs to set the pose for. If None, set for all environments.
        """
        logger.log_error("`set_local_pose` for contact sensor is not implemented yet.")

    def get_data(self) -> TensorDict:
        """Retrieve data from the sensor.

        Returns:
            Dict:{
                "position": Tensor of float32 (num_envs, num_contacts, 3) representing the contact positions,
                "normal": Tensor of float32 (num_envs, num_contacts, 3) representing the contact normals,
                "friction": Tensor of float32 (num_envs, num_contacts, 3) representing the contact friction,
                "impulse": Tensor of float32 (num_envs, num_contacts) representing the contact impulses,
                "distance": Tensor of float32 (num_envs, num_contacts) representing the contact distances,
                "user_ids": Tensor of int32 (num_envs, num_contacts, 2) representing contact user ids
                        , use rigid_object.get_user_id() and find which object it belongs to.
                "is_valid": Tensor of bool (num_envs, num_contacts) indicating which contacts are valid.
            }
        """
        return self._data_buffer

    def filter_by_user_ids(
        self, item_user_ids: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> TensorDict:
        """Filter contact report by specific user IDs.

        Args:
            item_user_ids (torch.Tensor): Tensor of user IDs to filter by.
            env_ids (Sequence[int] | None): Environment IDs to filter. If None, filter all environments.

        Returns:
            data: A TensorDict containing only the filtered contacts for the specified environments.
        """
        if env_ids is None:
            env_ids = range(self.num_instances)

        # Vectorized filtering across all specified environments
        env_ids_tensor = (
            torch.tensor(env_ids, device=self.device)
            if isinstance(env_ids, list)
            else env_ids
        )

        # Flatten data across all specified environments
        env_data = {
            "position": self._data_buffer["position"][env_ids_tensor].flatten(0, 1),
            "normal": self._data_buffer["normal"][env_ids_tensor].flatten(0, 1),
            "friction": self._data_buffer["friction"][env_ids_tensor].flatten(0, 1),
            "impulse": self._data_buffer["impulse"][env_ids_tensor].flatten(0, 1),
            "distance": self._data_buffer["distance"][env_ids_tensor].flatten(0, 1),
            "user_ids": self._data_buffer["user_ids"][env_ids_tensor].flatten(0, 1),
            "is_valid": self._data_buffer["is_valid"][env_ids_tensor].flatten(0, 1),
        }

        # Create valid mask (only slots up to _num_contacts_per_env are valid)
        num_envs_to_filter = len(env_ids_tensor)
        valid_mask = (
            torch.arange(self.cfg.max_contacts_per_env, device=self.device).expand(
                num_envs_to_filter, -1
            )
            < self._num_contacts_per_env[env_ids_tensor][:, None]
        )
        valid_mask = valid_mask.flatten()

        # Create user ID filter mask
        user_ids_flat = env_data["user_ids"]
        filter0_mask = torch.isin(user_ids_flat[:, 0], item_user_ids)
        filter1_mask = torch.isin(user_ids_flat[:, 1], item_user_ids)

        if self.cfg.filter_need_both_actor:
            filter_mask = torch.logical_and(filter0_mask, filter1_mask)
        else:
            filter_mask = torch.logical_or(filter0_mask, filter1_mask)

        # Combine valid and user ID filters
        combined_mask = torch.logical_and(valid_mask, filter_mask)

        if not combined_mask.any():
            # Return empty TensorDict if no matches
            return TensorDict(
                {
                    "position": torch.empty((0, 3), device=self.device),
                    "normal": torch.empty((0, 3), device=self.device),
                    "friction": torch.empty((0, 3), device=self.device),
                    "impulse": torch.empty((0,), device=self.device),
                    "distance": torch.empty((0,), device=self.device),
                    "user_ids": torch.empty(
                        (0, 2), dtype=torch.int32, device=self.device
                    ),
                    "is_valid": torch.empty((0,), dtype=torch.bool, device=self.device),
                },
                batch_size=[0],
                device=self.device,
            )

        # Extract filtered data using the combined mask
        filtered_data = {key: value[combined_mask] for key, value in env_data.items()}

        return TensorDict(
            filtered_data,
            batch_size=[filtered_data["position"].shape[0]],
            device=self.device,
        )

    def set_contact_point_visibility(
        self,
        visible: bool = True,
        rgba: Optional[Sequence[int]] = None,
        point_size: float = 3.0,
        env_ids: Sequence[int] | None = None,
    ):
        if env_ids is None:
            env_ids = range(self.num_instances)

        if visible:
            contact_position_arena = self._data_buffer["position"][
                : self.total_current_contacts
            ]
            contact_offsets = self._sim.arena_offsets[
                self._data_buffer["env_ids"][: self.total_current_contacts]
            ]
            contact_position_world = contact_position_arena + contact_offsets

            if self._visualizer is None:
                # create new visualizer
                temp_str = uuid.uuid4().hex
                self._visualizer = self._sim.get_env().create_point_cloud(name=temp_str)
            else:
                # update existing visualizer points
                self._visualizer.clear()
            rgba = rgba if rgba is not None else (0.8, 0.2, 0.2, 1.0)
            if len(rgba) != 4:
                logger.log_error(
                    f"Invalid rgba {rgba}, should be a sequence of 4 floats."
                )
            rgba = np.array(
                [
                    rgba[0],
                    rgba[1],
                    rgba[2],
                    rgba[3],
                ]
            )
            self._visualizer.add_points(
                points=contact_position_world.to("cpu").numpy(), color=rgba
            )
            # self._visualizer.set_point_size(point_size)
        else:
            if isinstance(self._visualizer, dexsim.models.PointCloud):
                self._visualizer.clear()
