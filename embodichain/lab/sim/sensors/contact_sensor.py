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

import dexsim
import math
import torch

from typing import Union, Tuple, Sequence, List, Optional, Dict

from embodichain.lab.sim.sensors import BaseSensor, SensorCfg
from embodichain.utils import logger, configclass
import uuid
import numpy as np


@configclass
class ContactSensorCfg(SensorCfg):
    """Base class for sensor abstraction in the simulation engine.

    Sensors should inherit from this class and implement the `update` and `get_data` methods.
    """

    rigid_uid_list: List[str] = []
    """rigid body contact filter configs"""

    articulation_cfg_list: List[ArticulationContactFilterCfg] = []
    """articulation link contact filter configs"""

    filter_need_both_actor: bool = True
    """Whether to filter contact only when both actors are in the filter list."""

    sensor_type: str = "ContactSensor"


@configclass
class ArticulationContactFilterCfg:
    articulation_uid: str = ""
    """Articulation unique identifier."""

    link_name_list: List[str] = []
    """link names in the articulation whose contacts need to be filtered."""


class ContactSensor(BaseSensor):
    """Sensor to get contacts from rigid body and articulation links."""

    SUPPORTED_DATA_TYPES = [
        "position",
        "normal",
        "friction",
        "impulse",
        "distance",
        "user_ids",
        "env_ids",
    ]

    def __init__(
        self, config: ContactSensorCfg, device: torch.device = torch.device("cpu")
    ) -> None:
        from embodichain.lab.sim import SimulationManager

        self._sim = SimulationManager.get_instance()
        """simulation manager reference"""

        self.item_user_ids: Optional[torch.Tensor] = None
        """Dexsim userid of the contact filter items."""

        self.item_env_ids: Optional[torch.Tensor] = None
        """Environment ids of the contact filter items."""

        self.item_user_env_ids_map: Optional[torch.Tensor] = None
        """Map from dexsim userid to environment id."""

        self._data_buffer = {
            "position": torch.empty((0, 3), device=device),
            "normal": torch.empty((0, 3), device=device),
            "friction": torch.empty((0, 3), device=device),
            "impulse": torch.empty((0,), device=device),
            "distance": torch.empty((0,), device=device),
            "user_ids": torch.empty((0, 2), dtype=torch.int32, device=device),
            "env_ids": torch.empty((0,), dtype=torch.int32, device=device),
        }
        """
            position: [num_contacts, 3] tensor, contact position in arena frame
            normal: [num_contacts, 3] tensor, contact normal
            friction: [num_contacts, 3] tensor, contact friction. Currently this value is not accurate.
            impulse: [num_contacts, ] tensor, contact impulse
            distance: [num_contacts, ] tensor, contact distance
            user_ids: [num_contacts, 2] of int, contact user ids
                , use rigid_object.get_user_id() and find which object it belongs to.
            env_ids: [num_contacts, ] of int, which arena the contact belongs to.
        """

        self._visualizer: Optional[dexsim.models.PointCloud] = None
        """contact point visualizer. Default to None"""
        self.device = device
        super().__init__(config, device)

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
            MAX_CONTACT = 65536
            self.contact_data_buffer = torch.zeros(
                MAX_CONTACT, 11, dtype=torch.float32, device=device
            )
            self.contact_user_ids_buffer = torch.zeros(
                MAX_CONTACT, 2, dtype=torch.int32, device=device
            )
        else:
            self._ps.enable_contact_data_update_on_cpu(True)

    def update(self, **kwargs) -> None:
        """Update the sensor state based on the current simulation state.

        This method is called periodically to ensure the sensor data is up-to-date.

        Args:
            **kwargs: Additional keyword arguments for sensor update.
        """

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
            self._data_buffer = {
                "position": torch.empty((0, 3), device=self.device),
                "normal": torch.empty((0, 3), device=self.device),
                "friction": torch.empty((0, 3), device=self.device),
                "impulse": torch.empty((0,), device=self.device),
                "distance": torch.empty((0,), device=self.device),
                "user_ids": torch.empty((0, 2), dtype=torch.int32, device=self.device),
                "env_ids": torch.empty((0,), dtype=torch.int32, device=self.device),
            }
            return

        filter0_mask = torch.isin(body_user_indices[:, 0], self.item_user_ids)
        filter1_mask = torch.isin(body_user_indices[:, 1], self.item_user_ids)
        if self.cfg.filter_need_both_actor:
            filter_mask = torch.logical_and(filter0_mask, filter1_mask)
        else:
            filter_mask = torch.logical_or(filter0_mask, filter1_mask)

        filtered_contact_data = contact_data[filter_mask]
        filtered_user_ids = body_user_indices[filter_mask]
        filtered_env_ids = self.item_user_env_ids_map[filtered_user_ids[:, 0]]
        # generate contact report
        contact_offsets = self._sim.arena_offsets[filtered_env_ids]
        filtered_contact_data[:, 0:3] = (
            filtered_contact_data[:, 0:3] - contact_offsets
        )  # minus arean offsets
        self._data_buffer["position"] = filtered_contact_data[:, 0:3]
        self._data_buffer["normal"] = filtered_contact_data[:, 3:6]
        self._data_buffer["friction"] = filtered_contact_data[:, 6:9]
        self._data_buffer["impulse"] = filtered_contact_data[:, 9]
        self._data_buffer["distance"] = filtered_contact_data[:, 10]
        self._data_buffer["user_ids"] = filtered_user_ids
        self._data_buffer["env_ids"] = filtered_env_ids

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
        return None

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
        return None

    def get_data(self, copy: bool = True) -> Dict[str, torch.Tensor]:
        """Retrieve data from the sensor.

        Args:
            copy: If True, return a copy of the data buffer. Defaults to True.
        Returns:
            Dict:{
                "position": Tensor of float32 (num_contact, 3) representing the contact positions,
                "normal": Tensor of float32 (num_contact, 3) representing the contact normals,
                "friction": Tensor of float32 (num_contact, 3) representing the contact friction,
                "impulse": Tensor of float32 (num_contact, ) representing the contact impulses,
                "distance": Tensor of float32 (num_contact, ) representing the contact distances,
                "user_ids": Tensor of int32 (num_contact, ) representing contact user ids
                        , use rigid_object.get_user_id() and find which object it belongs to.
                "env_ids": [num_contacts, ] of int, which arena the contact belongs to.
            }
        """
        if copy:
            return {key: value.clone() for key, value in self._data_buffer.items()}
        return self._data_buffer

    def filter_by_user_ids(self, item_user_ids: torch.Tensor):
        """Filter contact report by specific user IDs.

        Args:
            item_user_ids (torch.Tensor): Tensor of user IDs to filter by.

        Returns:
            data: A new ContactReport instance containing only the filtered contacts.
        """
        filter0_mask = torch.isin(self._data_buffer["user_ids"][:, 0], item_user_ids)
        filter1_mask = torch.isin(self._data_buffer["user_ids"][:, 1], item_user_ids)
        if self.cfg.filter_need_both_actor:
            filter_mask = torch.logical_and(filter0_mask, filter1_mask)
        else:
            filter_mask = torch.logical_or(filter0_mask, filter1_mask)
        return {
            "position": self._data_buffer["position"][filter_mask],
            "normal": self._data_buffer["normal"][filter_mask],
            "friction": self._data_buffer["friction"][filter_mask],
            "impulse": self._data_buffer["impulse"][filter_mask],
            "distance": self._data_buffer["distance"][filter_mask],
            "user_ids": self._data_buffer["user_ids"][filter_mask],
            "env_ids": self._data_buffer["env_ids"][filter_mask],
        }

    def set_contact_point_visibility(
        self,
        visible: bool = True,
        rgba: Optional[Sequence[int]] = None,
        point_size: float = 3.0,
    ):
        if visible:
            contact_position_arena = self._data_buffer["position"]
            contact_offsets = self._sim.arena_offsets[self._data_buffer["env_ids"]]
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
