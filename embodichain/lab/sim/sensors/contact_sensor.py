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

import uuid
from typing import TYPE_CHECKING, Sequence

import dexsim
import numpy as np
import torch
import warp as wp
from tensordict import TensorDict

from embodichain.lab.sim.sensors.base_sensor import BaseSensor, SensorCfg
from embodichain.utils import configclass, logger
from embodichain.utils.warp.kernels import scatter_contact_data

if TYPE_CHECKING:
    from dexsim.scene import ContactActorInfo, ContactQuery, ContactQueryCapabilities

    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "ArticulationContactFilterCfg",
    "ContactSensor",
    "ContactSensorCfg",
]


@configclass
class ContactSensorCfg(SensorCfg):
    """Configuration class for contact sensors.

    This class defines the configuration for contact sensors that detect
    collisions between rigid bodies and articulation links.
    """

    rigid_uid_list: list[str] = []
    """rigid body contact filter configs"""

    articulation_cfg_list: list[ArticulationContactFilterCfg] = []
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

    link_name_list: list[str] = []
    """link names in the articulation whose contacts need to be filtered."""

    @classmethod
    def from_dict(
        cls, init_dict: dict[str, str | list[str]]
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
        self,
        config: ContactSensorCfg,
        device: torch.device = torch.device("cpu"),
        *,
        owner: "SimulationManager | None" = None,
    ) -> None:
        if owner is None:
            from embodichain.lab.sim.sim_manager import SimulationManager

            owner = SimulationManager.get_instance()
        self._sim = owner

        self.item_user_ids: torch.Tensor | None = None
        """Backend-neutral actor IDs selected by the contact query."""

        self.item_env_ids: torch.Tensor | None = None
        """Environment IDs of the selected contact actors."""

        self.item_user_env_ids_map: torch.Tensor | None = None
        """Compatibility map from contact actor ID to environment ID."""

        self._visualizer: dexsim.models.PointCloud | None = None
        """contact point visualizer. Default to None"""
        self.device = device
        self.cfg = config
        self._query: ContactQuery | None = None

        self._num_contacts_per_env: torch.Tensor | None = None
        """Number of contacts per environment."""

        super().__init__(config, device, num_instances=owner.num_envs)

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
        assert self._num_contacts_per_env is not None
        return int(self._num_contacts_per_env.sum().item())

    @property
    def contact_capabilities(self) -> "ContactQueryCapabilities":
        """Capabilities reported by the active DexSim contact binding."""
        assert self._query is not None
        return self._query.capabilities

    def _build_sensor_from_config(
        self,
        config: ContactSensorCfg,
        device: torch.device,
    ) -> None:
        result = self._sim.spawn_result
        if result is None:
            raise RuntimeError("ContactSensor requires SimulationManager.prepare().")

        targets: list[object] = []
        for uid in config.rigid_uid_list:
            try:
                handles = self._sim._spawn_scene.handles(uid)
            except KeyError as exc:
                raise KeyError(f"Contact rigid-body UID not found: {uid!r}.") from exc
            if not handles:
                raise RuntimeError(f"Contact rigid body {uid!r} is not materialized.")
            targets.extend(handles)

        for articulation_cfg in config.articulation_cfg_list:
            uid = articulation_cfg.articulation_uid
            try:
                handles = self._sim._spawn_scene.handles(uid)
            except KeyError as exc:
                raise KeyError(f"Contact articulation UID not found: {uid!r}.") from exc
            if not handles:
                raise RuntimeError(f"Contact articulation {uid!r} is not materialized.")
            if not articulation_cfg.link_name_list:
                targets.extend(handles)
                continue
            for handle in handles:
                available = set(handle.get_link_names())
                missing = set(articulation_cfg.link_name_list) - available
                if missing:
                    raise ValueError(
                        f"Contact articulation {uid!r} has no links: {sorted(missing)}."
                    )
                targets.extend(
                    (handle, link_name) for link_name in articulation_cfg.link_name_list
                )

        if not targets:
            raise ValueError(
                "ContactSensor requires at least one rigid or link target."
            )

        self._query = result.create_contact_query(
            targets,
            match="all" if config.filter_need_both_actor else "any",
            capacity=self.max_total_contacts,
            device=device,
            frame="arena",
        )
        self._sync_filter_actor_metadata()

        num_envs = self.num_instances
        self._num_contacts_per_env = torch.zeros(
            num_envs, dtype=torch.int32, device=device
        )

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

    def _sync_filter_actor_metadata(self) -> None:
        assert self._query is not None
        actor_ids = self._query.selected_actor_ids
        self.item_user_ids = torch.as_tensor(
            actor_ids, dtype=torch.int32, device=self.device
        )
        self.item_env_ids = torch.as_tensor(
            [self._query.actor_info(actor_id).env_id for actor_id in actor_ids],
            dtype=torch.int32,
            device=self.device,
        )
        self.item_user_env_ids_map = torch.full(
            (max(actor_ids, default=-1) + 1,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        if actor_ids:
            self.item_user_env_ids_map[self.item_user_ids.to(torch.long)] = (
                self.item_env_ids
            )

    def update(self, **kwargs) -> None:
        """Update the sensor state based on the current simulation state.

        This method is called periodically to ensure the sensor data is up-to-date.

        Args:
            **kwargs: Additional keyword arguments for sensor update.
        """

        assert self._query is not None and self._num_contacts_per_env is not None
        self._num_contacts_per_env.zero_()
        self._data_buffer["is_valid"].zero_()

        contact_buffer = self._query.fetch()
        self._sync_filter_actor_metadata()
        if contact_buffer.count == 0:
            return
        env_ids = contact_buffer.env_ids[: contact_buffer.count]
        valid = (env_ids >= 0) & (env_ids < self.num_instances)
        if not bool(valid.any()):
            return
        contact_data = contact_buffer.data[: contact_buffer.count][valid].contiguous()
        actor_ids = contact_buffer.actor_ids[: contact_buffer.count][valid].contiguous()
        env_ids = env_ids[valid].contiguous()

        wp.launch(
            kernel=scatter_contact_data,
            dim=contact_data.shape[0],
            inputs=[
                wp.from_torch(contact_data),
                wp.from_torch(actor_ids),
                wp.from_torch(env_ids),
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
            device=str(self.device),
        )

    def get_arena_pose(self, to_matrix: bool = False) -> torch.Tensor | None:
        """Not used.

        Args:
            to_matrix: If True, return the pose as a 4x4 transformation matrix.

        Returns:
            A tensor representing the pose of the sensor in the arena frame.
        """
        logger.log_error("`get_arena_pose` for contact sensor is not implemented yet.")
        return None

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor | None:
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
            Batched contact data. ``position`` is in the Arena frame;
            ``normal`` points from actor 0 toward actor 1; ``friction`` and
            ``impulse`` are impulses; ``distance`` is signed separation;
            ``user_ids`` contains backend-neutral contact actor IDs; and
            ``is_valid`` marks populated rows.
        """
        return self._data_buffer

    def get_actor_info(self, actor_id: int) -> "ContactActorInfo":
        """Resolve an ID from the ``user_ids`` field to its Spawn identity."""
        assert self._query is not None
        return self._query.actor_info(actor_id)

    def filter_by_user_ids(
        self, item_user_ids: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> TensorDict:
        """Filter contact report by backend-neutral contact actor IDs.

        Args:
            item_user_ids: Actor IDs from this sensor's ``user_ids`` field.
            env_ids: Environment IDs to filter. If None, filter all environments.

        Returns:
            data: A TensorDict containing only the filtered contacts for the specified environments.
        """
        env_ids_tensor = (
            torch.arange(self.num_instances, device=self.device)
            if env_ids is None
            else torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        )
        item_user_ids = item_user_ids.to(device=self.device, dtype=torch.int32)

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

        if not bool(combined_mask.any()):
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
        rgba: Sequence[float] | None = None,
        point_size: float = 3.0,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = range(self.num_instances)

        if visible:
            # Convert env_ids to tensor if needed
            env_ids_tensor = (
                torch.tensor(env_ids, device=self.device)
                if not isinstance(env_ids, torch.Tensor)
                else env_ids
            )

            # Get number of contacts for each environment
            num_contacts = self._num_contacts_per_env[env_ids_tensor]

            # Create mask for valid contacts across all environments
            # Shape: [num_envs, max_contacts_per_env]
            contact_mask = torch.arange(
                self.cfg.max_contacts_per_env, device=self.device
            ).unsqueeze(0) < num_contacts.unsqueeze(1)

            if not contact_mask.any():
                # No contacts to visualize
                if isinstance(self._visualizer, dexsim.models.PointCloud):
                    self._visualizer.clear()
                return

            # Extract contact positions for all specified environments
            # Shape: [num_envs, max_contacts_per_env, 3]
            contact_position_arena = self._data_buffer["position"][env_ids_tensor]

            # Get arena offsets and broadcast to match positions shape
            # Shape: [num_envs, 1, 3] -> [num_envs, max_contacts_per_env, 3]
            contact_offsets = self._sim.arena_offsets[env_ids_tensor].unsqueeze(1)

            # Convert to world coordinates and apply mask in one go
            contact_position_world = (contact_position_arena + contact_offsets)[
                contact_mask
            ]

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
            self._visualizer.set_point_size(point_size)
        else:
            if isinstance(self._visualizer, dexsim.models.PointCloud):
                self._visualizer.clear()
