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

import torch
import numpy as np

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

__all__ = ["MatchObjectContainerEnv"]


@register_env("MatchObjectContainer-v1", max_episode_steps=600)
class MatchObjectContainerEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed.

        This is a classification task: place blocks into matching shaped containers.
        The task is successful if:
        1. Both cube blocks are inside container_cube (block_cube_1 and block_cube_2 -> container_cube)
        2. Both sphere blocks are inside container_sphere (block_sphere_1 and block_sphere_2 -> container_sphere)
        3. Both containers are up

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            block_cube_1 = self.sim.get_rigid_object("block_cube_1")
            block_sphere_1 = self.sim.get_rigid_object("block_sphere_1")
            block_cube_2 = self.sim.get_rigid_object("block_cube_2")
            block_sphere_2 = self.sim.get_rigid_object("block_sphere_2")
            container_cube = self.sim.get_rigid_object("container_cube")
            container_sphere = self.sim.get_rigid_object("container_sphere")
        except Exception as e:
            logger.log_warning(f"Blocks or containers not found: {e}, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get poses
        block_cube_1_pose = block_cube_1.get_local_pose(to_matrix=True)
        block_sphere_1_pose = block_sphere_1.get_local_pose(to_matrix=True)
        block_cube_2_pose = block_cube_2.get_local_pose(to_matrix=True)
        block_sphere_2_pose = block_sphere_2.get_local_pose(to_matrix=True)
        container_cube_pose = container_cube.get_local_pose(to_matrix=True)
        container_sphere_pose = container_sphere.get_local_pose(to_matrix=True)

        # Extract positions
        block_cube_1_pos = block_cube_1_pose[:, :3, 3]  # (num_envs, 3)
        block_sphere_1_pos = block_sphere_1_pose[:, :3, 3]
        block_cube_2_pos = block_cube_2_pose[:, :3, 3]
        block_sphere_2_pos = block_sphere_2_pose[:, :3, 3]
        container_cube_pos = container_cube_pose[:, :3, 3]
        container_sphere_pos = container_sphere_pose[:, :3, 3]

        container_cube_fallen = self._is_fall(container_cube_pose)
        container_sphere_fallen = self._is_fall(container_sphere_pose)

        # Check if blocks are inside their matching containers
        # I got radius and height from the container_metal.obj file
        container_bottom_radius = 0.1067  # Inner radius
        z_tolerance = 0.05  # Vertical tolerance
        container_height = 0.068  # Container height
        container_half_height = container_height / 2

        # Check if blocks are in their matching containers
        cube_1_in_container = self._is_block_in_container(
            block_cube_1_pos,
            container_cube_pos,
            container_bottom_radius,
            container_half_height,
            z_tolerance,
        )
        cube_2_in_container = self._is_block_in_container(
            block_cube_2_pos,
            container_cube_pos,
            container_bottom_radius,
            container_half_height,
            z_tolerance,
        )
        sphere_1_in_container = self._is_block_in_container(
            block_sphere_1_pos,
            container_sphere_pos,
            container_bottom_radius,
            container_half_height,
            z_tolerance,
        )
        sphere_2_in_container = self._is_block_in_container(
            block_sphere_2_pos,
            container_sphere_pos,
            container_bottom_radius,
            container_half_height,
            z_tolerance,
        )

        # Task success if cubes and spheres are in containers and containers are up
        success = (
            cube_1_in_container
            & cube_2_in_container
            & sphere_1_in_container
            & sphere_2_in_container
            & ~container_cube_fallen
            & ~container_sphere_fallen
        )

        return success

    def _is_block_in_container(
        self,
        block_pos: torch.Tensor,
        container_pos: torch.Tensor,
        container_bottom_radius: float,
        container_half_height: float,
        z_tolerance: float,
    ) -> torch.Tensor:
        """Check if a block is inside a container.

        Args:
            block_pos: Block position (num_envs, 3)
            container_pos: Container center position (num_envs, 3)
            container_bottom_radius: Inner radius of container bottom in meters
            container_half_height: Half height of container in meters
            z_tolerance: Vertical tolerance for bottom check in meters

        Returns:
            Boolean tensor indicating if block is in container (num_envs,)
        """
        # XY plane distance check
        xy_diff = torch.norm(block_pos[:, :2] - container_pos[:, :2], dim=1)

        # Z coordinate check: container center is at container_pos[:, 2]
        # Container bottom = container_pos[:, 2] - container_half_height
        # Container top = container_pos[:, 2] + container_half_height
        z_above_bottom = (
            block_pos[:, 2] > container_pos[:, 2] - container_half_height - z_tolerance
        )
        z_within_height = block_pos[:, 2] < container_pos[:, 2] + container_half_height

        return (xy_diff < container_bottom_radius) & z_above_bottom & z_within_height

    def _is_fall(self, pose: torch.Tensor) -> torch.Tensor:
        # Extract z-axis from rotation matrix (last column, first 3 elements)
        pose_rz = pose[:, :3, 2]
        world_z_axis = torch.tensor([0, 0, 1], dtype=pose.dtype, device=pose.device)

        # Compute dot product for each batch element
        dot_product = torch.sum(pose_rz * world_z_axis, dim=-1)  # Shape: (batch_size,)

        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Compute angle and check if fallen
        angle = torch.arccos(dot_product)
        return angle >= torch.pi / 4
