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
import numpy as np
import warp as wp
import dexsim.render as dr

from typing import Dict, Tuple, List, Sequence

from dexsim.utility import inv_transform
from embodichain.lab.sim.sensors import Camera, CameraCfg
from embodichain.utils.warp.kernels import reshape_tiled_image
from embodichain.utils.math import matrix_from_euler
from embodichain.utils import logger, configclass
from embodichain.lab.sim.utility.sim_utils import is_rt_enabled


@configclass
class StereoCameraCfg(CameraCfg):
    """Configuration class for StereoCamera."""

    sensor_type: str = "StereoCamera"

    # The camera intrinsics of the right camera.
    # The default camera is the left camera.
    intrinsics_right: Tuple[float, float, float, float] = (600, 600, 320.0, 240.0)

    left_to_right_pos: Tuple[float, float, float] = (0.05, 0.0, 0.0)
    # The rotation from left camera to right camera in degrees.
    left_to_right_rot: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    enable_disparity: bool = False

    fx_r: float = intrinsics_right[0]
    fy_r: float = intrinsics_right[1]
    cx_r: float = intrinsics_right[2]
    cy_r: float = intrinsics_right[3]

    @property
    def left_to_right(self) -> torch.Tensor:
        """Get the transformation matrix from left camera to right camera."""
        left_to_right = torch.eye(4, dtype=torch.float32)
        left_to_right[:3, 3] = torch.tensor(self.left_to_right_pos, dtype=torch.float32)
        rot = torch.tensor(self.left_to_right_rot, dtype=torch.float32)
        left_to_right[:3, :3] = matrix_from_euler(rot.unsqueeze(0)).squeeze(0)
        return left_to_right

    @property
    def right_to_left(self) -> torch.Tensor:
        """Get the transformation matrix from right camera to left camera."""
        return torch.inverse(self.left_to_right)

    def get_data_types(self) -> List[str]:
        data_types = []
        if self.enable_color:
            data_types.append("color")
            data_types.append("color_right")
        if self.enable_depth:
            data_types.append("depth")
            data_types.append("depth_right")
        if self.enable_mask:
            data_types.append("mask")
            data_types.append("mask_right")
        if self.enable_normal:
            data_types.append("normal")
            data_types.append("normal_right")
        if self.enable_position:
            data_types.append("position")
            data_types.append("position_right")
        if self.enable_disparity:
            data_types.append("disparity")
        return data_types


class PairCameraView:
    def __init__(
        self,
        left_view: dr.CameraView,
        right_view: dr.CameraView,
        left_to_right: np.ndarray,
    ) -> PairCameraView:
        self._left_view = left_view
        self._right_view = right_view
        self._left_to_right = left_to_right

        self._left_to_center = np.eye(4, dtype=np.float32)
        self._left_to_center[:3, 3] = left_to_right[:3, 3] * -0.5

        self._right_to_center = np.eye(4, dtype=np.float32)
        self._right_to_center[:3, 3] = left_to_right[:3, 3] * 0.5

    def set_local_pose(self, pose: np.ndarray) -> None:
        left_pose = pose @ self._left_to_center
        right_pose = pose @ self._right_to_center
        self._left_view.set_local_pose(left_pose)
        self._right_view.set_local_pose(right_pose)

    def get_local_pose(self) -> np.ndarray:
        left_pose = self._left_view.get_local_pose()
        return left_pose @ inv_transform(self._left_to_center)

    def set_world_pose(self, pose: np.ndarray) -> None:
        left_pose = pose @ self._left_to_center
        right_pose = pose @ self._right_to_center
        self._left_view.set_world_pose(left_pose)
        self._right_view.set_world_pose(right_pose)

    def get_world_pose(self) -> np.ndarray:
        left_pose = self._left_view.get_world_pose()
        return left_pose @ inv_transform(self._left_to_center)

    def get_node(self) -> dexsim.engine.Node:
        return self._left_view.get_node()

    def attach_node(self, parent: dexsim.engine.Node) -> None:
        self._left_view.attach_node(parent)
        self._right_view.attach_node(parent)


class StereoCamera(Camera):
    """Base class for sensor abstraction in the simulation engine.

    Sensors should inherit from this class and implement the `update` and `get_data` methods.
    """

    SUPPORTED_DATA_TYPES = [
        "color",
        "depth",
        "mask",
        "normal",
        "position",
        "color_right",
        "depth_right",
        "mask_right",
        "normal_right",
        "position_right",
        "disparity",
    ]

    def __init__(
        self,
        config: StereoCameraCfg,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(config, device)

        # check valid config
        if self.cfg.enable_disparity and not self.cfg.enable_depth:
            logger.log_error("Disparity can only be enabled when depth is enabled.")

    def _build_sensor_from_config(
        self, config: StereoCameraCfg, device: torch.device
    ) -> None:
        self._world = dexsim.default_world()
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        num_instances = len(arenas)

        if self.is_rt_enabled:
            self._frame_buffer = self._world.create_camera_group(
                [config.width, config.height], num_instances * 2, True
            )
            view_attrib = config.get_view_attrib()
            left_list = []
            right_list = []
            for i, arena in enumerate(arenas):
                left_view_name = f"{self.uid}_left_view{i + 1}"
                left_view = arena.create_camera(
                    left_view_name,
                    config.width,
                    config.height,
                    True,
                    view_attrib,
                    self._frame_buffer,
                )
                left_view.set_intrinsic(config.intrinsics)
                left_view.set_near(config.near)
                left_view.set_far(config.far)
                left_list.append(left_view)

            for i, arena in enumerate(arenas):
                right_view_name = f"{self.uid}_right_view{i + 1}"
                right_view = arena.create_camera(
                    right_view_name,
                    config.width,
                    config.height,
                    True,
                    view_attrib,
                    self._frame_buffer,
                )
                right_view.set_intrinsic(config.intrinsics_right)
                right_view.set_near(config.near)
                right_view.set_far(config.far)
                right_list.append(right_view)

            for i in range(num_instances):
                self._entities[i] = PairCameraView(
                    left_list[i], right_list[i], config.left_to_right.cpu().numpy()
                )

        else:
            self._grid_size = math.ceil(math.sqrt(num_instances))

            # stereo camera has two views, we append the right camera to the left camera's view list
            frame_width = self._grid_size * config.width * 2
            frame_height = self._grid_size * config.height
            view_attrib = config.get_view_attrib()

            # Create the data frame
            self._frame_buffer = self._world.create_frame_buffer(
                [frame_width, frame_height], view_attrib, True
            )
            self._frame_buffer.set_read_able(view_attrib)

            # Create camera views
            for i, arena in enumerate(arenas):
                col = i // self._grid_size
                row = i % self._grid_size
                x = row * config.width * 2
                y = col * config.height
                left_view_name = f"{self.uid}_left_view{i + 1}"

                left_view = arena.create_camera_view(
                    left_view_name,
                    (x, y),
                    (config.width, config.height),
                    self._frame_buffer,
                )

                left_view.set_intrinsic(config.intrinsics)
                left_view.set_near(config.near)
                left_view.set_far(config.far)
                left_view.enable_postprocessing(True)

                right_view_name = f"{self.uid}_right_view{i + 1}"
                right_view = arena.create_camera_view(
                    right_view_name,
                    (x + config.width, y),
                    (config.width, config.height),
                    self._frame_buffer,
                )
                right_view.set_intrinsic(config.intrinsics_right)
                right_view.set_near(config.near)
                right_view.set_far(config.far)
                right_view.enable_postprocessing(True)

                self._entities[i] = PairCameraView(
                    left_view, right_view, config.left_to_right.cpu().numpy()
                )

        # Define a mapping of data types to their respective shapes and dtypes
        buffer_specs = {
            "color": (
                (self.num_instances, config.height, config.width, 4),
                torch.uint8,
            ),
            "depth": (
                (self.num_instances, config.height, config.width, 1),
                torch.float32,
            ),
            "mask": (
                (self.num_instances, config.height, config.width, 1),
                torch.int32,
            ),
            "normal": (
                (self.num_instances, config.height, config.width, 3),
                torch.float32,
            ),
            "position": (
                (self.num_instances, config.height, config.width, 3),
                torch.float32,
            ),
            "disparity": (
                (self.num_instances, config.height, config.width, 1),
                torch.float32,
            ),
        }
        buffer_specs.update(
            {
                f"{data_type}_right": buffer_specs[data_type]
                for data_type in ["color", "depth", "mask", "normal", "position"]
            }
        )
        data_types = config.get_data_types()

        # stereo buffer to store data for left and right cameras
        # the data in `_data_buffer` is shared with the data in `_data_buffer_stereo`.
        self._data_buffer_stereo: Dict[str, torch.Tensor] = {}

        # Iterate through enabled data types and initialize buffers
        for data_type in data_types:
            if "right" in data_type:
                continue
            if getattr(config, f"enable_{data_type}", False):
                shape, dtype = buffer_specs[data_type]
                if data_type == "disparity":
                    self._data_buffer[data_type] = torch.empty(
                        shape, dtype=dtype, device=device
                    )

                # create new shape with width * 2 for stereo camera
                shape_ = (shape[0], shape[1], shape[2] * 2, shape[3])

                self._data_buffer_stereo[data_type] = torch.empty(
                    shape_, dtype=dtype, device=device
                )
                self._data_buffer[data_type] = self._data_buffer_stereo[data_type][
                    :, :, : config.width, :
                ]
                self._data_buffer[f"{data_type}_right"] = self._data_buffer_stereo[
                    data_type
                ][:, :, config.width :, :]

        self.cfg: CameraCfg = config
        if self.cfg.extrinsics.parent is not None:
            self._attach_to_entity()

    def update(self, **kwargs) -> None:
        """Update the sensor data.

        The supported data types are:
            - color: RGB images with shape (B, H, W, 4) and dtype torch.uint8
            - depth: Depth images with shape (B, H, W, 1) and dtype torch.float32
            - mask: Instance segmentation masks with shape (B, H, W, 1) and dtype torch.int32
            - normal: Normal images with shape (B, H, W, 3) and dtype torch.float32
            - position: Position images with shape (B, H, W, 3) and dtype torch.float32
            - disparity: Disparity images with shape (B, H, W, 1) and dtype torch.float32
        Args:
            **kwargs: Additional keyword arguments for sensor update.
                - fetch_only (bool): If True, only fetch the data from dexsim internal frame buffer without performing rendering.
        """

        fetch_only = kwargs.get("fetch_only", False)
        if not fetch_only:
            if self.is_rt_enabled:
                self._frame_buffer.apply()
            else:
                self._frame_buffer.apply_frame()

        self.cfg: StereoCameraCfg
        if self.cfg.enable_color:
            if self.is_rt_enabled:
                data = self._frame_buffer.get_rgb_gpu_buffer().to(self.device)
                self._data_buffer["color"] = data[: self.num_instances, ...]
                self._data_buffer[f"color_right"] = data[self.num_instances :, ...]
            else:
                data = self._frame_buffer.get_color_gpu_buffer().to(self.device)
                self._update_buffer_impl(data, self._data_buffer_stereo["color"])
        if self.cfg.enable_depth:
            data = self._frame_buffer.get_depth_gpu_buffer().to(self.device)
            if self.is_rt_enabled:
                self._data_buffer["depth"] = data[: self.num_instances, ...].unsqueeze_(
                    -1
                )
                self._data_buffer[f"depth_right"] = data[
                    self.num_instances :, ...
                ].unsqueeze_(-1)
            else:
                self._update_buffer_impl(data, self._data_buffer_stereo["depth"])
        if self.cfg.enable_mask:
            if self.is_rt_enabled:
                data = self._frame_buffer.get_visible_mask_gpu_buffer().to(
                    self.device, torch.int32
                )
                self._data_buffer["mask"] = data[: self.num_instances, ...].unsqueeze_(
                    -1
                )
                self._data_buffer[f"mask_right"] = data[
                    self.num_instances :, ...
                ].unsqueeze_(-1)
            else:
                data = self._frame_buffer.get_visible_gpu_buffer().to(
                    self.device, torch.int32
                )
                self._update_buffer_impl(data, self._data_buffer_stereo["mask"])
        if self.cfg.enable_normal:
            data = self._frame_buffer.get_normal_gpu_buffer().to(self.device)
            if self.is_rt_enabled:
                self._data_buffer["normal"] = data[: self.num_instances, ...]
                self._data_buffer[f"normal_right"] = data[self.num_instances :, ...]
            else:
                self._update_buffer_impl(data, self._data_buffer_stereo["normal"])
        if self.cfg.enable_position:
            data = self._frame_buffer.get_position_gpu_buffer().to(self.device)
            if self.is_rt_enabled:
                self._data_buffer["position"] = data[: self.num_instances, ...]
                self._data_buffer[f"position_right"] = data[self.num_instances :, ...]
            else:
                self._update_buffer_impl(data, self._data_buffer_stereo["position"])
        if self.cfg.enable_disparity:
            disparity = self._data_buffer["disparity"]
            disparity.fill_(0.0)
            distance = torch.sqrt(
                torch.sum(torch.square(self.cfg.left_to_right[:3, 3]))
            )
            # Compute disparity only for non-zero depth values
            depth = self._data_buffer["depth"]
            valid_depth_mask = depth > 0
            disparity[valid_depth_mask] = (
                self.cfg.fx * distance / depth[valid_depth_mask]
            )

    def _update_buffer_impl(
        self, data_buffer: torch.Tensor, data_buffer_out: torch.Tensor
    ) -> None:
        device = str(self.device)
        channel = data_buffer.shape[-1] if data_buffer.dim() >= 3 else 1
        wp.launch(
            kernel=reshape_tiled_image,
            dim=(self.num_instances, self.cfg.height, self.cfg.width * 2),
            inputs=[
                wp.from_torch(data_buffer).flatten(),
                wp.from_torch(data_buffer_out),
                self.cfg.height,
                self.cfg.width * 2,
                channel,
                self._grid_size,
            ],
            device="cuda:0" if device == "cuda" else device,
        )

    def get_left_right_arena_pose(self) -> torch.Tensor:
        """Get the local pose of the left and right cameras.

        Returns:
            torch.Tensor: The local pose of the left camera with shape (num_envs, 4, 4).
        """
        from embodichain.lab.sim.utility import get_dexsim_arenas

        arenas = get_dexsim_arenas()

        left_poses = []
        right_poses = []
        for i, entity in enumerate(self._entities):
            arena_pose = arenas[i].get_root_node().get_local_pose()
            left_pose = entity._left_view.get_world_pose()
            left_pose[:2, 3] -= arena_pose[:2, 3]
            left_poses.append(
                torch.as_tensor(
                    left_pose,
                    dtype=torch.float32,
                )
            )
            right_pose = entity._right_view.get_world_pose()
            right_pose[:2, 3] -= arena_pose[:2, 3]
            right_poses.append(
                torch.as_tensor(
                    right_pose,
                    dtype=torch.float32,
                )
            )
        return torch.stack(left_poses, dim=0).to(self.device), torch.stack(
            right_poses, dim=0
        ).to(self.device)

    def set_intrinsics(
        self,
        intrinsics: torch.Tensor,
        right_intrinsics: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """
        Set the camera intrinsics for both left and right cameras.

        Args:
            intrinsics (torch.Tensor): The intrinsics for the left camera with shape (4,) / (3, 3) or (B, 4) / (B, 3, 3).
            right_intrinsics (torch.Tensor | None): The intrinsics for the right camera with shape (4,) / (3, 3) or (B, 4) / (B, 3, 3). If None, use the same intrinsics as the left camera.
            env_ids (Sequence[int] | None): The environment ids to set the intrinsics. If None, set for all environments.
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)

        if intrinsics.dim() == 2 and intrinsics.shape[1] == 3:
            intrinsics = intrinsics.unsqueeze(0).repeat(len(ids), 1, 1)

        if intrinsics.dim() == 1:
            intrinsics = intrinsics.unsqueeze(0).repeat(len(ids), 1)

        if len(ids) != intrinsics.shape[0]:
            logger.log_error(
                f"Intrinsics shape {intrinsics.shape} does not match env_ids length {len(ids)}"
            )

        if right_intrinsics is None:
            right_intrinsics = intrinsics
        else:
            if right_intrinsics.dim() == 2 and right_intrinsics.shape[1] == 3:
                right_intrinsics = right_intrinsics.unsqueeze(0).repeat(len(ids), 1, 1)

            if right_intrinsics.dim() == 1:
                right_intrinsics = right_intrinsics.unsqueeze(0).repeat(len(ids), 1)

            if len(ids) != right_intrinsics.shape[0]:
                logger.log_error(
                    f"Right intrinsics shape {right_intrinsics.shape} does not match env_ids length {len(ids)}"
                )

        for i, env_id in enumerate(ids):
            entity = self._entities[env_id]
            if intrinsics.shape[1] == 3:
                entity._left_view.set_intrinsic(intrinsics[i].cpu().numpy())
                entity._right_view.set_intrinsic(right_intrinsics[i].cpu().numpy())
            else:
                entity._left_view.set_intrinsic(intrinsics[i].cpu().tolist())
                entity._right_view.set_intrinsic(right_intrinsics[i].cpu().tolist())

    def get_intrinsics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the camera intrinsics for both left and right cameras.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The intrinsics for the left and right cameras with shape (B, 3, 3).
        """
        intrinsics_left = []
        intrinsics_right = []
        for entity in self._entities:
            intrinsics_left.append(
                torch.as_tensor(entity._left_view.get_intrinsic(), dtype=torch.float32)
            )
            intrinsics_right.append(
                torch.as_tensor(entity._right_view.get_intrinsic(), dtype=torch.float32)
            )

        return (
            torch.stack(intrinsics_left, dim=0).to(self.device),
            torch.stack(intrinsics_right, dim=0).to(self.device),
        )
