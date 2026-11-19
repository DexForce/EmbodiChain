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
import torch
import numpy as np
import dexsim.render as dr

from dexsim.utility import inv_transform

from typing import TYPE_CHECKING, Any, Dict, Tuple, List, Sequence

from embodichain.lab.sim.sensors import Camera, CameraCfg
from embodichain.lab.sim._render_selection import (
    _normalize_render_rows,
    _select_render_values,
)
from embodichain.utils.math import matrix_from_euler
from embodichain.utils import logger, configclass

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["StereoCamera", "StereoCameraCfg", "PairCameraView"]


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
    """Forward one stereo instance's pose and exclusion policy to both views.

    The owning :class:`StereoCamera` manages the borrowed native views and
    their lifetime. Pose writes account for the configured eye separation.
    """

    def __init__(
        self,
        left_view: dr.CameraView,
        right_view: dr.CameraView,
        left_to_right: np.ndarray,
    ) -> None:
        self._left_view = left_view
        self._right_view = right_view
        self._left_to_right = left_to_right

        self._left_to_center = np.eye(4, dtype=np.float32)
        self._left_to_center[:3, 3] = left_to_right[:3, 3] * -0.5

        self._right_to_center = np.eye(4, dtype=np.float32)
        self._right_to_center[:3, 3] = left_to_right[:3, 3] * 0.5

    def set_excluded_render_categories(self, categories: int) -> None:
        """Apply one environment's visibility policy to both stereo eyes."""
        self._left_view.set_excluded_render_categories(categories)
        self._right_view.set_excluded_render_categories(categories)

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

    def _detach_from_parent_nodes(self) -> None:
        """Detach both native views before their shared parent is rebuilt."""
        for entity in self._entities:
            entity._left_view.get_node().detach_parent()
            entity._right_view.get_node().detach_parent()
        self._is_attached = False

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
        *,
        owner: SimulationManager,
    ) -> None:
        super().__init__(
            config,
            device,
            owner=owner,
        )

        # check valid config
        if self.cfg.enable_disparity and not self.cfg.enable_depth:
            logger.log_error("Disparity can only be enabled when depth is enabled.")

    def _build_sensor_from_config(
        self, config: StereoCameraCfg, device: torch.device
    ) -> None:
        self._create_camera_views(config, (config.intrinsics, config.intrinsics_right))
        left_list = [h.native for h in self._camera_handles[: self.num_instances]]
        right_list = [h.native for h in self._camera_handles[self.num_instances :]]
        self._configure_visual_overlays(
            left_list + right_list, config.include_visual_overlays
        )
        for i in range(self.num_instances):
            self._entities[i] = PairCameraView(
                left_list[i],
                right_list[i],
                config.left_to_right.cpu().numpy(),
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

    def update(self, **kwargs: Any) -> None:
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
        """
        self._capture_metadata = None
        fetch_only = kwargs.get("fetch_only", False)
        if not fetch_only:
            self._pending_capture_metadata = None
            self._apply_camera_groups()

        self.cfg: StereoCameraCfg
        if self.cfg.enable_color:
            data = self._read_groups("get_rgb_gpu_buffer")
            self._data_buffer["color"] = data[: self.num_instances, ...]
            self._data_buffer[f"color_right"] = data[self.num_instances :, ...]
        if self.cfg.enable_depth:
            data = self._read_groups("get_depth_gpu_buffer")
            self._data_buffer["depth"] = data[: self.num_instances, ...].unsqueeze_(-1)
            self._data_buffer[f"depth_right"] = data[
                self.num_instances :, ...
            ].unsqueeze_(-1)
        if self.cfg.enable_mask:
            data = self._read_groups("get_visible_mask_gpu_buffer", torch.int32)
            self._data_buffer["mask"] = data[: self.num_instances, ...].unsqueeze_(-1)
            self._data_buffer[f"mask_right"] = data[
                self.num_instances :, ...
            ].unsqueeze_(-1)
        if self.cfg.enable_normal:
            data = self._read_groups("get_normal_gpu_buffer")[..., :3]
            self._data_buffer["normal"] = data[: self.num_instances, ...]
            self._data_buffer[f"normal_right"] = data[self.num_instances :, ...]
        if self.cfg.enable_position:
            data = self._read_groups("get_position_gpu_buffer")[..., :3]
            self._data_buffer["position"] = data[: self.num_instances, ...]
            self._data_buffer[f"position_right"] = data[self.num_instances :, ...]
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

        self._capture_metadata = self._pending_capture_metadata

    def get_left_right_arena_pose(self) -> torch.Tensor:
        """Get the local pose of the left and right cameras.

        Returns:
            torch.Tensor: The local pose of the left camera with shape (num_envs, 4, 4).
        """
        left_poses = []
        right_poses = []
        for i, entity in enumerate(self._entities):
            arena_pose = self._arenas[i].get_root_node().get_local_pose()
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

    def _view_rows(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Expand selections into left-then-right native view order."""
        if env_ids is None:
            return None
        rows = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=env_ids.device if torch.is_tensor(env_ids) else self.device,
        )
        return torch.cat((rows, rows + self.num_instances))

    @staticmethod
    def _intrinsics_matrices(values: torch.Tensor, count: int) -> torch.Tensor:
        """Preserve complete K validation while packing different eye formats."""
        if values.shape == (3, 3):
            return values.expand(count, 3, 3)
        if values.ndim == 3 and values.shape[1:] == (3, 3):
            if len(values) != count:
                raise ValueError(
                    "Intrinsics row count does not match selected environments."
                )
            return values
        if values.shape == (4,):
            values = values.expand(count, 4)
        if values.shape != (count, 4):
            raise ValueError("Intrinsics must be (4,), (3,3), (N,4) or (N,3,3).")
        result = torch.zeros((count, 3, 3), dtype=values.dtype, device=values.device)
        result[:, 0, 0], result[:, 1, 1] = values[:, 0], values[:, 1]
        result[:, 0, 2], result[:, 1, 2] = values[:, 2], values[:, 3]
        result[:, 2, 2] = 1
        return result

    def set_intrinsics(
        self,
        intrinsics: torch.Tensor,
        right_intrinsics: torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """
        Set the camera intrinsics for both left and right cameras.

        Args:
            intrinsics (torch.Tensor): The intrinsics for the left camera with shape (4,) / (3, 3) or (B, 4) / (B, 3, 3).
            right_intrinsics (torch.Tensor | None): The intrinsics for the right camera with shape (4,) / (3, 3) or (B, 4) / (B, 3, 3). If None, use the same intrinsics as the left camera.
            env_ids (Sequence[int] | None): The environment ids to set the intrinsics. If None, set for all environments.
        """
        count = self.num_instances if env_ids is None else len(env_ids)
        left = self._intrinsics_matrices(intrinsics, count)
        right = (
            left
            if right_intrinsics is None
            else self._intrinsics_matrices(right_intrinsics, count).to(left)
        )
        rows, offsets = _normalize_render_rows(env_ids, self.num_instances)
        left = _select_render_values(left, offsets, 2)
        right = _select_render_values(right, offsets, 2)
        self._batch.apply_intrinsics(
            torch.cat((left, right)), rows=self._view_rows(rows)
        )

    def get_intrinsics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the camera intrinsics for both left and right cameras.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The intrinsics for the left and right cameras with shape (B, 3, 3).
        """
        intrinsics = torch.as_tensor(
            self._batch.fetch_intrinsics(matrix=True, device=str(self.device)),
            dtype=torch.float32,
            device=self.device,
        )
        return intrinsics[: self.num_instances], intrinsics[self.num_instances :]
