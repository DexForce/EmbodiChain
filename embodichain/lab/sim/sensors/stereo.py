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
import weakref
import dexsim.render as dr

from typing import TYPE_CHECKING, Any, Dict, Tuple, List, Sequence

from embodichain.lab.sim.sensors import Camera, CameraCfg
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
    """Compatibility view of one stereo pair owned by a live Scene camera.

    Pose and attachment operations publish both eyes through the owning
    sensor's native batch. Pose getters return an owned center-pose matrix.

    Args:
        left_view: Native left-eye view used for node access.
        right_view: Native right-eye view paired with ``left_view``.
        left_to_right: Calibration matrix; its translation defines the
            symmetric eye offsets, matching the stereo sensor contract.
        owner: Live sensor owning both Scene camera handles. Required for
            state and attachment operations.
        index: Logical environment row within the owning sensor.
    """

    def __init__(
        self,
        left_view: dr.CameraView,
        right_view: dr.CameraView,
        left_to_right: np.ndarray,
        *,
        owner: Camera | None = None,
        index: int = 0,
    ) -> None:
        self._owner_ref = None if owner is None else weakref.ref(owner)
        self._index = index
        self._left_view = left_view
        self._right_view = right_view
        self._left_to_right = left_to_right

        self._left_to_center = np.eye(4, dtype=np.float32)
        self._left_to_center[:3, 3] = left_to_right[:3, 3] * -0.5

        self._right_to_center = np.eye(4, dtype=np.float32)
        self._right_to_center[:3, 3] = left_to_right[:3, 3] * 0.5

    def _owner(self) -> Camera:
        """Resolve the owning sensor without keeping it alive through a cycle."""
        owner = None if self._owner_ref is None else self._owner_ref()
        if owner is None:
            raise RuntimeError(
                "Stereo view operations require a live Scene-owned sensor."
            )
        return owner

    def _apply_pose(self, pose: np.ndarray, frame: str) -> None:
        """Submit both eye offsets as one native pose operation."""
        owner = self._owner()
        owner._batch.apply_pose(
            np.stack((pose @ self._left_to_center, pose @ self._right_to_center)),
            rows=[self._index, self._index + owner.num_instances],
            frame=frame,
        )

    def _fetch_pose(self, frame: str) -> np.ndarray:
        """Recover the center from one snapshot of both symmetric eye offsets."""
        owner = self._owner()
        poses = owner._batch.fetch_pose(
            rows=[self._index, self._index + owner.num_instances], frame=frame
        )
        center = poses[0].copy()
        center[:3, 3] = (poses[0, :3, 3] + poses[1, :3, 3]) * 0.5
        return center

    def set_local_pose(self, pose: np.ndarray) -> None:
        self._apply_pose(pose, "local")

    def get_local_pose(self) -> np.ndarray:
        return self._fetch_pose("local")

    def set_world_pose(self, pose: np.ndarray) -> None:
        self._apply_pose(pose, "world")

    def get_world_pose(self) -> np.ndarray:
        return self._fetch_pose("world")

    def get_node(self) -> dexsim.engine.Node:
        return self._left_view.get_node()

    def attach_node(self, parent: dexsim.engine.Node) -> None:
        owner = self._owner()
        indices = [self._index, self._index + owner.num_instances]
        owner._scene.attach_render_objects(
            [owner._camera_handles[i] for i in indices],
            [parent.path_name()] * 2,
            local_poses=owner._batch.fetch_pose(rows=indices, frame="local"),
        )


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
        self._left_to_center = torch.eye(4)
        self._right_to_center = torch.eye(4)
        self._left_to_center[:3, 3] = config.left_to_right[:3, 3] * -0.5
        self._right_to_center[:3, 3] = config.left_to_right[:3, 3] * 0.5
        self._offsets = {}
        for i in range(self.num_instances):
            self._entities[i] = PairCameraView(
                left_list[i],
                right_list[i],
                config.left_to_right.numpy(),
                owner=self,
                index=i,
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
        fetch_only = kwargs.get("fetch_only", False)
        if not fetch_only:
            self._frame_buffer.apply_ordered()

        self.cfg: StereoCameraCfg
        if self.cfg.enable_color:
            data = self._frame_buffer.get_rgb_gpu_buffer().to(self.device)
            self._data_buffer["color"] = data[: self.num_instances, ...]
            self._data_buffer[f"color_right"] = data[self.num_instances :, ...]
        if self.cfg.enable_depth:
            data = self._frame_buffer.get_depth_gpu_buffer().to(self.device)
            self._data_buffer["depth"] = data[: self.num_instances, ...].unsqueeze_(-1)
            self._data_buffer[f"depth_right"] = data[
                self.num_instances :, ...
            ].unsqueeze_(-1)
        if self.cfg.enable_mask:
            data = self._frame_buffer.get_visible_mask_gpu_buffer().to(
                self.device, torch.int32
            )
            self._data_buffer["mask"] = data[: self.num_instances, ...].unsqueeze_(-1)
            self._data_buffer[f"mask_right"] = data[
                self.num_instances :, ...
            ].unsqueeze_(-1)
        if self.cfg.enable_normal:
            data = self._frame_buffer.get_normal_gpu_buffer().to(self.device)[..., :3]
            self._data_buffer["normal"] = data[: self.num_instances, ...]
            self._data_buffer[f"normal_right"] = data[self.num_instances :, ...]
        if self.cfg.enable_position:
            data = self._frame_buffer.get_position_gpu_buffer().to(self.device)[..., :3]
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

    def get_left_right_arena_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read both eye poses in arena coordinates from one native snapshot.

        Returns:
            Left and right pose tensors, each with shape (num_envs, 4, 4).
        """
        poses = torch.as_tensor(
            self._batch.fetch_pose(frame="arena", device=str(self.device)),
            dtype=torch.float32,
            device=self.device,
        )
        return poses[: self.num_instances], poses[self.num_instances :]

    @property
    def _eye_count(self) -> int:
        """Two native views share each logical stereo row."""
        return 2

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

    def _eye_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Apply the existing fixed eye offsets in one tensor operation."""
        key = (poses.device, poses.dtype)
        if key not in self._offsets:
            self._offsets[key] = torch.stack(
                (self._left_to_center, self._right_to_center)
            ).to(poses)
        offsets = self._offsets[key]
        return torch.cat((poses @ offsets[0], poses @ offsets[1]))

    def _logical_poses(self, frame: str) -> torch.Tensor:
        """Recover the center frame from a single native pose snapshot."""
        poses = super()._logical_poses(frame)
        left, right = poses[: self.num_instances], poses[self.num_instances :]
        center = left.clone()
        # Symmetric eye positions avoid subtracting a stored baseline from a
        # world-to-arena float32 read, which loses zero-center precision.
        center[:, :3, 3] = (left[:, :3, 3] + right[:, :3, 3]) * 0.5
        return center

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
        self._batch.apply_intrinsics(
            torch.cat((left, right)), rows=self._view_rows(env_ids)
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
