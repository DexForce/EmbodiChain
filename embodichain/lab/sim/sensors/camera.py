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
import dexsim.render as dr
import torch.nn.functional as F

from functools import cached_property
from typing import Tuple, Sequence, List

from embodichain.lab.sim.sensors import BaseSensor, SensorCfg
from embodichain.utils.math import matrix_from_quat, quat_from_matrix, look_at_to_pose
from embodichain.utils import logger, configclass


# Shared camera groups, populated by plan_camera_groups() and consumed by
# Camera._build_sensor_from_config(). Each slot owns one fixed-size layered
# framebuffer; mixed-resolution slots render at their largest camera resolution.
_SHARED_GROUPS: dict = {}
_MIXED_RESOLUTION_GROUP = "mixed_resolution"


def plan_camera_groups(
    sensor_cfgs: Sequence[SensorCfg],
    enabled: bool,
    merge_different_resolutions: bool = False,
) -> None:
    """Pre-allocate one shared camera group per distinct Camera resolution.

    Must be called after the arenas exist and before any sensor is constructed,
    because a group's layer count is fixed at creation and the cameras are then
    created into it in order.

    A group is a single layered image with one extent on every layer. By
    default, only same-resolution cameras share a group. When
    ``merge_different_resolutions`` is enabled, all Camera sensors share one
    group at the largest requested extent. Smaller cameras render with scaled
    intrinsics and their returned tensors are resized back to their configured
    size.

    Args:
        sensor_cfgs: every sensor config that will be built, in build order.
        enabled: RenderCfg.merge_camera_groups. When False this clears the
            registry and each Camera falls back to its own group.
        merge_different_resolutions: RenderCfg.merge_different_resolution_camera_groups.
            This is opt-in because it trades small-camera render cost for fewer
            group submissions.
    """
    _SHARED_GROUPS.clear()
    if not enabled:
        return

    world = dexsim.default_world()
    env = world.get_env()
    arenas = env.get_all_arenas()
    num_instances = len(arenas) if len(arenas) > 0 else 1

    cameras = [
        cfg
        for cfg in sensor_cfgs
        if getattr(cfg, "sensor_type", "Camera") == "Camera"
    ]
    if merge_different_resolutions and len(cameras) >= 2:
        width = max(cfg.width for cfg in cameras)
        height = max(cfg.height for cfg in cameras)
        _SHARED_GROUPS[_MIXED_RESOLUTION_GROUP] = {
            "frame_buffer": world.create_camera_group(
                [width, height], num_instances * len(cameras), True
            ),
            "next_offset": 0,
            "capacity": num_instances * len(cameras),
            "num_instances": num_instances,
            "resolution": (width, height),
        }
        logger.log_info(
            f"[camera] merged {len(cameras)} cameras across resolutions into one "
            f"{width}x{height} group ({num_instances * len(cameras)} layers)"
        )
        return

    counts: dict = {}
    for cfg in cameras:
        counts.setdefault((cfg.width, cfg.height), 0)
        counts[(cfg.width, cfg.height)] += 1

    for (width, height), count in counts.items():
        if count < 2:
            continue
        _SHARED_GROUPS[(width, height)] = {
            "frame_buffer": world.create_camera_group(
                [width, height], num_instances * count, True
            ),
            "next_offset": 0,
            "capacity": num_instances * count,
            "num_instances": num_instances,
            "resolution": (width, height),
        }
        logger.log_info(
            f"[camera] merged {count} x {width}x{height} cameras into one camera group "
            f"({num_instances * count} layers)"
        )
    if _SHARED_GROUPS:
        merged = sum(c for (r, c) in counts.items() if c >= 2)
        logger.log_info(
            f"[camera] merge_camera_groups: {len(counts)} resolutions, "
            f"{merged} of {sum(counts.values())} cameras share {len(_SHARED_GROUPS)} group(s)"
        )


def _resize_camera_output(
    output: torch.Tensor,
    height: int,
    width: int,
    mode: str,
    normalize: bool = False,
) -> torch.Tensor:
    """Resize a camera-group layer while preserving its channel-last layout."""
    if tuple(output.shape[1:3]) == (height, width):
        return output

    dtype = output.dtype
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    if output.ndim == 4:
        source = output.permute(0, 3, 1, 2).float()
        resized = F.interpolate(source, size=(height, width), mode=mode, **kwargs)
        resized = resized.permute(0, 2, 3, 1)
        if normalize:
            resized = F.normalize(resized, dim=-1)
    else:
        source = output.unsqueeze(1).float()
        resized = F.interpolate(source, size=(height, width), mode=mode, **kwargs)
        resized = resized[:, 0]

    if dtype == torch.uint8:
        return resized.round().clamp_(0, 255).to(dtype)
    return resized.to(dtype)


@configclass
class CameraCfg(SensorCfg):
    """Configuration class for Camera."""

    @configclass
    class ExtrinsicsCfg(SensorCfg.OffsetCfg):
        """Configuration class for camera extrinsics.

        The extrinsics define the position and orientation of the camera in the 3D world.
        If eye, target, and up are provided, they will be used to compute the extrinsics.
        Otherwise, the position and orientation will be set to the defaults.
        """

        eye: Tuple[float, float, float] | None = None
        target: Tuple[float, float, float] | None = None
        up: Tuple[float, float, float] | None = None
        """Alternative way to specify the camera extrinsics using eye, target, and up vectors."""

        @property
        def transformation(self) -> torch.Tensor:
            if self.eye:
                self.up = (0.0, 0.0, 1.0) if self.up is None else self.up
                return look_at_to_pose(self.eye, self.target, self.up).squeeze(0)
            else:
                return super().transformation

    sensor_type: str = "Camera"

    # Camera parameters
    width: int = 640
    height: int = 480
    near: float = 0.005
    far: float = 100.0

    # The camera intrinsics are defined as (fx, fy, cx, cy)
    intrinsics: Tuple[float, float, float, float] = (600, 600, 320.0, 240.0)
    extrinsics: ExtrinsicsCfg = ExtrinsicsCfg()

    enable_color: bool = True
    enable_depth: bool = False
    enable_mask: bool = False
    enable_normal: bool = False
    enable_position: bool = False

    fx: float = intrinsics[0]
    fy: float = intrinsics[1]
    cx: float = intrinsics[2]
    cy: float = intrinsics[3]

    def get_view_attrib(self) -> dr.ViewFlags:
        """Get the view attributes for the camera.

        The camera view whcich is used to render the scene
        Default view attributes for the camera are: [COLOR, DEPTH, MASK]
        The supported view attributes are:
            - COLOR: RGBA images
            - DEPTH: Depth images
            - MASK: Instance segmentation masks
            - NORMAL: Normal images
            - POSITION: Position images with 3D coordinates.

        Returns:
            The view attributes for the camera.
        """
        view_attrib: dr.ViewFlags = dr.ViewFlags.COLOR
        if self.enable_color:
            view_attrib |= dr.ViewFlags.COLOR
        if self.enable_depth:
            view_attrib |= dr.ViewFlags.DEPTH
        if self.enable_mask:
            view_attrib |= dr.ViewFlags.MASK
        if self.enable_normal:
            view_attrib |= dr.ViewFlags.NORMAL
        if self.enable_position:
            view_attrib |= dr.ViewFlags.POSITION
        return view_attrib

    def get_data_types(self) -> List[str]:
        data_types = []
        if self.enable_color:
            data_types.append("color")
        if self.enable_depth:
            data_types.append("depth")
        if self.enable_mask:
            data_types.append("mask")
        if self.enable_normal:
            data_types.append("normal")
        if self.enable_position:
            data_types.append("position")
        return data_types


class Camera(BaseSensor):
    """Base class for sensor abstraction in the simulation engine.

    Sensors should inherit from this class and implement the `update` and `get_data` methods.
    """

    SUPPORTED_DATA_TYPES = ["color", "depth", "mask", "normal", "position"]

    def __init__(
        self, config: CameraCfg, device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__(config, device)

    def _build_sensor_from_config(
        self, config: CameraCfg, device: torch.device
    ) -> None:
        self._world = dexsim.default_world()
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        num_instances = len(arenas)

        # Claim a layer slice of the group shared by every camera at this
        # resolution, if plan_camera_groups() reserved one. Otherwise keep the
        # one-group-per-camera behaviour.
        #
        # The capacity check matters: a group's layers are fixed at creation for
        # the cameras plan_camera_groups() knew about, but cameras are also added
        # outside that pass (e.g. the recording camera in gym managers/record.py).
        # An unplanned camera must get its own group rather than claim a layer
        # the group does not have.
        slot = _SHARED_GROUPS.get(_MIXED_RESOLUTION_GROUP)
        if slot is None:
            slot = _SHARED_GROUPS.get((config.width, config.height))
        if slot is not None and slot["next_offset"] + num_instances <= slot["capacity"]:
            self._frame_buffer = slot["frame_buffer"]
            self._layer_offset = slot["next_offset"]
            slot["next_offset"] += num_instances
            self._render_width, self._render_height = slot["resolution"]
        else:
            if slot is not None:
                logger.log_info(
                    f"[camera] {config.uid}: shared {config.width}x{config.height} group is "
                    "full (camera added outside plan_camera_groups); using its own group"
                )
            self._frame_buffer = self._world.create_camera_group(
                [config.width, config.height], num_instances, True
            )
            self._layer_offset = 0
            self._render_width, self._render_height = config.width, config.height

        view_attrib = config.get_view_attrib()
        scale_x = self._render_width / config.width
        scale_y = self._render_height / config.height
        self._intrinsic_scale_x = scale_x
        self._intrinsic_scale_y = scale_y
        render_intrinsics = (
            config.intrinsics[0] * scale_x,
            config.intrinsics[1] * scale_y,
            config.intrinsics[2] * scale_x,
            config.intrinsics[3] * scale_y,
        )
        for i, arena in enumerate(arenas):
            view_name = f"{self.uid}_view{i + 1}"
            view = arena.create_camera(
                view_name,
                self._render_width,
                self._render_height,
                True,
                view_attrib,
                self._frame_buffer,
            )
            view.set_intrinsic(render_intrinsics)
            view.set_near(config.near)
            view.set_far(config.far)
            self._entities[i] = view

        # Define a mapping of data types to their respective shapes and dtypes
        buffer_specs = {
            "color": (
                (self.num_instances, config.height, config.width, 4),
                torch.uint8,
            ),
            "depth": (
                (self.num_instances, config.height, config.width),
                torch.float32,
            ),
            "mask": (
                (self.num_instances, config.height, config.width),
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
        }
        data_types = config.get_data_types()

        # Iterate through enabled data types and initialize buffers
        for data_type in data_types:
            if getattr(config, f"enable_{data_type}", False):
                shape, dtype = buffer_specs[data_type]
                self._data_buffer[data_type] = torch.empty(
                    shape, dtype=dtype, device=device
                )

        self.cfg: CameraCfg = config
        if self.cfg.extrinsics.parent is not None:
            self._attach_to_entity()

    @cached_property
    def group_id(self) -> int:
        """Get the camera group ID in the dexsim world.

        Returns:
            int: The camera group ID.
        """
        return self._frame_buffer.get_group_id()

    @property
    def is_attached(self) -> bool:
        """Check if the camera is attached to a parent entity.

        Returns:
            bool: True if the camera is attached to a parent entity, False otherwise.
        """
        return self.cfg.extrinsics.parent is not None

    def update(self, **kwargs) -> None:
        """Update the sensor data.

        The supported data types are:
            - color: RGB images with shape (B, H, W, 4) and dtype torch.uint8
            - depth: Depth images with shape (B, H, W) and dtype torch.float32
            - mask: Instance segmentation masks with shape (B, H, W) and dtype torch.int32
            - normal: Normal images with shape (B, H, W, 3) and dtype torch.float32
            - position: Position images with shape (B, H, W, 3) and dtype torch.float32

        Args:
            **kwargs: Additional keyword arguments for sensor update.
        """
        fetch_only = kwargs.get("fetch_only", False)
        if not fetch_only:
            self._frame_buffer.apply()
        self.cfg: CameraCfg

        # When the group is shared with other cameras it holds every camera's
        # layers, so take only this camera's slice. Unshared groups have
        # _layer_offset == 0 and exactly num_instances layers, so this is a no-op.
        lo = self._layer_offset
        hi = lo + self.num_instances

        if self.cfg.enable_color:
            self._data_buffer["color"] = _resize_camera_output(
                self._frame_buffer.get_rgb_gpu_buffer().to(self.device)[lo:hi, ...],
                self.cfg.height,
                self.cfg.width,
                "bilinear",
            )

        if self.cfg.enable_depth:
            self._data_buffer["depth"] = _resize_camera_output(
                self._frame_buffer.get_depth_gpu_buffer().to(self.device)[lo:hi, ...],
                self.cfg.height,
                self.cfg.width,
                "bilinear",
            )

        if self.cfg.enable_mask:
            self._data_buffer["mask"] = _resize_camera_output(
                self._frame_buffer.get_visible_mask_gpu_buffer().to(
                    self.device, torch.int32
                )[lo:hi, ...],
                self.cfg.height,
                self.cfg.width,
                "nearest",
            )

        if self.cfg.enable_normal:
            self._data_buffer["normal"] = _resize_camera_output(
                self._frame_buffer.get_normal_gpu_buffer().to(self.device)[
                    lo:hi, ..., :3
                ],
                self.cfg.height,
                self.cfg.width,
                "bilinear",
                normalize=True,
            )

        if self.cfg.enable_position:
            self._data_buffer["position"] = _resize_camera_output(
                self._frame_buffer.get_position_gpu_buffer().to(self.device)[
                    lo:hi, ..., :3
                ],
                self.cfg.height,
                self.cfg.width,
                "bilinear",
            )

    def _attach_to_entity(self) -> None:
        """Attach the sensor to the parent entity in each environment."""
        env = self._world.get_env()
        for i, entity in enumerate(self._entities):

            parent = None
            if i == 0:
                parent = env.find_node(f"{self.cfg.extrinsics.parent}")
            else:
                parent = env.find_node(f"{self.cfg.extrinsics.parent}.{i-1}")
            if parent is None:
                logger.log_error(
                    f"Failed to find parent entity {self.cfg.extrinsics.parent} for sensor {self.cfg.uid}."
                )

            entity.attach_node(parent)

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set the local pose of the camera.

        Note: The pose should be in the OpenGL coordinate system, which means the Y is up and Z is forward.

        Args:
            pose (torch.Tensor): The local pose to set, should be a 4x4 transformation matrix.
            env_ids (Sequence[int] | None): The environment IDs to set the pose for. If None, set for all environments.
        """
        if env_ids is None:
            local_env_ids = range(len(self._entities))
        else:
            local_env_ids = env_ids

        pose = pose.cpu()
        if pose.dim() == 2 and pose.shape[1] == 7:
            pose_matrix = torch.eye(4).unsqueeze(0).repeat(pose.shape[0], 1, 1)
            pose_matrix[:, :3, 3] = pose[:, :3]
            pose_matrix[:, :3, :3] = matrix_from_quat(pose[:, 3:7])
            for i, env_idx in enumerate(local_env_ids):
                self._entities[env_idx].set_local_pose(pose_matrix[i].numpy())
        elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
            for i, env_idx in enumerate(local_env_ids):
                self._entities[env_idx].set_local_pose(pose[i].numpy())
        else:
            logger.log_error(
                f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
            )

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get the local pose of the camera.

        Args:
            to_matrix (bool): If True, return the pose as a 4x4 matrix. If False, return as a quaternion.

        Returns:
            torch.Tensor: The local pose of the camera.
        """
        poses = []
        for entity in self._entities:
            pose = entity.get_local_pose()
            poses.append(torch.as_tensor(pose, dtype=torch.float32))

        poses = torch.stack(poses, dim=0).to(self.device)
        if to_matrix is False:
            xyz = poses[:, :3, 3]
            quat = quat_from_matrix(poses[:, :3, :3])
            return torch.cat((xyz, quat), dim=-1)
        return poses

    def get_arena_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Get the pose of the sensor in the arena frame.

        Args:
            to_matrix (bool): If True, return the pose as a 4x4 transformation matrix.

        Returns:
            A tensor representing the pose of the sensor in the arena frame.
        """
        from embodichain.lab.sim.utility import get_dexsim_arenas

        arenas = get_dexsim_arenas()

        poses = []
        for i, entity in enumerate(self._entities):
            pose = entity.get_world_pose()
            pose[:2, 3] -= arenas[i].get_root_node().get_local_pose()[:2, 3]
            poses.append(torch.as_tensor(pose, dtype=torch.float32))

        poses = torch.stack(poses, dim=0).to(self.device)
        if to_matrix is False:
            xyz = poses[:, :3, 3]
            quat = quat_from_matrix(poses[:, :3, :3])
            return torch.cat((xyz, quat), dim=-1)
        return poses

    def look_at(
        self,
        eye: torch.Tensor,
        target: torch.Tensor,
        up: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set the camera to look at a target point.

        Args:
            eye (torch.Tensor): The position of the camera (eye) with shape (N, 3).
            target (torch.Tensor): The point the camera should look at (target) with shape (N, 3).
            up (torch.Tensor | None): The up direction vector. If None, defaults to [0, 0, 1].
            env_ids (Sequence[int] | None): The environment IDs to set the look at for. If None, set for all environments.
        """
        if up is None:
            up = torch.tensor([[0.0, 0.0, 1.0]]).repeat(eye.shape[0], 1)

        pose = look_at_to_pose(eye, target, up)
        # To opengl coordinate system.
        pose[:, :3, 1] = -pose[:, :3, 1]
        pose[:, :3, 2] = -pose[:, :3, 2]
        self.set_local_pose(pose, env_ids=env_ids)

    def set_intrinsics(
        self,
        intrinsics: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """
        Set the camera intrinsics for both left and right cameras.

        Args:
            intrinsics (torch.Tensor): The intrinsics for the left camera with shape (4,) / (3, 3) or (N, 4) / (N, 3, 3).
            env_ids (Sequence[int] | None): The environment ids to set the intrinsics. If None, set for all environments.
        """
        ids = env_ids if env_ids is not None else range(self.num_instances)

        if intrinsics.dim() == 2 and intrinsics.shape[1] == 3:
            intrinsics = intrinsics.unsqueeze(0).repeat(len(ids), 1, 1)

        if intrinsics.dim() == 1:
            intrinsics = intrinsics.unsqueeze(0).repeat(len(ids), 1)

        if len(ids) != intrinsics.shape[0]:
            logger.log_error(
                f"Invalid intrinsics shape {intrinsics.shape} for {len(ids)} environments."
            )

        scaled_intrinsics = intrinsics.clone()
        if scaled_intrinsics.shape[1] == 3:
            scaled_intrinsics[:, 0, :] *= self._intrinsic_scale_x
            scaled_intrinsics[:, 1, :] *= self._intrinsic_scale_y
        else:
            scaled_intrinsics[:, 0] *= self._intrinsic_scale_x
            scaled_intrinsics[:, 1] *= self._intrinsic_scale_y
            scaled_intrinsics[:, 2] *= self._intrinsic_scale_x
            scaled_intrinsics[:, 3] *= self._intrinsic_scale_y

        for i, env_id in enumerate(ids):
            entity = self._entities[env_id]
            if scaled_intrinsics.shape[1] == 3:
                entity.set_intrinsic(scaled_intrinsics[i].cpu().numpy())
            else:
                entity.set_intrinsic(scaled_intrinsics[i].cpu().tolist())

    def get_intrinsics(self) -> torch.Tensor:
        """
        Get the camera intrinsics for both left and right cameras.

        Returns:
            torch.Tensor: The intrinsics for the left camera with shape (N, 3, 3).
        """
        intrinsics = []
        for entity in self._entities:
            intrinsics.append(
                torch.as_tensor(entity.get_intrinsic(), dtype=torch.float32)
            )

        result = torch.stack(intrinsics, dim=0).to(self.device)
        result[:, 0, :] /= self._intrinsic_scale_x
        result[:, 1, :] /= self._intrinsic_scale_y
        return result

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.cfg: CameraCfg

        if self.cfg.extrinsics.eye is not None:
            eye = (
                torch.tensor(self.cfg.extrinsics.eye, dtype=torch.float32)
                .squeeze_(0)
                .repeat(self.num_instances, 1)
            )
            target = (
                torch.tensor(self.cfg.extrinsics.target, dtype=torch.float32)
                .squeeze_(0)
                .repeat(self.num_instances, 1)
            )
            up = (
                torch.tensor(self.cfg.extrinsics.up, dtype=torch.float32)
                .squeeze_(0)
                .repeat(self.num_instances, 1)
                if self.cfg.extrinsics.up is not None
                else None
            )
            self.look_at(eye, target, up, env_ids=env_ids)
        else:
            pose = self.cfg.extrinsics.transformation
            pose = pose.unsqueeze_(0).repeat(self.num_instances, 1, 1)

            if self.cfg.extrinsics.parent is None:
                # To opengl coordinate system.
                pose[:, :3, 1] = -pose[:, :3, 1]
                pose[:, :3, 2] = -pose[:, :3, 2]

            self.set_local_pose(pose, env_ids=env_ids)
