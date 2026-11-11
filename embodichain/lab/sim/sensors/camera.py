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
import operator
import weakref
import numpy as np
import torch
import dexsim.render as dr

from typing import TYPE_CHECKING, Any, List, Literal, Sequence, Tuple

from embodichain.lab.sim.sensors import BaseSensor, SensorCfg
from embodichain.lab.visualization.markers._native import _MARKER_RENDER_CATEGORY
from embodichain.lab.sim._render_selection import (
    _normalize_render_rows,
    _select_render_values,
)
from embodichain.utils.math import matrix_from_quat, quat_from_matrix, look_at_to_pose
from embodichain.utils import logger, configclass

if TYPE_CHECKING:
    from dexsim.scene import CameraBatch, Scene, SpawnedCamera, SpawnedCameraGroup
    from embodichain.lab.sim.sim_manager import SimulationManager
    from ._camera_group_pool import _CameraGroupLease

__all__ = ["Camera", "CameraCfg"]


@configclass
class CameraCfg(SensorCfg):
    """Configuration class for Camera."""

    @configclass
    class ExtrinsicsCfg(SensorCfg.OffsetCfg):
        """Configuration class for camera extrinsics.

        The extrinsics define the position and orientation of the camera in the 3D world.
        If eye, target, and up are provided, they will be used to compute the extrinsics.
        Otherwise, the position and orientation will be set to the defaults.

        SimulationManager resolves ``parent`` as a Robot or Articulation link
        name. Use ``"<asset_uid>/<link_name>"`` when the link name is ambiguous.
        Explicit mounted poses use OpenGL axes (-Z forward, +Y up).
        Unparented explicit poses use optical axes (+Z forward, +Y down) and
        are converted in :meth:`Camera.reset`. Look-at inputs specify viewing
        direction directly in either case.
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

    visualization_role: Literal["sensor", "record"] = "sensor"
    """Role used to group this camera in live visualization previews."""

    include_visual_overlays: bool = False
    """Include native visual overlays and scene-lit markers in captured images."""

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

    ground_plane_visible: bool = True
    """Include the shared default ground in this camera (both stereo eyes).

    The manager's global ground visibility remains a master switch. Hiding
    the ground requires DexSim camera render-category support; collision
    geometry and ground embedded in scanned assets are unaffected.
    """

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
        self,
        config: CameraCfg,
        device: torch.device = torch.device("cpu"),
        *,
        owner: SimulationManager,
    ) -> None:
        owner._require_native_renderer("Native cameras")
        if not isinstance(config.ground_plane_visible, bool):
            raise ValueError("ground_plane_visible must be a bool")
        if not config.ground_plane_visible:
            owner._require_ground_plane_filtering()
        self._owner_ref = weakref.ref(owner)
        self._ground_plane_visibility = [True] * owner.num_envs
        self._capture_metadata: dict[str, object] | None = None
        self._capture_sequence = 0
        self._pending_capture_metadata: dict[str, object] | None = None
        self._world = owner.get_world()
        self._scene: Scene = owner._spawn_scene.builder.result
        self._arena_names = owner._spawn_scene.arena_names
        self._arenas = [owner.get_env(i) for i in range(owner.num_envs)]
        if len(self._arenas) == 0:
            raise ValueError("Camera requires at least one materialized Arena.")
        self._camera_names: list[tuple[dexsim.environment.Arena, str]] = []
        self._camera_handles: tuple[SpawnedCamera, ...] = ()
        self._camera_batch: CameraBatch | None = None
        self._batch_revision = -1
        self._camera_group = None
        self._camera_groups: tuple[SpawnedCameraGroup, ...] = ()
        self._frame_buffers: tuple[Any, ...] = ()
        self._camera_lease: _CameraGroupLease | None = None
        self._is_attached = False
        self._is_destroyed = False
        try:
            super().__init__(config, device, num_instances=len(self._arenas))
            self.reset()
        except Exception:
            self.destroy()
            raise

    @property
    def _batch(self) -> CameraBatch:
        """Refresh the weak native selection after a physical topology change."""
        if self._is_destroyed:
            raise RuntimeError("Camera has been destroyed.")
        revision = self._scene.topology_revision
        if self._camera_batch is None or self._batch_revision != revision:
            self._camera_batch = self._scene.create_camera_batch(self._camera_handles)
            self._batch_revision = revision
        return self._camera_batch

    def _create_camera_views(
        self, config: CameraCfg, intrinsics: Sequence[Sequence[float]]
    ) -> None:
        """Create all eyes together in the already prepared, Scene-owned Arenas."""
        from dexsim.spawn import CameraDesc

        total_views = self.num_instances * len(intrinsics)
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("Camera owner is unavailable.")
        lease = owner._camera_group_pool.acquire(config, total_views, self._scene)
        self._camera_lease = lease
        self._camera_groups = lease.groups
        self._camera_group = lease.groups[0]
        self._frame_buffers = tuple(group.native for group in lease.groups)
        descriptors = []
        for eye, values in enumerate(intrinsics):
            matrix = np.eye(3, dtype=np.float32)
            matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2] = values
            for i in range(self.num_instances):
                suffix = (
                    "" if len(intrinsics) == 1 else ("_left" if eye == 0 else "_right")
                )
                descriptors.append(
                    CameraDesc(
                        name=f"{config.uid}{suffix}_view{i + 1}",
                        width=config.width,
                        height=config.height,
                        intrinsic=matrix,
                        near=config.near,
                        far=config.far,
                        view_flags=config.get_view_attrib(),
                    )
                )
        self._camera_handles = self._scene.add_cameras(
            descriptors,
            arena_names=list(self._arena_names) * len(intrinsics),
            camera_groups=lease.camera_groups,
        )
        self._frame_buffer = self._frame_buffers[0]

    def _build_sensor_from_config(
        self, config: CameraCfg, device: torch.device
    ) -> None:
        self._create_camera_views(config, (config.intrinsics,))
        self._entities = [handle.native for handle in self._camera_handles]

        self._configure_visual_overlays(self._entities, config.include_visual_overlays)

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

    def _configure_visual_overlays(
        self, views: Sequence[dr.CameraView], include: bool
    ) -> None:
        if include:
            for frame_buffer in self._frame_buffers:
                frame_buffer.set_visual_overlay_enabled(True)
        else:
            for view in views:
                view.set_excluded_render_categories(_MARKER_RENDER_CATEGORY)

    def set_ground_plane_visibility(
        self, visible: bool, env_ids: Sequence[int] | None = None
    ) -> None:
        """Include or exclude the shared ground for selected camera instances.

        This changes every image output, shadow and reflection consistently.
        It preserves collisions and other cameras. ``reset`` restores the
        configured policy; topology reattachment preserves the current policy.
        Marker filtering retains the configured ``include_visual_overlays`` policy.

        Args:
            visible: Whether this camera permits rendering the default ground.
            env_ids: Environment indices, or all instances when omitted.

        Raises:
            ValueError: The flag or environment indices are invalid.
            RuntimeError: The camera is destroyed or the SDK lacks filtering.
        """
        if self._is_destroyed:
            raise RuntimeError("Camera is destroyed")
        if not isinstance(visible, bool):
            raise ValueError("visible must be a bool")
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            # One host transfer for a selected reset, rather than one .item()
            # synchronization per environment when the indices are on CUDA.
            env_ids = env_ids.detach().cpu().tolist()
        if env_ids is not None and any(isinstance(index, bool) for index in env_ids):
            raise ValueError("env_ids must contain integers, not a boolean mask")
        try:
            ids = (
                list(range(self.num_instances))
                if env_ids is None
                else [operator.index(index) for index in env_ids]
            )
        except TypeError as error:
            raise ValueError("env_ids must contain integers") from error
        if any(index < 0 or index >= self.num_instances for index in ids):
            raise ValueError("env_ids contains an out-of-range environment")
        if len(set(ids)) != len(ids):
            raise ValueError("env_ids must not contain duplicates")
        changed = [
            index for index in ids if self._ground_plane_visibility[index] != visible
        ]
        if not changed:
            return
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("Camera owner is unavailable")
        owner._require_ground_plane_filtering()
        excluded = 0 if visible else owner._GROUND_RENDER_CATEGORY
        if not self.cfg.include_visual_overlays:
            excluded |= _MARKER_RENDER_CATEGORY
        self._capture_metadata = None
        for index in changed:
            self._entities[index].set_excluded_render_categories(excluded)
            self._ground_plane_visibility[index] = visible

    def get_capture_metadata(self) -> dict[str, object] | None:
        """Return a copy of metadata for the last successful image update.

        ``ground_plane_visible`` records the effective per-environment policy,
        including the global switch; it does not assert floor pixel coverage.
        Metadata is unavailable before acquisition or after a policy change.
        """
        return None if self._capture_metadata is None else dict(self._capture_metadata)

    def _snapshot_render_metadata(self) -> None:
        owner = self._owner_ref()
        global_visible = owner is not None and owner._spawn_default_plane_visibility
        self._capture_sequence += 1
        self._pending_capture_metadata = {
            "capture_sequence": self._capture_sequence,
            "camera_uid": self.cfg.uid,
            "ground_plane_visible": tuple(
                visible and global_visible for visible in self._ground_plane_visibility
            ),
        }

    @property
    def group_id(self) -> int:
        """Get the camera group ID when this sensor owns one native group.

        Returns:
            int: The camera group ID.
        """
        if len(self._camera_groups) != 1:
            raise RuntimeError("This camera has multiple groups; use group_ids.")
        return self._camera_groups[0].get_group_id()

    @property
    def group_ids(self) -> tuple[int, ...]:
        """Return native group IDs in view order."""
        return tuple(group.get_group_id() for group in self._camera_groups)

    def _read_groups(
        self, getter_name: str, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Read only this camera's slots from shared native targets."""
        outputs = []
        if self._camera_lease is None:
            raise RuntimeError("Camera has no live group allocation.")
        for segment in self._camera_lease.slices:
            tensor = getattr(segment.group.native, getter_name)()[
                segment.slot_start : segment.slot_start + segment.count
            ]
            if self.device.type == "cpu" and tensor.device.type == "cuda":
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                output = torch.empty(
                    tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
                )
                output.copy_(tensor, non_blocking=False)
            else:
                output = (
                    tensor.to(self.device)
                    if dtype is None
                    else tensor.to(self.device, dtype)
                )
            outputs.append(output)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def _apply_camera_groups(self) -> None:
        """Capture shared groups and publish every affected camera's metadata."""
        owner = self._owner_ref()
        if owner is None or self._camera_lease is None:
            raise RuntimeError("Camera owner is unavailable.")
        pool = owner._camera_group_pool
        if pool.is_shared(self._camera_lease):
            owner.render_camera_group(
                list(pool.group_ids_for_bucket(self._camera_lease))
            )
        else:
            for group in self._camera_groups:
                group.apply_ordered()
            self._snapshot_render_metadata()

    @property
    def is_attached(self) -> bool:
        """Return whether all camera views are attached to parent nodes.

        Returns:
            True after parent attachment and extrinsics application succeed.
        """
        return self._is_attached

    def update(self, **kwargs: Any) -> None:
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
        self._capture_metadata = None
        fetch_only = kwargs.get("fetch_only", False)
        if not fetch_only:
            self._pending_capture_metadata = None
            self._apply_camera_groups()
        self.cfg: CameraCfg

        if self.cfg.enable_color:
            self._data_buffer["color"] = self._read_groups("get_rgb_gpu_buffer")

        if self.cfg.enable_depth:
            self._data_buffer["depth"] = self._read_groups("get_depth_gpu_buffer")

        if self.cfg.enable_mask:
            self._data_buffer["mask"] = self._read_groups(
                "get_visible_mask_gpu_buffer", torch.int32
            )

        if self.cfg.enable_normal:
            self._data_buffer["normal"] = self._read_groups("get_normal_gpu_buffer")[
                ..., :3
            ]

        if self.cfg.enable_position:
            self._data_buffer["position"] = self._read_groups(
                "get_position_gpu_buffer"
            )[..., :3]

        self._capture_metadata = self._pending_capture_metadata

    def _detach_from_parent_nodes(self) -> None:
        """Retain camera views while their Spawn-owned parents are rebuilt."""
        for entity in self._entities:
            entity.get_node().detach_parent()
        self._is_attached = False

    def attach_to_parent_nodes(self, parent_nodes: Sequence[object]) -> None:
        """Attach camera views to one resolved parent node per environment.

        Args:
            parent_nodes: Parent render nodes ordered by environment index.

        Raises:
            RuntimeError: If the number of parent nodes does not match the
                number of camera instances.
            ValueError: If any parent node is missing.
        """
        nodes = list(parent_nodes)
        if len(nodes) != self.num_instances:
            raise RuntimeError(
                f"Camera attachment received {len(nodes)} parent nodes for "
                f"{self.num_instances} camera instances."
            )
        if any(node is None for node in nodes):
            raise ValueError("Camera attachment requires a parent node in every arena.")
        for entity, parent in zip(self._entities, nodes, strict=True):
            entity.attach_node(parent)
        # Extrinsics are expressed in the parent frame. Reapply them after
        # reparenting because the camera was initially reset in Arena space.
        self._reset_pose()
        self._is_attached = True

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set the local pose of the camera.

        Note:
            The pose uses OpenGL camera axes: +Y is up and -Z is forward.

        Args:
            pose (torch.Tensor): The local pose as ``(N, 4, 4)`` matrices or
                ``(N, 7)`` vectors in ``(x, y, z, qx, qy, qz, qw)`` order.
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
            to_matrix (bool): If True, return the pose as a 4x4 matrix. If
                False, return ``(x, y, z, qx, qy, qz, qw)``.

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
            to_matrix (bool): If True, return the pose as a 4x4 transformation
                matrix. If False, return ``(x, y, z, qx, qy, qz, qw)``.

        Returns:
            A tensor representing the pose of the sensor in the arena frame.
        """
        poses = []
        for i, entity in enumerate(self._entities):
            pose = entity.get_world_pose()
            pose[:2, 3] -= self._arenas[i].get_root_node().get_local_pose()[:2, 3]
            poses.append(torch.as_tensor(pose, dtype=torch.float32))

        poses = torch.stack(poses, dim=0).to(self.device)
        if to_matrix is False:
            xyz = poses[:, :3, 3]
            quat = quat_from_matrix(poses[:, :3, :3])
            return torch.cat((xyz, quat), dim=-1)
        return poses

    def destroy(self) -> None:
        """Remove render cameras before releasing their World-owned group."""
        if self._is_destroyed:
            return
        self._camera_batch = None
        if self._camera_handles:
            self._scene.remove_cameras(self._camera_handles)
            self._camera_handles = ()
        if hasattr(self, "_data_buffer"):
            self._data_buffer.clear()
        if hasattr(self, "_data_buffer_stereo"):
            self._data_buffer_stereo.clear()
        self._frame_buffer = None
        if self._camera_lease is not None:
            owner = self._owner_ref()
            if owner is not None:
                owner._camera_group_pool.release(self._camera_lease)
            self._camera_lease = None
        self._camera_groups = ()
        self._camera_group = None
        self._frame_buffers = ()
        self._is_destroyed = True
        self._capture_metadata = None
        self._entities = []
        self._camera_names = []
        self._frame_buffer = None
        self._is_attached = False
        self._arenas = []
        self._world = None
        self._scene = None

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
            up = torch.tensor([[0.0, 0.0, 1.0]], device=eye.device).repeat(
                eye.shape[0], 1
            )

        pose = look_at_to_pose(eye, target, up)
        # To opengl coordinate system.
        pose[:, :3, 1] = -pose[:, :3, 1]
        pose[:, :3, 2] = -pose[:, :3, 2]
        self.set_local_pose(pose, env_ids=env_ids)

    def set_intrinsics(
        self,
        intrinsics: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """
        Set intrinsics for the selected camera instances in one native batch.

        Args:
            intrinsics (torch.Tensor): The intrinsics for the left camera with shape (4,) / (3, 3) or (N, 4) / (N, 3, 3).
            env_ids (Sequence[int] | None): The environment ids to set the intrinsics. If None, set for all environments.
        """
        rows, offsets = _normalize_render_rows(env_ids, self.num_instances)
        broadcast_ndim = 2 if intrinsics.shape[-2:] == (3, 3) else 1
        self._batch.apply_intrinsics(
            _select_render_values(intrinsics, offsets, broadcast_ndim), rows=rows
        )

    def get_intrinsics(self) -> torch.Tensor:
        """
        Read owned camera intrinsic matrices in environment order.

        Returns:
            torch.Tensor: The intrinsics for the left camera with shape (N, 3, 3).
        """
        return torch.as_tensor(
            self._batch.fetch_intrinsics(matrix=True, device=str(self.device)),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore configured extrinsics and ground visibility for selected views."""
        self.set_ground_plane_visibility(self.cfg.ground_plane_visible, env_ids)
        self._reset_pose(env_ids)

    def _reset_pose(self, env_ids: Sequence[int] | None = None) -> None:
        self.cfg: CameraCfg
        ids = (
            None
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )

        if self.cfg.extrinsics.eye is not None:
            eye = (
                torch.tensor(
                    self.cfg.extrinsics.eye, dtype=torch.float32, device=self.device
                )
                .squeeze_(0)
                .repeat(self.num_instances, 1)
            )
            target = (
                torch.tensor(
                    self.cfg.extrinsics.target, dtype=torch.float32, device=self.device
                )
                .squeeze_(0)
                .repeat(self.num_instances, 1)
            )
            up = (
                torch.tensor(
                    self.cfg.extrinsics.up, dtype=torch.float32, device=self.device
                )
                .squeeze_(0)
                .repeat(self.num_instances, 1)
                if self.cfg.extrinsics.up is not None
                else None
            )
            if ids is not None:
                eye, target = eye[ids], target[ids]
                if up is not None:
                    up = up[ids]
            self.look_at(eye, target, up, env_ids=env_ids)
        else:
            pose = self.cfg.extrinsics.transformation.to(self.device)
            pose = pose.unsqueeze_(0).repeat(self.num_instances, 1, 1)

            if self.cfg.extrinsics.parent is None:
                # To opengl coordinate system.
                pose[:, :3, 1] = -pose[:, :3, 1]
                pose[:, :3, 2] = -pose[:, :3, 2]

            self.set_local_pose(pose if ids is None else pose[ids], env_ids=env_ids)
