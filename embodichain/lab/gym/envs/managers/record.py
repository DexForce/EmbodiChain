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

import os
import math
import random
import threading
from collections import deque
from typing import TYPE_CHECKING, Literal, Union, List

import numpy as np
import torch

from dexsim.utility import images_to_video
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.lab.sim.sensors.camera import CameraCfg, Camera

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = ["record_camera_data", "record_camera_data_async", "validation_cameras"]


class record_camera_data(Functor):
    """Capture a third-person camera through an interval event.

    By default, views are tiled into one video at 20 fps. Set ``per_env=True``
    to save one video per environment, with independent episode counters and
    selected-row capture/save/discard. ``fps=None`` uses the environment's control
    frequency; use an explicit fps when capturing less often than every step.

    Frames are committed by :meth:`save_and_clear` before reset. Closing the
    environment discards uncommitted frames, so callers ending without reset
    must explicitly commit the last recording.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
        """Initialize the functor.

        Args:
            cfg: The configuration of the functor.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        self._per_env = cfg.params.get("per_env", False)
        fps = cfg.params.get("fps", 20.0)
        self._fps = float(env.control_frequency if fps is None else fps)
        if not math.isfinite(self._fps) or self._fps <= 0:
            raise ValueError("Recording fps must be finite and positive.")
        self._env_frames: dict[int, list[np.ndarray]] = {}
        self._env_episode_ids: dict[int, int] = {}

        # extract the used quantities (to enable type-hinting)
        self._name = cfg.params.get("name", "default")
        resolution = cfg.params.get("resolution", (640, 480))
        eye = cfg.params.get("eye", (0, 0, 2))
        target = cfg.params.get("target", (0, 0, 0))
        up = cfg.params.get("up", (0, 0, 1))
        intrinsics = cfg.params.get(
            "intrinsics", (600, 600, int(resolution[0] / 2), int(resolution[1] / 2))
        )

        self.camera: Camera = env.sim.add_sensor(
            sensor_cfg=CameraCfg(
                uid=self._name,
                width=resolution[0],
                height=resolution[1],
                extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target, up=up),
                intrinsics=intrinsics,
                visualization_role="record",
            )
        )

        # Add this camera's group ID to the environment for batch rendering when RT is enabled.
        env.add_camera_group_id(self.camera.group_id)

        self._save_path = cfg.params.get("save_path", "./outputs/videos")
        self._current_episode = 0
        self._frames: List[np.ndarray] = []
        self._finalize_lock = threading.Lock()
        self._finalized = False

    def _ensure_open(self) -> None:
        """Raise when capture or commit is attempted after finalization."""
        if getattr(self, "_finalized", False):
            raise RuntimeError(f"{type(self).__name__} is already finalized")

    def _draw_frames_into_one_image(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Concatenate multiple frames into a single image with nearly square arrangement.

        Args:
            frames: Tensor with shape (B, H, W, 4) where B is batch size

        Returns:
            Single concatenated image tensor with shape (grid_h * H, grid_w * W, 4)
        """
        if frames.numel() == 0:
            return frames

        B, H, W, C = frames.shape

        # Calculate grid dimensions for nearly square arrangement
        grid_w = int(torch.ceil(torch.sqrt(torch.tensor(B, dtype=torch.float32))))
        grid_h = int(torch.ceil(torch.tensor(B, dtype=torch.float32) / grid_w))

        # Create empty grid to hold all frames
        result = torch.zeros(
            (grid_h * H, grid_w * W, C), dtype=frames.dtype, device=frames.device
        )

        # Fill the grid with frames
        for i in range(B):
            row = i // grid_w
            col = i % grid_w

            start_h = row * H
            end_h = start_h + H
            start_w = col * W
            end_w = start_w + W

            result[start_h:end_h, start_w:end_w] = frames[i]

        return result

    def save_and_clear(self, env_ids: Union[torch.Tensor, None] = None) -> None:
        """Save recorded frames as video and clear the buffer.

        This method is called from :meth:`EmbodiedEnv._initialize_episode` to ensure
        frames are saved before the episode is reset. This avoids the issue where the
        final episode's frames are lost because the save previously relied on detecting
        a reset inside :meth:`__call__`.

        Args:
            env_ids: Selected rows for separate videos, or all rows when None.
                Tiled recording saves the whole shared frame buffer.
        """
        self._ensure_open()
        if getattr(self, "_per_env", False):
            for env_id in self._selected_recorded_envs(env_ids):
                frames = self._env_frames[env_id]
                if not frames:
                    continue
                episode = self._env_episode_ids.get(env_id, 0)
                images_to_video(
                    frames,
                    self._save_path,
                    f"episode_{episode}_{self._name}_env{env_id}",
                    fps=self._fps,
                )
                self._env_episode_ids[env_id] = episode + 1
                self._env_frames[env_id] = []
            return
        if len(self._frames) > 0:
            video_name = f"episode_{self._current_episode}_{self._name}"
            images_to_video(
                self._frames, self._save_path, video_name, fps=getattr(self, "_fps", 20)
            )

            self._current_episode += 1
            self._frames = []

    def discard_and_clear(self, env_ids: Union[torch.Tensor, None] = None) -> None:
        """Discard frames without creating an episode video.

        Args:
            env_ids: Selected rows for separate videos, or all rows when None.
                Tiled recording discards the whole shared frame buffer.
        """
        if getattr(self, "_per_env", False):
            for env_id in self._selected_recorded_envs(env_ids):
                self._env_frames[env_id] = []
            return
        self._frames = []

    def _selected_recorded_envs(self, env_ids: torch.Tensor | None) -> list[int]:
        if env_ids is None:
            return list(self._env_frames)
        return [
            int(env_id)
            for env_id in torch.as_tensor(env_ids).reshape(-1).cpu().tolist()
            if int(env_id) in self._env_frames
        ]

    def finalize(self) -> None:
        """Discard uncommitted frames and close the recorder exactly once."""
        if not hasattr(self, "_finalize_lock"):
            self._finalize_lock = threading.Lock()
        with self._finalize_lock:
            if getattr(self, "_finalized", False):
                return
            self.discard_and_clear()
            self._finalized = True

    def close(self) -> None:
        """Finalize the recorder; repeated calls are safe."""
        self.finalize()

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        name: str,
        resolution: tuple[int, int] = (640, 480),
        eye: tuple[float, float, float] = (0, 0, 2),
        target: tuple[float, float, float] = (0, 0, 0),
        up: tuple[float, float, float] = (0, 0, 1),
        intrinsics: tuple[float, float, float, float] = (
            600,
            600,
            320,
            240,
        ),
        max_env_num: int = 16,
        save_path: str = "./outputs/videos",
        per_env: bool = False,
        fps: float | None = 20.0,
    ) -> None:
        """Capture up to ``max_env_num`` arena-relative views at this step.

        Args:
            env: Owning environment.
            env_ids: Rows to capture in per-environment mode, or all camera rows
                when None. Tiled recording always captures all camera rows.
            name: Camera name, configured at initialization.
            resolution: Image width and height, configured at initialization.
            eye: Arena-relative camera position.
            target: Arena-relative look-at position.
            up: Camera up direction.
            intrinsics: Pinhole intrinsics.
            max_env_num: Maximum number of environment views to capture.
            save_path: Output directory, configured at initialization.
            per_env: Whether to save separate environment videos.
            fps: Video cadence, or None for the environment control frequency.
        """
        self._ensure_open()
        # Interval events run before observation rendering. Render this group
        # from the current physics state before reading its reusable buffers.
        env.sim.render_camera_group([self.camera.group_id])
        self.camera.update(fetch_only=True)
        data = self.camera.get_data()
        rgb = data["color"]

        num_frames = min(rgb.shape[0], max_env_num)
        rgb = rgb[:num_frames]
        if self._per_env:
            selected_env_ids = list(range(num_frames))
            frames = rgb[..., :3]
            if env_ids is not None:
                selected_env_ids = [
                    int(env_id)
                    for env_id in torch.as_tensor(env_ids).reshape(-1).cpu().tolist()
                    if 0 <= int(env_id) < num_frames
                ]
                frames = frames[selected_env_ids]
            for env_id, frame in zip(selected_env_ids, frames.detach().cpu().numpy()):
                self._env_frames.setdefault(env_id, []).append(frame.copy())
            return
        rgb = self._draw_frames_into_one_image(rgb)[..., :3].cpu().numpy()
        self._frames.append(rgb)


class record_camera_data_async(record_camera_data):
    """Record camera data for multiple environments, merge and save as a single video at episode end."""

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._num_envs = min(4, getattr(env, "num_envs", 1))
        self._frames_list = [[] for _ in range(self._num_envs)]
        self._ep_idx = [0 for _ in range(self._num_envs)]
        self._committed_env_episodes = [deque() for _ in range(self._num_envs)]
        self._async_camera_finalize_lock = threading.Lock()
        self._async_camera_finalized = False
        self._async_camera_finalize_error: str | None = None

    def _normalize_env_ids(self, env_ids: Union[torch.Tensor, None]) -> list[int]:
        """Return recorder-local environment IDs for a transaction boundary."""
        if env_ids is None:
            return list(range(self._num_envs))
        if isinstance(env_ids, torch.Tensor):
            values = env_ids.reshape(-1).cpu().tolist()
        else:
            values = list(env_ids)
        return [int(env_id) for env_id in values if int(env_id) < self._num_envs]

    def _flush_committed_episodes(self) -> None:
        """Persist every complete FIFO set of explicitly committed env episodes."""
        while all(self._committed_env_episodes):
            episode_frames = [queue[0] for queue in self._committed_env_episodes]
            min_len = min(len(frames) for frames in episode_frames)
            big_frames = []
            for frame_id in range(min_len):
                frames = [frames[frame_id] for frames in episode_frames]
                frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.uint8)
                big_frame = (
                    self._draw_frames_into_one_image(frames_tensor)[..., :3]
                    .cpu()
                    .numpy()
                )
                big_frames.append(big_frame)

            video_name = f"ep{self._current_episode}_{self._name}_allenvs"
            # Peek above and pop only after persistence succeeds. A failed
            # write therefore remains observable and retryable at finalize().
            images_to_video(big_frames, self._save_path, video_name, fps=20)
            for queue in self._committed_env_episodes:
                queue.popleft()
            self._current_episode += 1

    def save_and_clear(self, env_ids: Union[torch.Tensor, None] = None) -> None:
        """Commit selected rows immediately instead of waiting for a later step."""
        self._ensure_open()
        for env_id in self._normalize_env_ids(env_ids):
            frames = self._frames_list[env_id]
            if frames:
                self._committed_env_episodes[env_id].append(frames)
                self._frames_list[env_id] = []
                self._ep_idx[env_id] += 1
        self._flush_committed_episodes()

    def discard_and_clear(self, env_ids: Union[torch.Tensor, None] = None) -> None:
        """Discard live frames while preserving already committed episodes."""
        super().discard_and_clear()
        for env_id in self._normalize_env_ids(env_ids):
            self._frames_list[env_id] = []

    def finalize(self) -> None:
        """Flush committed frame sets and reject incomplete committed batches."""
        if not hasattr(self, "_async_camera_finalize_lock"):
            self._async_camera_finalize_lock = threading.Lock()
            self._async_camera_finalized = False
            self._async_camera_finalize_error = None
        with self._async_camera_finalize_lock:
            if self._async_camera_finalized:
                if self._async_camera_finalize_error is not None:
                    raise RuntimeError(self._async_camera_finalize_error)
                return

            errors: list[str] = []
            try:
                self._flush_committed_episodes()
            except Exception as error:  # noqa: BLE001 - finish recorder cleanup
                errors.append(f"video persistence: {error}")

            pending_counts = [len(queue) for queue in self._committed_env_episodes]
            if any(pending_counts):
                errors.append(
                    "incomplete committed environment batch "
                    f"(pending episodes per env: {pending_counts})"
                )

            try:
                super().finalize()
            except Exception as error:  # noqa: BLE001 - aggregate cleanup
                errors.append(f"recorder cleanup: {error}")

            self._async_camera_finalized = True
            if errors:
                self._async_camera_finalize_error = (
                    "Async camera recorder finalization failed: " + "; ".join(errors)
                )
                raise RuntimeError(self._async_camera_finalize_error)

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        name: str,
        resolution: tuple[int, int] = (640, 480),
        eye: tuple[float, float, float] = (0, 0, 2),
        target: tuple[float, float, float] = (0, 0, 0),
        up: tuple[float, float, float] = (0, 0, 1),
        intrinsics: tuple[float, float, float, float] = (
            600,
            600,
            320,
            240,
        ),
        max_env_num: int = 16,
        save_path: str = "./outputs/videos",
    ):
        self._ensure_open()
        self.camera.update(fetch_only=True)
        data = self.camera.get_data()
        rgb = data["color"]  # shape: (num_envs, H, W, 4)
        if isinstance(rgb, torch.Tensor):
            rgb_np = rgb.cpu().numpy()
        else:
            rgb_np = rgb
        # Only collect frames for the first 4 environments
        for i in range(self._num_envs):
            self._frames_list[i].append(rgb_np[i][..., :])


class validation_cameras(Functor):
    """
    This functor creates validation cameras during initialization and captures
    their data when called. The cameras are created once and reused for subsequent calls.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        # Store camera configurations
        self.cameras_cfg = cfg.params.get("cameras", [])
        # Create each camera in __init__
        self.camera_uids = []
        for cam_cfg in self.cameras_cfg:
            uid = cam_cfg.get("uid", "validation_camera")
            width = cam_cfg.get("width", 1280)
            height = cam_cfg.get("height", 960)
            enable_mask = cam_cfg.get("enable_mask", False)
            intrinsics = cam_cfg.get("intrinsics", [1400, 1400, 640, 480])
            extrinsics_cfg = cam_cfg.get("extrinsics", {})
            extrinsics = CameraCfg.ExtrinsicsCfg(**extrinsics_cfg)

            camera = env.sim.add_sensor(
                sensor_cfg=CameraCfg(
                    uid=uid,
                    width=width,
                    height=height,
                    enable_mask=enable_mask,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                )
            )
            if camera is not None:
                self.camera_uids.append(uid)

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
    ):
        """Update cameras and return their data."""
        camera_data = {}
        for i, cam_uid in enumerate(self.camera_uids, start=1):
            camera = env.sim.get_sensor(cam_uid)
            camera.update()
            data = camera.get_data()
            camera_data[f"valid_rgb_{i}"] = data["color"]

        return camera_data
