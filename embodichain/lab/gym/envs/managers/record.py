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

import torch
import os
import random
import numpy as np
from typing import TYPE_CHECKING, Literal, Union, List

from dexsim.utility import images_to_video
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.lab.sim.sensors.camera import CameraCfg, Camera

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


class record_camera_data(Functor):
    """Record camera data in the environment. The camera is usually setup with third-person view, and
    is used to record the scene during the episode. It is helpful for debugging and visualization.

    Note:
        Currently, the functor is implemented in `interval' mode such that, it can only save the
        recorded frames when in :meth:`env.step()` function call. For example:
        ```python
        env.step()
        # perform multiple steps in the same episode
        env.reset()
        env.step()  # the video of the first episode will be saved here.
        ```
        The final episode frames will not be saved in the current implementation.
        We may improve it in the future.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        """Initialize the functor.

        Args:
            cfg: The configuration of the functor.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

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
            )
        )

        # Add this camera's group ID to the environment for batch rendering when RT is enabled.
        if getattr(env.sim, "is_rt_enabled", False):
            env.add_camera_group_id(self.camera.group_id)

        self._current_episode = 0
        self._frames: List[np.ndarray] = []

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
        # TODO: the current implementation will lost the final episode frames recording.
        # Check if the frames should be saved for the current episode
        if env.elapsed_steps.sum().item() == len(env_ids) and len(self._frames) > 0:
            video_name = f"episode_{self._current_episode}_{self._name}"
            images_to_video(self._frames, save_path, video_name, fps=20)

            self._current_episode += 1
            self._frames = []

        self.camera.update(fetch_only=self.camera.is_rt_enabled)
        data = self.camera.get_data()
        rgb = data["color"]

        num_frames = max(rgb.shape[0], max_env_num)
        rgb = rgb[:num_frames]
        rgb = self._draw_frames_into_one_image(rgb)[..., :3].cpu().numpy()
        self._frames.append(rgb)


class record_camera_data_async(record_camera_data):
    """Record camera data for multiple environments, merge and save as a single video at episode end."""

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._num_envs = min(4, getattr(env, "num_envs", 1))
        self._frames_list = [[] for _ in range(self._num_envs)]
        self._ep_idx = [0 for _ in range(self._num_envs)]

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
        self.camera.update(fetch_only=self.camera.is_rt_enabled)
        data = self.camera.get_data()
        rgb = data["color"]  # shape: (num_envs, H, W, 4)
        if isinstance(rgb, torch.Tensor):
            rgb_np = rgb.cpu().numpy()
        else:
            rgb_np = rgb
        # Only collect frames for the first 4 environments
        for i in range(self._num_envs):
            self._frames_list[i].append(rgb_np[i][..., :])

        # Check if elapsed_steps==1 (just reset)
        elapsed = env.elapsed_steps
        if isinstance(elapsed, torch.Tensor):
            elapsed_np = elapsed.cpu().numpy()
        else:
            elapsed_np = elapsed
        # Only check reset for the first 4 environments
        ready_envs = [
            i
            for i in range(self._num_envs)
            if elapsed_np[i] == 1 and len(self._frames_list[i]) > 1
        ]
        # Used to temporarily store episode frames for each env
        if not hasattr(self, "_pending_env_episodes"):
            self._pending_env_episodes = {}
        for i in ready_envs:
            if i not in self._pending_env_episodes:
                self._pending_env_episodes[i] = self._frames_list[i][:-1]
                self._frames_list[i] = [
                    self._frames_list[i][-1]
                ]  # Only keep the first frame after reset
                self._ep_idx[i] += 1
        # If all specified envs have collected frames, concatenate and save
        if len(self._pending_env_episodes) == self._num_envs:
            min_len = min(len(frames) for frames in self._pending_env_episodes.values())
            big_frames = []
            for j in range(min_len):
                frames = [
                    self._pending_env_episodes[i][j] for i in range(self._num_envs)
                ]
                frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.uint8)
                big_frame = (
                    self._draw_frames_into_one_image(frames_tensor)[..., :3]
                    .cpu()
                    .numpy()
                )
                big_frames.append(big_frame)
            video_name = f"ep{self._ep_idx[0]-1}_{self._name}_allenvs"
            images_to_video(big_frames, save_path, video_name, fps=20)
            self._pending_env_episodes.clear()


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
