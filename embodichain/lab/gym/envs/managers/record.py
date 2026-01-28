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

import torch
import os
import random
import numpy as np
from typing import TYPE_CHECKING, Literal, Union, List

from dexsim.utility import images_to_video
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.lab.sim.sensors.camera import CameraCfg

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

        self.camera = env.sim.add_sensor(
            sensor_cfg=CameraCfg(
                uid=self._name,
                width=resolution[0],
                height=resolution[1],
                extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target, up=up),
                intrinsics=intrinsics,
            )
        )

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
    """Record camera data for multiple environments with lazy merging strategy.
    
    Strategy:
    1. During recording: Save each environment's video independently (fast, no blocking)
    2. After evaluation: Merge videos into grid layout, then delete individual videos
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._num_envs = min(4, getattr(env, "num_envs", 1))
        self._frames_list = [[] for _ in range(self._num_envs)]
        self._ep_idx = [0 for _ in range(self._num_envs)]
        self._saved_videos = []  # Track saved videos for merging

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
        # Check which environments just completed an episode
        ready_envs = [
            i
            for i in range(self._num_envs)
            if elapsed_np[i] == 1 and len(self._frames_list[i]) > 1
        ]
        
        # Save completed episodes immediately (lazy merge mode)
        for i in ready_envs:
            frames = self._frames_list[i][:-1]  # Exclude reset frame
            if len(frames) > 0:
                # Save individual video
                video_name = f"env{i}_ep{self._ep_idx[i]}_{self._name}"
                video_path = os.path.join(save_path, f"{video_name}.mp4")
                images_to_video(frames, save_path, video_name, fps=20)
                
                # Track for later merging
                self._saved_videos.append({
                    'env_id': i,
                    'episode': self._ep_idx[i],
                    'path': video_path,
                    'name': video_name
                })
                
                # Reset frame list
                self._frames_list[i] = [self._frames_list[i][-1]]
                self._ep_idx[i] += 1


    def flush(self, save_path: str = "./outputs/videos"):
        """Save any remaining frames (called at evaluation end)."""
        for i in range(self._num_envs):
            if len(self._frames_list[i]) > 1:  # Has unsaved frames
                frames = self._frames_list[i]
                video_name = f"env{i}_ep{self._ep_idx[i]}_{self._name}_final"
                video_path = os.path.join(save_path, f"{video_name}.mp4")
                images_to_video(frames, save_path, video_name, fps=20)
                
                self._saved_videos.append({
                    'env_id': i,
                    'episode': self._ep_idx[i],
                    'path': video_path,
                    'name': video_name
                })
                
                self._frames_list[i] = []
                self._ep_idx[i] += 1
    
    def finalize(self, save_path: str = "./outputs/videos"):
        """Merge individual videos into grid layout, then delete individual files."""
        # Group videos by episode
        episodes = {}
        for video in self._saved_videos:
            ep = video['episode']
            if ep not in episodes:
                episodes[ep] = []
            episodes[ep].append(video)
        
        # Merge each episode's videos into grid
        for ep_idx, videos in episodes.items():
            if len(videos) == self._num_envs:
                # All environments have this episode, merge to grid
                output_name = f"eval_ep{ep_idx}_{self._name}_grid"
                output_path = os.path.join(save_path, f"{output_name}.mp4")
                
                video_paths = [v['path'] for v in sorted(videos, key=lambda x: x['env_id'])]
                
                try:
                    self._merge_videos_to_grid(video_paths, output_path)
                    
                    # Always remove individual videos after successful merge
                    for path in video_paths:
                        if os.path.exists(path):
                            os.remove(path)
                except Exception as e:
                    print(f"Warning: Failed to merge episode {ep_idx}: {e}")
                    # Keep individual videos on failure
    
    def _merge_videos_to_grid(self, video_paths: list, output_path: str):
        """Merge multiple videos into grid layout using numpy/cv2."""
        import cv2
        
        # Read all videos
        caps = [cv2.VideoCapture(path) for path in video_paths]
        
        if not all(cap.isOpened() for cap in caps):
            for cap in caps:
                cap.release()
            raise RuntimeError("Failed to open some videos")
        
        # Get video properties from first video
        fps = int(caps[0].get(cv2.CAP_PROP_FPS))
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate grid layout
        n = len(caps)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Output dimensions
        out_width = width * cols
        out_height = height * rows
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        # Read and merge frames
        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if len(frames) < n:
                break
            
            # Create grid
            grid_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            for idx, frame in enumerate(frames):
                row = idx // cols
                col = idx % cols
                y1 = row * height
                y2 = y1 + height
                x1 = col * width
                x2 = x1 + width
                grid_frame[y1:y2, x1:x2] = frame
            
            out.write(grid_frame)
        
        # Cleanup
        for cap in caps:
            cap.release()
        out.release()


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
