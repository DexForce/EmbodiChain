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
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
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
    """Record independent environment videos, committed explicitly before reset.

    Frames and provenance are owned per row. Normal commits obey the environment's
    EpisodeContext; ``purpose="diagnostic"`` labels isolated audit exports.
    Uncommitted frames are discarded at close, never implicitly published.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
        """Initialize the functor.

        Args:
            cfg: The configuration of the functor.
            env: The environment instance.

        Raises:
            ValueError: If the video frame rate is not finite and positive.
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
        fps = cfg.params.get("fps", 20.0)
        self._fps = float(env.control_frequency if fps is None else fps)
        if not math.isfinite(self._fps) or self._fps <= 0:
            raise ValueError("Camera recording fps must be finite and positive.")

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
        self._num_envs = getattr(env, "num_envs", 1)
        self._frames_list: list[list[np.ndarray]] = [[] for _ in range(self._num_envs)]
        self._ep_idx = [0] * self._num_envs
        self._jobs: dict[tuple[int, int, str], tuple] = {}
        self._receipts: dict[tuple[int, int, str], int] = {}
        self._finalize_error: BaseException | None = None
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

    def _normalize_env_ids(self, env_ids: torch.Tensor | list[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self._num_envs))
        values = (
            env_ids.detach().cpu().tolist()
            if isinstance(env_ids, torch.Tensor)
            else list(env_ids)
        )
        if any(
            type(row) is not int or not 0 <= row < self._num_envs for row in values
        ) or len(set(values)) != len(values):
            raise ValueError("Camera row IDs must be unique valid integers.")
        return values

    def _write_job(self, key: tuple[int, int, str], job: tuple) -> None:
        frames, metadata = job
        row, episode, purpose = key
        directory = Path(self._save_path)
        if purpose == "diagnostic":
            directory /= "diagnostics"
        directory /= f"env_{row:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"episode_{episode:06d}_{self._name}"
        # Publish complete files only. Failed jobs retain their frames and name.
        with tempfile.TemporaryDirectory(prefix=".camera-", dir=directory) as temporary:
            images_to_video(frames, temporary, name, fps=self._fps)
            staged = Path(temporary)
            (staged / (name + ".json")).write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            os.replace(staged / (name + ".mp4"), directory / (name + ".mp4"))
            os.replace(staged / (name + ".json"), directory / (name + ".json"))

    def _flush_jobs(self) -> None:
        for key, job in list(self._jobs.items()):
            self._write_job(key, job)
            self._receipts[key] = len(job[0])
            del self._jobs[key]

    def save_and_clear(
        self,
        env_ids: torch.Tensor | list[int] | None = None,
        *,
        purpose: str = "recording",
    ) -> None:
        """Commit selected eligible rows, retaining failed writes for retry.

        Args:
            env_ids: Environment rows to commit; None selects all rows.
            purpose: ``recording`` obeys eligibility; ``diagnostic`` writes
                explicitly labeled artifacts under a separate diagnostics directory.
        """
        self._ensure_open()
        if purpose not in ("recording", "diagnostic"):
            raise ValueError("Camera purpose must be recording or diagnostic.")
        rows = self._normalize_env_ids(env_ids)
        context = getattr(self._env, "episode_context", None)
        if purpose == "recording" and context is not None:
            rows = context.filter_recordable(rows).cpu().tolist()
        snapshots = (
            context.snapshot(rows) if context is not None else [{} for _ in rows]
        )
        for row, snapshot in zip(rows, snapshots, strict=True):
            frames = self._frames_list[row]
            if not frames:
                continue
            episode = snapshot.get("episode", self._ep_idx[row])
            key = (row, episode, purpose)
            if key in self._jobs or key in self._receipts:
                raise RuntimeError(
                    "A camera episode is already committed; reset before recording another fragment."
                )
            metadata = {
                "artifact_purpose": purpose,
                "env_id": row,
                "episode_context": deepcopy(snapshot),
                "frames": len(frames),
                "fps": self._fps,
            }
            self._jobs[key] = (tuple(frame.copy() for frame in frames), metadata)
            self._frames_list[row] = []
            self._ep_idx[row] += 1
        self._flush_jobs()

    def discard_and_clear(
        self, env_ids: torch.Tensor | list[int] | None = None
    ) -> None:
        """Discard selected live frames while preserving committed jobs.

        Args:
            env_ids: Environment rows to discard; None selects all rows.
        """
        for row in self._normalize_env_ids(env_ids):
            self._frames_list[row] = []

    def finalize(self) -> None:
        """Flush committed jobs and discard live frames; never commit on close."""
        with self._finalize_lock:
            if self._finalized:
                if self._finalize_error is not None:
                    raise self._finalize_error
                return
            try:
                self._flush_jobs()
            except BaseException as error:
                self._finalize_error = error
                raise
            finally:
                self.discard_and_clear()
                self._finalized = True

    def close(self) -> None:
        """Finalize committed jobs and discard uncommitted frames."""
        self.finalize()

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor | list[int] | None,
        name: str,
        resolution: tuple[int, int] = (640, 480),
        eye: tuple[float, float, float] = (0, 0, 2),
        target: tuple[float, float, float] = (0, 0, 0),
        up: tuple[float, float, float] = (0, 0, 1),
        intrinsics: tuple[float, float, float, float] = (600, 600, 320, 240),
        max_env_num: int = 16,
        save_path: str = "./outputs/videos",
        fps: float | None = 20.0,
    ) -> None:
        """Copy selected camera rows without consuming any other row's frames.

        Args:
            env: Owning environment.
            env_ids: Selected rows; None captures all rows up to max_env_num.
            name: Camera name from construction parameters.
            resolution: Construction-only image size.
            eye: Construction-only camera position.
            target: Construction-only look-at point.
            up: Construction-only vertical direction.
            intrinsics: Construction-only camera calibration.
            max_env_num: Upper bound on captured row indices.
            save_path: Construction-only output directory.
            fps: Video frame rate, or None for the environment control rate.
        """
        self._ensure_open()
        env.sim.render_camera_group([self.camera.group_id])
        self.camera.update(fetch_only=True)
        rgb = self.camera.get_data()["color"]
        for row in self._normalize_env_ids(env_ids):
            if row >= max_env_num:
                continue
            frame = rgb[row, ..., :3]
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            self._frames_list[row].append(np.array(frame, copy=True))


class record_camera_data_async(record_camera_data):
    """Persist independent row snapshots through one background video writer."""

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="camera-writer"
        )
        self._futures: dict[tuple[int, int, str], Future[None]] = {}

    def _flush_jobs(self) -> None:
        for key, job in list(self._jobs.items()):
            future = self._futures.get(key)
            if future is not None:
                if not future.done():
                    continue
                try:
                    future.result()
                except BaseException:
                    # Report the failed write once; its owned job remains available
                    # for the next explicit retry under the original episode ID.
                    del self._futures[key]
                    raise
                self._receipts[key] = len(job[0])
                del self._jobs[key]
                del self._futures[key]
            else:
                self._futures[key] = self._executor.submit(self._write_job, key, job)

    def finalize(self) -> None:
        """Wait for every committed row; independent rows never wait for a mosaic."""
        with self._finalize_lock:
            if self._finalized:
                if self._finalize_error is not None:
                    raise self._finalize_error
                return
            try:
                self._flush_jobs()
                for key, future in list(self._futures.items()):
                    future.result()
                    self._receipts[key] = len(self._jobs[key][0])
                    del self._jobs[key]
                    del self._futures[key]
            except BaseException as error:
                self._finalize_error = error
                raise
            finally:
                self._executor.shutdown(wait=True)
                self.discard_and_clear()
                self._finalized = True


class validation_cameras(Functor):
    """
    This functor creates validation cameras during initialization and captures
    their data when called. The cameras are created once and reused for subsequent calls.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
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
