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

"""Bounded-memory recording selected only by the GenSim E8 runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import imageio
from imageio_ffmpeg import count_frames_and_secs
import torch

from embodichain.lab.gym.envs.managers.record import record_camera_data

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.envs.managers import FunctorCfg

__all__: list[str] = []


class E8CameraRecorder(record_camera_data):
    """Stream unchanged camera frames instead of retaining a whole episode.

    This private subclass preserves the existing environment's recorder
    transaction boundaries and ``isinstance(record_camera_data)`` dispatch.
    Encoder errors never publish a partially written movie as a committed one.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
        """Initialize the original camera and a lazy single-episode encoder.

        Args:
            cfg: Original recorder configuration, without additional options.
            env: Environment owning the camera and recording lifecycle.
        """
        super().__init__(cfg, env)
        self._writer: Any = None
        self._partial_path: Path | None = None
        self._frame_count = 0
        self._encoding_error: BaseException | None = None

    def _video_path(self) -> Path:
        name = f"episode_{self._current_episode}_{self._name}"
        name = name.replace(" ", "_").replace("\n", "_")
        return Path(self._save_path) / f"{name}.mp4"

    def _start_writer(self) -> None:
        final_path = self._video_path()
        if final_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite committed video: {final_path}"
            )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        partial_directory = final_path.parent.parent / "video_in_progress"
        partial_directory.mkdir(parents=True, exist_ok=True)
        partial = partial_directory / f"{final_path.stem}.{uuid4().hex}.partial.mp4"
        partial.touch(exist_ok=False)
        self._partial_path = partial
        # These are exactly images_to_video's primary writer parameters. Its
        # existing codec, RGB handling and macroblock defaults stay unchanged.
        self._writer = imageio.get_writer(partial.as_posix(), fps=20, quality=5)

    def _close_writer(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor | None,
        name: str,
        **params: Any,
    ) -> None:
        """Capture with the original camera path and encode one frame in order.

        Args:
            env: Environment owning the camera.
            env_ids: Original recorder environment selection.
            name: Original recording camera name.
            **params: Unchanged original capture parameters.
        """
        self._ensure_open()
        if self._encoding_error is not None:
            raise RuntimeError(
                "E8 camera encoder previously failed"
            ) from self._encoding_error
        try:
            super().__call__(env, env_ids, name, **params)
            frame = self._frames.pop()
            if self._writer is None:
                self._start_writer()
            self._writer.append_data(frame)
            self._frame_count += 1
        except BaseException as error:
            self._frames.clear()
            self._encoding_error = error
            raise

    def save_and_clear(self, env_ids: torch.Tensor | None = None) -> None:
        """Commit only after the complete encoder stream closes successfully.

        Args:
            env_ids: Existing shared recorder transaction selection.
        """
        self._ensure_open()
        if self._encoding_error is not None:
            raise RuntimeError(
                "Cannot commit a failed E8 camera stream"
            ) from self._encoding_error
        if not self._frame_count:
            return
        try:
            self._close_writer()
            final_path = self._video_path()
            if final_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite committed video: {final_path}"
                )
            assert self._partial_path is not None
            encoded_frames, _ = count_frames_and_secs(self._partial_path)
            if encoded_frames != self._frame_count:
                raise RuntimeError(
                    "E8 camera frame count mismatch: "
                    f"captured={self._frame_count}, encoded={encoded_frames}"
                )
            os.replace(self._partial_path, final_path)
        except BaseException as error:
            self._encoding_error = error
            raise
        self._partial_path = None
        self._frame_count = 0
        self._current_episode += 1

    def discard_and_clear(self, env_ids: torch.Tensor | None = None) -> None:
        """Release the encoder and discard only this recorder's uncommitted file.

        Args:
            env_ids: Existing shared recorder transaction selection.

        Failed encoding files remain outside the committed video directory as
        audit artifacts; they are never retried or relabelled as valid movies.
        """
        try:
            self._close_writer()
        except BaseException as error:
            self._encoding_error = error
            raise
        finally:
            self._frames.clear()
            self._frame_count = 0
        if self._partial_path is not None and self._encoding_error is None:
            self._partial_path.unlink(missing_ok=True)
            self._partial_path = None

    def finalize(self) -> None:
        """Release uncommitted recording state exactly once, without committing."""
        with self._finalize_lock:
            if self._finalized:
                return
            try:
                self.discard_and_clear()
            finally:
                self._finalized = True
