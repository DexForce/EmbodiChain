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

"""Resident reference motion clips with explicit joint order and frame boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal
import hashlib
import json
import math

import numpy as np
import torch

from embodichain.utils.math import matrix_from_quat

__all__ = ["MotionLibrary"]


class MotionLibrary:
    """Own floating-base reference states and query them on the target device.

    State fields are ``pos`` (world root position), ``quat`` (XYZW), ``rot``
    (rotation matrix), ``q`` (joint position), ``v`` / ``w`` (world root-origin
    linear/angular velocity) and ``qd`` (joint velocity). Queries return owned
    tensors; future offsets hold both clip endpoints and never cross clips.

    Attributes:
        content_digest: Versioned identity of the selected float32 motion data,
            clip layout, timing and joint order, cached at construction.

    Args:
        qpos: Packed ``[root XYZ, quaternion, joint positions]`` array.
        qvel: Packed ``[world linear, world angular, joint velocities]`` array.
        lengths: Frame count of each consecutive clip.
        names: Unique clip names in storage order.
        joint_names: Joint names in the supplied position/velocity column order.
        fps: Sampling frequency in Hz.
        quaternion_order: Input quaternion convention; outputs always use XYZW.
        device: Device for resident data and frame indices.
    """

    def __init__(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        lengths: Sequence[int],
        names: Sequence[str],
        joint_names: Sequence[str],
        fps: float,
        *,
        quaternion_order: Literal["xyzw", "wxyz"] = "xyzw",
        device: str | torch.device = "cpu",
    ) -> None:
        self.names = tuple(str(n) for n in names)
        self.joint_names = tuple(joint_names)
        lengths_array = np.asarray(lengths, dtype=np.int64)
        dof = len(self.joint_names)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Motion fps must be finite and positive")
        if (
            not len(lengths_array)
            or lengths_array.ndim != 1
            or np.any(lengths_array < 2)
            or len(self.names) != len(lengths_array)
            or len(set(self.names)) != len(self.names)
        ):
            raise ValueError("Motion clips need unique names and at least two frames")
        if not dof or len(set(self.joint_names)) != dof:
            raise ValueError("Motion joint names must be nonempty and unique")
        total = int(lengths_array.sum())
        if qpos.shape != (total, 7 + dof) or qvel.shape != (total, 6 + dof):
            raise ValueError("Motion arrays do not match clip lengths and joint order")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Motion states must be finite")
        if quaternion_order not in ("xyzw", "wxyz"):
            raise ValueError("Quaternion order must be xyzw or wxyz")
        quaternion = np.asarray(qpos[:, 3:7], dtype=np.float32)
        if quaternion_order == "wxyz":
            quaternion = quaternion[:, [1, 2, 3, 0]]
        if np.any(np.linalg.norm(quaternion, axis=-1) < 1e-8):
            raise ValueError("Motion quaternions must be nonzero")
        self.device = torch.empty(0, device=device).device
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.num_frames = total
        self.num_clips = len(self.names)
        self.lengths = torch.tensor(lengths_array, device=self.device)
        self.starts = torch.cat(
            (self.lengths.new_zeros(1), self.lengths.cumsum(0)[:-1])
        )
        self.ends = self.starts + self.lengths
        columns = {
            "pos": qpos[:, :3],
            "quat": quaternion,
            "q": qpos[:, 7:],
            "v": qvel[:, :3],
            "w": qvel[:, 3:6],
            "qd": qvel[:, 6:],
        }
        # Hash selected float32 source states once on the CPU, before device
        # upload. Lengths and joint order give the byte stream its interpretation.
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "version": 1,
                    "dtype": "<f4",
                    "names": self.names,
                    "joint_names": self.joint_names,
                    "lengths": lengths_array.tolist(),
                    "fps": self.fps,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for name, values in columns.items():
            digest.update(name.encode("ascii"))
            digest.update(memoryview(np.ascontiguousarray(values, dtype="<f4")))
        self.content_digest = "motion-v1:" + digest.hexdigest()
        self._states = {
            k: torch.tensor(v, device=self.device, dtype=torch.float32)
            for k, v in columns.items()
        }
        self._states["quat"] = torch.nn.functional.normalize(
            self._states["quat"], dim=-1
        )

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        joint_names: Sequence[str],
        *,
        quaternion_order: Literal["xyzw", "wxyz"],
        device: str | torch.device = "cpu",
        clips: Sequence[str] | None = None,
    ) -> MotionLibrary:
        """Load packed qpos/qvel clips, optionally selecting a named subset.

        The archive contains qpos, qvel, lengths, names and fps. Optional starts
        must describe consecutive clips. Optional joint_names must match the
        requested order; archives without it require the caller's explicit order.
        """
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            if (
                "quaternion_order" in data
                and str(data["quaternion_order"]) != quaternion_order
            ):
                raise ValueError(
                    "Motion archive quaternion convention differs from the requested order"
                )
            lengths = data["lengths"].astype(np.int64)
            names = data["names"].astype(str).tolist()
            starts = np.r_[0, np.cumsum(lengths)[:-1]]
            if "starts" in data and not np.array_equal(data["starts"], starts):
                raise ValueError("Motion archive must contain consecutive clips")
            if "joint_names" in data and tuple(
                data["joint_names"].astype(str)
            ) != tuple(joint_names):
                raise ValueError("Motion archive joint order does not match the robot")
            qpos, qvel = data["qpos"], data["qvel"]
            if clips is not None:
                selected = [names.index(name) for name in clips]
                rows = np.concatenate(
                    [np.arange(starts[i], starts[i] + lengths[i]) for i in selected]
                )
                qpos, qvel = qpos[rows], qvel[rows]
                lengths, names = lengths[selected], [names[i] for i in selected]
            return cls(
                qpos,
                qvel,
                lengths,
                names,
                joint_names,
                float(data["fps"]),
                quaternion_order=quaternion_order,
                device=device,
            )

    @classmethod
    def from_csv(
        cls,
        paths: Sequence[str | Path],
        joint_names: Sequence[str],
        *,
        source_fps: float = 30.0,
        target_fps: float = 50.0,
        device: str | torch.device = "cpu",
    ) -> MotionLibrary:
        """Read headerless XYZ/XYZW/joint CSV clips and resample on the CPU.

        Paths determine clip order and their stems provide unique clip names.
        Joint columns follow ``joint_names``. Resident output moves to ``device``
        after preprocessing; parsing does not require a simulator or GPU.
        """
        from embodichain.compute.trajectory.floating import resample_floating_trajectory

        positions, velocities, lengths, names = [], [], [], []
        for path in paths:
            path = Path(path).expanduser()
            raw = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
            if raw.shape[1] != 7 + len(joint_names):
                raise ValueError(
                    f"Motion CSV joint width differs from joint_names: {path}"
                )
            qpos, qvel = resample_floating_trajectory(
                torch.from_numpy(raw), source_fps, target_fps
            )
            positions.append(qpos.numpy())
            velocities.append(qvel.numpy())
            lengths.append(len(qpos))
            names.append(path.stem)
        if not names:
            raise ValueError("At least one motion CSV is required")
        return cls(
            np.concatenate(positions),
            np.concatenate(velocities),
            lengths,
            names,
            joint_names,
            target_fps,
            device=device,
        )

    def save_npz(self, path: str | Path) -> None:
        """Write a portable XYZW archive with explicit joint and clip metadata."""
        s = {k: v.detach().cpu().numpy() for k, v in self._states.items()}
        np.savez_compressed(
            Path(path),
            qpos=np.concatenate((s["pos"], s["quat"], s["q"]), -1),
            qvel=np.concatenate((s["v"], s["w"], s["qd"]), -1),
            lengths=self.lengths.cpu().numpy(),
            starts=self.starts.cpu().numpy(),
            names=np.asarray(self.names),
            joint_names=np.asarray(self.joint_names),
            fps=self.fps,
            quaternion_order="xyzw",
        )

    def clip_ids(self, frames: torch.Tensor) -> torch.Tensor:
        """Map valid global frame indices to clips, preserving leading axes."""
        return torch.searchsorted(self.starts, frames.contiguous(), right=True) - 1

    def frame(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read valid global indices, returning independent float32 state tensors."""
        state = {
            k: v.index_select(0, frames.reshape(-1)).reshape(
                *frames.shape, *v.shape[1:]
            )
            for k, v in self._states.items()
        }
        state["rot"] = matrix_from_quat(state["quat"])
        return state

    def future(
        self, frames: torch.Tensor, offsets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Query offsets from each frame with endpoint hold inside that clip."""
        clips = self.clip_ids(frames)
        indices = frames.unsqueeze(-1) + offsets
        indices = torch.maximum(indices, self.starts[clips].unsqueeze(-1))
        indices = torch.minimum(indices, (self.ends[clips] - 1).unsqueeze(-1))
        return self.frame(indices)

    def remaining(self, frames: torch.Tensor) -> torch.Tensor:
        """Return the number of frames after each valid index in its clip."""
        return self.ends[self.clip_ids(frames)] - frames - 1
