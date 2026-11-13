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

"""Owned command inputs and progress shared by Gym replay adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDictBase

__all__: list[str] = []


class _ReplayCursor:
    """Track row-local transition boundaries without owning physical state."""

    def __init__(self, lengths: torch.Tensor):
        self.lengths = lengths
        self.steps = torch.zeros_like(lengths)

    @property
    def exhausted(self) -> torch.Tensor:
        return self.steps >= self.lengths

    def indices(self) -> torch.Tensor:
        return self.steps.clamp(max=self.lengths - 1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.steps.zero_()
        else:
            self.steps[env_ids] = 0

    def advance(self) -> None:
        self.steps.copy_((self.steps + 1).clamp(max=self.lengths))

    def seek(self, step: int) -> None:
        if step < 0 or bool((step > self.lengths).any()):
            raise ValueError("Replay frame is outside the authored interval.")
        self.steps.fill_(step)


class _ReplayCommands:
    """Sample actual controller inputs; authored inputs need no recorded states."""

    def __init__(self, samples: torch.Tensor):
        self._samples = samples

    @classmethod
    def from_recording(cls, trajectory: dict) -> _ReplayCommands:
        return cls(trajectory["actions"])

    def sample(self, indices: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(len(indices), device=indices.device)
        return self._samples[rows, indices]

    def interval(self, start: int, stop: int) -> torch.Tensor:
        if not 0 <= start < stop <= self._samples.shape[1]:
            raise ValueError("Replay interval exceeds the owned command stream.")
        return self._samples[:, start:stop].clone()

    @staticmethod
    def position_target(command):
        if isinstance(command, (Mapping, TensorDictBase)):
            if set(command.keys()) != {"qpos"}:
                raise ValueError("Native position replay accepts only qpos targets.")
            qpos = command["qpos"]
        else:
            qpos = command
        return qpos

    def require_position(self, frame: int, command) -> None:
        qpos = self.position_target(command)
        if (
            not 0 <= frame < self._samples.shape[1]
            or not isinstance(qpos, torch.Tensor)
            or not torch.equal(qpos, self._samples[:, frame])
        ):
            raise ValueError(f"Command differs from native replay at frame {frame}.")


@dataclass(frozen=True)
class _AuthoredJointStream:
    """Execution-free physical declaration of one position-command input."""

    stream_id: str
    trajectory: str
    sha256: str
    fps: int

    def load(self) -> _JointCommandSnapshot:
        content = Path(self.trajectory).read_bytes()
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise ValueError("Replay trajectory digest does not match.")
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            names = tuple(data["joint_names"].tolist())
            positions = data["positions"]
        if (
            not names
            or not all(type(name) is str and name for name in names)
            or len(set(names)) != len(names)
            or positions.ndim != 2
            or positions.shape[0] < 2
            or positions.shape[1] != len(names)
            or positions.dtype != np.float32
            or not np.isfinite(positions).all()
        ):
            raise ValueError("Invalid authored joint position stream.")
        return _JointCommandSnapshot(
            self, names, torch.from_numpy(positions.copy()), None
        )


@dataclass(frozen=True)
class _JointCommandSnapshot:
    """One environment-owned read, ordered once for the selected robot."""

    source: _AuthoredJointStream
    joint_names: tuple[str, ...]
    positions: torch.Tensor
    robot: object | None
    source_joint_names: tuple[str, ...] = ()

    @property
    def steps(self) -> int:
        return self.positions.shape[0] - 1

    @property
    def step_dt(self) -> float:
        return 1 / self.source.fps

    def bind(self, robot) -> _JointCommandSnapshot:
        if self.robot is not None:
            if self.robot is not robot:
                raise ValueError("Replay snapshot is bound to a different robot.")
            return self
        if set(self.joint_names) != set(robot.joint_names):
            raise ValueError("Replay requires exactly the recorded robot joints.")
        columns = [self.joint_names.index(name) for name in robot.joint_names]
        return _JointCommandSnapshot(
            self.source,
            tuple(robot.joint_names),
            self.positions[:, columns].to(robot.device),
            robot,
            self.source_joint_names or self.joint_names,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "id": self.source.stream_id,
            "sha256": self.source.sha256,
            "fps": self.source.fps,
            "steps": self.steps,
            "source_joint_names": list(self.source_joint_names or self.joint_names),
            "joint_names": list(self.joint_names),
        }

    def commands(self) -> _ReplayCommands:
        return _ReplayCommands(self.positions[None])


def _decode_authored_joint_stream(value, *, base_dir: Path) -> _AuthoredJointStream:
    from ._component_composition import _identifier, _mapping

    config = _mapping(
        value,
        path="simulation.replay",
        required=frozenset({"kind", "id", "trajectory", "sha256", "fps"}),
    )
    if config["kind"] != "native_joint_positions":
        raise ValueError("Unsupported simulation.replay.kind.")
    digest = config["sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("Replay requires a lowercase SHA-256 digest.")
    fps = config["fps"]
    if type(fps) is not int or fps <= 0:
        raise ValueError("Replay fps must be a positive integer.")
    trajectory = _identifier(config["trajectory"], path="simulation.replay.trajectory")
    return _AuthoredJointStream(
        _identifier(config["id"], path="simulation.replay.id"),
        str((base_dir / trajectory).resolve()),
        digest,
        fps,
    )
