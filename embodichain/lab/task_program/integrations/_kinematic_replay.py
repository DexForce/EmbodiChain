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


"""Configured, synchronized replay of sampled joint positions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    JointPositionGoal,
    MoveJoints,
    SkillDescriptor,
    TimedTrajectory,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)

__all__: list[str] = []


@dataclass(frozen=True)
class _KinematicReplayFactory:
    trajectory: str
    sha256: str
    fps: int
    phases: tuple[tuple[str, int, int], ...]

    call_id: ClassVar[str] = "simulation.kinematic_joint_replay"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    @property
    def steps(self) -> int:
        return self.phases[-1][2]

    def load(self) -> tuple[np.ndarray, tuple[str, ...]]:
        import io

        content = Path(self.trajectory).read_bytes()
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise ValueError("Kinematic replay trajectory digest does not match.")
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            names = tuple(data["joint_names"].tolist())
            positions = data["positions"]
        if (
            not names
            or not all(type(name) is str and name for name in names)
            or len(set(names)) != len(names)
            or positions.shape != (self.steps + 1, len(names))
            or positions.dtype != np.float32
            or not np.isfinite(positions).all()
        ):
            raise ValueError("Invalid kinematic replay joint trajectory.")
        return positions, names

    def ordered(self, positions, names, robot):
        if set(names) != set(robot.joint_names):
            raise ValueError(
                "Kinematic replay requires exactly the recorded robot joints."
            )
        columns = [names.index(name) for name in robot.joint_names]
        return torch.as_tensor(positions[:, columns], device=robot.device)

    def create(self, *, simulation, robot, scene_registry, engine):
        del simulation, scene_registry
        if engine.robot is not robot:
            raise ValueError("Kinematic replay must use the bound robot.")
        return _KinematicReplayLowerer(self, self.ordered(*self.load(), robot))


class _KinematicReplayLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = _KinematicReplayFactory.call_id
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    def __init__(self, factory, positions):
        self._factory = factory
        self._positions = positions

    def lower(self, call, *, context, bound, option_template):
        del bound, option_template
        phase = call.arguments.get("phase")
        if set(call.arguments) != {"phase"} or type(phase) is not str:
            raise ValueError("Kinematic replay requires a phase name.")
        for name, start, stop in self._factory.phases:
            if name == phase:
                break
        else:
            raise ValueError(f"Unknown kinematic replay phase {phase!r}.")
        dt = 1 / self._factory.fps
        if context.batch_size != 1 or context.require_control_dt() != dt:
            raise ValueError(
                "Kinematic replay requires one arena at the recorded rate."
            )
        commands = self._positions[None, start + 1 : stop + 1]
        return SemanticLowering(
            goal=JointPositionGoal(
                TimedTrajectory(
                    positions=commands,
                    velocities=None,
                    accelerations=None,
                    dt=torch.full_like(commands[..., 0], dt),
                    env_ids=context.env_ids,
                )
            )
        )


def _decode_kinematic_replay(value, *, path):
    from .configured import _identifier, _mapping

    config = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "trajectory", "sha256", "fps", "phases"}),
    )
    digest = _identifier(config["sha256"], path=f"{path}.sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{path}.sha256 must be a lowercase SHA-256 digest.")
    fps = config["fps"]
    if type(fps) is not int or fps <= 0:
        raise ValueError(f"{path}.fps must be a positive integer.")
    phases = config["phases"]
    if not isinstance(phases, dict) or not phases:
        raise ValueError(f"{path}.phases must be a nonempty mapping.")
    ranges = []
    previous_stop = 0
    for name, bounds in phases.items():
        _identifier(name, path=f"{path}.phases name")
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(type(frame) is not int for frame in bounds)
            or bounds[0] != previous_stop
            or bounds[1] <= bounds[0]
        ):
            raise ValueError(f"{path}.phases must be ordered contiguous frame ranges.")
        ranges.append((name, *bounds))
        previous_stop = bounds[1]
    return _KinematicReplayFactory(
        trajectory=_identifier(config["trajectory"], path=f"{path}.trajectory"),
        sha256=digest,
        fps=fps,
        phases=tuple(ranges),
    )
