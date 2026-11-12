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


"""Phase selection over an already bound environment-owned command snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_skills import (
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
    source: str
    phases: tuple[tuple[str, int, int], ...]

    call_id: ClassVar[str] = "simulation.kinematic_joint_replay"
    revision: ClassVar[str] = "2"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        raise ValueError(
            "Joint replay requires its environment-owned command snapshot."
        )

    def create_for_environment(
        self, *, environment, simulation, robot, scene_registry, engine
    ):
        del simulation, scene_registry
        if engine.robot is not robot:
            raise ValueError("Joint replay must use the bound robot.")
        snapshot = environment._replay_snapshot(self.source)
        if snapshot.robot is not robot or self.phases[-1][2] != snapshot.steps:
            raise ValueError("Replay phases must cover the bound command snapshot.")
        return _KinematicReplayLowerer(self.phases, snapshot)


class _KinematicReplayLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = _KinematicReplayFactory.call_id
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    def __init__(self, phases, snapshot):
        self._phases = phases
        self._snapshot = snapshot
        self._commands = snapshot.commands()

    def lower(self, call, *, context, bound, option_template):
        del bound, option_template
        phase = call.arguments.get("phase")
        if set(call.arguments) != {"phase"} or type(phase) is not str:
            raise ValueError("Joint replay requires a phase name.")
        for name, start, stop in self._phases:
            if name == phase:
                break
        else:
            raise ValueError(f"Unknown joint replay phase {phase!r}.")
        dt = self._snapshot.step_dt
        if context.batch_size != 1 or context.require_control_dt() != dt:
            raise ValueError("Joint replay requires one arena at the recorded rate.")
        commands = self._commands.interval(start + 1, stop + 1)
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
        value, path=path, required=frozenset({"kind", "source", "phases"})
    )
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
        source=_identifier(config["source"], path=f"{path}.source"),
        phases=tuple(ranges),
    )
