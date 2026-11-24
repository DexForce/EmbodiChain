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
"""Configured timed joint trajectories with optional grasp-pose feedback."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import ClassVar, TYPE_CHECKING
from uuid import uuid4
from weakref import proxy

import numpy as np
import torch

from embodichain.lab.sim.atomic_skills import (
    AtomicSkillEngine,
    JointPositionGoal,
    MoveJoints,
    ObservedJointTrajectory,
    PlanningContext,
    SkillDescriptor,
    SkillOptions,
    TimedTrajectory,
)
from embodichain.lab.sim.motion._cartesian_track import _TrackLimits
from embodichain.lab.sim.motion._grasp_trajectory_feedback import (
    _GraspCapture,
    _GraspTrack,
    _GraspTrajectoryFeedback,
    _PoseOffset,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.language.semantic_contracts import (
    SemanticCallContract,
)
from embodichain.lab.task_program.semantics import (
    BoundSemanticCall,
    RegisteredSemanticCall,
    SceneRegistry,
)

from ._configured_track import (
    _CartesianTrack,
    _decode_cartesian_track,
    _PlannedTrack,
    _resolve_and_plan_track,
)

if TYPE_CHECKING:
    from embodichain.lab.task_program.semantics import SceneManifest
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Articulation, Robot

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _Phase:
    id: str
    start: int
    end: int
    join_current: bool
    feedback: bool


@dataclass(frozen=True, slots=True)
class _Feedback:
    articulation_uid: str
    link: str
    initial_offset: _PoseOffset | None
    tracks: tuple[_GraspTrack, ...]


@dataclass(frozen=True, slots=True)
class _TrajectoryPhaseFactory:
    motion: _CartesianTrack
    phases: tuple[_Phase, ...]
    feedback: _Feedback | None

    call_id: ClassVar[str] = "simulation.joint_trajectory_phase"
    revision: ClassVar[str] = "2"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    @property
    def argument_contract(self) -> SemanticCallContract:
        return SemanticCallContract(
            required={"phase": tuple(phase.id for phase in self.phases)}, optional={}
        )

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Validate the trajectory's bound pose references before assembly."""
        self.motion.validate_scene(scene)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        raise ValueError("An authored joint trajectory requires a live environment.")

    def create_for_environment(
        self,
        *,
        environment: EmbodiedEnv,
        simulation: SimulationManager,
        robot: Robot,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        if (
            engine.robot is not robot
            or environment.robot is not robot
            or environment.num_envs != 1
        ):
            raise ValueError(
                "An authored joint trajectory requires its bound single-arena robot."
            )
        dt = environment.step_dt
        if not math.isclose(dt, self.motion.step_dt, rel_tol=1e-8):
            raise ValueError(
                "Trajectory planning requires the declared environment cadence."
            )
        if self.motion.steps > environment.max_episode_steps:
            raise ValueError(
                "The planned path exceeds the environment episode horizon."
            )
        plan = _resolve_and_plan_track(self.motion, robot, scene_registry)
        source = _TrajectorySource(
            robot, self.phases, plan.positions, plan.poses, dt, limits=plan.limits
        )
        if self.feedback is not None:
            feedback = self.feedback
            obj = simulation.get_articulation(feedback.articulation_uid)
            if obj is None:
                raise ValueError(
                    f"Trajectory feedback articulation {feedback.articulation_uid!r} is missing."
                )
            source.set_feedback(feedback, obj, self.motion.tool_link, plan)
        source_id = uuid4().hex
        engine.planning_services.register_joint_trajectory_source(source_id, source)
        # Registrations retain lowerer identities. The engine owns each source.
        return _TrajectoryPhaseLowerer(source_id, proxy(source))


class _TrajectoryPhaseLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = _TrajectoryPhaseFactory.call_id
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, source_id: str, source: _TrajectorySource) -> None:
        self._source_id = source_id
        self._source = source

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        phase = self._source.phases[call.arguments["phase"]]
        if phase.feedback:
            cuts = (
                phase.start,
                *(
                    step - 1
                    for step in self._source.observation_steps
                    if phase.start + 1 < step <= phase.end
                ),
                phase.end,
            )
            lengths = tuple(b - a for a, b in zip(cuts, cuts[1:]))
            goal = ObservedJointTrajectory(self._source_id, phase.id, lengths)
        else:
            goal = self._source.resolve(
                phase.id, 0, phase.end - phase.start, context, context.planning_mask
            )
        return SemanticLowering(goal=JointPositionGoal(goal))


class _TrajectorySource:
    def __init__(
        self,
        robot: Robot,
        phases: tuple[_Phase, ...],
        positions: np.ndarray,
        poses: np.ndarray,
        dt: float,
        *,
        limits: _TrackLimits | None = None,
    ) -> None:
        self._robot = proxy(robot)
        self.phases = {phase.id: phase for phase in phases}
        self._positions = positions
        self._poses = poses
        self._dt = dt
        self._limits = limits
        self._hz = 1 / dt
        self._feedback: _GraspTrajectoryFeedback | None = None
        self.observation_steps: tuple[int, ...] = ()
        self._last_chunk: tuple[int, int, TimedTrajectory] | None = None

    def set_feedback(
        self, spec: _Feedback, obj: Articulation, tool_link: str, plan: _PlannedTrack
    ) -> None:
        def resolve_offset(offset: _PoseOffset | None) -> _PoseOffset | None:
            if offset is None or not isinstance(offset.reference, str):
                return offset
            reference = tuple(
                tuple(row) for row in plan.targets[offset.reference].tolist()
            )
            return replace(offset, reference=reference)

        self._object = proxy(obj)
        self._tool_link, self._object_link = tool_link, spec.link
        self._feedback = _GraspTrajectoryFeedback(
            plan.kinematics,
            plan.arm_ids,
            obj.get_link_pose(spec.link, to_matrix=True)[0].cpu().numpy(),
            resolve_offset(spec.initial_offset),
            tuple(
                replace(track, object_offset=resolve_offset(track.object_offset))
                for track in spec.tracks
            ),
            self._hz,
        )
        self.observation_steps = tuple(sorted(self._feedback.observation_steps()))

    def resolve(
        self,
        phase: str,
        start: int,
        stop: int,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> TimedTrajectory:
        if context.batch_size != 1 or not math.isclose(
            context.require_control_dt(), self._dt, rel_tol=1e-8
        ):
            raise ValueError(
                "The authored joint trajectory requires its configured control grid."
            )
        bounds = self.phases[phase]
        if not 0 <= start < stop <= bounds.end - bounds.start:
            raise ValueError(
                f"Invalid command interval for trajectory phase {phase!r}."
            )
        first, end = bounds.start + start + 1, bounds.start + stop + 1
        current = round(context.robot.timestamp * self._hz)
        positions = self._positions[first - 1 : end]
        if not context.is_preview and bounds.join_current:
            pad = first - 1 - current
            if pad < 0:
                raise ValueError(f"Trajectory handoff missed phase {phase!r}'s start.")
            last = self._robot.get_qpos(target=True).index_select(0, context.env_ids)[
                :, None
            ]
            nominal = torch.as_tensor(positions.copy(), device=last.device)[None]
            u = torch.linspace(
                0, 1, nominal.shape[1], device=last.device, dtype=nominal.dtype
            )
            weight = (1 - u) ** 3 * (1 + 3 * u + 6 * u.square())
            joined = nominal[:, 1:] + weight[None, 1:, None] * (last - nominal[:, :1])
            joined = torch.cat((last.expand(-1, pad + 1, -1), joined), dim=1)
            if self._limits is not None:
                self._limits.validate(joined[0].cpu().numpy())
            return self._trajectory(joined, context)
        if bounds.feedback and not context.is_preview:
            if current + 1 != first:
                raise ValueError(
                    f"Trajectory phase {phase!r} expected frame {first}, observed {current + 1}."
                )
            if self._last_chunk is not None and self._last_chunk[:2] == (first, end):
                return self._last_chunk[2]
            feedback = self._feedback
            assert feedback is not None
            corrected = np.empty_like(positions)
            corrected[:1] = (
                self._robot.get_qpos(target=True).cpu().numpy()
                if first > 1
                else positions[:1]
            )

            def observe() -> tuple[np.ndarray, np.ndarray]:
                if frame != first:
                    raise RuntimeError(
                        f"Trajectory feedback frame {frame} requires a chunk boundary."
                    )
                return (
                    self._robot.get_link_pose(self._tool_link, to_matrix=True)[0]
                    .cpu()
                    .numpy(),
                    self._object.get_link_pose(self._object_link, to_matrix=True)[0]
                    .cpu()
                    .numpy(),
                )

            for i, frame in enumerate(range(first, end), 1):
                corrected[i] = feedback.correct(
                    frame, positions[i], self._poses[frame], observe
                )
            if self._limits is not None:
                self._limits.validate(corrected)
            targets = self._trajectory(
                torch.as_tensor(corrected, device=context.robot.qpos.device)[None],
                context,
            )
            self._last_chunk = (first, end, targets)
            return targets
        if self._feedback is not None and not context.is_preview and first > 1:
            # Finite-difference across a transition from a corrected phase.
            positions = np.concatenate(
                (self._robot.get_qpos(target=True).cpu().numpy(), positions[1:])
            )
            if self._limits is not None:
                self._limits.validate(positions)
        return self._trajectory(
            torch.as_tensor(positions.copy(), device=context.robot.qpos.device)[None],
            context,
        )

    def _trajectory(
        self, targets: torch.Tensor, context: PlanningContext
    ) -> TimedTrajectory:
        return TimedTrajectory(
            positions=targets[:, 1:],
            velocities=targets.diff(dim=1) * self._hz,
            accelerations=None,
            dt=torch.full(
                (1, targets.shape[1] - 1),
                self._dt,
                dtype=torch.float64,
                device=targets.device,
            ),
            env_ids=context.env_ids,
        )


def _decode_trajectory_phase(value: object, *, path: str) -> _TrajectoryPhaseFactory:
    from .configured import _boolean, _identifier, _integer, _mapping, _sequence

    cfg = _mapping(
        value,
        path=path,
        required=frozenset({"kind", "motion", "phases"}),
        optional=frozenset({"feedback"}),
    )
    motion = _decode_cartesian_track(cfg["motion"], path=f"{path}.motion")
    if not isinstance(cfg["phases"], dict) or not cfg["phases"]:
        raise ValueError(f"{path}.phases must be a nonempty mapping.")
    phases = []
    previous = 0
    for name, item in cfg["phases"].items():
        key = f"{path}.phases.{name}"
        props = _mapping(
            item,
            path=key,
            required=frozenset({"steps"}),
            optional=frozenset({"join_current", "feedback"}),
        )
        steps = _sequence(props["steps"], path=f"{key}.steps")
        if len(steps) != 2:
            raise ValueError(f"{key}.steps must provide start and end control indices.")
        start, end = (_integer(step, path=f"{key}.steps", minimum=0) for step in steps)
        if not previous <= start < end <= motion.steps:
            raise ValueError(
                f"{key} is outside the trajectory or overlaps a previous phase."
            )
        join = _boolean(props.get("join_current", False), path=f"{key}.join_current")
        adaptive = _boolean(props.get("feedback", False), path=f"{key}.feedback")
        if join and adaptive:
            raise ValueError(f"{key} cannot join current state and use grasp feedback.")
        phases.append(_Phase(_identifier(name, path=key), start, end, join, adaptive))
        previous = end
    feedback = (
        None
        if "feedback" not in cfg
        else _decode_feedback(
            cfg["feedback"],
            f"{path}.feedback",
            motion.steps + 1,
            {"home", *(name for name, _ in motion.targets)},
        )
    )
    if feedback is None and any(phase.feedback for phase in phases):
        raise ValueError(f"{path} declares feedback phases with no feedback source.")
    return _TrajectoryPhaseFactory(motion, tuple(phases), feedback)


def _decode_feedback(
    value: object, path: str, frames: int, targets: set[str]
) -> _Feedback:
    from .configured import (
        _finite_tuple,
        _identifier,
        _integer,
        _mapping,
        _real,
        _sequence,
    )

    def transform(matrix: object, key: str) -> tuple[tuple[float, ...], ...]:
        rows = tuple(
            _finite_tuple(row, path=key, expected_length=4)
            for row in _sequence(matrix, path=key)
        )
        if len(rows) != 4 or not np.allclose(rows[3], [0, 0, 0, 1]):
            raise ValueError(f"{key} must be a homogeneous rigid transform.")
        rotation = np.array(rows)[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3)) or not math.isclose(
            np.linalg.det(rotation), 1.0, rel_tol=1e-7
        ):
            raise ValueError(f"{key} must have a proper rotation.")
        return rows

    def offset(item: object, key: str) -> _PoseOffset:
        spec = _mapping(
            item, path=key, required=frozenset({"reference", "rise", "fall"})
        )
        rise = _finite_tuple(spec["rise"], path=f"{key}.rise", expected_length=2)
        fall = _finite_tuple(spec["fall"], path=f"{key}.fall", expected_length=2)
        if not 0 <= rise[0] < rise[1] <= fall[0] < fall[1]:
            raise ValueError(f"{key} needs ordered, nonzero blend intervals.")
        reference = spec["reference"]
        if isinstance(reference, str):
            if reference not in targets:
                raise ValueError(f"{key}.reference must name a declared target.")
        else:
            reference = transform(reference, f"{key}.reference")
        return _PoseOffset(reference, rise, fall)

    cfg = _mapping(
        value,
        path=path,
        required=frozenset({"articulation_uid", "link", "tracks"}),
        optional=frozenset({"initial_offset"}),
    )
    tracks = []
    previous = 0
    for i, item in enumerate(_sequence(cfg["tracks"], path=f"{path}.tracks")):
        key = f"{path}.tracks[{i}]"
        track = _mapping(
            item,
            path=key,
            required=frozenset({"first", "last", "tool_in_object", "captures"}),
            optional=frozenset({"object_offset"}),
        )
        first = _integer(
            track["first"],
            path=f"{key}.first",
            minimum=previous + 1,
            maximum=frames - 1,
        )
        last = _integer(
            track["last"], path=f"{key}.last", minimum=first, maximum=frames - 1
        )
        captures = []
        previous_capture = first - 1
        for j, item in enumerate(_sequence(track["captures"], path=f"{key}.captures")):
            capture_key = f"{key}.captures[{j}]"
            capture = _mapping(
                item, path=capture_key, required=frozenset({"first", "last", "weight"})
            )
            a = _integer(
                capture["first"],
                path=f"{capture_key}.first",
                minimum=previous_capture + 1,
                maximum=last,
            )
            b = _integer(
                capture["last"], path=f"{capture_key}.last", minimum=a, maximum=last
            )
            weight = _real(
                capture["weight"],
                path=f"{capture_key}.weight",
                minimum=0,
                maximum=1,
                strict_minimum=True,
            )
            captures.append(_GraspCapture(a, b, weight))
            previous_capture = b
        tracks.append(
            _GraspTrack(
                first,
                last,
                transform(track["tool_in_object"], f"{key}.tool_in_object"),
                tuple(captures),
                (
                    offset(track["object_offset"], f"{key}.object_offset")
                    if "object_offset" in track
                    else None
                ),
            )
        )
        previous = last
    return _Feedback(
        *(
            _identifier(cfg[name], path=f"{path}.{name}")
            for name in ("articulation_uid", "link")
        ),
        (
            offset(cfg["initial_offset"], f"{path}.initial_offset")
            if "initial_offset" in cfg
            else None
        ),
        tuple(tracks),
    )
