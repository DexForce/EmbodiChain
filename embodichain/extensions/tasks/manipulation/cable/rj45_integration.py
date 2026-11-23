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
"""Execute calibrated RJ45 motion via the standard timed MoveJoints skill."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
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

from .rj45_motion import (
    Rj45Feedback,
    Rj45Motion,
    _feedback_observation_steps,
    _JOINT_NAMES,
    plan_rj45_motion,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.sim.objects.articulation import Articulation
    from embodichain.lab.sim.objects.robot import Robot

__all__ = ["FrankaRj45PhaseFactory"]

_PHASE_BOUNDS_S = {
    "home": (0, 1),
    "floor_pickup": (1, 6),
    "approach_socket": (6, 10),
    "insert_and_release": (10, 14),
    "seated_hold": (14, 19),
    "latch_grasp": (19, 25),
    "withdraw": (25, 28),
    "place_and_release": (28, 34),
    "return_home": (34, 40),
}
_NOMINAL_PHASES = frozenset({"home", "seated_hold", "return_home"})


def _phase_steps(phase: str, hz: int) -> tuple[int, int]:
    start_s, end_s = _PHASE_BOUNDS_S[phase]
    return start_s * hz, end_s * hz


@dataclass(frozen=True, slots=True)
class FrankaRj45PhaseFactory:
    """Bind the calibrated RJ45 reference and live plug to the Gym clock."""

    call_id: ClassVar[str] = "simulation.franka_rj45_phase"
    revision: ClassVar[str] = "2"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    @property
    def argument_contract(self) -> SemanticCallContract:
        """Publish the authored phase names for offline program preflight."""
        return SemanticCallContract(
            required={"phase": tuple(_PHASE_BOUNDS_S)}, optional={}
        )

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        raise ValueError("RJ45 motion requires the settled Gym scene.")

    def create_for_environment(
        self,
        *,
        environment: EmbodiedEnv,
        simulation: object,
        robot: Robot,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        """Build a fresh controller from the current episode's settled plug pose."""
        if (
            engine.robot is not robot
            or environment.robot is not robot
            or environment.num_envs != 1
            or list(robot.joint_names) != _JOINT_NAMES
        ):
            raise ValueError("RJ45 motion requires the scene's single-arena Franka.")
        config = environment.franka_rj45_config
        hz = config["control_hz"]
        if hz != 60 or config["duration_s"] != _PHASE_BOUNDS_S["return_home"][1]:
            raise ValueError("The RJ45 program uses the calibrated 40 s / 60 Hz grid.")
        plug = environment.sim.get_articulation("plug")
        if plug is None:
            raise ValueError("RJ45 motion requires the plug articulation.")
        robot_path = Path(robot.cfg.fpath)
        motion = plan_rj45_motion(config, robot_path)
        settled_plug = plug.get_link_pose("plug", to_matrix=True)[0].cpu().numpy()
        feedback = Rj45Feedback(config, robot_path, settled_plug)
        source = _Rj45JointTrajectorySource(
            proxy(robot), proxy(plug), motion, feedback, hz
        )
        source_id = uuid4().hex
        engine.planning_services.register_joint_trajectory_source(source_id, source)
        # The active episode owns its command arrays and measured grasp state.
        environment._rj45_control = source
        return _FrankaRj45PhaseLowerer(source_id, hz)


class _FrankaRj45PhaseLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = FrankaRj45PhaseFactory.call_id
    target_descriptor: ClassVar[SkillDescriptor] = (
        FrankaRj45PhaseFactory.target_descriptor
    )
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, source_id: str, control_hz: int) -> None:
        self._source_id = source_id
        self._hz = control_hz
        self._observation_steps = _feedback_observation_steps(control_hz)

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        """Create one MoveJoints invocation for the named phase."""
        phase = call.arguments["phase"]
        start, end = _phase_steps(phase, self._hz)
        cuts = (
            (start + 1,)
            + tuple(step for step in self._observation_steps if start + 1 < step <= end)
            + (end + 1,)
        )
        lengths = tuple(stop - first for first, stop in zip(cuts, cuts[1:]))
        return SemanticLowering(
            goal=JointPositionGoal(
                ObservedJointTrajectory(self._source_id, phase, lengths)
            )
        )


class _Rj45JointTrajectorySource:
    """Resolve bounded command chunks on the same grid as the measured feedback."""

    def __init__(
        self,
        robot: Robot,
        plug: Articulation,
        motion: Rj45Motion,
        feedback: Rj45Feedback,
        control_hz: int,
    ) -> None:
        self._robot = robot
        self._plug = plug
        self._motion = motion
        self._feedback = feedback
        self._hz = control_hz
        self._last_chunk: tuple[int, int, TimedTrajectory] | None = None

    def resolve(
        self,
        phase: str,
        start: int,
        stop: int,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> TimedTrajectory:
        """Read live state only at the chunk's acknowledged start boundary."""
        hz = self._hz
        dt = context.require_control_dt()
        if context.batch_size != 1 or not math.isclose(dt, 1 / hz, rel_tol=1e-8):
            raise ValueError("RJ45 commands require one arena at 60 Hz.")
        phase_start, phase_end = _phase_steps(phase, hz)
        if not 0 <= start < stop <= phase_end - phase_start:
            raise ValueError(f"RJ45 phase {phase!r} received invalid command bounds.")
        first, end = phase_start + start + 1, phase_start + stop + 1
        if not context.is_preview:
            frame = round(context.robot.timestamp * hz) + 1
            if frame != first:
                raise ValueError(
                    f"RJ45 phase {phase!r} expected frame {first}, observed {frame}."
                )
            if self._last_chunk is not None:
                previous_first, previous_end, previous_chunk = self._last_chunk
                # Re-preparing a chunk must not apply the grasp correction twice.
                if (first, end) == (previous_first, previous_end):
                    return previous_chunk
                if first != previous_end:
                    raise ValueError(
                        f"RJ45 phase {phase!r} skipped a command boundary."
                    )

        positions = self._motion.positions[first:end]
        if not context.is_preview and phase not in _NOMINAL_PHASES:
            positions = self._corrected_positions(positions, first)
        previous = self._motion.positions[first - 1 : first]
        if not context.is_preview and first > 1:
            previous = self._robot.get_qpos(target=True).cpu().numpy()
        targets = torch.as_tensor(
            np.concatenate((previous, positions)),
            device=context.robot.qpos.device,
        ).unsqueeze(0)
        trajectory = TimedTrajectory(
            positions=targets[:, 1:],
            velocities=targets.diff(dim=1) * hz,
            accelerations=None,
            dt=torch.full(
                (1, len(positions)), dt, dtype=torch.float64, device=targets.device
            ),
            env_ids=context.env_ids,
        )
        if not context.is_preview:
            self._last_chunk = (first, end, trajectory)
        return trajectory

    def _corrected_positions(self, positions: np.ndarray, first: int) -> np.ndarray:
        def observe() -> tuple[np.ndarray, np.ndarray]:
            if frame != first:
                raise RuntimeError(f"RJ45 feedback frame {frame} needs its own chunk.")
            return (
                self._robot.get_link_pose("fr3_hand_tcp", to_matrix=True)[0]
                .cpu()
                .numpy(),
                self._plug.get_link_pose("plug", to_matrix=True)[0].cpu().numpy(),
            )

        corrected = np.empty_like(positions)
        for offset, command in enumerate(positions):
            frame = first + offset
            corrected[offset] = self._feedback.correct(
                frame / self._hz, command, self._motion.poses[frame], observe
            )
        return corrected
