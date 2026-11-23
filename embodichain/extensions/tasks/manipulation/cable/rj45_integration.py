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
from weakref import proxy

import numpy as np
import torch

from embodichain.lab.sim.atomic_skills import (
    AtomicSkillEngine,
    JointPositionGoal,
    MoveJoints,
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

from .rj45_motion import Rj45Feedback, Rj45Motion, _JOINT_NAMES, plan_rj45_motion

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.sim.objects.articulation import Articulation
    from embodichain.lab.sim.objects.robot import Robot

__all__ = ["FrankaRj45PhaseFactory"]

_PHASE_END_S = {
    "home": 1,
    "floor_pickup": 6,
    "approach_socket": 10,
    "insert_and_release": 14,
    "seated_hold": 19,
    "latch_grasp": 25,
    "withdraw": 28,
    "place_and_release": 34,
    "return_home": 40,
}
_NOMINAL_PHASES = frozenset({"home", "seated_hold", "return_home"})


@dataclass(frozen=True, slots=True)
class FrankaRj45PhaseFactory:
    """Bind the calibrated RJ45 reference and live plug to the Gym clock."""

    call_id: ClassVar[str] = "simulation.franka_rj45_phase"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    @property
    def argument_contract(self) -> SemanticCallContract:
        """Publish the authored phase names for offline program preflight."""
        return SemanticCallContract(
            required={"phase": tuple(_PHASE_END_S)}, optional={}
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
        if hz != 60 or config["duration_s"] != _PHASE_END_S["return_home"]:
            raise ValueError("The RJ45 program uses the calibrated 40 s / 60 Hz grid.")
        plug = environment.sim.get_articulation("plug")
        if plug is None:
            raise ValueError("RJ45 motion requires the plug articulation.")
        robot_path = Path(robot.cfg.fpath)
        motion = plan_rj45_motion(config, robot_path)
        settled_plug = plug.get_link_pose("plug", to_matrix=True)[0].cpu().numpy()
        feedback = Rj45Feedback(config, robot_path, settled_plug)
        # The registration keeps lowerer identities for freshness checks. The
        # active environment owns its command arrays and kinematics instead.
        environment._rj45_control = (motion, feedback)
        return _FrankaRj45PhaseLowerer(
            proxy(robot), proxy(plug), proxy(motion), proxy(feedback), hz
        )


class _FrankaRj45PhaseLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = FrankaRj45PhaseFactory.call_id
    target_descriptor: ClassVar[SkillDescriptor] = (
        FrankaRj45PhaseFactory.target_descriptor
    )
    preserves_symbolic_state: ClassVar[bool] = True

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
        self._corrected_frame = 0
        self._corrected_position = motion.positions[0]

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._robot.get_link_pose("fr3_hand_tcp", to_matrix=True)[0].cpu().numpy(),
            self._plug.get_link_pose("plug", to_matrix=True)[0].cpu().numpy(),
        )

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        """Lower a nominal phase or one feedback tick to full-joint commands."""
        phase = call.arguments["phase"]
        hz = self._hz
        dt = context.require_control_dt()
        if context.batch_size != 1 or not math.isclose(dt, 1 / hz, rel_tol=1e-8):
            raise ValueError("RJ45 commands require one arena at 60 Hz.")
        preceding_s = 0
        for name, end_s in _PHASE_END_S.items():
            if name == phase:
                break
            preceding_s = end_s
        else:
            raise ValueError(f"Unknown RJ45 phase {phase!r}.")
        first, last = preceding_s * hz + 1, end_s * hz
        frame = first if context.is_preview else round(context.robot.timestamp * hz) + 1
        if not first <= frame <= last or (phase in _NOMINAL_PHASES and frame != first):
            raise ValueError(
                f"RJ45 phase {phase!r} received out-of-sequence frame {frame}."
            )

        if phase in _NOMINAL_PHASES:
            positions = self._motion.positions[frame : last + 1]
        else:
            command = self._motion.positions[frame]
            if not context.is_preview:
                # Re-lowering the same tick must not integrate grasp smoothing twice.
                if frame != self._corrected_frame:
                    self._corrected_position = self._feedback.correct(
                        frame / hz, command, self._motion.poses[frame], self._observe
                    )
                    self._corrected_frame = frame
                command = self._corrected_position
            positions = command[None]

        previous = self._motion.positions[frame - 1 : frame]
        if not context.is_preview and frame > 1:
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
        return SemanticLowering(goal=JointPositionGoal(trajectory))
