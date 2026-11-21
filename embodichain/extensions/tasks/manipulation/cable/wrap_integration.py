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
"""Bind a checked, single-post cable wrap plan to the standard MoveJoints skill."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING
from weakref import proxy

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
from .wrap_motion import plan_wrap_motion

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.sim.objects.robot import Robot
    from .wrap_evaluation import _WrapMonitor

__all__ = ["CableWrapPhaseFactory"]

_WRAP_PHASES = ("home", "approach_post", "orbit", "tighten", "settle", "hold_closed")


@dataclass(frozen=True, slots=True)
class CableWrapPhaseFactory:
    """Create a phase lowerer with the physical scene's robot model and control grid."""

    call_id: ClassVar[str] = "simulation.cable_wrap_phase"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    @property
    def argument_contract(self) -> SemanticCallContract:
        """Expose the supported semantic phases during static preflight."""
        return SemanticCallContract(required={"phase": _WRAP_PHASES}, optional={})

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        raise ValueError(
            "Cable wrap planning requires its environment's scene configuration."
        )

    def create_for_environment(
        self,
        *,
        environment: EmbodiedEnv,
        simulation: object,
        robot: Robot,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        from .wrap_evaluation import attach_wrap_monitor

        if (
            engine.robot is not robot
            or environment.robot is not robot
            or environment.num_envs != 1
        ):
            raise ValueError(
                "The Franka wrap deployment requires its bound, single-arena robot."
            )
        config = environment.franka_rope_config
        monitor = getattr(environment, "_wrap_monitor", None)
        if monitor is None or not monitor.active:
            robot_path = Path(robot.cfg.fpath)
            motion = plan_wrap_motion(config, robot_path)
            monitor = attach_wrap_monitor(environment, config, motion, robot_path)
        # The registration retains its lowerer identities. The episode monitor owns
        # the full trajectory so expired episodes can release their command arrays.
        return _CableWrapPhaseLowerer(proxy(monitor), int(config["control_hz"]))


class _CableWrapPhaseLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = CableWrapPhaseFactory.call_id
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, monitor: _WrapMonitor, hz: int) -> None:
        self._monitor = monitor
        self._hz = hz

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        if set(call.arguments) != {"phase"} or not isinstance(
            call.arguments["phase"], str
        ):
            raise ValueError("A cable wrap call requires one named phase.")
        dt = context.require_control_dt()
        if context.batch_size != 1 or not math.isclose(
            dt, 1.0 / self._hz, rel_tol=1e-8
        ):
            raise ValueError(
                "Franka wrap commands must run at the scene's configured control rate."
            )
        phase_name = call.arguments["phase"]
        phase = self._monitor.motion.phase_slice(phase_name)
        # Include the last command of the preceding phase so its incoming
        # finite difference is preserved at every segment boundary.
        samples = self._monitor.motion.positions[phase.start - 1 : phase.stop]
        positions = torch.as_tensor(
            samples.copy(), device=context.robot.qpos.device
        ).unsqueeze(0)
        if phase_name == "approach_post" and not context.is_preview:
            positions = self._join_pickup(positions, phase.start - 1, context)
        trajectory = TimedTrajectory(
            positions=positions[:, 1:],
            velocities=positions.diff(dim=1) * self._hz,
            accelerations=None,
            dt=torch.full(
                (1, positions.shape[1] - 1),
                dt,
                dtype=torch.float64,
                device=positions.device,
            ),
            env_ids=context.env_ids,
        )
        return SemanticLowering(goal=JointPositionGoal(trajectory))

    def _join_pickup(
        self, nominal: torch.Tensor, lift_frame: int, context: PlanningContext
    ) -> torch.Tensor:
        # Hold the loaded command until the authored lift boundary. Blend out
        # the pickup offset during the tangential approach to the post.
        pad = lift_frame - (len(self._monitor.rows) - 1)
        if pad < 0:
            raise RuntimeError(
                f"Franka cable pickup exceeded the {lift_frame / self._hz:g}s lift boundary."
            )
        last = self._monitor.robot.get_qpos(target=True).index_select(
            0, context.env_ids
        )[:, None]
        u = torch.linspace(
            0, 1, nominal.shape[1], dtype=nominal.dtype, device=nominal.device
        )
        remaining = (1 - u) ** 3 * (1 + 3 * u + 6 * u.square())
        corrected = nominal[:, 1:] + remaining[None, 1:, None] * (last - nominal[:, :1])
        return torch.cat((last.expand(-1, pad + 1, -1), corrected), dim=1)
