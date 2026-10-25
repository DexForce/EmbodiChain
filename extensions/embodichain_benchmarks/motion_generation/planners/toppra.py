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

"""IK plus TOPPRA time-parameterization diagnostic adapter."""

from __future__ import annotations

from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners import (
    MoveType,
    PlanResult,
    PlanState,
    ToppraPlannerCfg,
    ToppraPlanOptions,
)
from embodichain.lab.sim.motion.planners.utils import TrajectorySampleMethod

from ..config import PlannerSpecCfg
from ..models import BenchmarkCase
from ..registry import register_planner_adapter
from .base import PlannerAdapter, PlannerContext

__all__ = ["ToppraAdapter"]


class ToppraAdapter(PlannerAdapter):
    """Pre-interpolate EEF targets through IK, then run TOPPRA.

    TOPPRA is the dual-arm Atomic Skill backend: the coordinated skills reject
    cuRobo, so the adapter also exposes its MotionGenerator to
    :class:`AtomicSkillEngine` and plans joint-space cases from their frozen
    reference configurations.
    """

    capabilities = frozenset(
        {"eef_waypoint", "joint_waypoint", "batched", "empty_world", "atomic_skill"}
    )
    timing_is_solved = True
    """TOPPRA solves the time parameterization under velocity and acceleration
    limits."""

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        super().__init__(spec, context)
        self.motion_generator: MotionGenerator | None = None

    def availability(self) -> tuple[bool, str | None]:
        """The in-tree TOPPRA implementation needs no optional external package."""
        return True, None

    def build(self) -> None:
        """Construct the TOPPRA MotionGenerator."""
        self.resolve_motion_limits(require_timing=True)
        self.motion_generator = MotionGenerator(
            MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.context.robot.uid))
        )

    def _plan_options(self) -> ToppraPlanOptions:
        """Return the suite-configured TOPPRA limits and sample count."""
        limits = self.resolve_motion_limits(require_timing=True)
        return ToppraPlanOptions(
            constraints={
                "velocity": limits["velocity"],
                "acceleration": limits["acceleration"],
            },
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=self.context.sample_interval,
        )

    def atomic_plan_options(self) -> ToppraPlanOptions:
        """Use the suite limits and sample count for Atomic Skill calls."""
        return self._plan_options()

    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Plan ordered EEF or joint waypoints through the IK-to-TOPPRA pipeline."""
        if self.motion_generator is None:
            raise RuntimeError("TOPPRA adapter must be built before plan().")
        validity = case.case_parameters.get(
            "motion_validity", "ordered_cartesian_waypoints"
        )
        if validity == "ordered_joint_waypoints":
            targets = [
                PlanState.from_qpos(
                    case.reference_qpos[:, index], move_type=MoveType.JOINT_MOVE
                )
                for index in range(case.num_waypoints)
            ]
        elif validity == "ordered_cartesian_waypoints":
            targets = [
                PlanState.from_xpos(case.target_waypoints[:, index])
                for index in range(case.num_waypoints)
            ]
        else:
            raise ValueError(f"Unsupported motion_validity mode {validity!r}.")
        return self.motion_generator.generate(
            targets,
            MotionGenOptions(
                start_qpos=case.start_qpos,
                control_part=self.context.control_part,
                is_interpolate=True,
                is_linear=validity == "ordered_cartesian_waypoints",
                plan_opts=self._plan_options(),
            ),
        )

    def close(self) -> None:
        """Drop the motion generator through the common planner lifecycle."""
        self._close_motion_generator()


register_planner_adapter("toppra", ToppraAdapter)
