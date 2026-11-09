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

import math

import torch

from embodichain.compute.kinematics._masked_ik import _solve_masked_ik

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
from ._constraints import _case_modalities, _case_targets

__all__ = ["ToppraAdapter"]


class ToppraAdapter(PlannerAdapter):
    """Pre-interpolate EEF targets through IK, then run TOPPRA.

    The optional masked-constraint route solves each requested component from
    the preceding public joint state, then times the resulting joint path.
    Unused target components and Cartesian reference solutions are never inputs.
    """

    configuration_features = frozenset({"masked_constraints"})

    capabilities = frozenset(
        {"eef_waypoint", "joint_waypoint", "batched", "empty_world", "atomic_action"}
    )
    timing_is_solved = True
    """TOPPRA solves the time parameterization under velocity and acceleration
    limits."""

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        super().__init__(spec, context)
        self.motion_generator: MotionGenerator | None = None
        if spec.config.get("masked_constraints", False):
            self.capabilities = self.capabilities | {"masked_constraints"}

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

    def atomic_plan_options(
        self, *, control_dt: float | None = None
    ) -> ToppraPlanOptions:
        """Sample Atomic motion on the host's control clock under suite limits.

        Rest boundaries preserve timing when Atomic Skills insert hand holds.
        Each stopped segment uses the exact destination period after conservative
        duration rounding. Direct callers without a host period retain the suite
        quantity default.
        """
        options = self._plan_options()
        if control_dt is not None:
            if (
                isinstance(control_dt, bool)
                or not math.isfinite(control_dt)
                or control_dt <= 0
            ):
                raise ValueError("control_dt must be positive and finite.")
            options.sample_method = TrajectorySampleMethod.TIME
            options.sample_interval = float(control_dt)
            options.stop_at_waypoints = True
        return options

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
        elif validity == "ordered_constraint_waypoints":
            if "masked_constraints" not in self.capabilities:
                raise ValueError("Enable config.masked_constraints for masked cases.")
            return self._plan_constraints(case)
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

    def _plan_constraints(self, case: BenchmarkCase) -> PlanResult:
        """Solve sequential component IK and retain failed rows as failures."""
        assert self.motion_generator is not None
        robot, part = self.context.robot, self.context.control_part
        lower, upper = robot.get_qpos_limits(name=part).to(case.start_qpos).unbind(-1)
        current = case.start_qpos.clone()
        alive = torch.ones(case.batch_size, dtype=torch.bool, device=current.device)
        targets = []
        settings = dict(self.spec.config.get("masked_ik", {}))
        for modality, target in zip(_case_modalities(case), _case_targets(case)):
            if modality == "joint":
                solved = target.qpos
                valid = torch.isfinite(solved).all(-1) & (
                    (solved >= lower) & (solved <= upper)
                ).all(-1)
            else:
                valid, solved = _solve_masked_ik(
                    target.xpos,
                    current,
                    lower,
                    upper,
                    lambda q: robot.compute_batch_fk(qpos=q, name=part, to_matrix=True),
                    modality,
                    **settings,
                )
            alive &= valid
            current = torch.where(alive[:, None], solved, current)
            targets.append(PlanState.from_qpos(current.clone()))
        # Match MotionGenerator's failed-row contract even when an earlier
        # component was solved: a failed request is a stationary start hold.
        for target in targets:
            target.qpos = torch.where(alive[:, None], target.qpos, case.start_qpos)
        result = self.motion_generator.generate(
            targets,
            MotionGenOptions(
                start_qpos=case.start_qpos,
                control_part=part,
                is_interpolate=True,
                is_linear=False,
                plan_opts=self._plan_options(),
            ),
        )
        result.success &= alive.to(result.success.device)
        return result

    def close(self) -> None:
        """Drop the motion generator through the common planner lifecycle."""
        self._close_motion_generator()


register_planner_adapter("toppra", ToppraAdapter)
