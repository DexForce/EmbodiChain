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

"""Offline atomic candidate generation and legacy PickUp template export."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import torch

from embodichain.lab.sim.motion.expansion import (
    TrajectoryPhase,
    TrajectoryTemplate,
)
from .atomic_candidates import (
    AtomicCandidateGenerationCfg,
    AtomicCandidateRejection,
    AtomicGenerationResult,
    AtomicTrajectoryGenerator,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "AtomicCandidateGenerationCfg",
    "AtomicCandidateRejection",
    "AtomicGenerationResult",
    "AtomicTrajectoryGenerator",
    "export_pickup_templates",
]


def export_pickup_templates(
    compiled: CompiledTrajectory,
    robot: Robot,
    *,
    control_dt: float,
    source_id: str = "atomic_pickup",
    source_revision: str = "v1",
    template_id: str = "reference_0",
    validator_id: str = "task_success",
    hold_steps: int = 30,
    control_part: str = "arm",
) -> tuple[TrajectoryTemplate, ...]:
    """Export full-joint candidates with protected approach/close/lift/hold phases.

    Accepts exactly MoveEndEffector followed by PickUp. Segment boundaries come
    from the atomic plans; they are never inferred from motion. The duplicate
    zero-time boundary must be a shared position and becomes one control-period
    hold. Mimic geometry is expanded from the robot's declared affine coupling.
    Only the initial transit permits spatial residual augmentation.

    This exports an offline atomic plan for independently validated qpos replay.
    It does not commit projected symbolic effects or run AtomicActionRuntime's
    recovery/feedback state machine. Physical contact and task verification are
    required separately before an exported trajectory can become expert data.

    Args:
        compiled: Successful full-batch result from AtomicActionEngine.compile.
        robot: Robot whose full joint order and mimic model produced the plan.
        control_dt: Fixed replay period, matching every positive plan interval.
        source_id: Stable source identity.
        source_revision: Caller-owned source revision.
        template_id: Template identity shared across row-specific cases.
        validator_id: Required physical task check identity.
        hold_steps: Explicit terminal commands used to measure a stable grasp.
        control_part: Arm joints authorized for transit residuals.

    Returns:
        One independently owned template for each physical row.
    """
    if (
        not math.isfinite(control_dt)
        or control_dt <= 0
        or type(hold_steps) is not int
        or hold_steps < 2
    ):
        raise ValueError("Provide a positive control_dt and at least two hold steps")
    if tuple(plan.skill_id for plan in compiled.action_plans) != (
        "move_end_effector",
        "pick_up",
    ):
        raise ValueError("Export requires exactly MoveEndEffector followed by PickUp")
    if not bool(compiled.plan_success.all()):
        raise ValueError("Cannot export failed atomic plans")
    positions = compiled.trajectory.positions.clone()
    intervals = compiled.trajectory.dt.clone()
    if positions.shape[0] != robot.num_instances or positions.shape[2] != robot.dof:
        raise ValueError(
            "Compiled trajectory must cover the complete physical robot batch"
        )
    if not torch.equal(
        compiled.trajectory.env_ids.cpu(), torch.arange(robot.num_instances)
    ):
        raise ValueError("Compiled row order must be the physical environment order")
    transit = compiled.action_waypoint_offset(1)
    approach, close, lift = (
        compiled.segment(1, name) for name in ("approach", "close", "lift")
    )
    if (
        approach.start != transit
        or approach.stop != close.start
        or close.stop != lift.start
        or lift.stop != positions.shape[1]
    ):
        raise ValueError("PickUp segments must cover the complete second action")
    if not torch.allclose(
        positions[:, transit], positions[:, transit - 1], atol=1e-5, rtol=0
    ):
        raise ValueError("Atomic action boundary is not a shared position")
    if not bool((intervals[:, 0] == 0).all() and (intervals[:, transit] == 0).all()):
        raise ValueError("Expected explicit zero-time action anchors")
    intervals[:, transit] = control_dt
    if not torch.allclose(
        intervals[:, 1:].double(),
        torch.full_like(intervals[:, 1:].double(), control_dt),
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Atomic timing must match the replay control clock")
    if set(robot.mimic_ids) & set(robot.mimic_parents):
        raise ValueError("PickUp export requires direct active-to-mimic coupling")
    for child, parent, multiplier, offset in zip(
        robot.mimic_ids,
        robot.mimic_parents,
        robot.mimic_multipliers,
        robot.mimic_offsets,
    ):
        positions[..., child] = positions[..., parent] * multiplier + offset
    length = positions.shape[1]
    positions = torch.cat(
        (positions, positions[:, -1:].repeat(1, hold_steps, 1)), dim=1
    )
    intervals = torch.cat(
        (intervals, intervals.new_full((robot.num_instances, hold_steps), control_dt)),
        dim=1,
    )
    phases = (
        TrajectoryPhase("transit", 0, transit, allowed_operators=("joint_residual",)),
        TrajectoryPhase("approach", transit, close.start, kind="contact"),
        TrajectoryPhase("close", close.start, lift.start, kind="contact"),
        TrajectoryPhase("lift", lift.start, length, kind="contact"),
        TrajectoryPhase("hold", length, length + hold_steps, kind="hold"),
    )
    return tuple(
        TrajectoryTemplate(
            source_id,
            source_revision,
            template_id,
            tuple(robot.joint_names),
            positions[row],
            intervals[row],
            phases,
            allowed_operators=("joint_residual",),
            validator_id=validator_id,
            controlled_joint_indices=tuple(robot.get_joint_ids(control_part)),
        )
        for row in range(robot.num_instances)
    )
