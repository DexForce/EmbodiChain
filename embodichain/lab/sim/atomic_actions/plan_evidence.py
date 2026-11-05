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

"""Read-only complete action-plan proposals and velocity diagnostics."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from embodichain.compute.trajectory.limits import (
    joint_velocity_diagnostics,
    joint_velocity_validity,
)
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from .plans import ActionPlan

__all__ = ["action_plan_proposal_evidence"]


def action_plan_proposal_evidence(plan: ActionPlan, robot: Robot) -> dict[str, Any]:
    """Read complete joint-plan proposals without cloning trajectory tensors.

    Args:
        plan: Validated action plan retained by its planning/execution owner.
        robot: Exact borrowed Robot owning the plan's full joint columns.

    Returns:
        Independent plain data containing grasp proposals, velocity-qualified
        planning masks and final command velocity diagnostics. These values
        certify no acquired attachment, physical effect or task success. Source
        tensors are only read; no plan, policy or Robot state is modified.

    Raises:
        ValueError: If a successful plan lacks final joint trajectory evidence.
    """
    updates = getattr(
        getattr(plan, "expected_effects", None), "held_object_updates", {}
    )
    candidates = [
        dict(
            evidence_scope="proposal_only_not_observed_attachment",
            resource=resource,
            object_id=held.semantics.entity_id,
            object_to_eef=held.object_to_eef.detach().cpu().tolist(),
            grasp_xpos=held.grasp_xpos.detach().cpu().tolist(),
        )
        for resource, held in updates.items()
        if held is not None
    ]
    trajectory = plan.joint_trajectory
    if trajectory is None:
        if plan.plan_success.any():
            raise ValueError("Initial plan requires final joint trajectory evidence.")
        valid = plan.plan_success.clone()
        diagnostics = []
    else:
        limits = authored_joint_velocity_limits(robot, None).to(trajectory.positions)
        valid = joint_velocity_validity(trajectory.positions, trajectory.dt, limits)
        diagnostics = joint_velocity_diagnostics(
            trajectory.positions, trajectory.dt, limits
        )
    return dict(
        planned_grasp_candidates=candidates,
        plan_success=(plan.plan_success & valid).detach().cpu().tolist(),
        final_command_velocity=dict(
            valid_mask=valid.detach().cpu().tolist(),
            diagnostics=diagnostics,
        ),
    )
