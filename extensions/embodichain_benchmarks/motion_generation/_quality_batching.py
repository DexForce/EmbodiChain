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

"""Execution-only NMG batches, retaining the original statistical cases."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.motion.planners.neural_planner import NeuralPlanner
from embodichain.lab.sim.motion.planners.utils import PlanResult, normalize_success_mask

from .config import stable_hash
from .models import BenchmarkCase

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__: list[str] = []


def case_groups(
    cases: list[BenchmarkCase], capacity: int
) -> Iterator[list[BenchmarkCase]]:
    """Group only inputs with the same NMG planning semantics and horizon."""
    groups: dict[tuple, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        if case.batch_size != 1:
            raise ValueError("Quality batching requires frozen B=1 cases.")
        if case.case_parameters.get("measurement_purpose") == "performance_replay":
            raise ValueError("Performance replay cannot use quality batching.")
        modalities = case.case_parameters.get("constraint_modalities")
        key = (
            case.num_waypoints,
            case.full_start_qpos is None,
            None if modalities is None else tuple(modalities),
            case.case_parameters.get("motion_validity", "ordered_cartesian_waypoints"),
        )
        groups[key].append(case)
    for group in groups.values():
        for offset in range(0, len(group), capacity):
            yield group[offset : offset + capacity]


def pack_cases(cases: list[BenchmarkCase], capacity: int) -> BenchmarkCase:
    """Pack planner tensors, padding unused physical slots without new trials."""
    if not cases or len(cases) > capacity:
        raise ValueError("Quality batch must occupy between one and capacity rows.")
    rows = cases + [cases[-1]] * (capacity - len(cases))
    first = cases[0]
    full = [case.full_start_qpos for case in rows]
    if any(value is None for value in full) and not all(
        value is None for value in full
    ):
        raise ValueError("Quality batch mixes full and controlled start states.")
    return replace(
        first,
        case_id="execution_batch_" + stable_hash([case.case_id for case in cases])[:16],
        batch_size=capacity,
        start_qpos=torch.cat([case.start_qpos for case in rows]),
        reference_qpos=torch.cat([case.reference_qpos for case in rows]),
        target_waypoints=torch.cat([case.target_waypoints for case in rows]),
        full_start_qpos=None if full[0] is None else torch.cat(full),
    )


def split_result(result: PlanResult, row: int, *, retimed: bool) -> PlanResult:
    """Restore the standalone row's grids, traces and raw boundary derivatives."""
    trace, lengths = result.policy_rollout_positions, result.policy_rollout_lengths
    if trace is None or lengths is None:
        raise ValueError("NMG quality batches require original policy trace lengths.")
    native_length = int(lengths[row])
    updates: dict[str, object] = {
        "success": normalize_success_mask(
            result.success,
            num_envs=trace.shape[0],
            device=trace.device,
            name="Quality batch success",
        )[row : row + 1],
        "policy_rollout_positions": trace[row : row + 1, :native_length].clone(),
        "policy_rollout_lengths": lengths[row : row + 1].clone(),
    }
    if result.positions is not None:
        if result.dt is None:
            raise ValueError("Quality batch output requires timing.")
        if retimed:
            timed_indices = torch.nonzero(result.dt[row] > 0).flatten()
            # Native TOPPRA output has at least two samples, including failed
            # or stationary rows. Later zero-dt holds belong to other rows.
            count = max(2, int(timed_indices[-1]) + 1) if timed_indices.numel() else 2
            count = min(count, result.positions.shape[1])
        else:
            count = native_length
        for name in ("positions", "dt", "velocities", "accelerations", "xpos_list"):
            value = getattr(result, name)
            if value is not None:
                updates[name] = value[row : row + 1, :count].clone()
        if not retimed:
            # In a packed output the row's last native state was an interior
            # sample followed by holds. Restore standalone one-sided stencils.
            updates["velocities"], updates["accelerations"] = (
                NeuralPlanner._compute_vel_acc_via_finite_diff(
                    updates["positions"], updates["dt"]
                )
            )
    if result.constraint_report is not None:
        updates["constraint_report"] = {
            key: (
                value[row : row + 1].clone()
                if value.ndim and value.shape[0] == trace.shape[0]
                else value.clone()
            )
            for key, value in result.constraint_report.items()
        }
    return replace(result, **updates)


class EvaluationRobotRow:
    """Map one original B=1 validator onto its actual simulator environment."""

    def __init__(self, robot: Robot, row: int) -> None:
        self.robot = robot
        self.row = row
        self.device = robot.device

    def compute_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ) -> torch.Tensor:
        """Evaluate the original single-row grid in its environment frame."""
        return self.robot.compute_fk(
            qpos=qpos, name=name, to_matrix=to_matrix, env_ids=[self.row]
        )

    def compute_batch_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ) -> torch.Tensor:
        """Preserve every original validator grid and explicit environment map."""
        return self.robot.compute_batch_fk(
            qpos=qpos, name=name, to_matrix=to_matrix, env_ids=[self.row]
        )

    def get_qpos_limits(self, name: str) -> torch.Tensor:
        """Select per-environment bounds while retaining shared limits."""
        limits = self.robot.get_qpos_limits(name=name)
        return (
            limits
            if limits.ndim == 2 or limits.shape[0] == 1
            else limits[self.row : self.row + 1]
        )

    def get_qvel_limits(self, name: str) -> torch.Tensor:
        """Select per-environment speed caps while retaining shared limits."""
        limits = self.robot.get_qvel_limits(name=name)
        return (
            limits
            if limits.ndim == 1 or limits.shape[0] == 1
            else limits[self.row : self.row + 1]
        )
