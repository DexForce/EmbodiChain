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

"""Typed records shared by the motion-generation benchmark modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import torch

__all__ = [
    "AlgorithmRole",
    "BenchmarkCase",
    "CaseOutcome",
    "PlannerMetadata",
    "TrialPhase",
    "TrialRecord",
]


class AlgorithmRole(str, Enum):
    """Role an algorithm plays in benchmark comparisons."""

    CANDIDATE = "candidate"
    PRIMARY_BASELINE = "primary_baseline"
    DIAGNOSTIC_BASELINE = "diagnostic_baseline"


class TrialPhase(str, Enum):
    """Lifecycle phase represented by a raw trial record."""

    AVAILABILITY = "availability"
    CONSTRUCT = "construct"
    PREPARE = "prepare"
    COLD = "cold"
    WARMUP = "warmup"
    MEASURED = "measured"


@dataclass(frozen=True)
class PlannerMetadata:
    """Stable planner identity and capability metadata."""

    algorithm_id: str
    algorithm_role: AlgorithmRole
    adapter: str
    config_hash: str
    capabilities: frozenset[str]
    model_revision: str = "N/A"
    inference_dtype: str = "fp32"
    supported_robots: tuple[str, ...] = ("franka_panda",)
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    """One env-batched planner input frozen before execution.

    The first fields retain the free-space case contract.  The trailing fields
    describe execution/task tracks without forcing planner adapters to know
    about a particular robot, atomic skill, or object implementation.
    """

    suite_version: str
    track: str
    scenario_id: str
    case_id: str
    seed: int
    batch_size: int
    num_waypoints: int
    path_shape: str
    start_state_bin: str
    start_qpos: torch.Tensor
    target_waypoints: torch.Tensor
    reference_qpos: torch.Tensor
    robot_id: str = "franka_panda"
    skill_id: str = "N/A"
    object_id: str | None = None
    task_difficulty: str | None = None
    primary_success: str = "motion_valid"
    full_start_qpos: torch.Tensor | None = None
    case_parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseOutcome:
    """External validation result for one environment row in a batch."""

    env_index: int
    planning_success: bool
    finite: bool
    ordered_waypoints_reached: bool
    motion_valid: bool
    completed_waypoint_ratio: float
    final_translation_err_mm: float | None
    final_rotation_err_deg: float | None
    waypoint_translation_err_mm_mean: float | None
    waypoint_translation_err_mm_p95: float | None
    waypoint_translation_err_mm_max: float | None
    waypoint_rotation_err_deg_mean: float | None
    waypoint_rotation_err_deg_p95: float | None
    waypoint_rotation_err_deg_max: float | None
    joint_limit_violation: bool
    max_normalized_joint_violation: float | None
    joint_path_length_rad: float | None
    cartesian_path_length_m: float | None
    path_efficiency: float | None
    failure_code: str | None = None
    planner_failure_code: str | None = None
    execution_success: bool | None = None
    task_success: bool | None = None
    task_completion_time_s: float | None = None
    joint_tracking_rmse_rad: float | None = None
    object_lift_delta_m: float | None = None
    replan_count: int | None = None


@dataclass(frozen=True)
class TrialRecord:
    """Raw lifecycle or planning record written to ``trials.jsonl``."""

    suite_version: str
    track: str
    scenario_id: str
    case_id: str
    algorithm_id: str
    algorithm_role: AlgorithmRole
    model_revision: str
    planner_config_hash: str
    seed: int
    repeat: int
    batch_size: int
    waypoint_count: int
    path_shape: str
    start_state_bin: str
    phase: TrialPhase
    status: str = "ok"
    failure_code: str | None = None
    failure_message: str | None = None
    cost_time_ms: float | None = None
    cpu_delta_mb: float | None = None
    gpu_delta_mb: float | None = None
    peak_gpu_mb: float | None = None
    robot_id: str = "franka_panda"
    skill_id: str = "N/A"
    object_id: str | None = None
    task_difficulty: str | None = None
    primary_success: str = "motion_valid"
    execution_time_ms: float | None = None
    end_to_end_time_ms: float | None = None
    trajectory_duration_s: float | None = None
    trajectory_waypoints: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    outcomes: tuple[CaseOutcome, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping while retaining numeric values."""
        data = asdict(self)
        data["algorithm_role"] = self.algorithm_role.value
        data["phase"] = self.phase.value
        return data
