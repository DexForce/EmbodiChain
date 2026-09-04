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

"""Scenario provider contract for motion-generation tracks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from embodichain.lab.sim.planners.utils import PlanResult

from ..metrics.trajectory import compute_case_outcomes, make_failure_outcomes
from ..models import CaseOutcome

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot

    from ..config import SuiteCfg, TrackCfg
    from ..models import BenchmarkCase
    from ..planners.base import PlannerAdapter
    from ..video import VideoRecordCfg

__all__ = ["ScenarioEvaluation", "ScenarioProvider"]


@dataclass(frozen=True)
class ScenarioEvaluation:
    """Outcomes and execution metrics produced outside planner timing."""

    outcomes: tuple[CaseOutcome, ...]
    execution_time_ms: float | None = None
    end_to_end_time_ms: float | None = None
    trajectory_duration_s: float | None = None
    trajectory_waypoints: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class ScenarioProvider(ABC):
    """Generate, plan, execute, and evaluate one registered scenario kind."""

    required_capabilities: frozenset[str] = frozenset()

    @abstractmethod
    def batch_sizes(self, suite: "SuiteCfg", track: "TrackCfg") -> list[int]:
        """Return simulator batch sizes required by this track."""

    @abstractmethod
    def generate_cases(
        self,
        suite: "SuiteCfg",
        track: "TrackCfg",
        robot: "Robot",
        control_part: str,
        batch_size: int,
    ) -> list["BenchmarkCase"]:
        """Build the frozen case manifest for one batch size."""

    def configure_runtime(
        self,
        simulation: "SimulationManager",
        robot: "Robot",
        suite: "SuiteCfg",
        track: "TrackCfg",
        control_part: str,
    ) -> None:
        """Finalize scenario runtime state before case generation."""

    def create_runtime_entities(
        self,
        simulation: "SimulationManager",
        suite: "SuiteCfg",
        track: "TrackCfg",
    ) -> None:
        """Add scenario-owned entities before the first physics update."""

    def close_runtime(self) -> None:
        """Release references to scenario-owned simulation entities."""

    def prepare_planner(
        self, adapter: "PlannerAdapter", first_case: "BenchmarkCase"
    ) -> None:
        """Bind scenario resources to a built planner outside trial timing."""

    def close_planner(self, adapter: "PlannerAdapter") -> None:
        """Release scenario resources that retain a planner adapter."""

    def reset_case(
        self,
        simulation: "SimulationManager",
        robot: "Robot",
        case: "BenchmarkCase",
        control_part: str,
    ) -> None:
        """Restore the frozen robot start state before a planning call."""
        if case.full_start_qpos is not None:
            for target in (False, True):
                robot.set_qpos(case.full_start_qpos, target=target)
        else:
            for target in (False, True):
                robot.set_qpos(case.start_qpos, name=control_part, target=target)
        robot.clear_dynamics()
        simulation.update(step=1)

    def plan_case(self, adapter: "PlannerAdapter", case: "BenchmarkCase") -> object:
        """Plan one case through the selected adapter."""
        return adapter.plan(case)

    def plan_contract_error(self, result: object) -> str | None:
        """Return a diagnostic when a planner artifact violates this scenario."""
        if isinstance(result, PlanResult):
            return None
        return f"Expected PlanResult, got {type(result).__name__}."

    def failure_outcomes(
        self, case: "BenchmarkCase", failure_code: str
    ) -> tuple[CaseOutcome, ...]:
        """Create scenario-appropriate outcomes after a runner-level failure."""
        return make_failure_outcomes(case.batch_size, failure_code)

    def evaluate_case(
        self,
        result: object,
        case: "BenchmarkCase",
        robot: "Robot",
        control_part: str,
        suite: "SuiteCfg",
        *,
        planning_time_ms: float,
    ) -> ScenarioEvaluation:
        """Externally validate a planner-only trajectory."""
        if not isinstance(result, PlanResult):
            raise TypeError(self.plan_contract_error(result))
        outcomes = compute_case_outcomes(
            result,
            case,
            robot,
            control_part,
            validation_samples=suite.protocol.validation_samples,
            position_threshold_m=suite.protocol.position_threshold_m,
            rotation_threshold_rad=suite.protocol.rotation_threshold_rad,
            joint_limit_tolerance_rad=suite.protocol.joint_limit_tolerance_rad,
        )
        return ScenarioEvaluation(outcomes=outcomes)

    def record_replay(
        self,
        result: object,
        case: "BenchmarkCase",
        evaluation: ScenarioEvaluation | None,
        *,
        output_dir: Path,
        algorithm_id: str,
        video: "VideoRecordCfg",
    ) -> Path | None:
        """Optionally record a second, untimed replay. Default is a no-op.

        Args:
            result: Planner or compiled-action artifact from the measured trial.
            case: Frozen case identity used for the output filename.
            evaluation: Timed evaluation, or ``None`` after a runner-level failure.
            output_dir: Directory that should receive the mp4.
            algorithm_id: Planner id used in the filename.
            video: Recording policy and encoder settings.

        Returns:
            Path to a saved video, or ``None`` when this scenario does not record.
        """
        del result, case, evaluation, output_dir, algorithm_id, video
        return None
