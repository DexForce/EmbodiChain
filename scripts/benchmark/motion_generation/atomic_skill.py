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

"""Case construction contract for Atomic Skill benchmark integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from embodichain.lab.sim.atomic_actions import ActionInvocation, TaskState
    from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory
    from .config import SuiteCfg, TrackCfg
    from .contracts import ExecutionObservation, PhysicalStageEvaluator
    from .models import BenchmarkCase, CaseOutcome
    from .planners.base import PlannerAdapter
    from .scenarios.atomic_task import AtomicTaskScenario

__all__ = ["AtomicSkillCaseProvider"]


class AtomicSkillCaseProvider(ABC):
    """Generate and ground one Atomic Action without planner-specific logic."""

    skill_id: str
    requires_gripper = False

    @abstractmethod
    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        """Generate one frozen case and independent IK validity evidence."""

    @abstractmethod
    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        """Ground the case into one planner-independent action invocation."""

    def object_id(self, case: BenchmarkCase) -> str | None:
        """Return the manipulated object identifier when one exists."""
        return case.object_id

    def lift_segment_start(self, compiled: CompiledTrajectory) -> int | None:
        """Return the first lift waypoint that should release object dynamics."""
        return None

    def articulation_effect_segment(self, case: BenchmarkCase) -> str | None:
        """Return the segment whose live articulation displacement is scored."""
        del case
        return None

    def initial_task_state(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> TaskState | None:
        """Return an optional symbolic precondition for isolated action testing."""
        del scenario, case
        return None

    def physical_evaluator(self) -> PhysicalStageEvaluator | None:
        """Return a measured-stage evaluator, or use legacy task_result.

        Returns:
            An evaluator that receives no compiled planning state.
        """
        return None

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        """Return legacy task success until this provider supplies an evaluator."""
        raise NotImplementedError(
            "Provide physical_evaluator() or legacy task_result()."
        )
