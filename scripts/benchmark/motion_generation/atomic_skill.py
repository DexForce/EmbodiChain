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
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        SceneSnapshot,
        TaskState,
    )
    from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory
    from .config import SuiteCfg, TrackCfg
    from .contracts import ExecutionObservation, PhysicalStageEvaluator
    from .models import BenchmarkCase, CaseOutcome
    from .planners.base import PlannerAdapter
    from .scenarios.atomic_task import AtomicTaskScenario
    from .spatial import SpatialParameter

__all__ = ["AtomicSkillCaseProvider", "InfeasibleCaseError"]


class InfeasibleCaseError(RuntimeError):
    """The requested task cannot be posed, independently of any planner.

    Case generation raises it when independent IK cannot reach a frozen pose,
    no reachable grasp exists, or a start state leaves the joint-limit margin.
    A spatial domain draws another sample instead; a track without spatial
    sampling still stops, as for any other ``RuntimeError``.

    Args:
        message: Human-readable description of the rejected case.
        reason: Short code a spatial domain counts its rejections by:
            ``ik_unreachable``, ``no_reachable_grasp`` or ``start_joint_margin``.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class AtomicSkillCaseProvider(ABC):
    """Generate and ground one Atomic Action without planner-specific logic."""

    skill_id: str
    requires_gripper = False
    samples_grasps = False
    """Whether the skill samples grasps itself and needs a grasp generator."""
    spatial_parameters: tuple["SpatialParameter", ...] = ()
    """Spatial quantities a domain may sample for this skill's cases."""

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

    def prefix_invocations(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> tuple[ActionInvocation, ...]:
        """Return actions compiled before this skill to establish its precondition.

        A skill that needs a held object, such as Pour, returns the PickUp that
        acquires it.  The prefix is replayed and scored with the skill.

        Returns:
            Invocations compiled in order before :meth:`build_invocation`.
        """
        del scenario, case
        return ()

    def initial_scene(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> SceneSnapshot | None:
        """Return explicit scene observations the skill plans against.

        Returns:
            A snapshot that replaces the engine's rigid-object observations, or
            ``None`` to keep them.
        """
        del scenario, case
        return None

    def frozen_grasps(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> Mapping[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return the grasps a self-sampling skill must use, by gripper part.

        Skills that sample their own grasps, such as OpenDoor or HandOver,
        receive these manifest grasps instead, so every planner plans the same
        contacts.

        Returns:
            ``{gripper_part: (target_pose, grasp_pose)}`` batches; empty when
            the skill accepts explicit grasps or needs none.
        """
        del scenario, case
        return {}

    def object_rotation_segment(
        self, compiled: CompiledTrajectory
    ) -> tuple[int, int] | None:
        """Return the waypoint window whose object rotation is measured.

        Returns:
            ``(start, stop)`` trajectory indices, or ``None`` when the skill is
            not scored on how far it rotates the held object.
        """
        del compiled
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
