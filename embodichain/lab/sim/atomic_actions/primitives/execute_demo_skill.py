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

"""Generic Atomic Skill for a task-specific expert-demonstration asset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from copy import deepcopy
from types import MappingProxyType
from typing import ClassVar

from embodichain.lab.sim.atomic_actions.core import AtomicAction, ObjectSemantics
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.nominal import (
    NominalMotionSource,
    MonitoredNominalMotionSource,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    EffectVerificationRequirement,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.requirements import (
    SkillBindingContract,
    JOINT_POSITION_CAPABILITY,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)

__all__ = ["DemoSkillGoal", "ExecuteDemoSkill"]


@dataclass(frozen=True, slots=True)
class DemoSkillGoal:
    """Pinned asset identity and logical scene roles, without hardware names."""

    asset_sha256: str
    role_bindings: Mapping[str, str]
    role_uids: Mapping[str, str] = field(default_factory=dict)
    terminal_semantics: ObjectSemantics | None = None
    goal_kind: ClassVar[str] = "demo_skill"

    def __post_init__(self) -> None:
        from embodichain.data_pipeline.real2sim.demo_skill import _sha256

        _sha256(self.asset_sha256, "asset_sha256")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.role_bindings.items()
        ):
            raise ValueError("DemoSkill roles require nonempty logical entity IDs")
        object.__setattr__(
            self, "role_bindings", MappingProxyType(dict(self.role_bindings))
        )
        object.__setattr__(self, "role_uids", MappingProxyType(dict(self.role_uids)))

    def __deepcopy__(self, memo: dict[int, object]) -> DemoSkillGoal:
        return DemoSkillGoal(
            self.asset_sha256,
            dict(self.role_bindings),
            dict(self.role_uids),
            deepcopy(self.terminal_semantics, memo),
        )


class ExecuteDemoSkill(AtomicAction[DemoSkillGoal, ActionOptions]):
    """Execute one demonstration contract as one planning/recovery boundary."""

    skill_id: ClassVar[str] = "execute_demo_skill"
    GoalType: ClassVar[type] = DemoSkillGoal
    OptionsType: ClassVar[type] = ActionOptions
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
                grasp_commands={},
            ),
        )
    )

    def _plan(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ActionPlan:
        raise ValueError("ExecuteDemoSkill requires a pinned, monitored nominal source")

    def _plan_from_nominal(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        nominal_source: NominalMotionSource,
    ) -> ActionPlan:
        self.require_goal(request)
        if not isinstance(nominal_source, MonitoredNominalMotionSource):
            raise TypeError(
                "DemoSkill requires measured execution and terminal verification"
            )
        # The source declares arbitrary internal phase names. The common validator
        # still checks full-joint ownership, clock, limits and recovery policy.
        artifact = nominal_source.artifact
        from embodichain.lab.sim.atomic_actions.primitives._helpers import (
            require_shared_task_state_key,
        )

        entry_key = require_shared_task_state_key(
            request.binding.endpoint("primary", "motion"),
            request.binding.endpoint("primary", "grasp"),
            participant="DemoSkill",
        )
        held = context.get_held_object(entry_key)
        expected = artifact.execution_spec.initial_held_role
        if expected is None and held is not None:
            raise ValueError("DemoSkill requires an empty input manipulator")
        if expected is not None and (
            held is None
            or held.semantics.entity_id != request.goal.role_bindings[expected]
            or not bool(context.task.exclusive_held_object_mask(entry_key).all())
        ):
            raise ValueError(
                "DemoSkill requires its declared exclusive input attachment"
            )
        motion = self._validate_nominal_motion(
            request,
            context,
            nominal_source,
            segments=None,
        )
        from embodichain.lab.sim.atomic_actions.effects import StateDelta
        from embodichain.lab.sim.atomic_actions.state import HeldObjectState
        from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
        from embodichain.lab.sim.atomic_actions.primitives._helpers import (
            require_shared_task_state_key,
        )
        import torch
        import numpy as np

        spec = nominal_source.monitor.spec
        key = require_shared_task_state_key(
            request.binding.endpoint("primary", "motion"),
            request.binding.endpoint("primary", "grasp"),
            participant="DemoSkill",
        )
        effects = StateDelta()
        if spec.release_on_exit:
            effects = StateDelta(held_object_updates={key: None})
        elif spec.terminal_held_role is not None:
            if request.goal.terminal_semantics is None:
                raise ValueError(
                    "terminal attachment requires scene-owned object semantics"
                )
            constraint = next(
                c
                for c in spec.constraints
                if c.scope == "terminal"
                and c.kind == "relative_pose"
                and c.subject == "tcp"
                and c.other == spec.terminal_held_role
            )
            relation = torch.as_tensor(
                np.asarray(constraint.reference_pose).reshape(4, 4),
                device=context.robot.qpos.device,
                dtype=context.robot.qpos.dtype,
            ).unsqueeze(0)
            arm = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            terminal = self.robot.compute_batch_fk(
                motion.trajectory.positions[:, -1:, list(arm.joint_ids)],
                name=arm.control_part,
                env_ids=context.env_ids.tolist(),
                to_matrix=True,
            ).reshape(1, 4, 4)
            effects = StateDelta(
                held_object_updates={
                    key: HeldObjectState(
                        request.goal.terminal_semantics, relation, terminal
                    )
                }
            )
        metadata = motion.provenance()
        metadata["demo_skill_phase_kinds"] = {
            phase.name: phase.kind for phase in spec.phases
        }
        metadata["demo_skill_events"] = [event.sample_index for event in spec.events]
        metadata["demo_skill"] = {
            "asset_id": artifact.execution_spec.asset_id,
            "revision": artifact.execution_spec.revision,
            "content_sha256": artifact.content_sha256,
            "spec": artifact.spec_json,
            "recording_sha256": artifact.demonstration.recording_sha256,
        }
        return self.build_plan(
            request,
            context,
            success=motion.success,
            trajectory=motion.trajectory,
            segment_lengths=motion.segment_lengths,
            replannable=False,
            effect_verification=EffectVerificationRequirement("demo_skill"),
            expected_effects=effects,
            diagnostics=PlannerDiagnostics(backend="demo_skill", metadata=metadata),
        )
