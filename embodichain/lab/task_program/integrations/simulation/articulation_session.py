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

"""Explicit compiled-occurrence wiring for installed prismatic continuation facts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from embodichain.lab.sim.atomic_actions.articulation_continuation import (
    ArticulationContinuationContext,
    ArticulationInstalledPlan,
    capture_installed_articulation_plan,
)
from embodichain.lab.sim.atomic_actions import ExecutionPlanAttempt
from embodichain.lab.task_program.semantics import SemanticIntegrationManifest
from .prismatic_lowering import (
    ArticulationSlideFactory,
    ArticulationWithdrawFactory,
    ArticulationParkFactory,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        PlanningContext,
        ResolvedActionRequest,
    )
    from embodichain.lab.task_program.runtime.expansion import TaskProgramPlanRequest
    from embodichain.lab.task_program.compiler import CompiledTaskProgram
    from ..catalog import SimulationTaskProgramRegistration
    from ..invocation_policy import InvocationPolicyBinding

__all__ = [
    "ArticulationInvocationBinding",
    "ArticulationContinuationRegistry",
    "bind_articulation_invocations",
]


@dataclass(frozen=True, slots=True)
class ArticulationInvocationBinding:
    """One exact compiled call and optional adjacent predecessor/followup budget."""

    invocation_id: str
    call_id: str
    kind: Literal["slide", "withdraw", "park"]
    predecessor_id: str | None = None
    withdrawal_sample_count: int | None = None

    def __post_init__(self) -> None:
        for value in (self.invocation_id, self.call_id):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Articulation occurrence identities must be exact.")
        if self.predecessor_id is not None and (
            type(self.predecessor_id) is not str
            or not self.predecessor_id
            or self.predecessor_id != self.predecessor_id.strip()
        ):
            raise ValueError("Articulation predecessor identity must be exact.")
        if self.kind not in ("slide", "withdraw", "park"):
            raise ValueError("Unknown articulation occurrence role.")
        if self.kind != "slide" and self.withdrawal_sample_count is not None:
            raise ValueError("Only Slide declares its following cleanup sample count.")
        if self.withdrawal_sample_count is not None and (
            type(self.withdrawal_sample_count) is not int
            or self.withdrawal_sample_count < 1
        ):
            raise ValueError(
                "Cleanup sample count must be an explicit positive integer."
            )


def bind_articulation_invocations(
    program: CompiledTaskProgram,
    registration: SimulationTaskProgramRegistration,
    invocation_policy_bindings: tuple[InvocationPolicyBinding, ...] = (),
) -> tuple[ArticulationInvocationBinding, ...]:
    """Bind only adjacent source calls with unchanged target and resource intent.

    Required sample counts come from the actually selected declared presets,
    including explicit InvocationPolicyBinding selections. No live observations,
    task indices, object-name conventions or inferred resource capabilities select
    recovery behavior.

    Args:
        program: Exact declaration-only compiled program.
        registration: Source-qualified static factories and robot profile.
        invocation_policy_bindings: Explicit per-occurrence preset selections.

    Returns:
        Ordered serial occurrence bindings; no live history is created.
    """
    return _bind_articulation_declarations(
        program,
        registration.robot_profile_binding.declare(),
        registration.catalog.scene,
        registration.call_catalog,
        registration.registered_semantic_lowerer_factories,
        invocation_policy_bindings,
    )


def _bind_articulation_declarations(
    program,
    profile,
    scene,
    call_catalog,
    declared_factories,
    invocation_policy_bindings,
):
    from ..invocation_policy import BoundInvocationPolicies

    factories = {
        value.call_id: value
        for value in declared_factories
        if isinstance(value, (ArticulationSlideFactory, ArticulationParkFactory))
    }
    if not factories:
        return ()
    manifest = SemanticIntegrationManifest(
        scene=scene,
        robot_profile=profile,
        call_catalog=call_catalog,
        runtime_preset=program.integration.runtime_preset,
    )
    selected = {}
    if invocation_policy_bindings:
        policies = BoundInvocationPolicies(program, invocation_policy_bindings)
        policies.validate_manifest(program, manifest)
        selected = {value.invocation_id: value.preset_id for value in policies.bindings}
    segments = tuple(program.iter_segments())
    entries = {}
    for segment in segments:
        for call in segment.calls:
            factory = factories.get(getattr(call.call, "call_id", None))
            if factory is None:
                continue
            if segment.parallel_block is not None or len(segment.calls) != 1:
                raise ValueError(
                    "Source articulation continuation requires one sequential call per segment."
                )
            arguments = dict(call.call.arguments)
            if isinstance(factory, ArticulationSlideFactory):
                if set(arguments) not in (
                    {"object", "state"},
                    {"object", "state", "part"},
                ):
                    raise ValueError(
                        "Prismatic calls require exact object/state/optional-part arguments."
                    )
                key = (
                    arguments["object"]
                    if "part" not in arguments
                    else f"{arguments['object']}::{arguments['part']}"
                )
                selected_binding = next(
                    (value for value in factory.bindings if value.selection_key == key),
                    None,
                )
                if selected_binding is None:
                    raise ValueError(
                        "Prismatic call selects an undeclared source part."
                    )
                selected_binding.source.target(arguments["state"])
            elif arguments:
                raise ValueError("Prismatic Park arguments must be empty.")
            identity = (
                f"{program.program_id}/{segment.segment_id}:{call.segment_call_index}"
            )
            linked = manifest.link_call(
                call.call, path=call.source_path, preset_id=selected.get(identity)
            )
            preset = profile.presets[linked.preset_id]
            if (
                type(preset.recovery_policy.planning_recovery)
                is not factory.recovery_type
            ):
                raise ValueError(
                    "Source articulation calls require their explicit typed planning recovery."
                )
            motion = preset.motion_policy
            expected_retries = () if factory.continuation_role == "park" else (260, 320)
            if (
                motion.strategy != "ik_interp"
                or motion.interpolation_mode != "fixed_grid"
                or motion.ik_seed_mode != "sequential"
                or motion.dynamic_collision_mode != "off"
                or motion.segment_sample_count_retries != expected_retries
                or motion.segment_continuous_ik_recovery
            ):
                raise ValueError(
                    "Source articulation presets require their exact fixed-grid segment recovery contract."
                )
            entries[segment.segment_index] = (
                identity,
                call.call,
                factory.continuation_role,
                motion.sample_count,
            )
    predecessors, followups = {}, {}
    for first, second in zip(segments, segments[1:]):
        if first.segment_index not in entries or second.segment_index not in entries:
            continue
        left, right = entries[first.segment_index], entries[second.segment_index]
        same_resource = left[1].resources == right[1].resources
        if (
            left[2] == "slide"
            and right[2] == "withdraw"
            and same_resource
            and left[1].arguments == right[1].arguments
        ):
            predecessors[right[0]] = left[0]
            followups[left[0]] = right[3]
        elif (
            left[2] == "withdraw"
            and right[2] == "park"
            and same_resource
            and not right[1].arguments
        ):
            predecessors[right[0]] = left[0]
    return tuple(
        ArticulationInvocationBinding(
            identity,
            call.call_id,
            kind,
            predecessors.get(identity),
            followups.get(identity),
        )
        for identity, call, kind, _ in entries.values()
    )


class ArticulationContinuationRegistry:
    """Own actual installation history for one B1 program/episode instance.

    The same object is injected as the Atomic provider and Task Program lifecycle
    factory. Preview reads do not change history; only validated installed attempts
    produce immutable facts. Physical task success is never inferred here.
    """

    def __init__(
        self,
        robot: Robot,
        bindings: Sequence[ArticulationInvocationBinding],
        *,
        program_fingerprint: str,
        episode_id: str,
        validate_scope: Callable[[], None],
        validate_observation: Callable[[PlanningContext], None],
    ) -> None:
        values = tuple(bindings)
        if (
            robot.num_instances != 1
            or not values
            or any(type(value) is not ArticulationInvocationBinding for value in values)
        ):
            raise ValueError(
                "Articulation continuation requires explicit B1 occurrence bindings."
            )
        by_id = {value.invocation_id: value for value in values}
        if len(by_id) != len(values):
            raise ValueError("Articulation invocation identities must be unique.")
        for value in values:
            if value.predecessor_id is not None:
                predecessor = by_id.get(value.predecessor_id)
                if predecessor is None or (predecessor.kind, value.kind) not in (
                    ("slide", "withdraw"),
                    ("withdraw", "park"),
                ):
                    raise ValueError(
                        "Articulation predecessor must name its declared prior stage."
                    )
        for identity in (program_fingerprint, episode_id):
            if (
                type(identity) is not str
                or not identity
                or identity != identity.strip()
            ):
                raise ValueError(
                    "Articulation ownership requires actual program/episode identity."
                )
        if not callable(validate_scope) or not callable(validate_observation):
            raise TypeError(
                "Articulation continuation requires explicit owner/observation guards."
            )
        self._robot, self._bindings = robot, MappingProxyType(by_id)
        self._scope_id = f"{program_fingerprint}:{episode_id}:{uuid4().hex}"
        self._guard, self._observation_guard = validate_scope, validate_observation
        self._facts: dict[str, tuple[ArticulationInstalledPlan, ...]] = {}
        self._version, self._closed = 0, False

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed engine robot."""
        return self._robot

    @property
    def installed_facts(self) -> tuple[ArticulationInstalledPlan, ...]:
        """Return immutable historical facts without reading native state."""
        return tuple(value for values in self._facts.values() for value in values)

    def _open(self):
        if self._closed:
            raise RuntimeError("Articulation continuation registry is closed.")
        self._guard()

    def _context(self, binding):
        return ArticulationContinuationContext(
            self._scope_id,
            self._version,
            binding.kind,
            binding.invocation_id,
            binding.withdrawal_sample_count,
            self._facts.get(binding.predecessor_id, ()),
        )

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ArticulationContinuationContext | None:
        """Read facts for this exact occurrence without committing a proposal."""
        binding = self._bindings.get(request.invocation_id)
        if binding is None:
            return None
        self._open()
        self._observation_guard(context)
        expected = {
            "slide": "slide",
            "withdraw": "move_end_effector",
            "park": "move_joints",
        }[binding.kind]
        if request.skill_id != expected or context.env_ids.tolist() != [0]:
            raise ValueError(
                "Articulation request differs from its declared occurrence."
            )
        return self._context(binding)

    def create_plan_lifecycle(
        self, request: TaskProgramPlanRequest, *, engine: AtomicActionEngine
    ) -> ArticulationContinuationRegistry | None:
        """Select this same owner only for its exact declared semantic occurrence."""
        binding = self._bindings.get(request.invocation.invocation_id)
        if binding is None:
            return None
        self._open()
        if (
            engine.robot is not self._robot
            or request.call.semantic_id != binding.call_id
        ):
            raise ValueError(
                "Articulation lifecycle differs from its actual engine/call binding."
            )
        return self

    def on_plan_installed(self, attempt: ExecutionPlanAttempt) -> None:
        """Append facts only at the actual successful/failed installation boundary."""
        self._open()
        if type(attempt) is not ExecutionPlanAttempt:
            raise TypeError(
                "Articulation installation requires the real owned attempt snapshot."
            )
        request = attempt.request
        if (
            attempt.plan.invocation_id != request.invocation_id
            or attempt.plan.invocation_revision != request.revision
            or attempt.plan.commands.env_ids.tolist() != [0]
        ):
            raise ValueError(
                "Installed articulation plan differs from its actual request or native row."
            )
        binding = self._bindings.get(request.invocation_id)
        if binding is None or attempt.planned_mask.shape != (1,):
            raise ValueError(
                "Articulation installation is not declared in this B1 scope."
            )
        fact = capture_installed_articulation_plan(
            self._context(binding),
            attempt.plan.planning_recovery_outcome,
            invocation_id=request.invocation_id,
            invocation_revision=attempt.plan.invocation_revision,
            skill_id=request.skill_id,
            planned=bool(attempt.planned_mask.any()),
        )
        self._facts[request.invocation_id] = (
            *self._facts.get(request.invocation_id, ()),
            fact,
        )
        self._version += 1

    def on_execution_aborted(self, reason: str) -> None:
        """Invalidate future continuation after the existing abort handshake."""
        del reason
        self.close()

    def close(self) -> None:
        """End this logical scope while retaining immutable installation history."""
        self._closed = True
