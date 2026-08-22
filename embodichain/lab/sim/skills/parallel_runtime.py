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

"""Branch-local semantic execution joined by one deterministic barrier."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from embodichain.lab.sim.atomic_actions import (
    CommandAcknowledgement,
    CommandSink,
    ExecutionClock,
    PlanningContext,
    RuntimeCommandFrame,
    RuntimeEndpointTarget,
    StateDelta,
    TaskState,
    TimedCommandSequence,
)

from .calls import SemanticCallSpec
from .compiler import SemanticSkillCompiler
from .effects import SymbolicStateKey
from .integration import (
    PathPart,
    SemanticDiagnostic,
    SemanticValidationError,
)
from .parallel import (
    ParallelBranchPlan,
    ParallelTimingPolicy,
    align_parallel_commands,
    merge_parallel_effects,
    resolve_parallel_barrier,
)
from .profiles import ResourceClaim
from .runtime import SkillResult, SkillRuntime, SkillStatus, task_state_to_metadata


def _validate_identifier(value: str, *, field_name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty stable identifier.")


def _snapshot_target(target: RuntimeEndpointTarget) -> RuntimeEndpointTarget:
    snapshot = target.snapshot()
    if type(snapshot) is not type(target) or snapshot is target:
        raise TypeError("Runtime target snapshots must be independent exact values.")
    return snapshot


def _target_fingerprint(target: RuntimeEndpointTarget) -> Hashable:
    """Return one validated target address and safe-hold fingerprint."""
    fingerprint = target.address_fingerprint
    try:
        hash(fingerprint)
    except TypeError as exc:
        raise TypeError(
            "RuntimeEndpointTarget.address_fingerprint must be hashable."
        ) from exc
    return fingerprint


@runtime_checkable
class ParallelBranchRuntime(Protocol):
    """Minimal branch-local runtime surface required by the coordinator."""

    @property
    def result(self) -> SkillResult:
        """Return the current immutable branch result."""

    def start(
        self,
        *calls: SemanticCallSpec,
        workflow_id: str,
        eligible_mask: torch.Tensor | None = None,
    ) -> SkillResult:
        """Start one branch-local semantic workflow."""

    def step(self) -> SkillResult:
        """Advance the branch by one due runtime cycle."""

    def deactivate_rows(
        self,
        env_mask: torch.Tensor,
        *,
        reason: str,
    ) -> SkillResult:
        """Remove peer-failed rows while other rows continue."""

    def cancel(self, reason: str) -> SkillResult:
        """Cancel the complete branch and apply its safe stop."""


@runtime_checkable
class ParallelCommandSafetyValidator(Protocol):
    """Fail-closed physical-safety boundary for one merged command tick.

    Resource claims prevent controller arbitration conflicts but cannot prove
    that independently generated robot motions are collision-free when
    executed together.  Environment integrations must install a validator
    backed by their authoritative robot/collision model before parallel
    commands can leave the coordinator.
    """

    def validate(
        self,
        *,
        branch_frames: Mapping[str, RuntimeCommandFrame],
        merged_frame: RuntimeCommandFrame,
    ) -> None:
        """Raise when the synchronized command is not physically safe."""


class ParallelSafetyError(RuntimeError):
    """Raised when physical parallel-command safety cannot be established."""


class ParallelLaneCommandSink:
    """Acknowledge one branch locally and expose its frame to a coordinator.

    The coordinator is the only object allowed to forward commands to the real
    transport. A lane retains its last frame so shorter or temporarily waiting
    branches use deterministic hold-last padding.
    """

    def __init__(self) -> None:
        self._fresh_frame: RuntimeCommandFrame | None = None
        self._last_frame: RuntimeCommandFrame | None = None
        self._hold_requests: list[
            tuple[tuple[RuntimeEndpointTarget, ...], PlanningContext]
        ] = []
        self._cancel_targets: tuple[RuntimeEndpointTarget, ...] = ()

    @property
    def last_frame(self) -> RuntimeCommandFrame | None:
        """Return an owned hold-last frame, if this lane has sent one."""
        return None if self._last_frame is None else self._last_frame.snapshot()

    @property
    def hold_request(
        self,
    ) -> tuple[tuple[RuntimeEndpointTarget, ...], PlanningContext | None]:
        """Return all pending targets and their latest planning context."""
        targets: dict[Hashable, RuntimeEndpointTarget] = {}
        context: PlanningContext | None = None
        for requested, request_context in self._hold_requests:
            for target in requested:
                targets[_target_fingerprint(target)] = target
            context = request_context
        return (
            tuple(_snapshot_target(target) for target in targets.values()),
            context,
        )

    @property
    def cancel_targets(self) -> tuple[RuntimeEndpointTarget, ...]:
        """Return target snapshots from the most recent cancel request."""
        return tuple(_snapshot_target(target) for target in self._cancel_targets)

    def send(
        self,
        command: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Capture exactly one fresh frame for the current coordinator tick."""
        del timeout
        if not isinstance(command, RuntimeCommandFrame):
            raise TypeError("command must be a RuntimeCommandFrame.")
        if self._fresh_frame is not None:
            raise RuntimeError(
                "A parallel lane emitted multiple command frames before drain."
            )
        self._fresh_frame = command.snapshot()
        self._last_frame = command.snapshot()
        return CommandAcknowledgement.accepted_ack("buffered by parallel lane")

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Capture a target-scoped hold; hold-last remains the grid command."""
        del timeout
        if not isinstance(context, PlanningContext):
            raise TypeError("context must be a PlanningContext.")
        self._hold_requests.append(
            (
                tuple(_snapshot_target(target) for target in targets),
                context,
            )
        )
        return CommandAcknowledgement.accepted_ack("buffered parallel hold")

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Capture cancellation ownership for coordinator-level safe stop."""
        del timeout
        self._fresh_frame = None
        self._cancel_targets = tuple(_snapshot_target(target) for target in targets)
        return CommandAcknowledgement.accepted_ack("buffered parallel cancel")

    def drain_frame(self) -> RuntimeCommandFrame | None:
        """Consume the frame emitted since the previous coordinator step."""
        frame = self._fresh_frame
        self._fresh_frame = None
        return None if frame is None else frame.snapshot()

    def drain_hold_requests(
        self,
    ) -> tuple[tuple[tuple[RuntimeEndpointTarget, ...], PlanningContext], ...]:
        """Consume every completion/safe hold buffered since the last tick."""
        requests = tuple(
            (
                tuple(_snapshot_target(target) for target in targets),
                context,
            )
            for targets, context in self._hold_requests
        )
        self._hold_requests.clear()
        return requests


@dataclass(frozen=True, slots=True)
class ParallelRuntimeBranch:
    """One semantic-call lane and its exclusive resource claim."""

    branch_id: str
    calls: tuple[SemanticCallSpec, ...]
    claim: ResourceClaim
    runtime: ParallelBranchRuntime = field(repr=False, compare=False)
    command_sink: ParallelLaneCommandSink = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.branch_id, field_name="branch_id")
        calls = tuple(self.calls)
        if not calls or not all(isinstance(call, SemanticCallSpec) for call in calls):
            raise TypeError("calls must contain SemanticCallSpec values.")
        if not isinstance(self.claim, ResourceClaim):
            raise TypeError("claim must be a ResourceClaim.")
        if not isinstance(self.runtime, ParallelBranchRuntime):
            raise TypeError("runtime must implement ParallelBranchRuntime.")
        if type(self.command_sink) is not ParallelLaneCommandSink:
            raise TypeError("command_sink must be ParallelLaneCommandSink.")
        object.__setattr__(self, "calls", calls)


@dataclass(frozen=True, slots=True)
class ParallelBranchStaticAnalysis:
    """Provider-free physical and symbolic claims for one semantic lane."""

    branch_id: str
    calls: tuple[SemanticCallSpec, ...]
    claim: ResourceClaim
    symbolic_writes: frozenset[SymbolicStateKey]
    opaque_symbolic_call_indices: tuple[int, ...]
    source_path: tuple[PathPart, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.branch_id, field_name="branch_id")
        calls = tuple(self.calls)
        if not calls or not all(isinstance(call, SemanticCallSpec) for call in calls):
            raise TypeError("calls must contain SemanticCallSpec values.")
        if not isinstance(self.claim, ResourceClaim):
            raise TypeError("claim must be a ResourceClaim.")
        if type(self.symbolic_writes) is not frozenset or not all(
            type(write) is SymbolicStateKey for write in self.symbolic_writes
        ):
            raise TypeError(
                "symbolic_writes must be an exact frozenset of "
                "SymbolicStateKey values."
            )
        opaque_indices = tuple(self.opaque_symbolic_call_indices)
        if not all(
            type(index) is int and 0 <= index < len(calls) for index in opaque_indices
        ):
            raise ValueError(
                "opaque_symbolic_call_indices must select branch call indices."
            )
        if len(set(opaque_indices)) != len(opaque_indices):
            raise ValueError("opaque_symbolic_call_indices must be unique.")
        source_path = tuple(self.source_path)
        if not source_path or not all(
            (type(part) is str and bool(part)) or type(part) is int
            for part in source_path
        ):
            raise ValueError("source_path must contain valid diagnostic components.")
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "opaque_symbolic_call_indices", opaque_indices)
        object.__setattr__(self, "source_path", source_path)


def analyze_parallel_branches(
    compiler: SemanticSkillCompiler,
    branch_calls: Mapping[str, tuple[SemanticCallSpec, ...]],
    *,
    workflow_id: str = "parallel_static_analysis",
    branch_paths: Mapping[str, tuple[PathPart, ...]] | None = None,
) -> tuple[ParallelBranchStaticAnalysis, ...]:
    """Reject overlapping physical claims and exact symbolic write keys.

    This is the canonical provider-free parallel preflight shared by the core
    runtime factory and higher-level declarative frontends.  Dynamic command
    collision safety remains the responsibility of
    :class:`ParallelCommandSafetyValidator`.

    Args:
        compiler: Canonical semantic compiler owning the current integration.
        branch_calls: Ordered branch IDs and their complete semantic calls.
        workflow_id: Stable diagnostic prefix for branch workflows.
        branch_paths: Optional exact source path for every supplied branch.

    Returns:
        Ordered owned branch analyses with combined resource claims.

    Raises:
        ValueError: If fewer than two branches are supplied or claims overlap.
        SemanticValidationError: If branches write one exact symbolic key.
    """
    if not isinstance(compiler, SemanticSkillCompiler):
        raise TypeError("compiler must be a SemanticSkillCompiler.")
    if not isinstance(branch_calls, Mapping) or len(branch_calls) < 2:
        raise ValueError("branch_calls must contain at least two branches.")
    _validate_identifier(workflow_id, field_name="workflow_id")
    if branch_paths is not None:
        if not isinstance(branch_paths, Mapping):
            raise TypeError("branch_paths must be a mapping or None.")
        if set(branch_paths) != set(branch_calls):
            raise ValueError("branch_paths keys must exactly match branch_calls.")

    analyses: list[ParallelBranchStaticAnalysis] = []
    for branch_index, (branch_id, supplied_calls) in enumerate(branch_calls.items()):
        _validate_identifier(branch_id, field_name="parallel branch IDs")
        calls = tuple(supplied_calls)
        if not calls or not all(isinstance(call, SemanticCallSpec) for call in calls):
            raise TypeError(
                "parallel branch calls must contain SemanticCallSpec values."
            )
        source_path = (
            ("parallel", "branches", branch_index)
            if branch_paths is None
            else tuple(branch_paths[branch_id])
        )
        workflow = compiler.analyze(
            calls,
            workflow_id=f"{workflow_id}:{branch_index}:{branch_id}",
            path=source_path,
        )
        analyses.append(
            ParallelBranchStaticAnalysis(
                branch_id=branch_id,
                calls=calls,
                claim=ResourceClaim.combine(
                    tuple(call.bound.binding.claim for call in workflow.calls)
                ),
                symbolic_writes=frozenset(
                    write
                    for analyzed_call in workflow.calls
                    for write in analyzed_call.symbolic_writes
                ),
                opaque_symbolic_call_indices=tuple(
                    analyzed_call.index
                    for analyzed_call in workflow.calls
                    if analyzed_call.opaque_symbolic_effect
                ),
                source_path=source_path,
            )
        )
    for index, left in enumerate(analyses):
        for right in analyses[index + 1 :]:
            if left.claim.conflicts_with(right.claim):
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        "parallel_resource_conflict",
                        right.source_path,
                        f"Parallel branches {left.branch_id!r} and "
                        f"{right.branch_id!r} have overlapping resource claims.",
                        (left.branch_id, right.branch_id),
                    )
                )
            shared_writes = left.symbolic_writes & right.symbolic_writes
            if shared_writes:
                conflict = min(
                    shared_writes,
                    key=lambda write: (write.domain.value, write.address),
                )
                raise SemanticValidationError(
                    SemanticDiagnostic(
                        "parallel_symbolic_write_conflict",
                        right.source_path,
                        f"Parallel branches {left.branch_id!r} and "
                        f"{right.branch_id!r} both write symbolic TaskState key "
                        f"{conflict.rendered}.",
                        (left.branch_id, right.branch_id),
                    )
                )
    return tuple(analyses)


@dataclass(frozen=True, slots=True, eq=False)
class ParallelSkillResult:
    """Owned coordinator status at one explicit barrier."""

    status: SkillStatus
    env_ids: torch.Tensor
    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    cancelled_mask: torch.Tensor
    pending_mask: torch.Tensor
    task_state: TaskState
    branch_results: Mapping[str, SkillResult]
    elapsed_steps: int
    command_count: int
    wait_duration: float = 0.0
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, SkillStatus):
            raise TypeError("status must be a SkillStatus.")
        if (
            not isinstance(self.env_ids, torch.Tensor)
            or self.env_ids.dtype != torch.long
            or self.env_ids.dim() != 1
        ):
            raise ValueError("env_ids must be a one-dimensional int64 tensor.")
        batch_size = int(self.env_ids.numel())
        for field_name in (
            "success_mask",
            "failure_mask",
            "cancelled_mask",
            "pending_mask",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.bool
                or value.shape != (batch_size,)
                or value.device != self.env_ids.device
            ):
                raise ValueError(f"{field_name} must match env_ids.")
        if (
            (self.success_mask & (self.failure_mask | self.cancelled_mask)).any()
            or (self.failure_mask & self.cancelled_mask).any()
            or (
                self.pending_mask
                & (self.success_mask | self.failure_mask | self.cancelled_mask)
            ).any()
        ):
            raise ValueError("parallel result masks must be disjoint.")
        if not isinstance(self.task_state, TaskState):
            raise TypeError("task_state must be a TaskState.")
        if (
            self.task_state.batch_size != batch_size
            or self.task_state.device != self.env_ids.device
        ):
            raise ValueError("task_state must match env_ids.")
        if type(self.elapsed_steps) is not int or self.elapsed_steps < 0:
            raise ValueError("elapsed_steps must be non-negative.")
        if type(self.command_count) is not int or self.command_count < 0:
            raise ValueError("command_count must be non-negative.")
        if not math.isfinite(self.wait_duration) or self.wait_duration < 0.0:
            raise ValueError("wait_duration must be finite and non-negative.")
        if self.message is not None and type(self.message) is not str:
            raise TypeError("message must be a string or None.")
        branches: dict[str, SkillResult] = {}
        for branch_id, result in self.branch_results.items():
            _validate_identifier(branch_id, field_name="branch result IDs")
            if not isinstance(result, SkillResult):
                raise TypeError("branch_results values must be SkillResult values.")
            branches[branch_id] = result
        object.__setattr__(self, "env_ids", self.env_ids.clone())
        for field_name in (
            "success_mask",
            "failure_mask",
            "cancelled_mask",
            "pending_mask",
        ):
            object.__setattr__(self, field_name, getattr(self, field_name).clone())
        object.__setattr__(
            self,
            "task_state",
            TaskState(
                batch_size=self.task_state.batch_size,
                device=self.task_state.device,
                held_objects=self.task_state.held_objects,
                coordinated_held_objects=self.task_state.coordinated_held_objects,
                articulation_joints=self.task_state.articulation_joints,
            ),
        )
        object.__setattr__(self, "branch_results", MappingProxyType(branches))

    @property
    def terminal(self) -> bool:
        """Whether every row has left the barrier."""
        return self.status in {
            SkillStatus.COMPLETED,
            SkillStatus.FAILED,
            SkillStatus.CANCELLED,
        }

    def to_metadata(self) -> dict[str, object]:
        """Return a fresh deterministic JSON-safe parallel barrier result."""
        return {
            "schema_version": 1,
            "kind": "parallel_skill_result",
            "status": self.status.value,
            "env_ids": self.env_ids.detach().cpu().tolist(),
            "masks": {
                "success": self.success_mask.detach().cpu().tolist(),
                "failure": self.failure_mask.detach().cpu().tolist(),
                "cancelled": self.cancelled_mask.detach().cpu().tolist(),
                "pending": self.pending_mask.detach().cpu().tolist(),
            },
            "task_state": task_state_to_metadata(self.task_state),
            "branches": {
                branch_id: result.to_metadata()
                for branch_id, result in sorted(self.branch_results.items())
            },
            "elapsed_steps": self.elapsed_steps,
            "command_count": self.command_count,
            "wait_duration": self.wait_duration,
            "message": self.message,
        }


def _optional_tensor_equal(
    left: torch.Tensor | None, right: torch.Tensor | None
) -> bool:
    return (left is None and right is None) or (
        left is not None and right is not None and torch.equal(left, right)
    )


def _state_value_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if left is None or right is None:
        return left is right
    if hasattr(left, "position"):
        return torch.equal(left.position, right.position) and _optional_tensor_equal(
            left.env_mask,
            right.env_mask,
        )
    if hasattr(left, "left_object_to_eef"):
        return (
            left.semantics.entity_id == right.semantics.entity_id
            and torch.equal(left.left_object_to_eef, right.left_object_to_eef)
            and torch.equal(left.right_object_to_eef, right.right_object_to_eef)
            and torch.equal(left.left_grasp_xpos, right.left_grasp_xpos)
            and torch.equal(left.right_grasp_xpos, right.right_grasp_xpos)
            and _optional_tensor_equal(left.env_mask, right.env_mask)
        )
    return (
        left.semantics.entity_id == right.semantics.entity_id
        and torch.equal(left.object_to_eef, right.object_to_eef)
        and torch.equal(left.grasp_xpos, right.grasp_xpos)
        and _optional_tensor_equal(left.env_mask, right.env_mask)
    )


def _mapping_delta(
    before: Mapping[object, object], after: Mapping[object, object]
) -> dict:
    updates: dict[object, object | None] = {}
    for key in set(before) | set(after):
        if key not in after:
            updates[key] = None
        elif key not in before or not _state_value_equal(before[key], after[key]):
            updates[key] = after[key]
    return updates


def _task_state_delta(before: TaskState, after: TaskState) -> StateDelta:
    if before.batch_size != after.batch_size or before.device != after.device:
        raise ValueError("Parallel branch TaskState changed batch or device.")
    return StateDelta(
        held_object_updates=_mapping_delta(
            before.held_objects,
            after.held_objects,
        ),
        coordinated_held_object_updates=_mapping_delta(
            before.coordinated_held_objects,
            after.coordinated_held_objects,
        ),
        articulation_joint_updates=_mapping_delta(
            before.articulation_joints,
            after.articulation_joints,
        ),
    )


class ParallelSkillRuntime:
    """Run independent JIT semantic lanes on one synchronized command grid.

    Schema v2 deliberately uses conservative barrier ownership: branches are
    not assigned disjoint environment-row partitions, so two branches that
    write the same symbolic key conflict for the complete started batch even
    when their observed value masks happen to be disjoint.  A future schema
    may add explicit row partitioning before relaxing this invariant.

    A lane completion hold is forwarded as an explicit grid action.  Other
    lanes therefore receive deterministic hold-padding for that environment
    step; a merged frame generated in the same coordinator cycle is retained
    and dispatched only after the clock advances.  Branch runners are not
    stepped while that retained frame is being dispatched.  This keeps the
    physical order ``observed hold -> next command`` and limits every normal
    coordinator step to one action-producing transport operation.
    """

    def __init__(
        self,
        branches: tuple[ParallelRuntimeBranch, ...],
        command_sink: CommandSink,
        clock: ExecutionClock,
        timing_policy: ParallelTimingPolicy,
        safety_validator: ParallelCommandSafetyValidator,
        *,
        timeout_steps: int,
        failure_policy: str = "fail_fast",
    ) -> None:
        if not isinstance(branches, tuple) or len(branches) < 2:
            raise ValueError("ParallelSkillRuntime requires at least two branches.")
        if not all(type(branch) is ParallelRuntimeBranch for branch in branches):
            raise TypeError("branches must contain ParallelRuntimeBranch values.")
        branch_ids = tuple(branch.branch_id for branch in branches)
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("Parallel branch IDs must be unique.")
        for index, left in enumerate(branches):
            for right in branches[index + 1 :]:
                if left.claim.conflicts_with(right.claim):
                    raise ValueError(
                        f"Parallel branches {left.branch_id!r} and "
                        f"{right.branch_id!r} have overlapping resource claims."
                    )
        if not isinstance(command_sink, CommandSink):
            raise TypeError("command_sink must implement CommandSink.")
        if not isinstance(clock, ExecutionClock):
            raise TypeError("clock must implement ExecutionClock.")
        if not isinstance(timing_policy, ParallelTimingPolicy):
            raise TypeError("timing_policy must be ParallelTimingPolicy.")
        if not isinstance(safety_validator, ParallelCommandSafetyValidator):
            raise TypeError(
                "safety_validator must implement ParallelCommandSafetyValidator; "
                "resource claims alone do not establish collision safety."
            )
        if type(timeout_steps) is not int or timeout_steps <= 0:
            raise ValueError("timeout_steps must be positive.")
        if failure_policy != "fail_fast":
            raise ValueError("failure_policy must be exactly 'fail_fast'.")
        initial = branches[0].runtime.result
        for branch in branches[1:]:
            result = branch.runtime.result
            if (
                result.env_ids.device != initial.env_ids.device
                or not torch.equal(result.env_ids, initial.env_ids)
                or result.task_state.batch_size != initial.task_state.batch_size
                or result.task_state.device != initial.task_state.device
            ):
                raise ValueError(
                    "Parallel branch runtimes must share env_ids, batch, and device."
                )
            if not _task_state_delta(initial.task_state, result.task_state).is_empty:
                raise ValueError(
                    "Parallel branch runtimes must start from the same verified "
                    "TaskState barrier snapshot."
                )
        self._branches = branches
        self._command_sink = command_sink
        self._clock = clock
        self._timing_policy = timing_policy
        self._safety_validator = safety_validator
        self._timeout_steps = timeout_steps
        self._initial_state = initial.task_state
        self._task_state = initial.task_state
        self._env_ids = initial.env_ids
        self._status = SkillStatus.IDLE
        self._success = torch.zeros_like(initial.success_mask)
        self._failure = torch.zeros_like(initial.failure_mask)
        self._cancelled = torch.zeros_like(initial.cancelled_mask)
        self._pending = torch.ones_like(initial.success_mask)
        self._started_eligible = torch.zeros_like(initial.success_mask)
        self._elapsed_steps = 0
        self._start_timestamp: float | None = None
        self._command_count = 0
        self._wait_duration = 0.0
        self._message: str | None = None
        self._force_mask_dispatch = False
        self._terminal_stop_forwarded = False
        self._held_target_fingerprints: set[Hashable] = set()
        self._last_hold_context: PlanningContext | None = None
        self._deferred_frame: RuntimeCommandFrame | None = None
        self._deferred_lane_frames: dict[str, RuntimeCommandFrame] = {}
        self._terminal_hold_pending = False
        self._next_transport_at: float | None = None

    @classmethod
    def from_template(
        cls,
        template_runtime: SkillRuntime,
        branch_calls: Mapping[str, tuple[SemanticCallSpec, ...]],
        command_sink: CommandSink,
        timing_policy: ParallelTimingPolicy,
        safety_validator: ParallelCommandSafetyValidator,
        *,
        timeout_steps: int,
        failure_policy: str = "fail_fast",
        workflow_id: str = "parallel_static_analysis",
        branch_paths: Mapping[str, tuple[PathPart, ...]] | None = None,
    ) -> ParallelSkillRuntime:
        """Analyze claims and derive independent lanes from one runtime.

        This factory deliberately accepts semantic calls instead of compiled
        Gym-program types.  It keeps the simulation runtime independent of the
        higher-level configuration package while giving every frontend one
        canonical resource-conflict and lane-construction path.

        Args:
            template_runtime: Idle runtime providing shared compiler and ports.
            branch_calls: Ordered branch ID to semantic-call sequence mapping.
            command_sink: The sole outbound merged command sink.
            timing_policy: Exact shared environment grid.
            safety_validator: Required physical/collision safety gate for each
                synchronized outbound command.
            timeout_steps: Maximum environment steps at the barrier.
            failure_policy: Row-local barrier failure policy.
            workflow_id: Stable prefix for provider-free claim analysis.
            branch_paths: Optional exact source path for every branch.

        Returns:
            A one-shot parallel runtime whose branches share no mutable runner
            state.
        """
        if not isinstance(template_runtime, SkillRuntime):
            raise TypeError("template_runtime must be a SkillRuntime.")
        if template_runtime.status is SkillStatus.RUNNING:
            raise RuntimeError("template_runtime must not be running.")
        branches: list[ParallelRuntimeBranch] = []
        for analysis in analyze_parallel_branches(
            template_runtime.compiler,
            branch_calls,
            workflow_id=workflow_id,
            branch_paths=branch_paths,
        ):
            lane_sink = ParallelLaneCommandSink()
            lane_runtime = template_runtime.fork(
                lane_sink,
                task_state=template_runtime.task_state,
            )
            branches.append(
                ParallelRuntimeBranch(
                    branch_id=analysis.branch_id,
                    calls=analysis.calls,
                    claim=analysis.claim,
                    runtime=lane_runtime,
                    command_sink=lane_sink,
                )
            )
        return cls(
            tuple(branches),
            command_sink,
            template_runtime.clock,
            timing_policy,
            safety_validator,
            timeout_steps=timeout_steps,
            failure_policy=failure_policy,
        )

    @property
    def result(self) -> ParallelSkillResult:
        """Return an owned barrier snapshot."""
        return ParallelSkillResult(
            status=self._status,
            env_ids=self._env_ids,
            success_mask=self._success,
            failure_mask=self._failure,
            cancelled_mask=self._cancelled,
            pending_mask=self._pending,
            task_state=self._task_state,
            branch_results={
                branch.branch_id: branch.runtime.result for branch in self._branches
            },
            elapsed_steps=self._elapsed_steps,
            command_count=self._command_count,
            wait_duration=self._wait_duration,
            message=self._message,
        )

    @property
    def clock(self) -> ExecutionClock:
        """Return the exact clock shared by the coordinator and every lane."""
        return self._clock

    @property
    def branch_claims(self) -> Mapping[str, ResourceClaim]:
        """Return immutable statically analyzed claims in branch order."""
        return MappingProxyType(
            {branch.branch_id: branch.claim for branch in self._branches}
        )

    def start(
        self,
        *,
        workflow_id: str = "parallel_workflow",
        eligible_mask: torch.Tensor | None = None,
    ) -> ParallelSkillResult:
        """Start all lanes from the same verified barrier state."""
        if self._status is not SkillStatus.IDLE:
            raise RuntimeError("ParallelSkillRuntime instances are one-shot.")
        _validate_identifier(workflow_id, field_name="workflow_id")
        if eligible_mask is None:
            eligible = torch.ones_like(self._pending)
        else:
            if (
                not isinstance(eligible_mask, torch.Tensor)
                or eligible_mask.dtype != torch.bool
                or eligible_mask.shape != self._pending.shape
                or eligible_mask.device != self._pending.device
            ):
                raise ValueError("eligible_mask must match the parallel batch.")
            eligible = eligible_mask.clone()
        if not eligible.any():
            raise ValueError("eligible_mask must contain an active row.")
        self._success.zero_()
        self._failure.zero_()
        self._cancelled.zero_()
        self._pending = eligible.clone()
        self._started_eligible = eligible.clone()
        self._elapsed_steps = 0
        self._start_timestamp = self._read_clock()
        self._command_count = 0
        self._wait_duration = 0.0
        self._message = None
        self._force_mask_dispatch = False
        self._terminal_stop_forwarded = False
        self._held_target_fingerprints.clear()
        self._last_hold_context = None
        self._deferred_frame = None
        self._deferred_lane_frames.clear()
        self._terminal_hold_pending = False
        self._next_transport_at = None
        self._status = SkillStatus.RUNNING
        started: list[ParallelRuntimeBranch] = []
        try:
            for branch in self._branches:
                branch.runtime.start(
                    *branch.calls,
                    workflow_id=f"{workflow_id}:{branch.branch_id}",
                    eligible_mask=eligible,
                )
                started.append(branch)
        except Exception as exc:
            reason = "Parallel branch startup failed: " f"{type(exc).__name__}: {exc}"
            for branch in started:
                branch.runtime.cancel(reason)
            self._failure = eligible.clone()
            self._pending.zero_()
            self._status = SkillStatus.FAILED
            self._message = reason
            return self.result
        try:
            self._sync_branch_identity()
            self._update_barrier()
            self._finish_if_complete()
        except Exception as exc:
            self._abort_coordinator("Parallel startup coordination failed", exc)
        return self.result

    def step(self) -> ParallelSkillResult:
        """Advance one deterministic coordinator state-machine transition."""
        if self._status is not SkillStatus.RUNNING:
            return self.result
        try:
            self._update_elapsed_steps()
            if self._elapsed_steps >= self._timeout_steps and (
                self._pending.any() or self._transport_flush_pending
            ):
                self._timeout_pending_rows()
                self._finish_if_complete()
                return self.result
            transport_wait = self._remaining_transport_wait()
            if transport_wait > 0.0:
                self._wait_duration = transport_wait
                return self.result
            if self._deferred_frame is not None:
                accepted = self._dispatch_deferred_frame()
                if (
                    accepted
                    and not self._pending.any()
                    and self._status is SkillStatus.RUNNING
                ):
                    self._terminal_hold_pending = True
                self._finish_if_complete()
                return self.result
            if self._terminal_hold_pending:
                self._terminal_hold_pending = False
                self._dispatch_requested_hold(
                    required=True,
                    include_last_targets=True,
                )
                self._finish_if_complete()
                return self.result
            for branch in self._branches:
                if not branch.runtime.result.terminal:
                    branch.runtime.step()
            self._update_barrier()
            self._dispatch_grid_frame()
            self._finish_if_complete()
        except Exception as exc:
            self._abort_coordinator("Parallel coordinator step failed", exc)
        return self.result

    def _timeout_pending_rows(self) -> None:
        """Fail and safe-stop deadline-expired rows before another command."""
        timed_out = self._pending.clone()
        if not timed_out.any() and self._transport_flush_pending:
            timed_out = self._started_eligible.clone()
        if not timed_out.any():
            return
        self._failure |= timed_out
        self._success &= ~timed_out
        self._pending &= ~timed_out
        self._deferred_frame = None
        self._deferred_lane_frames.clear()
        self._terminal_hold_pending = False
        self._next_transport_at = None
        self._message = f"Parallel barrier timed out after {self._timeout_steps} steps."
        errors: list[str] = []
        for branch in self._branches:
            if branch.runtime.result.terminal:
                continue
            try:
                branch.runtime.cancel(self._message)
            except Exception as exc:
                errors.append(f"{branch.branch_id}: {type(exc).__name__}: {exc}")
        stopped, stop_message = self._forward_safe_stop()
        self._terminal_stop_forwarded = True
        if not stopped and stop_message is not None:
            errors.append(stop_message)
        if errors:
            self._message += " Safe stop errors: " + "; ".join(errors)

    def _read_clock(self) -> float:
        """Read one finite non-negative timestamp from the shared clock."""
        now = float(self._clock.now())
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("ExecutionClock.now() must be finite and non-negative.")
        return now

    def _update_elapsed_steps(self) -> None:
        """Measure completed environment-grid intervals since start."""
        assert self._start_timestamp is not None
        now = self._read_clock()
        elapsed = now - self._start_timestamp
        if elapsed < -self._timing_policy.tolerance:
            raise RuntimeError("Parallel execution clock moved backwards.")
        ratio = max(0.0, elapsed) / self._timing_policy.step_dt
        tolerance = self._timing_policy.tolerance / self._timing_policy.step_dt
        self._elapsed_steps = max(
            self._elapsed_steps,
            int(math.floor(ratio + tolerance)),
        )

    def _sync_branch_identity(self) -> None:
        """Adopt and verify env IDs after every lane's first observation."""
        reference = self._branches[0].runtime.result
        for branch in self._branches[1:]:
            result = branch.runtime.result
            if (
                result.env_ids.device != reference.env_ids.device
                or not torch.equal(result.env_ids, reference.env_ids)
                or result.task_state.batch_size != reference.task_state.batch_size
                or result.task_state.device != reference.task_state.device
            ):
                raise ValueError(
                    "Parallel branch observations must share env_ids, batch, "
                    "and device."
                )
        self._env_ids = reference.env_ids.clone()

    def cancel(
        self,
        reason: str = "Parallel workflow cancelled by caller.",
    ) -> ParallelSkillResult:
        """Cancel every lane and forward one target-scoped transport cancel."""
        if type(reason) is not str or not reason:
            raise ValueError("reason must be a non-empty string.")
        if self._status is not SkillStatus.RUNNING:
            return self.result
        had_transport_flush = self._transport_flush_pending
        cancelled = self._pending.clone()
        if not cancelled.any() and had_transport_flush:
            cancelled = self._started_eligible.clone()
        self._deferred_frame = None
        self._deferred_lane_frames.clear()
        self._terminal_hold_pending = False
        self._next_transport_at = None
        errors: list[str] = []
        for branch in self._branches:
            try:
                branch.runtime.cancel(reason)
            except Exception as exc:
                errors.append(f"{branch.branch_id}: {type(exc).__name__}: {exc}")
        stopped, stop_message = self._forward_safe_stop()
        self._terminal_stop_forwarded = True
        if stop_message is not None:
            errors.append(stop_message)
        self._pending &= ~cancelled
        self._success &= ~cancelled
        merge_succeeded = self._merge_verified_state()
        if errors or not stopped or not merge_succeeded:
            self._failure |= cancelled
            self._cancelled &= ~cancelled
            self._status = SkillStatus.FAILED
            if errors or not stopped:
                stop_detail = "; ".join(errors) or "unknown safe-stop failure"
                self._message = reason + " Safe stop failed: " + stop_detail
            elif self._message is None:
                self._message = reason + " Verified-state merge failed."
        else:
            self._cancelled |= cancelled
            self._status = SkillStatus.CANCELLED
            self._message = reason
        self._wait_duration = 0.0
        return self.result

    @property
    def _transport_flush_pending(self) -> bool:
        """Whether a retained command or mandatory final hold is outstanding."""
        return self._deferred_frame is not None or self._terminal_hold_pending

    def _remaining_transport_wait(self) -> float:
        """Return time until another normal grid action may be forwarded."""
        ready_at = self._next_transport_at
        if ready_at is None:
            return 0.0
        remaining = ready_at - self._read_clock()
        if remaining <= self._timing_policy.tolerance:
            self._next_transport_at = None
            return 0.0
        return remaining

    def _record_transport_action(self) -> None:
        """Arm the next physical grid boundary after one accepted action."""
        self._next_transport_at = self._read_clock() + self._timing_policy.step_dt
        self._wait_duration = self._timing_policy.step_dt

    def _update_barrier(self) -> None:
        results = {branch.branch_id: branch.runtime.result for branch in self._branches}
        pending = {
            branch_id: (
                result.eligible_mask
                & ~result.success_mask
                & ~result.failure_mask
                & ~result.cancelled_mask
            )
            for branch_id, result in results.items()
        }
        update = resolve_parallel_barrier(
            pending_masks=pending,
            success_masks={
                branch_id: result.success_mask for branch_id, result in results.items()
            },
            failure_masks={
                branch_id: result.failure_mask | result.cancelled_mask
                for branch_id, result in results.items()
            },
        )
        new_failure = update.failure_mask & ~self._failure
        self._failure |= update.failure_mask
        self._success |= update.completed_mask & ~update.failure_mask
        self._pending &= ~update.completed_mask
        if new_failure.any():
            self._force_mask_dispatch = True
            reason = "A peer parallel branch failed for these environment rows."
            for branch in self._branches:
                mask = update.cancellation_masks[branch.branch_id]
                if mask.any():
                    branch.runtime.deactivate_rows(mask, reason=reason)
        running = tuple(result for result in results.values() if not result.terminal)
        if not running or any(result.wait_duration <= 0.0 for result in running):
            self._wait_duration = 0.0
        else:
            self._wait_duration = min(result.wait_duration for result in running)

    def _dispatch_grid_frame(self) -> None:
        fresh: dict[str, RuntimeCommandFrame] = {}
        for branch in self._branches:
            frame = branch.command_sink.drain_frame()
            if frame is not None:
                if branch.runtime.result.terminal:
                    raise ParallelSafetyError(
                        f"Parallel branch {branch.branch_id!r} became terminal "
                        "while emitting a fresh command frame. A post-command "
                        "observation is required before a safe terminal hold."
                    )
                fresh[branch.branch_id] = frame
        force_mask_dispatch = self._force_mask_dispatch
        self._force_mask_dispatch = False
        if not fresh and not force_mask_dispatch:
            self._dispatch_requested_hold()
            return
        plans: list[ParallelBranchPlan] = []
        lane_frames: dict[str, RuntimeCommandFrame] = {}
        requested_holds = {
            _target_fingerprint(target)
            for branch in self._branches
            for target in branch.command_sink.hold_request[0]
        }
        for branch in self._branches:
            frame = fresh.get(branch.branch_id)
            is_fresh = frame is not None
            if frame is None:
                frame = branch.command_sink.last_frame
            if frame is None:
                continue
            if not is_fresh:
                commands = tuple(
                    command
                    for command in frame.commands
                    if _target_fingerprint(command.target)
                    not in self._held_target_fingerprints | requested_holds
                )
                if not commands:
                    continue
                frame = RuntimeCommandFrame(
                    commands=commands,
                    active_mask=frame.active_mask,
                    env_ids=frame.env_ids,
                    hold_duration=frame.hold_duration,
                )
            frame = frame.with_active_mask(frame.active_mask & ~self._failure)
            lane_frames[branch.branch_id] = frame.snapshot()
            plans.append(
                ParallelBranchPlan(
                    branch_id=branch.branch_id,
                    claim=branch.claim,
                    commands=TimedCommandSequence(
                        frames=(frame,),
                        env_ids=frame.env_ids,
                    ),
                )
            )
        if not plans:
            self._dispatch_requested_hold()
            return
        if len(plans) == 1:
            frame = plans[0].commands.frames[0]
            durations = frame.hold_duration
            expected = torch.full_like(
                durations,
                self._timing_policy.step_dt,
            )
            if not torch.allclose(
                durations,
                expected,
                atol=self._timing_policy.tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    "Parallel command frames must equal the environment step grid."
                )
            merged = plans[0].commands
        else:
            merged = align_parallel_commands(tuple(plans), self._timing_policy)
        frame = merged.frames[0]
        if not frame.active_mask.any():
            self._dispatch_requested_hold(extra_targets=frame.targets)
            return

        if self._has_unforwarded_hold_targets():
            self._deferred_frame = frame.snapshot()
            self._deferred_lane_frames = {
                branch_id: branch_frame.snapshot()
                for branch_id, branch_frame in lane_frames.items()
            }
            if not self._dispatch_requested_hold():
                self._deferred_frame = None
                self._deferred_lane_frames.clear()
            return

        # Drain duplicate requests to refresh the latest synchronized context
        # without producing another action, then send exactly one grid frame.
        self._dispatch_requested_hold()
        accepted = self._send_merged_frame(frame, lane_frames)
        if accepted and not self._pending.any() and self._status is SkillStatus.RUNNING:
            self._terminal_hold_pending = True

    def _dispatch_deferred_frame(self) -> bool:
        """Send a frame retained behind one explicit hold-padding step."""
        frame = self._deferred_frame
        if frame is None:
            raise RuntimeError("No deferred parallel frame is available.")
        lane_frames = {
            branch_id: branch_frame.snapshot()
            for branch_id, branch_frame in self._deferred_lane_frames.items()
        }
        self._deferred_frame = None
        self._deferred_lane_frames.clear()
        return self._send_merged_frame(frame, lane_frames)

    def _send_merged_frame(
        self,
        frame: RuntimeCommandFrame,
        lane_frames: Mapping[str, RuntimeCommandFrame],
    ) -> bool:
        """Validate and forward one active synchronized command frame."""
        try:
            safety_result = self._safety_validator.validate(
                branch_frames=MappingProxyType(dict(lane_frames)),
                merged_frame=frame.snapshot(),
            )
        except ParallelSafetyError:
            raise
        except Exception as exc:
            raise ParallelSafetyError(
                "Parallel command safety validation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if safety_result is not None:
            raise ParallelSafetyError(
                "ParallelCommandSafetyValidator.validate() must return None."
            )
        acknowledgement = self._command_sink.send(frame, timeout=1.0)
        if not isinstance(acknowledgement, CommandAcknowledgement):
            raise TypeError("CommandSink.send() returned an invalid value.")
        if not acknowledgement.accepted:
            self._fail_transport(acknowledgement.message)
            return False
        self._command_count += 1
        self._record_transport_action()
        self._held_target_fingerprints.difference_update(
            _target_fingerprint(target) for target in frame.targets
        )
        return True

    def _has_unforwarded_hold_targets(self) -> bool:
        """Whether lane requests contain a target not already physically held."""
        for branch in self._branches:
            targets, _ = branch.command_sink.hold_request
            if any(
                _target_fingerprint(target) not in self._held_target_fingerprints
                for target in targets
            ):
                return True
        return False

    def _dispatch_requested_hold(
        self,
        *,
        extra_targets: tuple[RuntimeEndpointTarget, ...] = (),
        include_last_targets: bool = False,
        required: bool = False,
    ) -> bool:
        """Forward every lane hold without dropping earlier call targets."""
        targets: dict[Hashable, RuntimeEndpointTarget] = {
            _target_fingerprint(target): target for target in extra_targets
        }
        context: PlanningContext | None = None
        for branch in self._branches:
            for (
                branch_targets,
                branch_context,
            ) in branch.command_sink.drain_hold_requests():
                for target in branch_targets:
                    targets[_target_fingerprint(target)] = target
                context = branch_context
                self._last_hold_context = branch_context
            if include_last_targets:
                last_frame = branch.command_sink.last_frame
                if last_frame is not None:
                    for target in last_frame.targets:
                        targets[_target_fingerprint(target)] = target
        targets = {
            key: target
            for key, target in targets.items()
            if key not in self._held_target_fingerprints
        }
        if not targets:
            return True
        if context is None:
            context = self._last_hold_context
        if context is None:
            message = "Parallel hold targets have no synchronized planning context."
            if required or targets:
                self._fail_transport(message)
            return False
        acknowledgement = self._command_sink.hold(
            tuple(targets.values()),
            context,
            timeout=1.0,
        )
        if not isinstance(acknowledgement, CommandAcknowledgement):
            raise TypeError("CommandSink.hold() returned an invalid value.")
        if not acknowledgement.accepted:
            self._fail_transport(acknowledgement.message)
            return False
        self._held_target_fingerprints.update(targets)
        self._last_hold_context = context
        self._record_transport_action()
        return True

    def _fail_transport(self, message: str) -> None:
        self._deferred_frame = None
        self._deferred_lane_frames.clear()
        self._terminal_hold_pending = False
        self._next_transport_at = None
        failed = self._pending.clone()
        if not failed.any():
            failed = self._started_eligible.clone()
        self._failure |= failed
        self._success &= ~failed
        self._pending &= ~failed
        self._message = "Parallel command transport rejected the merged operation."
        if message:
            self._message += f" {message}"
        for branch in self._branches:
            branch.runtime.cancel(self._message)
        self._forward_safe_stop()
        self._terminal_stop_forwarded = True

    def _forward_safe_stop(self) -> tuple[bool, str | None]:
        """Forward lane-owned cancellation and hold once to the real sink."""
        targets: dict[Hashable, RuntimeEndpointTarget] = {}
        context: PlanningContext | None = self._last_hold_context
        for branch in self._branches:
            for target in branch.command_sink.cancel_targets:
                targets[_target_fingerprint(target)] = target
            branch_targets, branch_context = branch.command_sink.hold_request
            for target in branch_targets:
                targets[_target_fingerprint(target)] = target
            if branch_context is not None:
                context = branch_context
            last_frame = branch.command_sink.last_frame
            if last_frame is not None:
                for target in last_frame.targets:
                    targets[_target_fingerprint(target)] = target
        if not targets:
            return True, None
        snapshots = tuple(targets.values())
        errors: list[str] = []
        try:
            cancel_ack = self._command_sink.cancel(snapshots, timeout=1.0)
            if not isinstance(cancel_ack, CommandAcknowledgement):
                raise TypeError("CommandSink.cancel() returned an invalid value.")
            if not cancel_ack.accepted:
                errors.append(cancel_ack.message or "transport cancel was rejected")
        except Exception as exc:
            errors.append(f"cancel {type(exc).__name__}: {exc}")
        if context is None:
            errors.append("no planning context was available for final safe hold")
        else:
            try:
                hold_ack = self._command_sink.hold(
                    snapshots,
                    context,
                    timeout=1.0,
                )
                if not isinstance(hold_ack, CommandAcknowledgement):
                    raise TypeError("CommandSink.hold() returned an invalid value.")
                if not hold_ack.accepted:
                    errors.append(hold_ack.message or "transport hold was rejected")
            except Exception as exc:
                errors.append(f"hold {type(exc).__name__}: {exc}")
        if not errors:
            self._held_target_fingerprints.update(
                _target_fingerprint(target) for target in snapshots
            )
            self._last_hold_context = context
        return (not errors), (None if not errors else "; ".join(errors))

    def _abort_coordinator(self, prefix: str, exc: Exception) -> None:
        """Convert an internal tick exception into a safe terminal failure."""
        self._deferred_frame = None
        self._deferred_lane_frames.clear()
        self._terminal_hold_pending = False
        self._next_transport_at = None
        reason = f"{prefix}: {type(exc).__name__}: {exc}"
        failed = self._pending.clone()
        if not failed.any():
            failed = self._started_eligible.clone()
        self._failure |= failed
        self._success &= ~failed
        self._pending &= ~failed
        errors: list[str] = []
        for branch in self._branches:
            if branch.runtime.result.terminal:
                continue
            try:
                branch.runtime.cancel(reason)
            except Exception as cancel_exc:
                errors.append(
                    f"{branch.branch_id}: {type(cancel_exc).__name__}: {cancel_exc}"
                )
        stopped, stop_message = self._forward_safe_stop()
        self._terminal_stop_forwarded = True
        if not stopped and stop_message is not None:
            errors.append(stop_message)
        self._message = reason
        if errors:
            self._message += " Safe stop errors: " + "; ".join(errors)
        self._merge_verified_state()
        self._status = SkillStatus.FAILED
        self._wait_duration = 0.0

    def _merge_verified_state(self) -> bool:
        """Merge every branch-local verified patch at a terminal barrier."""
        effects = {
            branch.branch_id: (
                _task_state_delta(
                    self._initial_state,
                    branch.runtime.result.task_state,
                ),
                self._started_eligible,
            )
            for branch in self._branches
        }
        try:
            self._task_state = merge_parallel_effects(self._initial_state, effects)
        except Exception as exc:
            self._failure |= self._started_eligible
            self._success.zero_()
            merge_message = (
                "Parallel verified-state merge failed: " f"{type(exc).__name__}: {exc}"
            )
            self._message = (
                merge_message
                if self._message is None
                else f"{self._message} {merge_message}"
            )
            return False
        return True

    def _finish_if_complete(self) -> None:
        if self._pending.any():
            return
        if self._deferred_frame is not None or self._terminal_hold_pending:
            return
        self._merge_verified_state()
        if self._status is SkillStatus.RUNNING and not self._terminal_stop_forwarded:
            self._dispatch_requested_hold(required=True, include_last_targets=True)
        self._wait_duration = 0.0
        if self._failure.any():
            self._status = SkillStatus.FAILED
        elif self._cancelled.any():
            self._status = SkillStatus.CANCELLED
        else:
            self._status = SkillStatus.COMPLETED


__all__ = [
    "analyze_parallel_branches",
    "ParallelBranchRuntime",
    "ParallelBranchStaticAnalysis",
    "ParallelCommandSafetyValidator",
    "ParallelLaneCommandSink",
    "ParallelRuntimeBranch",
    "ParallelSkillResult",
    "ParallelSkillRuntime",
    "ParallelSafetyError",
]
