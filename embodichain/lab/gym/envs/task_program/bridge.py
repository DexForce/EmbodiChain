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

"""Gym ports and a lazy demo adapter for compiled Task Programs.

This module deliberately stops at the Gym action boundary.  It never calls
``env.step`` and never updates a simulator directly.  The demo executor owns
the environment step; when it asks the action generator for the next value,
the bridge treats the previously yielded value as consumed and advances the
environment-backed execution clock by exactly one step.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch

from embodichain.lab.gym.envs._json import json_safe_copy as _json_safe_copy
from embodichain.lab.gym.envs.demo import DemoSegment
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.sim.atomic_actions.bindings import (
    JointPositionTarget,
    RuntimeEndpointTarget,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUpOptions
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceOptions
from embodichain.lab.sim.atomic_actions.runner import (
    CommandAcknowledgement,
    ExecutionClock,
    ExecutionRunnerCfg,
)
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    RuntimeCommandPayload,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext, TaskState
from embodichain.lab.task_program.runtime.parallel import ParallelTimingPolicy
from embodichain.lab.task_program.runtime.parallel_executor import (
    ParallelCommandSafetyValidator,
    ParallelSemanticExecutionResult,
    ParallelSemanticExecutor,
)
from embodichain.lab.task_program.runtime.executor import SemanticCallExecutor
from embodichain.lab.task_program.runtime.results import (
    SemanticExecutionResult,
    SemanticExecutionStatus,
)
from embodichain.lab.task_program.semantics.calls import Pick, Place
from embodichain.lab.task_program.semantics.integration import SemanticValidationError
from embodichain.lab.task_program.semantics.profiles import EffectAssurance
from embodichain.lab.sim.types import EnvAction

_SAFE_HOLD_ACTION_KINDS = frozenset(
    {"runtime_safe_hold", "runtime_wait_hold", "runtime_abort_safe_hold"}
)


class TaskProgramBridgeError(RuntimeError):
    """Base error raised by the Task Program Gym bridge."""


class EnvironmentStepTimingError(TaskProgramBridgeError, ValueError):
    """Raised when runtime timing cannot be represented on the Gym step grid."""


class UnsupportedRuntimeTransportError(TaskProgramBridgeError, LookupError):
    """Raised when a command frame names an unregistered transport."""


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Validate and return one strict identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _validate_timeout(timeout: float) -> None:
    """Validate a runner-supplied acknowledgement timeout."""
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise TypeError("timeout must be a real number.")
    if not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("timeout must be finite and positive.")


@runtime_checkable
class CurrentQposProvider(Protocol):
    """Source of full robot positions aligned to explicit environment IDs."""

    def current_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return ``(batch_size, robot_dof)`` positions for ``env_ids``."""


@runtime_checkable
class RuntimeTransportActionEncoder(Protocol):
    """Extensible lowering boundary for one runtime transport kind.

    An encoder receives the action produced by earlier registered transports
    and returns the next owned action value.  This permits a future transport
    to promote the built-in tensor action to a ``TensorDict`` when the Gym
    action manager exposes a structured controller boundary.
    """

    transport_id: ClassVar[str]
    """Exact runtime transport ID handled by this encoder."""

    target_types: ClassVar[tuple[type[RuntimeEndpointTarget], ...]]
    """Exact runtime-target types accepted by this encoder."""

    payload_types: ClassVar[tuple[type[RuntimeCommandPayload], ...]]
    """Exact runtime-payload types accepted by this encoder."""

    def encode(
        self,
        command: EndpointCommand,
        *,
        base_action: EnvAction,
        active_mask: torch.Tensor,
    ) -> EnvAction:
        """Merge one addressed command into ``base_action``."""

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        base_action: EnvAction,
        context: PlanningContext,
    ) -> EnvAction:
        """Merge this transport's self-proven safe hold into ``base_action``.

        The transport remains authoritative for neutralizing its own controller;
        parallel command validation does not replace this transport-specific hold
        contract.
        """


@runtime_checkable
class CompiledTaskProgramPort(Protocol):
    """Minimal provider-free compiled-program surface consumed by the bridge."""

    program_id: str

    def iter_segments(self) -> Iterator[Any]:
        """Lazily yield compiled logical segments."""

    def sequential_execution_analysis(self, segment_index: int) -> Any:
        """Return current prefix plus downstream calls up to the next barrier."""


@runtime_checkable
class SequentialSemanticCallExecutorPort(Protocol):
    """Nonblocking semantic runtime surface used by sequential segments."""

    @property
    def result(self) -> SemanticExecutionResult:
        """Return the current immutable runtime result."""

    @property
    def status(self) -> SemanticExecutionStatus:
        """Return the current runtime status."""

    def start(
        self,
        *calls: Any,
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SemanticExecutionResult:
        """Start one semantic workflow without blocking on motion."""

    def step(self) -> SemanticExecutionResult:
        """Advance the workflow by at most one due runner cycle."""

    def cancel(self, reason: str) -> SemanticExecutionResult:
        """Cancel one running workflow through the runner's safe-stop path."""

    def adopt_verified_task_state(
        self, task_state: TaskState
    ) -> SemanticExecutionResult:
        """Install state merged at an independent parallel barrier."""


@runtime_checkable
class SegmentPostPolicyPort(Protocol):
    """Environment-aware program post-policy boundary.

    Implementations may observe the environment after each resumed yield, but
    must return every controller action to this iterable.  The bridge then
    routes those values through the ordinary demo executor and ``env.step``.
    """

    def validate_policy(
        self,
        policy: Any,
        *,
        segment: Any,
    ) -> None:
        """Validate one compiled policy without live observation or action."""

    def actions(
        self,
        policy: Any,
        *,
        segment: Any,
        active_mask: torch.Tensor,
    ) -> Iterable[Any]:
        """Yield holds until ``policy`` completes for the active rows only."""

    def post_policy_metadata(
        self,
        policy: Any,
        *,
        segment: Any,
    ) -> Mapping[str, Any]:
        """Return JSON-safe metadata after one policy has run."""

    def post_policy_result(self, policy: Any, *, segment: Any) -> Any:
        """Return one boolean or one boolean per environment row."""


@runtime_checkable
class SegmentValidatorPort(Protocol):
    """Environment-aware boundary for compiled program validators."""

    def validate_validator(
        self,
        validator: Any,
        *,
        segment: Any,
    ) -> None:
        """Validate one compiled validator without observing the environment."""

    def validate(self, validator: Any, *, segment: Any) -> Any:
        """Return one boolean or one boolean per environment row."""

    def validator_metadata(
        self,
        validator: Any,
        *,
        segment: Any,
    ) -> Mapping[str, Any]:
        """Return JSON-safe metadata after one validator has run."""


class GymPlanningObservationProvider:
    """Callback-backed observation port that also exposes the latest qpos.

    Args:
        capture: Callback accepting verified :class:`TaskState` and returning
            one fresh :class:`PlanningContext` from the Gym environment.

    The callback is intentionally explicit: environment-specific scene,
    simulator, and registry access remains in environment integration code.
    """

    def __init__(self, capture: Callable[[TaskState], PlanningContext]) -> None:
        if not callable(capture):
            raise TypeError("capture must be callable.")
        self._capture = capture
        self._latest: PlanningContext | None = None

    def observe(self, task_state: TaskState) -> PlanningContext:
        """Capture and retain one fresh planning context."""
        if not isinstance(task_state, TaskState):
            raise TypeError("task_state must be a TaskState.")
        context = self._capture(task_state)
        if not isinstance(context, PlanningContext):
            raise TypeError("capture must return a PlanningContext.")
        self._latest = context
        return context

    def current_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return latest full qpos rows in the requested stable-ID order."""
        context = self._latest
        if context is None:
            raise RuntimeError("No planning context has been observed yet.")
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if env_ids.dtype != torch.long or env_ids.dim() != 1 or env_ids.numel() == 0:
            raise ValueError("env_ids must be a non-empty one-dimensional long tensor.")
        if env_ids.device != context.env_ids.device:
            raise ValueError("env_ids must share the latest context device.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")

        row_by_id = {
            int(env_id): row
            for row, env_id in enumerate(context.env_ids.detach().cpu().tolist())
        }
        try:
            rows = [
                row_by_id[int(env_id)] for env_id in env_ids.detach().cpu().tolist()
            ]
        except KeyError as exc:
            raise ValueError(
                f"Environment ID {int(exc.args[0])} is absent from the latest context."
            ) from exc
        return context.robot.qpos[rows].clone()


class EnvironmentStepClock(ExecutionClock):
    """Monotonic execution clock advanced only by explicit Gym steps.

    ``sleep`` intentionally raises.  Calling synchronous ``SemanticCallExecutor.run``
    with this clock would otherwise advance execution without an environment
    transition.  Demo integrations must use the nonblocking ``start``/``step``
    path and call :meth:`advance_after_env_step` only after a yielded action was
    passed to ``env.step``.
    """

    def __init__(self, step_dt: float, *, initial_step: int = 0) -> None:
        if not isinstance(step_dt, (int, float)) or isinstance(step_dt, bool):
            raise TypeError("step_dt must be a real number.")
        if not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
            raise ValueError("step_dt must be finite and positive.")
        if type(initial_step) is not int or initial_step < 0:
            raise ValueError("initial_step must be a non-negative integer.")
        self._step_dt = float(step_dt)
        self._step_index = initial_step

    @property
    def step_dt(self) -> float:
        """Return the authoritative Gym control cadence."""
        return self._step_dt

    @property
    def step_index(self) -> int:
        """Return the number of explicitly acknowledged environment steps."""
        return self._step_index

    def now(self) -> float:
        """Return deterministic environment time in seconds."""
        return self._step_index * self._step_dt

    def sleep(self, duration: float) -> None:
        """Reject implicit waiting that is not backed by ``env.step``."""
        self.steps_for_duration(duration, field_name="sleep duration")
        raise RuntimeError(
            "EnvironmentStepClock cannot sleep or advance implicitly; use the "
            "nonblocking runtime and advance_after_env_step() after env.step()."
        )

    def steps_for_duration(
        self,
        duration: float,
        *,
        field_name: str = "duration",
    ) -> int:
        """Return an exact integer-grid representation of ``duration``.

        Float32 command tensors receive a small ratio-space tolerance, but an
        incompatible cadence is never rounded or resampled.
        """
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise TypeError(f"{field_name} must be a real number.")
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError(f"{field_name} must be finite and non-negative.")
        ratio = duration / self._step_dt
        nearest = round(ratio)
        tolerance = max(1.0e-6, abs(ratio) * 1.0e-6)
        if not math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=tolerance):
            raise EnvironmentStepTimingError(
                f"{field_name}={duration:.9g}s is not an integer multiple of "
                f"step_dt={self._step_dt:.9g}s; explicit resampling is not supported."
            )
        return int(nearest)

    def validate_frame(self, frame: RuntimeCommandFrame) -> None:
        """Validate every row's command hold duration against the step grid."""
        if not isinstance(frame, RuntimeCommandFrame):
            raise TypeError("frame must be a RuntimeCommandFrame.")
        for row, duration in enumerate(frame.hold_duration.detach().cpu().tolist()):
            self.steps_for_duration(
                float(duration),
                field_name=f"RuntimeCommandFrame.hold_duration[{row}]",
            )

    def advance_after_env_step(self, steps: int = 1) -> None:
        """Advance time after ``steps`` completed Gym environment transitions."""
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer.")
        self._step_index += steps


class JointPositionGymTransportEncoder:
    """Built-in ``robot.joint_position`` to full-qpos action encoder."""

    transport_id: ClassVar[str] = JointPositionTarget.TRANSPORT_ID
    target_types: ClassVar[tuple[type[RuntimeEndpointTarget], ...]] = (
        JointPositionTarget,
    )
    payload_types: ClassVar[tuple[type[RuntimeCommandPayload], ...]] = (
        JointPositionPayload,
    )

    def encode(
        self,
        command: EndpointCommand,
        *,
        base_action: EnvAction,
        active_mask: torch.Tensor,
    ) -> EnvAction:
        """Write addressed joints while holding every other qpos column."""
        if not isinstance(command.target, JointPositionTarget):
            raise TypeError("Joint-position transport requires JointPositionTarget.")
        if not isinstance(command.payload, JointPositionPayload):
            raise TypeError("Joint-position transport requires JointPositionPayload.")
        if not isinstance(base_action, torch.Tensor):
            raise TypeError(
                "The built-in joint-position encoder requires a tensor base action; "
                "register structured transports after it or provide a compatible "
                "custom composition encoder."
            )
        if base_action.dim() != 2 or base_action.shape[0] != command.batch_size:
            raise ValueError(
                "The full-qpos base action must have shape (batch_size, robot_dof)."
            )
        if active_mask.dtype != torch.bool or active_mask.shape != (
            command.batch_size,
        ):
            raise ValueError("active_mask must be bool with one value per command row.")
        if active_mask.device != base_action.device:
            raise ValueError("active_mask and base_action must share a device.")
        joint_ids = command.target.joint_ids
        if max(joint_ids) >= base_action.shape[1]:
            raise ValueError(
                f"Joint ID {max(joint_ids)} exceeds full qpos width "
                f"{base_action.shape[1]}."
            )
        positions = command.payload.positions
        if positions.device != base_action.device:
            raise ValueError("Joint payload and base action must share a device.")
        if not base_action.is_floating_point():
            raise TypeError("The full-qpos base action must be floating point.")

        action = base_action.clone()
        columns = torch.tensor(joint_ids, dtype=torch.long, device=action.device)
        selected = action.index_select(1, columns)
        selected[active_mask] = positions[active_mask].to(dtype=action.dtype)
        action[:, columns] = selected
        return action

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        base_action: EnvAction,
        context: PlanningContext,
    ) -> EnvAction:
        """Keep observed full qpos unchanged for addressed joint targets."""
        del context
        if not all(isinstance(target, JointPositionTarget) for target in targets):
            raise TypeError("Joint-position hold received an incompatible target.")
        return base_action.clone()


class RuntimeCommandFrameEncoder:
    """Encode transport-neutral command frames to controller-ready Gym actions.

    Args:
        qpos_provider: Full-qpos source aligned to a frame's explicit ``env_ids``.
        transports: Optional additional transport encoders.  The built-in
            joint-position encoder precedes them when enabled.
        include_joint_position: Whether to install the built-in joint-position
            encoder. Standard assemblies disable it when their exact profile uses
            only custom endpoint transports.
    """

    def __init__(
        self,
        qpos_provider: CurrentQposProvider,
        *,
        transports: Iterable[RuntimeTransportActionEncoder] = (),
        include_joint_position: bool = True,
    ) -> None:
        if not isinstance(qpos_provider, CurrentQposProvider):
            raise TypeError("qpos_provider must implement CurrentQposProvider.")
        if type(include_joint_position) is not bool:
            raise TypeError("include_joint_position must be a bool.")
        self._qpos_provider = qpos_provider
        self._transports: dict[str, RuntimeTransportActionEncoder] = {}
        self._frozen = False
        if include_joint_position:
            self.register_transport(JointPositionGymTransportEncoder())
        for transport in transports:
            self.register_transport(transport)

    @property
    def transport_ids(self) -> tuple[str, ...]:
        """Return registered transport IDs in deterministic encoding order."""
        return tuple(self._transports)

    @property
    def is_frozen(self) -> bool:
        """Return whether runtime transport registration is permanently closed."""
        return self._frozen

    def freeze(self) -> None:
        """Permanently close transport registration for a standard assembly."""
        self._frozen = True

    def register_transport(
        self,
        transport: RuntimeTransportActionEncoder,
        *,
        replace: bool = False,
    ) -> None:
        """Register one shared transport-to-Gym action encoder."""
        if self._frozen:
            raise RuntimeError(
                "Runtime transport registration is frozen for this command encoder."
            )
        if not isinstance(transport, RuntimeTransportActionEncoder):
            raise TypeError("transport must implement RuntimeTransportActionEncoder.")
        transport_type = type(transport)
        transport_id = _validate_identifier(
            getattr(transport_type, "transport_id", None),
            field_name="RuntimeTransportActionEncoder.transport_id",
        )
        self._validate_declared_types(
            getattr(transport_type, "target_types", None),
            base_type=RuntimeEndpointTarget,
            field_name="RuntimeTransportActionEncoder.target_types",
        )
        self._validate_declared_types(
            getattr(transport_type, "payload_types", None),
            base_type=RuntimeCommandPayload,
            field_name="RuntimeTransportActionEncoder.payload_types",
        )
        if type(replace) is not bool:
            raise TypeError("replace must be a bool.")
        if transport_id in self._transports and not replace:
            raise ValueError(f"Transport {transport_id!r} is already registered.")
        self._transports[transport_id] = transport

    @staticmethod
    def _validate_declared_types(
        values: object,
        *,
        base_type: type[object],
        field_name: str,
    ) -> None:
        """Validate one non-empty exact tuple of supported runtime types."""
        if type(values) is not tuple or not values:
            raise TypeError(f"{field_name} must be a non-empty exact tuple.")
        if not all(
            isinstance(value, type) and issubclass(value, base_type) for value in values
        ):
            raise TypeError(
                f"{field_name} must contain {base_type.__name__} subclasses."
            )
        if len(set(values)) != len(values):
            raise ValueError(f"{field_name} must not contain duplicate types.")

    @staticmethod
    def _validate_command_types(
        transport: RuntimeTransportActionEncoder,
        command: EndpointCommand,
    ) -> None:
        """Require exact target and payload coverage before transport routing."""
        transport_type = type(transport)
        if type(command.target) not in transport_type.target_types:
            raise TypeError(
                f"Transport {transport_type.transport_id!r} does not declare exact "
                f"target type {type(command.target).__name__}."
            )
        if type(command.payload) not in transport_type.payload_types:
            raise TypeError(
                f"Transport {transport_type.transport_id!r} does not declare exact "
                f"payload type {type(command.payload).__name__}."
            )

    @staticmethod
    def _validate_hold_target_types(
        transport: RuntimeTransportActionEncoder,
        targets: Iterable[RuntimeEndpointTarget],
    ) -> None:
        """Require exact target coverage before safe-hold routing."""
        transport_type = type(transport)
        for target in targets:
            if type(target) not in transport_type.target_types:
                raise TypeError(
                    f"Transport {transport_type.transport_id!r} does not declare "
                    f"exact hold target type {type(target).__name__}."
                )

    def _base_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Capture and validate one owned full-qpos hold action."""
        qpos = self._qpos_provider.current_qpos(env_ids)
        if not isinstance(qpos, torch.Tensor):
            raise TypeError("CurrentQposProvider.current_qpos() must return a tensor.")
        if qpos.dim() != 2 or qpos.shape[0] != env_ids.shape[0] or qpos.shape[1] == 0:
            raise ValueError(
                "Current qpos must have shape (batch_size, robot_dof) with non-zero DOF."
            )
        if qpos.device != env_ids.device:
            raise ValueError("Current qpos and env_ids must share a device.")
        if not qpos.is_floating_point() or not torch.isfinite(qpos).all().item():
            raise ValueError("Current qpos must contain finite floating-point values.")
        return qpos.clone()

    def encode(self, frame: RuntimeCommandFrame) -> EnvAction:
        """Encode one frame on top of a fresh full-qpos hold action."""
        if not isinstance(frame, RuntimeCommandFrame):
            raise TypeError("frame must be a RuntimeCommandFrame.")
        action: EnvAction = self._base_qpos(frame.env_ids)
        by_transport: dict[str, list[EndpointCommand]] = {}
        for command in frame.commands:
            transport = self._transports.get(command.transport_id)
            if transport is None:
                raise UnsupportedRuntimeTransportError(
                    f"No Gym action encoder is registered for runtime transport "
                    f"{command.transport_id!r}."
                )
            self._validate_command_types(transport, command)
            by_transport.setdefault(command.transport_id, []).append(command)
        for transport_id, transport in self._transports.items():
            for command in by_transport.get(transport_id, ()):
                action = transport.encode(
                    command,
                    base_action=action,
                    active_mask=frame.active_mask,
                )
        return action

    def encode_hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
    ) -> EnvAction:
        """Encode an observed-position safe hold for addressed transports."""
        if not isinstance(context, PlanningContext):
            raise TypeError("context must be a PlanningContext.")
        action: EnvAction = context.robot.qpos.clone()
        by_transport: dict[str, list[RuntimeEndpointTarget]] = {}
        for target in targets:
            if not isinstance(target, RuntimeEndpointTarget):
                raise TypeError("targets must contain RuntimeEndpointTarget values.")
            by_transport.setdefault(target.transport_id, []).append(target)
        for transport_id, grouped in by_transport.items():
            transport = self._transports.get(transport_id)
            if transport is None:
                raise UnsupportedRuntimeTransportError(
                    f"No Gym action encoder is registered for runtime transport "
                    f"{transport_id!r}."
                )
            self._validate_hold_target_types(transport, grouped)
        for transport_id, transport in self._transports.items():
            grouped = by_transport.get(transport_id)
            if grouped is None:
                continue
            action = transport.hold(
                tuple(grouped),
                base_action=action,
                context=context,
            )
        return action

    def encode_idle_hold(self, env_ids: torch.Tensor) -> EnvAction:
        """Return a fresh full-qpos hold when no transport was armed yet."""
        return self._base_qpos(env_ids)


class BufferedGymCommandSink:
    """Runner command sink that buffers actions for the Gym demo generator.

    Acceptance means the command was validated and copied into the local
    buffer; it does not claim that an environment transition already occurred.
    """

    def __init__(
        self,
        encoder: RuntimeCommandFrameEncoder,
        clock: EnvironmentStepClock,
    ) -> None:
        if not isinstance(encoder, RuntimeCommandFrameEncoder):
            raise TypeError("encoder must be a RuntimeCommandFrameEncoder.")
        if not isinstance(clock, EnvironmentStepClock):
            raise TypeError("clock must be an EnvironmentStepClock.")
        self._encoder = encoder
        self._clock = clock
        self._pending: deque[ControllerAction] = deque()
        self._last_emitted: ControllerAction | None = None
        self._accepted_action_count = 0

    @property
    def clock(self) -> EnvironmentStepClock:
        """Return the exact environment-step clock used for timing checks."""
        return self._clock

    @property
    def pending_count(self) -> int:
        """Return the number of accepted actions not yet yielded to Gym."""
        return len(self._pending)

    @property
    def accepted_action_count(self) -> int:
        """Return the monotonic count of actions accepted by this sink."""
        return self._accepted_action_count

    def send(
        self,
        command: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Validate, encode, and buffer one runtime command frame."""
        _validate_timeout(timeout)
        if not isinstance(command, RuntimeCommandFrame):
            raise TypeError("command must be a RuntimeCommandFrame.")
        self._clock.validate_frame(command)
        action = self._encoder.encode(command)
        metadata = {
            "bridge_action_kind": "runtime_command",
            "runtime_destinations": [
                [item.transport_id, item.target.target_id] for item in command.commands
            ],
            "active_mask": command.active_mask.detach().cpu().tolist(),
            "hold_duration": command.hold_duration.detach().cpu().tolist(),
        }
        self._pending.append(ControllerAction(value=action, metadata=metadata))
        self._accepted_action_count += 1
        return CommandAcknowledgement.accepted_ack("Buffered for the Gym step loop.")

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Buffer one observed-position safe hold action."""
        _validate_timeout(timeout)
        action = self._encoder.encode_hold(tuple(targets), context)
        metadata = {
            "bridge_action_kind": "runtime_safe_hold",
            "runtime_destinations": [
                [target.transport_id, target.target_id] for target in targets
            ],
        }
        self._pending.append(ControllerAction(value=action, metadata=metadata))
        self._accepted_action_count += 1
        return CommandAcknowledgement.accepted_ack("Safe hold buffered for Gym.")

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Discard accepted-but-not-yielded frames before a safe-stop hold."""
        _validate_timeout(timeout)
        if not all(isinstance(target, RuntimeEndpointTarget) for target in targets):
            raise TypeError("targets must contain RuntimeEndpointTarget values.")
        self._pending.clear()
        return CommandAcknowledgement.accepted_ack("Buffered commands cancelled.")

    def discard_pending(self) -> None:
        """Discard actions that were accepted locally but never yielded."""
        self._pending.clear()

    def drain_safe_stop_action(
        self,
        *,
        fallback: ControllerAction | None = None,
    ) -> ControllerAction | None:
        """Select one buffered safe hold and discard every other local action.

        This method is used only by the demo abort handshake. A runtime
        acknowledgement proves local buffering, not ``env.step`` consumption;
        therefore an interrupted generator must explicitly surface the final
        safe hold to the executor while dropping stale motion commands.
        """
        candidates: list[ControllerAction] = []
        for candidate in (self._last_emitted, fallback):
            if (
                candidate is not None
                and candidate.metadata.get("bridge_action_kind")
                in _SAFE_HOLD_ACTION_KINDS
            ):
                candidates.append(candidate.snapshot())
        while self._pending:
            candidate = self._pending.popleft()
            if candidate.metadata.get("bridge_action_kind") in _SAFE_HOLD_ACTION_KINDS:
                candidates.append(candidate.snapshot())
        return None if not candidates else candidates[-1].snapshot()

    def pop(self) -> ControllerAction:
        """Pop the next accepted action and remember it as the active hold."""
        if not self._pending:
            raise RuntimeError("No buffered Gym command is available.")
        action = self._pending.popleft().snapshot()
        self._last_emitted = action.snapshot()
        return action

    def wait_hold(self, env_ids: torch.Tensor) -> ControllerAction:
        """Return an owned hold action for one runtime waiting step."""
        if self._last_emitted is None:
            value = self._encoder.encode_idle_hold(env_ids)
        else:
            value = self._last_emitted.value
        return ControllerAction(
            value=value,
            metadata={"bridge_action_kind": "runtime_wait_hold"},
        )


@dataclass(slots=True)
class _SegmentLifecycle:
    """Mutable state shared by one lazy action generator and validator."""

    complete: bool = False
    result: SemanticExecutionResult | ParallelSemanticExecutionResult | None = None
    validation: torch.Tensor | None = None
    runtime: SequentialSemanticCallExecutorPort | ParallelSemanticExecutor | None = None
    pending_action: ControllerAction | None = None
    actions_started: bool = False
    sink_acceptance_baseline: int | None = None
    yielded_action_count: int = 0
    abort_started: bool = False
    abort_complete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    post_policy_success: torch.Tensor | None = None


def _validate_runtime_result(
    result: SemanticExecutionResult | ParallelSemanticExecutionResult,
) -> SemanticExecutionResult | ParallelSemanticExecutionResult:
    """Validate one exact sequential or parallel runtime boundary."""
    if not isinstance(
        result, (SemanticExecutionResult, ParallelSemanticExecutionResult)
    ):
        raise TypeError(
            "Runtime methods must return SemanticExecutionResult or ParallelSemanticExecutionResult values."
        )
    return result


def _normalize_validation(
    value: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize one validator output to an owned row-local boolean tensor."""
    tensor = torch.as_tensor(value, dtype=torch.bool, device=device).reshape(-1)
    if tensor.numel() == 1 and batch_size > 1:
        tensor = tensor.repeat(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(
            f"Segment validator returned {tensor.numel()} flags, expected "
            f"{batch_size}."
        )
    return tensor.clone()


def _runtime_result_metadata(
    result: SemanticExecutionResult | ParallelSemanticExecutionResult,
) -> dict[str, Any]:
    """Snapshot one core runtime result through its canonical serializer."""
    serializer = getattr(result, "to_metadata", None)
    if not callable(serializer):
        raise TypeError(
            f"{type(result).__name__} must provide to_metadata() for demo tracing."
        )
    metadata = _json_safe_copy(serializer(), field_name="runtime result metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Runtime result to_metadata() must return a mapping.")
    return metadata


class TaskProgramDemoBridge:
    """Adapt sequential compiled program segments to lazy Gym demonstrations.

    Args:
        program: Provider-free compiled Task Program.
        runtime: Nonblocking semantic :class:`SemanticCallExecutor` surface.
        command_sink: The same buffered sink installed in ``runtime``.
        clock: The same environment-step clock installed in ``runtime``.
        post_policy_port: Optional environment-aware post-policy executor.
        validator_port: Optional environment-aware validator executor.
        runner_cfg: Runner transport policy selected by the runtime preset.
        parallel_safety_validator: Optional authoritative physical-safety gate
            required before any parallel branch can start.

    Parallel blocks retain their branch lanes and explicit barrier.
    They are lowered through :class:`ParallelSemanticExecutor`; they are never
    flattened into a sequential semantic-call list.
    """

    def __init__(
        self,
        program: CompiledTaskProgramPort,
        runtime: SequentialSemanticCallExecutorPort,
        command_sink: BufferedGymCommandSink,
        clock: EnvironmentStepClock,
        *,
        post_policy_port: SegmentPostPolicyPort | None = None,
        validator_port: SegmentValidatorPort | None = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
        parallel_safety_validator: ParallelCommandSafetyValidator | None = None,
    ) -> None:
        if not isinstance(program, CompiledTaskProgramPort):
            raise TypeError("program must implement CompiledTaskProgramPort.")
        _validate_identifier(program.program_id, field_name="program.program_id")
        if not isinstance(runtime, SequentialSemanticCallExecutorPort):
            raise TypeError(
                "runtime must implement SequentialSemanticCallExecutorPort."
            )
        if not isinstance(command_sink, BufferedGymCommandSink):
            raise TypeError("command_sink must be a BufferedGymCommandSink.")
        if not isinstance(clock, EnvironmentStepClock):
            raise TypeError("clock must be an EnvironmentStepClock.")
        if command_sink.clock is not clock:
            raise ValueError("command_sink and bridge must share the exact clock.")
        if post_policy_port is not None and not isinstance(
            post_policy_port, SegmentPostPolicyPort
        ):
            raise TypeError("post_policy_port must implement SegmentPostPolicyPort.")
        if validator_port is not None and not isinstance(
            validator_port, SegmentValidatorPort
        ):
            raise TypeError("validator_port must implement SegmentValidatorPort.")
        if runner_cfg is not None and not isinstance(runner_cfg, ExecutionRunnerCfg):
            raise TypeError("runner_cfg must be an ExecutionRunnerCfg or None.")
        if parallel_safety_validator is not None and not isinstance(
            parallel_safety_validator, ParallelCommandSafetyValidator
        ):
            raise TypeError(
                "parallel_safety_validator must implement "
                "ParallelCommandSafetyValidator."
            )
        self._program = program
        self._runtime = runtime
        self._sink = command_sink
        self._clock = clock
        self._post_policy_port = post_policy_port
        self._validator_port = validator_port
        self._runner_cfg = deepcopy(runner_cfg or ExecutionRunnerCfg())
        self._parallel_safety_validator = parallel_safety_validator
        self._active_segment_id: str | None = None
        self._eligible_mask: torch.Tensor | None = None
        self._program_completed = False

    @property
    def clock(self) -> EnvironmentStepClock:
        """Return the environment-step clock used by this bridge."""
        return self._clock

    @property
    def program_completed(self) -> bool:
        """Return whether every compiled segment completed its full lifecycle."""
        return self._program_completed

    @property
    def completion_mask(self) -> torch.Tensor:
        """Return the final row-local program acceptance mask.

        Raises:
            RuntimeError: If the segment iterator has not completed normally.
        """
        if not self._program_completed or self._eligible_mask is None:
            raise RuntimeError(
                "Task Program completion is unavailable before all segments "
                "finish execution and validation."
            )
        return self._eligible_mask.clone()

    def iter_segments(self) -> Iterator[DemoSegment]:
        """Lazily adapt compiled program segments to ``DemoSegment`` values.

        Consumers must exhaust each segment's actions and invoke its validator
        before requesting the next segment. Skipping either lifecycle boundary
        raises :class:`TaskProgramBridgeError` instead of silently carrying stale row
        eligibility into downstream execution.
        """
        for segment in self._program.iter_segments():
            metadata = self._segment_metadata(segment)
            lifecycle = _SegmentLifecycle(metadata=metadata)
            validator = self._segment_validator(segment, lifecycle)
            yield DemoSegment(
                actions=self._segment_actions(segment, lifecycle),
                name=segment.name,
                metadata=metadata,
                validator=validator,
                abort_actions=self._segment_abort_actions(segment, lifecycle),
                failure_policy="row_independent",
                progress_total_steps=self._segment_progress_total_steps(segment),
            )
            self._require_consumed_segment_lifecycle(segment, lifecycle)
        if self._eligible_mask is None:
            raise TaskProgramBridgeError("Compiled Task Program produced no segments.")
        self._program_completed = True

    def __iter__(self) -> Iterator[DemoSegment]:
        """Delegate iteration to :meth:`iter_segments`."""
        return self.iter_segments()

    @staticmethod
    def _require_consumed_segment_lifecycle(
        segment: Any,
        lifecycle: _SegmentLifecycle,
    ) -> None:
        """Reject advancing past a segment with an unconsumed lifecycle.

        The public demo executor exhausts ``actions`` and then invokes the
        segment validator before requesting the next lazy segment.  Direct
        bridge consumers must preserve the same ordering because validation is
        also the commit point for runtime and post-policy row eligibility.
        """
        if not lifecycle.complete:
            raise TaskProgramBridgeError(
                f"Segment {segment.segment_id!r} actions must be exhausted before "
                "requesting the next compiled segment."
            )
        if lifecycle.validation is None:
            raise TaskProgramBridgeError(
                f"Segment {segment.segment_id!r} validator must be called after "
                "its actions are exhausted and before requesting the next "
                "compiled segment."
            )

    def _segment_progress_total_steps(self, segment: Any) -> int | None:
        """Count fixed Pick/Place samples without analyzing downstream calls."""
        if (
            getattr(segment, "parallel_block", None) is not None
            or segment.post_policies
            or self._runner_cfg.minimum_cycle_time != 0.0
            or self._runner_cfg.hold_on_completion
            or self._runner_cfg.hold_during_effect_verification
        ):
            return None

        compiler = getattr(self._runtime, "compiler", None)
        integration = getattr(compiler, "integration", None)
        if integration is None:
            return None

        total = 0
        try:
            for compiled_call in segment.calls:
                call = compiled_call.call
                if type(call) not in (Pick, Place):
                    return None
                preset = integration.link_call(call).preset
                motion = preset.motion_policy
                recovery = preset.recovery_policy
                tracking = preset.tracking_policy
                if (
                    preset.effect_assurance is not EffectAssurance.PROJECTED
                    or motion.strategy != "ik_interp"
                    or type(motion.sample_count) is not int
                    or recovery.max_replans != 0
                    or recovery.max_action_retries != 0
                    or preset.workflow_recovery_policy.max_recovery_attempts != 0
                    or tracking.in_flight is not None
                    or getattr(tracking.terminal, "settle_duration", None) != 0.0
                ):
                    return None
                options = preset.action_option_template(call.semantic_id)
                if type(options) is PickUpOptions:
                    total += motion.sample_count + options.grasp_settle_steps
                elif type(options) is PlaceOptions:
                    total += motion.sample_count + options.release_settle_steps
                else:
                    return None
        except SemanticValidationError:
            # Let ordinary execution report linking failures through its lifecycle.
            return None
        return total or None

    def _segment_metadata(self, segment: Any) -> dict[str, Any]:
        """Build mutable JSON-safe metadata completed at lifecycle boundaries."""
        return {
            "task_program_id": self._program.program_id,
            "program_segment_id": segment.segment_id,
            "program_segment_index": segment.segment_index,
            "segment_count": getattr(self._program, "segment_count", None),
            "program_segment_source_path": list(segment.source_path),
            "program_segment_implicit": bool(segment.implicit),
            "semantic_call_indices": [call.call_index for call in segment.calls],
            "post_policy_count": len(segment.post_policies),
            "validator_count": len(segment.validators),
            "parallel": getattr(segment, "parallel_block", None) is not None,
            "runtime": None,
            "post_policies": [],
            "validation": None,
        }

    @staticmethod
    def _record_runtime_result(
        lifecycle: _SegmentLifecycle,
        result: SemanticExecutionResult | ParallelSemanticExecutionResult,
    ) -> None:
        """Snapshot one runtime boundary into its owning segment metadata."""
        lifecycle.result = result
        lifecycle.metadata["runtime"] = _runtime_result_metadata(result)

    def _decorate_action(
        self,
        action: Any,
        *,
        segment: Any,
        result: SemanticExecutionResult | ParallelSemanticExecutionResult,
        action_kind: str | None = None,
    ) -> ControllerAction:
        """Own one action and attach stable program/runtime provenance."""
        if isinstance(action, ControllerAction):
            value = action.value
            metadata = dict(action.metadata)
        else:
            value = action
            metadata = {}
        if action_kind is not None:
            metadata["bridge_action_kind"] = action_kind
        metadata.update(
            {
                "task_program_id": self._program.program_id,
                "program_segment_id": segment.segment_id,
                "program_segment_index": segment.segment_index,
                "environment_step": self._clock.step_index,
                "runtime_status": result.status.value,
                "runtime_call_index": getattr(result, "current_call_index", None),
            }
        )
        return ControllerAction(value=value, metadata=metadata)

    def _yield_and_advance(
        self,
        action: ControllerAction,
        lifecycle: _SegmentLifecycle,
    ) -> Iterator[ControllerAction]:
        """Yield once and advance only after explicit consumption acknowledgement."""
        if lifecycle.pending_action is not None:
            raise RuntimeError("A prior demo action is still awaiting acknowledgement.")
        lifecycle.pending_action = action.snapshot()
        lifecycle.yielded_action_count += 1
        yield action
        if lifecycle.pending_action is not None:
            self._clock.advance_after_env_step()
            lifecycle.pending_action = None

    def _segment_actions(
        self,
        segment: Any,
        lifecycle: _SegmentLifecycle,
    ) -> Iterator[ControllerAction]:
        """Drive one semantic segment without bypassing the Gym step loop."""
        segment_id = segment.segment_id
        lifecycle.actions_started = True
        lifecycle.sink_acceptance_baseline = self._sink.accepted_action_count
        if self._active_segment_id is not None:
            raise RuntimeError(
                f"Segment {self._active_segment_id!r} is still active; exhaust or "
                "close it before starting another lazy segment."
            )
        self._active_segment_id = segment_id
        result: SemanticExecutionResult | ParallelSemanticExecutionResult | None = None
        segment_runtime: (
            SequentialSemanticCallExecutorPort | ParallelSemanticExecutor
        ) = self._runtime
        is_parallel = getattr(segment, "parallel_block", None) is not None
        try:
            if is_parallel:
                segment_runtime = self._parallel_runtime(segment)
                lifecycle.runtime = segment_runtime
                result = _validate_runtime_result(
                    segment_runtime.start(
                        workflow_id=f"{self._program.program_id}/{segment_id}",
                        eligible_mask=self._eligible_mask,
                    )
                )
            else:
                lifecycle.runtime = segment_runtime
                analysis = self._program.sequential_execution_analysis(
                    segment.segment_index
                )
                calls = tuple(analysis.calls)
                if not calls:
                    raise TaskProgramBridgeError(
                        f"Compiled segment {segment_id!r} contains no semantic calls."
                    )
                execution_prefix_length = analysis.execution_prefix_length
                if execution_prefix_length != len(segment.calls):
                    raise TaskProgramBridgeError(
                        f"Compiled segment {segment_id!r} analysis prefix length "
                        "does not match its owned semantic calls."
                    )
                result = _validate_runtime_result(
                    segment_runtime.start(
                        calls,
                        workflow_id=f"{self._program.program_id}/{segment_id}",
                        eligible_mask=self._eligible_mask,
                        execution_prefix_length=execution_prefix_length,
                    )
                )

            while True:
                emitted = False
                while self._sink.pending_count:
                    action = self._decorate_action(
                        self._sink.pop(),
                        segment=segment,
                        result=result,
                    )
                    yield from self._yield_and_advance(action, lifecycle)
                    emitted = True

                if emitted and not result.terminal:
                    # The result's wait duration was measured before the action
                    # just consumed by Gym.  Refresh it against the advanced
                    # environment clock before deciding whether another hold is due.
                    result = _validate_runtime_result(segment_runtime.step())
                    continue

                if result.terminal:
                    break

                if result.wait_duration > 0.0:
                    self._clock.steps_for_duration(
                        result.wait_duration,
                        field_name="SemanticExecutionResult.wait_duration",
                    )
                    hold = self._decorate_action(
                        self._sink.wait_hold(result.env_ids),
                        segment=segment,
                        result=result,
                        action_kind="runtime_wait_hold",
                    )
                    yield from self._yield_and_advance(hold, lifecycle)

                result = _validate_runtime_result(segment_runtime.step())

            self._record_runtime_result(lifecycle, result)
            self._retain_eligible_rows(result.success_mask)
            if is_parallel:
                self._runtime.adopt_verified_task_state(result.task_state)
            if result.status is SemanticExecutionStatus.COMPLETED:
                yield from self._post_policy_actions(segment, result, lifecycle)
            lifecycle.complete = True
        finally:
            if not lifecycle.abort_started and lifecycle.pending_action is not None:
                if result is not None and not result.terminal:
                    segment_runtime.cancel(
                        f"Demo segment {segment_id!r} action iteration stopped early."
                    )
                raise TaskProgramBridgeError(
                    f"Demo segment {segment_id!r} was closed with an unacknowledged "
                    "action. Consume DemoSegment.abort_actions through env.step() "
                    "before closing the action iterator."
                )
            self._active_segment_id = None

    def _segment_abort_actions(
        self,
        segment: Any,
        lifecycle: _SegmentLifecycle,
    ) -> Callable[..., Iterator[ControllerAction]]:
        """Create the explicit executor-to-runtime cancellation handshake."""

        def abort(
            reason: str,
            *,
            last_action_consumed: bool,
        ) -> Iterator[ControllerAction]:
            return self._abort_segment(
                segment,
                lifecycle,
                reason=reason,
                last_action_consumed=last_action_consumed,
            )

        return abort

    def _abort_segment(
        self,
        segment: Any,
        lifecycle: _SegmentLifecycle,
        *,
        reason: str,
        last_action_consumed: bool,
    ) -> Iterator[ControllerAction]:
        """Abort one segment, surfacing a safe hold only after controller activity."""
        if type(reason) is not str or not reason:
            raise ValueError("abort reason must be a non-empty string.")
        if type(last_action_consumed) is not bool:
            raise TypeError("last_action_consumed must be a bool.")
        if lifecycle.abort_started:
            raise RuntimeError(
                f"Segment {segment.segment_id!r} abort handshake already started."
            )
        if not lifecycle.actions_started:
            raise RuntimeError(
                f"Segment {segment.segment_id!r} has no started action iteration "
                "to abort."
            )
        baseline = lifecycle.sink_acceptance_baseline
        if baseline is None:
            raise RuntimeError(
                f"Segment {segment.segment_id!r} has no sink lifecycle baseline."
            )
        controller_activity_started = (
            lifecycle.yielded_action_count > 0
            or lifecycle.pending_action is not None
            or self._sink.accepted_action_count > baseline
        )
        if not controller_activity_started:
            # Runtime construction and preflight are deliberately observation- and
            # command-free.  If either fails before the first accepted or yielded
            # action, there is no physical controller state to safe-stop.  Mark the
            # handshake complete without touching the partially constructed runtime
            # so the original action-generation exception remains authoritative.
            lifecycle.abort_started = True
            lifecycle.abort_complete = True
            return
        runtime = lifecycle.runtime
        pending = lifecycle.pending_action
        if runtime is None:
            raise TaskProgramBridgeError(
                f"Segment {segment.segment_id!r} accepted or yielded a controller "
                "action without retaining a runtime capable of strict safe-stop."
            )
        lifecycle.abort_started = True
        if pending is not None:
            pending = pending.snapshot()
        if pending is not None and last_action_consumed:
            self._clock.advance_after_env_step()
        lifecycle.pending_action = None

        result = _validate_runtime_result(runtime.result)
        if not result.terminal:
            result = _validate_runtime_result(runtime.cancel(reason))
        self._record_runtime_result(lifecycle, result)

        pending_kind = (
            None if pending is None else pending.metadata.get("bridge_action_kind")
        )
        if (
            pending is not None
            and last_action_consumed
            and pending_kind in _SAFE_HOLD_ACTION_KINDS
        ):
            self._sink.discard_pending()
            lifecycle.abort_complete = True
            return

        safe_action = self._sink.drain_safe_stop_action(
            fallback=None if last_action_consumed else pending,
        )
        if safe_action is None:
            raise TaskProgramBridgeError(
                f"Segment {segment.segment_id!r} stopped before exhaustion, but "
                "no controller safe-hold action was available for env.step()."
            )
        processed = self._decorate_action(
            safe_action,
            segment=segment,
            result=result,
            action_kind="runtime_abort_safe_hold",
        )
        yield processed
        self._clock.advance_after_env_step()
        lifecycle.abort_complete = True

    def _parallel_runtime(self, segment: Any) -> ParallelSemanticExecutor:
        """Build one one-shot coordinator from a compiled explicit barrier."""
        if self._parallel_safety_validator is None:
            raise TaskProgramBridgeError(
                f"Parallel segment {segment.segment_id!r} requires an explicit "
                "ParallelCommandSafetyValidator; resource claims alone do not "
                "establish physical collision safety."
            )
        if not isinstance(self._runtime, SemanticCallExecutor):
            # Production integration always supplies SemanticCallExecutor.  Keeping the
            # sequential protocol permits lightweight tests and alternate
            # frontends, but the canonical parallel factory requires forkable
            # runtime internals by design.
            raise TypeError(
                "Parallel compiled segments require a concrete SemanticCallExecutor "
                "template."
            )
        block = segment.parallel_block
        branches = tuple(block.branches)
        if len(branches) < 2:
            raise TaskProgramBridgeError(
                f"Parallel segment {segment.segment_id!r} requires at least two "
                "compiled branches."
            )
        branch_calls = {
            f"branch_{branch.branch_index}": tuple(
                compiled.call for compiled in branch.calls
            )
            for branch in branches
        }
        branch_paths = {
            f"branch_{branch.branch_index}": tuple(
                getattr(branch, "source_path", segment.source_path)
            )
            for branch in branches
        }
        if any(not calls for calls in branch_calls.values()):
            raise TaskProgramBridgeError(
                f"Parallel segment {segment.segment_id!r} contains an empty branch."
            )
        barrier = block.barrier
        return ParallelSemanticExecutor.from_template(
            self._runtime,
            branch_calls,
            self._sink,
            ParallelTimingPolicy(self._clock.step_dt),
            self._parallel_safety_validator,
            timeout_steps=barrier.timeout_steps,
            failure_policy=barrier.failure_policy,
            runner_cfg=self._runner_cfg,
            workflow_id=(
                f"{self._program.program_id}/{segment.segment_id}:parallel_analysis"
            ),
            branch_paths=branch_paths,
        )

    def _retain_eligible_rows(self, accepted: torch.Tensor) -> None:
        """Permanently remove failed rows before a later lazy segment starts."""
        if not isinstance(accepted, torch.Tensor):
            raise TypeError("accepted must be a torch.Tensor.")
        if accepted.dtype != torch.bool or accepted.dim() != 1:
            raise ValueError("accepted must be a one-dimensional bool tensor.")
        if self._eligible_mask is None:
            self._eligible_mask = torch.ones_like(accepted)
        elif (
            self._eligible_mask.shape != accepted.shape
            or self._eligible_mask.device != accepted.device
        ):
            raise ValueError("Environment rows changed across program segments.")
        self._eligible_mask &= accepted

    def _post_policy_actions(
        self,
        segment: Any,
        result: SemanticExecutionResult | ParallelSemanticExecutionResult,
        lifecycle: _SegmentLifecycle,
    ) -> Iterator[ControllerAction]:
        """Route environment-aware post-policy actions through the same generator."""
        policies = tuple(segment.post_policies)
        if policies and self._post_policy_port is None:
            raise TaskProgramBridgeError(
                f"Segment {segment.segment_id!r} declares post-policies, but no "
                "SegmentPostPolicyPort was installed."
            )
        traces = lifecycle.metadata["post_policies"]
        if not isinstance(traces, list):
            raise TypeError("Segment post-policy metadata storage must be a list.")
        for policy_index, policy in enumerate(policies):
            assert self._post_policy_port is not None
            active_mask = (
                result.success_mask.clone()
                if self._eligible_mask is None
                else self._eligible_mask.clone()
            )
            if lifecycle.post_policy_success is not None:
                active_mask &= lifecycle.post_policy_success
            actions = self._post_policy_port.actions(
                policy,
                segment=segment,
                active_mask=active_mask,
            )
            if isinstance(actions, (str, bytes)):
                raise TypeError("Post-policy actions must be an iterable of actions.")
            action_iterator = iter(actions)
            iteration_error: BaseException | None = None
            try:
                for action in action_iterator:
                    processed = self._decorate_action(
                        action,
                        segment=segment,
                        result=result,
                        action_kind="program_post_policy",
                    )
                    yield from self._yield_and_advance(processed, lifecycle)
            except BaseException as exc:
                iteration_error = exc
                raise
            finally:
                close = getattr(action_iterator, "close", None)
                if callable(close):
                    close()
                cfg = getattr(policy, "cfg", None)
                trace: dict[str, Any] = {
                    "policy_index": policy_index,
                    "kind": getattr(cfg, "kind", type(policy).__name__),
                    "source_path": list(getattr(policy, "source_path", ())),
                    "result_mask": result.success_mask.detach().cpu().tolist(),
                    "result": None,
                }
                port = self._post_policy_port
                try:
                    policy_success = active_mask & _normalize_validation(
                        port.post_policy_result(policy, segment=segment),
                        batch_size=result.env_ids.numel(),
                        device=result.env_ids.device,
                    )
                except Exception:
                    if iteration_error is None:
                        raise
                    policy_success = active_mask.clone()
                if lifecycle.post_policy_success is None:
                    lifecycle.post_policy_success = policy_success.clone()
                else:
                    lifecycle.post_policy_success &= policy_success
                trace["result_mask"] = policy_success.detach().cpu().tolist()
                try:
                    trace["result"] = port.post_policy_metadata(
                        policy,
                        segment=segment,
                    )
                except Exception:
                    if iteration_error is None:
                        raise
                traces.append(
                    _json_safe_copy(
                        trace,
                        field_name=f"post-policy {policy_index} metadata",
                    )
                )

    def _segment_validator(
        self,
        segment: Any,
        lifecycle: _SegmentLifecycle,
    ) -> Callable[[], torch.Tensor]:
        """Create a demo-boundary validator including runtime row success."""

        def validate() -> torch.Tensor:
            if not lifecycle.complete or lifecycle.result is None:
                raise RuntimeError(
                    f"Segment {segment.segment_id!r} cannot be validated before its "
                    "action iterable is exhausted."
                )
            if lifecycle.validation is not None:
                return lifecycle.validation.clone()
            result = lifecycle.result
            accepted = result.success_mask.clone()
            runtime_success = result.success_mask.clone()
            eligible_before = (
                torch.ones_like(accepted)
                if self._eligible_mask is None
                else self._eligible_mask.clone()
            )
            if self._eligible_mask is not None:
                accepted &= self._eligible_mask
            if lifecycle.post_policy_success is not None:
                accepted &= lifecycle.post_policy_success
            validators = tuple(segment.validators)
            if validators and self._validator_port is None:
                raise TaskProgramBridgeError(
                    f"Segment {segment.segment_id!r} declares validators, but no "
                    "SegmentValidatorPort was installed."
                )
            validator_traces: list[dict[str, Any]] = []
            for validator_index, validator in enumerate(validators):
                assert self._validator_port is not None
                value = self._validator_port.validate(validator, segment=segment)
                validator_result = _normalize_validation(
                    value,
                    batch_size=result.env_ids.numel(),
                    device=result.env_ids.device,
                )
                accepted &= validator_result
                cfg = getattr(validator, "cfg", None)
                trace: dict[str, Any] = {
                    "validator_index": validator_index,
                    "kind": getattr(cfg, "kind", type(validator).__name__),
                    "source_path": list(getattr(validator, "source_path", ())),
                    "result_mask": validator_result.detach().cpu().tolist(),
                    "result": None,
                }
                trace["result"] = self._validator_port.validator_metadata(
                    validator,
                    segment=segment,
                )
                validator_traces.append(
                    _json_safe_copy(
                        trace,
                        field_name=f"validator {validator_index} metadata",
                    )
                )
            lifecycle.metadata["validation"] = _json_safe_copy(
                {
                    "env_ids": result.env_ids.detach().cpu().tolist(),
                    "runtime_success_mask": runtime_success.detach().cpu().tolist(),
                    "eligible_mask_before_validation": eligible_before.detach()
                    .cpu()
                    .tolist(),
                    "post_policy_success_mask": (
                        None
                        if lifecycle.post_policy_success is None
                        else lifecycle.post_policy_success.detach().cpu().tolist()
                    ),
                    "validators": validator_traces,
                    "accepted_mask": accepted.detach().cpu().tolist(),
                },
                field_name="segment validation metadata",
            )
            self._retain_eligible_rows(accepted)
            lifecycle.validation = accepted.clone()
            return accepted.clone()

        return validate


__all__: list[str] = []
