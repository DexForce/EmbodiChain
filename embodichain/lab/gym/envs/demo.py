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

"""Segment-aware expert demonstration protocol and executor."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, Literal

import torch
from tensordict import TensorDict

from embodichain.lab.sim.types import EnvAction

__all__ = [
    "DEMO_ANNOTATION_KEYS",
    "DEMO_SCHEMA_VERSION",
    "DemoEpisodeResult",
    "DemoSegment",
    "DemoSegmentResult",
    "ProcessedEnvAction",
    "execute_demo_episode",
    "resolve_demo_segments",
]

DEMO_SCHEMA_VERSION = 2
"""Current version of the segment-aware demonstration metadata schema."""

DEMO_ANNOTATION_KEYS = (
    "valid",
    "episode_step",
    "segment_id",
    "segment_step",
    "segment_start",
    "segment_end",
    "terminated",
    "truncated",
)
"""Per-frame annotation keys stored in expert rollout buffers."""


def _json_safe_copy(value: Any, *, field_name: str) -> Any:
    """Return an owned JSON value without implicit type coercion."""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key or key != key.strip():
                raise ValueError(
                    f"{field_name} mapping keys must be non-empty strings "
                    "without outer whitespace."
                )
            result[key] = _json_safe_copy(
                item,
                field_name=f"{field_name}.{key}",
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_copy(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field_name} contains non-JSON value {type(value).__name__}.")


@dataclass(frozen=True, slots=True, eq=False)
class ProcessedEnvAction:
    """Owned controller-ready action that must still pass through ``env.step``.

    Semantic runtimes and demonstration bridges may already have produced the
    action-manager output (for example, a full joint-position command assembled
    from typed runtime endpoints). Wrapping it prevents the environment from
    applying the pre-action transform a second time while retaining the normal
    simulation, manager, recorder, reward, and dataset step lifecycle.

    Args:
        value: Controller-ready tensor or ``TensorDict``.
        metadata: JSON-compatible provenance attached by the producer. The
            environment does not interpret this mapping.
    """

    value: EnvAction
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, (torch.Tensor, TensorDict)):
            raise TypeError("value must be a torch.Tensor or TensorDict.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        owned_value = self.value.clone()
        owned_metadata = _json_safe_copy(self.metadata, field_name="metadata")
        object.__setattr__(self, "value", owned_value)
        object.__setattr__(self, "metadata", MappingProxyType(owned_metadata))

    def snapshot(self) -> ProcessedEnvAction:
        """Return an independently owned processed-action envelope."""
        return ProcessedEnvAction(value=self.value, metadata=self.metadata)


@dataclass(frozen=True)
class DemoSegment:
    """One semantic subtask inside a demonstration episode.

    The action iterable may be lazy. This lets a task yield one segment, wait
    for it to execute, inspect the updated scene, and only then plan the next
    segment.

    Args:
        actions: Actions for this segment.
        name: Stable human-readable segment name.
        target_uid: Optional scene entity manipulated by this segment.
        instruction: Optional language instruction specific to this segment.
        metadata: Additional JSON-compatible task metadata.
        validator: Optional zero-argument callback that validates this segment
            after its actions are exhausted. It must return one boolean per
            parallel environment (or one scalar broadcast to every environment).
            Gym ``terminated`` and ``truncated`` remain episode-level signals;
            use this callback for subtask-level validation.
        abort_actions: Optional callback invoked when the executor stops after
            retrieving an action but before exhausting the iterable. It receives
            a reason and ``last_action_consumed`` flag, and must return any
            emergency controller actions that still need ordinary ``env.step``
            consumption. This is the explicit cancellation handshake for lazy
            runtimes whose command acknowledgements only mean locally buffered.
        failure_policy: ``"batch_abort"`` preserves legacy batch-atomic
            behavior. ``"row_independent"`` permanently freezes only failed
            environment rows while peers continue through the shared segment
            and later lazy segments.
    """

    actions: Iterable[Any]
    name: str = "segment"
    target_uid: str | None = None
    instruction: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validator: Callable[[], Any] | None = field(default=None, repr=False, compare=False)
    abort_actions: Callable[..., Iterable[Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    failure_policy: Literal["batch_abort", "row_independent"] = "batch_abort"

    def __post_init__(self) -> None:
        if self.abort_actions is not None and not callable(self.abort_actions):
            raise TypeError("abort_actions must be callable or None.")
        if self.failure_policy not in {"batch_abort", "row_independent"}:
            raise ValueError(
                "failure_policy must be 'batch_abort' or 'row_independent'."
            )


@dataclass(frozen=True)
class DemoSegmentResult:
    """Execution result and half-open frame range for one segment.

    Scalar span and status fields are batch aggregates kept for compatibility.
    The tuple fields preserve each vector-environment row independently.

    Args:
        segment_id: Zero-based segment index within the episode.
        name: Stable segment name supplied by the task.
        start_step: Earliest participating row start, inclusive.
        end_step: Latest participating row end, exclusive.
        success: Whether every participating row completed the segment.
        target_uid: Optional manipulated scene entity.
        instruction: Optional language instruction.
        failure_reason: First aggregate failure reason, if any.
        metadata: Additional JSON-compatible task metadata.
        active: Participation mask captured at segment start.
        start_steps: Per-environment inclusive starts.
        end_steps: Per-environment exclusive ends.
        successes: Per-environment segment status.
        failure_reasons: Per-environment failure reasons.
    """

    segment_id: int
    name: str
    start_step: int
    end_step: int
    success: bool
    target_uid: str | None = None
    instruction: str | None = None
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    active: tuple[bool, ...] = ()
    start_steps: tuple[int, ...] = ()
    end_steps: tuple[int, ...] = ()
    successes: tuple[bool, ...] = ()
    failure_reasons: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        owned_metadata = _json_safe_copy(
            self.metadata,
            field_name="segment result metadata",
        )
        object.__setattr__(self, "metadata", MappingProxyType(owned_metadata))

    def to_metadata(self, env_id: int | None = None) -> dict[str, Any]:
        """Return a JSON-compatible aggregate or per-environment representation.

        Args:
            env_id: Optional parallel-environment index. When provided, scalar
                spans and status are selected from the per-environment fields.

        Returns:
            JSON-compatible segment metadata.
        """
        metadata = {
            "segment_id": self.segment_id,
            "name": self.name,
            "target_uid": self.target_uid,
            "instruction": self.instruction,
            "metadata": _json_safe_copy(
                self.metadata,
                field_name="segment result metadata",
            ),
        }
        if env_id is not None and self.start_steps:
            metadata.update(
                {
                    "start_step": self.start_steps[env_id],
                    "end_step": self.end_steps[env_id],
                    "success": self.successes[env_id],
                    "failure_reason": self.failure_reasons[env_id],
                }
            )
            return metadata

        metadata.update(
            {
                "start_step": self.start_step,
                "end_step": self.end_step,
                "success": self.success,
                "failure_reason": self.failure_reason,
            }
        )
        if self.active:
            metadata.update(
                {
                    "active": list(self.active),
                    "start_steps": list(self.start_steps),
                    "end_steps": list(self.end_steps),
                    "successes": list(self.successes),
                    "failure_reasons": list(self.failure_reasons),
                }
            )
        return metadata


@dataclass(frozen=True)
class DemoEpisodeResult:
    """Result of executing all planned segments for one batched episode.

    Args:
        episode_index: Logical episode identifier.
        length: Maximum recorded row length.
        completed: Whether every environment completed successfully.
        success: Sticky per-environment success flags.
        terminated: Sticky per-environment Gym termination flags.
        truncated: Sticky per-environment Gym truncation flags.
        terminal_reason: Aggregate terminal reason.
        segments: Executed segment results.
        lengths: Independent per-environment recorded lengths.
        completed_by_env: Independent valid-completion flags.
        terminal_reasons: Independent terminal reasons.
    """

    episode_index: int
    length: int
    completed: bool
    success: tuple[bool, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    terminal_reason: str
    segments: tuple[DemoSegmentResult, ...] = ()
    lengths: tuple[int, ...] = ()
    completed_by_env: tuple[bool, ...] = ()
    terminal_reasons: tuple[str, ...] = ()

    @property
    def all_success(self) -> bool:
        """Whether every parallel environment completed successfully."""
        return bool(self.success) and all(self.success)

    @property
    def any_success(self) -> bool:
        """Whether at least one parallel environment completed successfully."""
        return any(self.success)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        metadata = {
            "schema_version": DEMO_SCHEMA_VERSION,
            "episode_index": self.episode_index,
            "length": self.length,
            "completed": self.completed,
            "success": list(self.success),
            "terminated": list(self.terminated),
            "truncated": list(self.truncated),
            "terminal_reason": self.terminal_reason,
            "segments": [segment.to_metadata() for segment in self.segments],
        }
        if self.lengths:
            metadata.update(
                {
                    "lengths": list(self.lengths),
                    "completed_by_env": list(self.completed_by_env),
                    "terminal_reasons": list(self.terminal_reasons),
                }
            )
        return metadata


ProgressWrapper = Callable[[Iterable[Any], str], Iterable[Any]]
StopPredicate = Callable[[], bool]


def _env_target(env: Any) -> Any:
    """Return the unwrapped environment when available."""
    return getattr(env, "unwrapped", env)


def _get_env_callable(env: Any, name: str) -> Callable[..., Any] | None:
    """Resolve an environment method through Gym wrappers when necessary."""
    getter = getattr(env, "get_wrapper_attr", None)
    if getter is not None:
        try:
            value = getter(name)
        except AttributeError:
            value = None
        if callable(value):
            return value

    value = getattr(_env_target(env), name, None)
    return value if callable(value) else None


def _as_bool_tuple(value: Any, num_envs: int) -> tuple[bool, ...]:
    """Normalize a scalar, sequence, or tensor to one flag per environment."""
    if value is None:
        return (False,) * num_envs
    tensor = torch.as_tensor(value, dtype=torch.bool).reshape(-1).cpu()
    if tensor.numel() == 1 and num_envs > 1:
        tensor = tensor.repeat(num_envs)
    if tensor.numel() != num_envs:
        raise ValueError(
            f"Expected {num_envs} environment flags, got {tensor.numel()}."
        )
    return tuple(bool(item) for item in tensor.tolist())


def _has_terminal_runtime_failure_trace(segment: DemoSegment) -> bool:
    """Return whether a lazy segment recorded a canonical failed runtime.

    Expert-program action iterables may terminate before yielding a controller
    command when planning fails.  Their bridge finalizes the runtime trace while
    exhausting the iterable and exposes a validator that commits row-local
    failure.  This marker distinguishes that outcome from an ordinary empty
    ``DemoSegment``, whose existing ``empty_segment`` guard remains unchanged.
    """
    runtime = segment.metadata.get("runtime")
    if not isinstance(runtime, Mapping):
        return False
    return (
        runtime.get("kind")
        in {
            "skill_result",
            "parallel_skill_result",
        }
        and runtime.get("status") == "failed"
    )


def _dataset_instruction(env: Any) -> str:
    """Return the dataset-level instruction used for legacy demo segments."""
    metadata = getattr(_env_target(env), "metadata", {})
    dataset_metadata = (
        metadata.get("dataset", {}) if isinstance(metadata, Mapping) else {}
    )
    instruction_cfg = (
        dataset_metadata.get("instruction")
        if isinstance(dataset_metadata, Mapping)
        else None
    )
    instruction = (
        instruction_cfg.get("lang")
        if isinstance(instruction_cfg, Mapping)
        else instruction_cfg
    )
    return str(instruction) if instruction else "unknown_task"


def resolve_demo_segments(env: Any, **kwargs: Any) -> Iterable[DemoSegment]:
    """Resolve a task's segment plan with legacy single-action-list fallback.

    Tasks implementing ``create_demo_segments`` own the number, order, and
    targets of segments. Older tasks that only implement
    ``create_demo_action_list`` are represented as one ``legacy`` segment.

    Args:
        env: Gym environment or wrapper.
        **kwargs: Planning arguments forwarded to the task method.

    Returns:
        A possibly lazy iterable of :class:`DemoSegment` objects.

    Raises:
        AttributeError: If the environment exposes neither planning API.
        TypeError: If a segment planner yields a value of the wrong type.
    """
    creator = _get_env_callable(env, "create_demo_segments")
    if creator is not None:
        segments = creator(**kwargs)
    else:
        legacy_creator = _get_env_callable(env, "create_demo_action_list")
        if legacy_creator is None:
            raise AttributeError(
                "Environment must implement create_demo_segments() or "
                "create_demo_action_list()."
            )
        actions = legacy_creator(**kwargs)
        segments = None if actions is None else (DemoSegment(actions, name="legacy"),)

    if segments is None:
        return ()
    if isinstance(segments, DemoSegment):
        segments = (segments,)

    fallback_instruction = _dataset_instruction(env)

    def _validate() -> Iterable[DemoSegment]:
        for segment in segments:
            if not isinstance(segment, DemoSegment):
                raise TypeError(
                    "create_demo_segments() must yield DemoSegment objects, "
                    f"got {type(segment).__name__}."
                )
            if segment.instruction is None:
                segment = replace(segment, instruction=fallback_instruction)
            yield segment

    return _validate()


def execute_demo_episode(
    env: Any,
    *,
    episode_index: int = 0,
    should_stop: StopPredicate | None = None,
    progress: ProgressWrapper | None = None,
    **plan_kwargs: Any,
) -> DemoEpisodeResult:
    """Plan and execute every segment in one environment episode.

    Auto-reset is suspended for the duration of execution. The caller owns the
    transaction boundary and must explicitly call ``env.reset()`` to commit a
    successful episode or ``env.reset(options={"save_data": False})`` to
    discard an invalid attempt.

    Args:
        env: Gym environment or wrapper.
        episode_index: Logical episode identifier used in metadata and logs.
        should_stop: Optional callback checked before every action.
        progress: Optional wrapper such as ``tqdm`` for action iterables.
        **plan_kwargs: Arguments forwarded to the task's planning method.

    Returns:
        A :class:`DemoEpisodeResult` describing segment spans and terminal
        state.
    """
    target = _env_target(env)
    num_envs = int(getattr(target, "num_envs", 1))
    begin_episode = _get_env_callable(env, "_begin_demo_episode_recording")
    begin_segment = _get_env_callable(env, "_begin_demo_segment_recording")
    end_segment = _get_env_callable(env, "_end_demo_segment_recording")
    end_episode = _get_env_callable(env, "_end_demo_episode_recording")
    normalize_action = _get_env_callable(env, "_normalize_demo_action")
    mask_action = _get_env_callable(env, "_mask_demo_action")
    set_active_mask = _get_env_callable(env, "_set_demo_active_mask")
    success_fn = _get_env_callable(env, "is_task_success")

    active = [True] * num_envs

    def publish_active_mask() -> None:
        """Publish executor liveness to recording hooks and action masking."""
        if set_active_mask is not None:
            set_active_mask(tuple(active))
            return
        previous = getattr(target, "_demo_active_mask", None)
        device = getattr(previous, "device", None)
        setattr(
            target,
            "_demo_active_mask",
            torch.tensor(active, dtype=torch.bool, device=device),
        )

    if begin_episode is not None:
        begin_episode(episode_index=episode_index)
    publish_active_mask()

    previous_no_auto_reset = bool(getattr(target, "_demo_no_auto_reset", False))
    setattr(target, "_demo_no_auto_reset", True)

    lengths = [0] * num_envs
    success = [False] * num_envs
    completed_by_env = [False] * num_envs
    terminated = [False] * num_envs
    truncated = [False] * num_envs
    terminal_reasons = ["pending"] * num_envs
    segment_results: list[DemoSegmentResult] = []
    last_info: Mapping[str, Any] = {}
    fatal_reason: str | None = None

    try:
        segment_count = 0
        segments = iter(resolve_demo_segments(env, **plan_kwargs))
        while any(active):
            if should_stop is not None and should_stop():
                fatal_reason = "interrupted"
                for env_id, is_active in enumerate(active):
                    if is_active:
                        terminal_reasons[env_id] = fatal_reason
                        active[env_id] = False
                publish_active_mask()
                break
            try:
                segment = next(segments)
            except StopIteration:
                break

            segment_id = segment_count
            segment_count += 1
            participants = tuple(active)
            start_steps = tuple(lengths)
            segment_successes = [False] * num_envs
            segment_failure_reasons: list[str | None] = [None] * num_envs
            segment_reason: str | None = None
            if begin_segment is not None:
                begin_segment(segment_id=segment_id, segment=segment)

            action_count = 0
            actions_exhausted = True
            actions: Iterable[Any] = segment.actions
            if actions is None:
                actions = ()
            if progress is not None:
                actions = progress(
                    actions,
                    f"Executing episode #{episode_index}, segment #{segment_id}: "
                    f"{segment.name}",
                )

            action_iterator = iter(actions)
            last_action_consumed: bool | None = None
            action_error: Exception | None = None
            while True:
                try:
                    action = next(action_iterator)
                except StopIteration:
                    break
                except Exception as exc:
                    action_error = exc
                    actions_exhausted = False
                    segment_reason = "action_generation_failed"
                    break
                last_action_consumed = False
                if should_stop is not None and should_stop():
                    actions_exhausted = False
                    fatal_reason = "interrupted"
                    segment_reason = fatal_reason
                    for env_id, is_active in enumerate(active):
                        if is_active:
                            terminal_reasons[env_id] = fatal_reason
                            segment_failure_reasons[env_id] = fatal_reason
                            active[env_id] = False
                    publish_active_mask()
                    break

                try:
                    if normalize_action is not None:
                        action = normalize_action(action)
                    if not all(active):
                        if mask_action is None:
                            raise RuntimeError(
                                "A vector demo environment completed asynchronously "
                                "but does not implement "
                                "_mask_demo_action(action, active_mask)."
                            )
                        action = mask_action(action, tuple(active))
                except Exception as exc:
                    action_error = exc
                    actions_exhausted = False
                    segment_reason = "action_processing_failed"
                    break

                active_before_step = tuple(active)
                try:
                    _, _, terminated_value, truncated_value, info = env.step(action)
                except Exception as exc:
                    action_error = exc
                    actions_exhausted = False
                    segment_reason = "action_execution_failed"
                    break
                last_action_consumed = True
                action_count += 1
                last_info = info
                for env_id, was_active in enumerate(active_before_step):
                    if was_active:
                        lengths[env_id] += 1

                step_terminated = _as_bool_tuple(terminated_value, num_envs)
                step_truncated = _as_bool_tuple(truncated_value, num_envs)
                step_success_source = last_info.get("success")
                if step_success_source is None and any(
                    step_terminated[env_id]
                    for env_id, was_active in enumerate(active_before_step)
                    if was_active
                ):
                    step_success_source = (
                        success_fn() if success_fn is not None else False
                    )
                step_success = _as_bool_tuple(step_success_source, num_envs)
                step_failure = _as_bool_tuple(last_info.get("fail"), num_envs)

                step_failed = False
                active_step_truncated = False
                for env_id, was_active in enumerate(active_before_step):
                    if not was_active:
                        continue
                    # Preserve both raw Gym flags when an environment reports
                    # terminated and truncated on the same transition.
                    terminated[env_id] |= step_terminated[env_id]
                    truncated[env_id] |= step_truncated[env_id]
                    if step_truncated[env_id]:
                        active_step_truncated = True
                        success[env_id] = False
                        terminal_reasons[env_id] = "truncated"
                        segment_failure_reasons[env_id] = "truncated"
                        active[env_id] = False
                        step_failed = True
                    elif step_failure[env_id]:
                        success[env_id] = False
                        terminal_reasons[env_id] = "failure"
                        segment_failure_reasons[env_id] = "failure"
                        active[env_id] = False
                        step_failed = True
                    elif step_terminated[env_id]:
                        active[env_id] = False
                        if step_success[env_id]:
                            success[env_id] = True
                            completed_by_env[env_id] = True
                            terminal_reasons[env_id] = "success"
                            segment_successes[env_id] = True
                        else:
                            success[env_id] = False
                            terminal_reasons[env_id] = "failure"
                            segment_failure_reasons[env_id] = "failure"
                            step_failed = True

                if step_failed:
                    if segment.failure_policy == "batch_abort":
                        actions_exhausted = False
                        fatal_reason = (
                            "truncated" if active_step_truncated else "failure"
                        )
                        segment_reason = fatal_reason
                        for env_id, is_active in enumerate(active):
                            if is_active:
                                terminal_reasons[env_id] = "batch_aborted"
                                segment_failure_reasons[env_id] = "batch_aborted"
                                active[env_id] = False
                    publish_active_mask()
                    if segment.failure_policy == "batch_abort" or not any(active):
                        actions_exhausted = False
                        break

                publish_active_mask()
                if not any(active):
                    # Every row reached episode-level success. Stop this segment
                    # and do not request another lazy segment.
                    actions_exhausted = False
                    break
                if should_stop is not None and should_stop():
                    actions_exhausted = False
                    fatal_reason = "interrupted"
                    segment_reason = fatal_reason
                    for env_id, is_active in enumerate(active):
                        if is_active:
                            terminal_reasons[env_id] = fatal_reason
                            segment_failure_reasons[env_id] = fatal_reason
                            active[env_id] = False
                    publish_active_mask()
                    break

            if not actions_exhausted:
                if segment.abort_actions is not None:
                    reason = (
                        segment_reason
                        or fatal_reason
                        or "demo segment execution stopped before exhaustion"
                    )
                    try:
                        emergency_actions = segment.abort_actions(
                            reason,
                            last_action_consumed=bool(last_action_consumed),
                        )
                        if isinstance(emergency_actions, (str, bytes)):
                            raise TypeError(
                                "abort_actions must return an iterable of actions."
                            )
                        emergency_iterator = iter(emergency_actions)
                        try:
                            for emergency_action in emergency_iterator:
                                if normalize_action is not None:
                                    emergency_action = normalize_action(
                                        emergency_action
                                    )
                                try:
                                    _, _, _, _, emergency_info = env.step(
                                        emergency_action
                                    )
                                except Exception as exc:
                                    raise RuntimeError(
                                        "Emergency demo safe-stop action failed "
                                        "during env.step()."
                                    ) from exc
                                action_count += 1
                                last_info = emergency_info
                                for env_id, is_participant in enumerate(participants):
                                    if is_participant:
                                        lengths[env_id] += 1
                        finally:
                            close_emergency = getattr(
                                emergency_iterator,
                                "close",
                                None,
                            )
                            if callable(close_emergency):
                                close_emergency()
                    finally:
                        close_actions = getattr(action_iterator, "close", None)
                        if callable(close_actions):
                            close_actions()
                else:
                    close_actions = getattr(action_iterator, "close", None)
                    if callable(close_actions):
                        close_actions()

            if action_error is not None:
                raise RuntimeError(
                    "Demo action generation, processing, or execution failed "
                    "after an emergency safe-stop attempt."
                ) from action_error

            traced_terminal_runtime_failure = (
                action_count == 0
                and actions_exhausted
                and segment_reason is None
                and segment.validator is not None
                and _has_terminal_runtime_failure_trace(segment)
            )
            if (
                action_count == 0
                and segment_reason is None
                and not traced_terminal_runtime_failure
            ):
                fatal_reason = "empty_segment"
                segment_reason = fatal_reason
                for env_id, is_participant in enumerate(participants):
                    if is_participant:
                        terminal_reasons[env_id] = fatal_reason
                        segment_failure_reasons[env_id] = fatal_reason
                        active[env_id] = False
                publish_active_mask()

            if actions_exhausted and segment_reason is None:
                validation = (
                    _as_bool_tuple(segment.validator(), num_envs)
                    if segment.validator is not None
                    else (True,) * num_envs
                )
                validation_failed = False
                for env_id, is_participant in enumerate(participants):
                    if not is_participant:
                        continue
                    if completed_by_env[env_id] and success[env_id]:
                        segment_successes[env_id] = True
                    elif active[env_id] and validation[env_id]:
                        segment_successes[env_id] = True
                    elif active[env_id]:
                        validation_failed = True
                        segment_failure_reasons[env_id] = "segment_validation_failed"
                        terminal_reasons[env_id] = "segment_validation_failed"

                if validation_failed:
                    if segment.failure_policy == "batch_abort":
                        fatal_reason = "segment_validation_failed"
                        segment_reason = fatal_reason
                        for env_id, is_active in enumerate(active):
                            if is_active:
                                if segment_failure_reasons[env_id] is None:
                                    segment_failure_reasons[env_id] = "batch_aborted"
                                    terminal_reasons[env_id] = "batch_aborted"
                                segment_successes[env_id] = False
                                active[env_id] = False
                    else:
                        for env_id, is_active in enumerate(active):
                            if (
                                is_active
                                and segment_failure_reasons[env_id]
                                == "segment_validation_failed"
                            ):
                                segment_successes[env_id] = False
                                active[env_id] = False
                    publish_active_mask()

            participant_ids = [
                env_id
                for env_id, is_participant in enumerate(participants)
                if is_participant
            ]
            segment_ok = bool(participant_ids) and all(
                segment_successes[env_id] for env_id in participant_ids
            )
            aggregate_failure = segment_reason or next(
                (
                    segment_failure_reasons[env_id]
                    for env_id in participant_ids
                    if segment_failure_reasons[env_id] is not None
                ),
                None,
            )
            end_steps = tuple(lengths)
            segment_result = DemoSegmentResult(
                segment_id=segment_id,
                name=segment.name,
                start_step=min(start_steps[env_id] for env_id in participant_ids),
                end_step=max(end_steps[env_id] for env_id in participant_ids),
                success=segment_ok,
                target_uid=segment.target_uid,
                instruction=segment.instruction,
                failure_reason=aggregate_failure,
                metadata=segment.metadata,
                active=participants,
                start_steps=start_steps,
                end_steps=end_steps,
                successes=tuple(segment_successes),
                failure_reasons=tuple(segment_failure_reasons),
            )
            segment_results.append(segment_result)
            if end_segment is not None:
                end_segment(result=segment_result)

            if segment_reason is not None or not any(active):
                break

        if segment_count == 0 and fatal_reason is None:
            fatal_reason = "empty_plan"
            for env_id, is_active in enumerate(active):
                if is_active:
                    terminal_reasons[env_id] = fatal_reason
                    active[env_id] = False
            publish_active_mask()
        elif fatal_reason is None and any(active):
            # Normal plan exhaustion validates only rows that have not already
            # reached sticky episode success. Legacy expert tasks use
            # is_task_success() for this final validation.
            success_source = (
                success_fn()
                if success_fn is not None
                else last_info.get("success", True)
            )
            final_success = _as_bool_tuple(success_source, num_envs)
            for env_id, is_active in enumerate(active):
                if not is_active:
                    continue
                success[env_id] = final_success[env_id]
                completed_by_env[env_id] = final_success[env_id]
                terminal_reasons[env_id] = (
                    "success" if final_success[env_id] else "task_incomplete"
                )
                active[env_id] = False
            publish_active_mask()

        completed = bool(completed_by_env) and all(completed_by_env)
        if fatal_reason is not None:
            terminal_reason = fatal_reason
        elif completed and all(success):
            terminal_reason = "success"
        else:
            terminal_reason = next(
                (reason for reason in terminal_reasons if reason != "success"),
                "task_incomplete",
            )

        result = DemoEpisodeResult(
            episode_index=episode_index,
            length=max(lengths, default=0),
            completed=completed,
            success=tuple(success),
            terminated=tuple(terminated),
            truncated=tuple(truncated),
            terminal_reason=terminal_reason,
            segments=tuple(segment_results),
            lengths=tuple(lengths),
            completed_by_env=tuple(completed_by_env),
            terminal_reasons=tuple(terminal_reasons),
        )
        if end_episode is not None:
            end_episode(result=result)
        return result
    finally:
        setattr(target, "_demo_no_auto_reset", previous_no_auto_reset)
