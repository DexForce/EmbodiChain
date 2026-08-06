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
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "DEMO_ANNOTATION_KEYS",
    "DEMO_SCHEMA_VERSION",
    "DemoEpisodeResult",
    "DemoSegment",
    "DemoSegmentResult",
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
    """

    actions: Iterable[Any]
    name: str = "segment"
    target_uid: str | None = None
    instruction: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DemoSegmentResult:
    """Execution result and half-open frame range for one segment."""

    segment_id: int
    name: str
    start_step: int
    end_step: int
    success: bool
    target_uid: str | None = None
    instruction: str | None = None
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "success": self.success,
            "target_uid": self.target_uid,
            "instruction": self.instruction,
            "failure_reason": self.failure_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DemoEpisodeResult:
    """Result of executing all planned segments for one batched episode."""

    episode_index: int
    length: int
    completed: bool
    success: tuple[bool, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    terminal_reason: str
    segments: tuple[DemoSegmentResult, ...] = ()

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
        return {
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

    def _validate() -> Iterable[DemoSegment]:
        for segment in segments:
            if not isinstance(segment, DemoSegment):
                raise TypeError(
                    "create_demo_segments() must yield DemoSegment objects, "
                    f"got {type(segment).__name__}."
                )
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

    if begin_episode is not None:
        begin_episode(episode_index=episode_index)

    previous_no_auto_reset = bool(getattr(target, "_demo_no_auto_reset", False))
    setattr(target, "_demo_no_auto_reset", True)

    total_steps = 0
    segment_results: list[DemoSegmentResult] = []
    terminated = (False,) * num_envs
    truncated = (False,) * num_envs
    last_info: Mapping[str, Any] = {}
    terminal_reason = "completed"
    completed = True

    try:
        segment_count = 0
        for segment_id, segment in enumerate(resolve_demo_segments(env, **plan_kwargs)):
            segment_count += 1
            start_step = total_steps
            if begin_segment is not None:
                begin_segment(segment_id=segment_id, segment=segment)

            action_count = 0
            segment_reason: str | None = None
            actions: Iterable[Any] = segment.actions
            if actions is None:
                actions = ()
            if progress is not None:
                actions = progress(
                    actions,
                    f"Executing episode #{episode_index}, segment #{segment_id}: "
                    f"{segment.name}",
                )

            for action in actions:
                if should_stop is not None and should_stop():
                    completed = False
                    terminal_reason = "interrupted"
                    segment_reason = terminal_reason
                    break

                if normalize_action is not None:
                    action = normalize_action(action)
                _, _, terminated_value, truncated_value, info = env.step(action)
                action_count += 1
                total_steps += 1
                last_info = info
                terminated = _as_bool_tuple(terminated_value, num_envs)
                truncated = _as_bool_tuple(truncated_value, num_envs)

                if any(terminated) or any(truncated):
                    completed = False
                    terminal_reason = "truncated" if any(truncated) else "terminated"
                    segment_reason = terminal_reason
                    break

            if action_count == 0 and segment_reason is None:
                completed = False
                terminal_reason = "empty_segment"
                segment_reason = terminal_reason

            segment_success = segment_reason is None
            if segment_reason == "terminated":
                segment_success = all(
                    _as_bool_tuple(last_info.get("success"), num_envs)
                )

            segment_result = DemoSegmentResult(
                segment_id=segment_id,
                name=segment.name,
                start_step=start_step,
                end_step=total_steps,
                success=segment_success,
                target_uid=segment.target_uid,
                instruction=segment.instruction,
                failure_reason=None if segment_success else segment_reason,
                metadata=segment.metadata,
            )
            segment_results.append(segment_result)
            if end_segment is not None:
                end_segment(result=segment_result)

            if segment_reason is not None:
                break

        if segment_count == 0:
            completed = False
            terminal_reason = "empty_plan"

        success_fn = _get_env_callable(env, "is_task_success")
        if any(terminated) or any(truncated):
            success_source = last_info.get("success")
            if success_source is None:
                success_source = success_fn() if success_fn is not None else False
        else:
            # Expert tasks historically validate the final planned state with
            # is_task_success(). EmbodiedEnv.get_info() always contains a
            # compute_task_state()-based success field, which remains false for
            # legacy tasks that only implement the expert-policy hook.
            success_source = (
                success_fn()
                if success_fn is not None
                else last_info.get("success", completed)
            )
        success = _as_bool_tuple(success_source, num_envs)

        if any(truncated):
            success = tuple(False for _ in success)
            terminal_reason = "truncated"
        elif any(terminated):
            if all(success):
                completed = True
                terminal_reason = "success"
            else:
                terminal_reason = "failure"
        elif completed and all(success):
            terminal_reason = "success"
        elif completed:
            terminal_reason = "task_incomplete"

        result = DemoEpisodeResult(
            episode_index=episode_index,
            length=total_steps,
            completed=completed,
            success=success,
            terminated=terminated,
            truncated=truncated,
            terminal_reason=terminal_reason,
            segments=tuple(segment_results),
        )
        if end_episode is not None:
            end_episode(result=result)
        return result
    finally:
        setattr(target, "_demo_no_auto_reset", previous_no_auto_reset)
