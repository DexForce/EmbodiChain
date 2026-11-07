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

"""Source-independent collection, admission and persistence limits."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import fields
import math
from typing import Any, get_args, get_origin, get_type_hints
from embodichain.utils import configclass

__all__ = [
    "ExpansionSessionCfg",
    "ExpansionExecutionCfg",
    "ExpansionValidationCfg",
    "ExpansionCollectionCfg",
    "ExpansionPersistenceCfg",
]


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )


def _count(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _boolean(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty registered ID")


def _fixed(value: str, supported: str, name: str) -> None:
    if value != supported:
        raise ValueError(f"{name} currently supports only {supported!r}")


def _decode_tuple_value(
    value: object,
    *,
    member_type: type,
    name: str,
) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    if any(type(item) is not member_type for item in value):
        raise ValueError(f"{name} contains an invalid value type")
    return tuple(value)


def _decode(cls: type, data: Mapping[str, Any], path: str = "") -> Any:
    if not isinstance(data, Mapping):
        raise ValueError(f"{path or cls.__name__} must be a mapping")
    hints = get_type_hints(cls)
    allowed = {field.name for field in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"unknown fields in {path or cls.__name__}: {sorted(unknown, key=str)}"
        )
    decoded = {}
    for key, value in data.items():
        expected = hints[key]
        name = f"{path}.{key}" if path else key
        if hasattr(expected, "__dataclass_fields__"):
            decoded[key] = _decode(expected, value, name)
        elif get_origin(expected) is tuple:
            member = get_args(expected)[0]
            # ``spatial.method`` used to name a single method, so a bare string
            # still decodes there. Every other sequence field requires a
            # sequence, which keeps its own schema check from being skipped.
            if key in getattr(cls, "__tuple_shorthand__", ()) and isinstance(
                value, str
            ):
                value = (value,)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{name} must be a sequence")
            if any(
                type(item) not in ((float, int) if member is float else (member,))
                for item in value
            ):
                raise ValueError(f"{name} contains an invalid value type")
            decoded[key] = tuple(value)
        elif get_origin(expected) is dict:
            key_type, value_type = get_args(expected)
            if key_type is not str or not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a string-keyed mapping")
            if any(type(item_key) is not str for item_key in value):
                raise ValueError(f"{name} must use string keys")
            if get_origin(value_type) is tuple:
                member_type = get_args(value_type)[0]
                decoded[key] = {
                    item_key: _decode_tuple_value(
                        item_value,
                        member_type=member_type,
                        name=f"{name}.{item_key}",
                    )
                    for item_key, item_value in value.items()
                }
            elif value_type is str:
                if any(type(item_value) is not str for item_value in value.values()):
                    raise ValueError(f"{name} must map strings to strings")
                decoded[key] = dict(value)
            else:
                raise TypeError(f"unsupported config mapping annotation: {expected}")
        elif expected is float and type(value) in (float, int):
            decoded[key] = float(value)
        elif type(value) is expected:
            decoded[key] = value
        else:
            raise ValueError(f"{name} has an invalid type")
    return cls(**decoded)


@configclass
class ExpansionExecutionCfg:
    """Bound the ready pool and retain the supported single-cohort execution policy."""

    pool_mode: str = "per_env_case"
    scheduler: str = "full_batch"
    ready_low_watermark: int = 1
    ready_high_watermark: int = 16
    ready_max_bytes: int = 268435456
    max_inflight: int = 1
    overlap_planning_and_physics: bool = False

    def __post_init__(self) -> None:
        _fixed(self.pool_mode, "per_env_case", "execution.pool_mode")
        _fixed(self.scheduler, "full_batch", "execution.scheduler")
        _count(self.ready_low_watermark, "ready_low_watermark", 0)
        _count(self.ready_high_watermark, "ready_high_watermark")
        _count(self.ready_max_bytes, "ready_max_bytes")
        _count(self.max_inflight, "execution.max_inflight")
        if self.max_inflight != 1:
            raise ValueError("execution.max_inflight currently supports only 1")
        if self.ready_low_watermark >= self.ready_high_watermark:
            raise ValueError(
                "ready_low_watermark must be less than ready_high_watermark"
            )
        _boolean(self.overlap_planning_and_physics, "overlap_planning_and_physics")
        if self.overlap_planning_and_physics:
            raise ValueError("overlapping planning and physics is not implemented")


@configclass
class ExpansionValidationCfg:
    """Required evidence gates for planning and measured expert acceptance."""

    validator_id: str = "task_success"
    profile_id: str = "verified_motion"
    motion_limits_profile_id: str = "robot_execution_limits"
    require_path_collision: bool = True
    require_task_success: bool = True
    path_length_ratio_max: float = 1.5
    duration_ratio_max: float = 1.5

    def __post_init__(self) -> None:
        for name in ("validator_id", "profile_id", "motion_limits_profile_id"):
            _id(getattr(self, name), name)
        for name in ("require_path_collision", "require_task_success"):
            _boolean(getattr(self, name), name)
            if not getattr(self, name):
                raise ValueError(f"expert collection requires {name}")
        _positive(self.path_length_ratio_max, "path_length_ratio_max")
        _positive(self.duration_ratio_max, "duration_ratio_max")


@configclass
class ExpansionCollectionCfg:
    """Bound proposals, physical attempts, confirmed targets and monotonic elapsed time."""

    target_committed_episodes: int = 1
    max_proposals: int = 100
    max_rollout_attempts: int = 100
    max_attempts_per_candidate: int = 1
    max_wall_time_s: float = 60.0

    def __post_init__(self) -> None:
        for name in (
            "target_committed_episodes",
            "max_proposals",
            "max_rollout_attempts",
            "max_attempts_per_candidate",
        ):
            _count(getattr(self, name), name)
        _positive(self.max_wall_time_s, "max_wall_time_s")
        if self.max_attempts_per_candidate != 1:
            raise ValueError(
                "rollout retries are not implemented; max_attempts_per_candidate must be 1"
            )
        if self.target_committed_episodes > min(
            self.max_proposals, self.max_rollout_attempts
        ):
            raise ValueError(
                "collection budgets cannot be smaller than the committed target"
            )


@configclass
class ExpansionPersistenceCfg:
    """Single-writer persistence policy and outstanding immutable-payload limits."""

    sink: str = "lerobot"
    accepted_only: bool = True
    async_write: bool = False
    pending_episode_limit: int = 8
    pending_max_bytes: int = 536870912
    save_audit: bool = True
    split_unit: str = "trajectory_family"

    def __post_init__(self) -> None:
        _id(self.sink, "sink")
        for name in ("accepted_only", "async_write", "save_audit"):
            _boolean(getattr(self, name), name)
        if not self.accepted_only or not self.save_audit:
            raise ValueError("expert collection requires accepted_only and save_audit")
        if self.async_write:
            raise ValueError("asynchronous persistence is not implemented")
        _count(self.pending_episode_limit, "pending_episode_limit")
        _count(self.pending_max_bytes, "pending_max_bytes")
        _fixed(self.split_unit, "trajectory_family", "split_unit")


def _validate_nested(value: Any, expected: type, name: str) -> None:
    if not isinstance(value, expected):
        raise ValueError(
            f"{name} must use its typed configuration; use from_mapping for dictionaries"
        )
    # Re-decode a plain mapping to reject mutated nested types and unsupported values.
    _decode(expected, value.to_dict(), name)


@configclass
class ExpansionSessionCfg:
    """Limits for one value-only expansion session, independent of trajectory algorithms.

    Args:
        seed: Logical candidate random seed, independent of physical slots.
        execution: Ready-pool bounds and supported cohort lifecycle.
        validation: Required measured acceptance evidence.
        collection: Attempt, target and elapsed-time budgets.
        persistence: Outstanding artifact-count and byte bounds.
    """

    seed: int = 0
    execution: ExpansionExecutionCfg = ExpansionExecutionCfg()
    validation: ExpansionValidationCfg = ExpansionValidationCfg()
    collection: ExpansionCollectionCfg = ExpansionCollectionCfg()
    persistence: ExpansionPersistenceCfg = ExpansionPersistenceCfg()

    def __post_init__(self) -> None:
        _count(self.seed, "seed", 0)
        for name in ("execution", "validation", "collection", "persistence"):
            _validate_nested(
                getattr(self, name), get_type_hints(type(self))[name], name
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ExpansionSessionCfg:
        """Decode nested session settings, rejecting unsupported fields and values."""
        return _decode(cls, data)
