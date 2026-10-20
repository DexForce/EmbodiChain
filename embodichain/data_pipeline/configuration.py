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

"""Derive bounded ODS consumption and cache settings from measured capacities.

This module plans settings; the sampler and consumer must enforce the returned
quota and active-rate cap. Profile qualification is supplied by the caller and
is never inferred from throughput. All counts refer to chunk selections, not
whole-trajectory passes or supervised frame appearances.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from fractions import Fraction
import math
from typing import Literal

from embodichain.utils import configclass

__all__ = [
    "ODSReuseProfileCfg",
    "ODSRuntimeCfg",
    "ODSConfiguration",
    "configure_ods",
    "ODSRateMeasurementCfg",
    "ODSThroughputMeasurement",
    "measure_ods_throughput",
]

_MAX_COUNT = (1 << 63) - 1


@configclass
class ODSReuseProfileCfg:
    """Explicit task/model/sampler qualification and selection-budget bounds.

    ``scope`` identifies task, model, history/horizon and trajectory-length
    assumptions. ``qualified=False`` is a supported provisional profile, not a
    claim that its recommended bounds are scientifically safe.
    """

    profile_id: str = MISSING
    scope: str = MISSING
    qualified: bool = MISSING
    min_chunks_per_episode: int = MISSING
    max_chunks_per_episode: int = MISSING
    hard_max_chunks_per_episode: int = MISSING
    latency_safety_factor: float = 1.25


@configclass
class ODSRuntimeCfg:
    """Measured runtime settings and optional total cache-storage allowance.

    ``cache_memory_bytes`` and ``bytes_per_episode`` must be supplied together.
    The allowance is for episode storage only; the caller must reserve worker,
    model and allocator memory separately. B and P are preserved unchanged.
    """

    batch_size: int = MISSING
    producer_envs: int = MISSING
    first_publication_p95_seconds: float = MISSING
    cache_memory_bytes: int | None = None
    bytes_per_episode: int | None = None
    min_cache_episodes: int = 1
    measurement_scope: str = "unspecified"


@dataclass(frozen=True)
class ODSConfiguration:
    """Derived settings and their qualifications; no runtime state is changed."""

    episode_sample_budget: int
    episode_sample_hard_limit: int
    balance_chunks_per_episode: float
    batch_size: int
    producer_envs: int
    buffer_size: int
    cache_bytes: int | None
    initial_chunk_budget: int
    initial_max_updates: int
    initial_remainder_chunks: int
    refill_remaining_chunks: int
    active_chunks_per_second_cap: float
    training_chunks_per_second_target: float
    training_updates_per_second_target: float
    publication_episodes_per_second_target: float
    training_duty_fraction: float
    producer_duty_fraction: float
    expected_training_throttle_fraction: float
    expected_producer_throttle_fraction: float
    profile_id: str
    profile_scope: str
    profile_qualified: bool
    measurement_scope: str
    reasons: tuple[str, ...]
    end_of_supply_rule: str = (
        "Stop when remaining unlocked per-episode quota cannot fund one full batch."
    )


def _integer(name: str, value: object, *, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_COUNT:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_MAX_COUNT}]")
    return value


def _number(name: str, value: object, *, zero: bool = False) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (value == 0 and not zero):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )
    return Fraction(value)


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _float(value: Fraction, name: str, *, conservative: bool = False) -> float:
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} exceeds finite floating-point range") from None
    if not math.isfinite(result) or (value > 0 and result == 0):
        raise ValueError(
            f"{name} is outside representable positive floating-point range"
        )
    if conservative and Fraction(result) > value:
        result = math.nextafter(result, 0.0)
        if value > 0 and result == 0:
            raise ValueError(f"{name} cannot be represented conservatively")
    return result


def _round_up(value: int, multiple: int) -> int:
    return _integer(
        "derived cache episodes", ((value + multiple - 1) // multiple) * multiple
    )


def configure_ods(
    accepted_episodes_per_second: float,
    training_chunks_per_second: float,
    *,
    profile: ODSReuseProfileCfg,
    runtime: ODSRuntimeCfg,
) -> ODSConfiguration:
    """Plan quota-protected ODS settings from two concurrent capacity estimates.

    Args:
        accepted_episodes_per_second: Published successful trajectories per
            production second, including retry cost; not an idle whole-run rate.
        training_chunks_per_second: Completed chunk selections per active learner
            second, including sampling, transfers and optimization.
        profile: Explicit qualified or provisional selection-budget policy.
        runtime: Batch, writer parallelism, latency and storage metadata.

    Returns:
        Budget, active and average pacing targets, cache and refill controls.
        The cache heuristic assumes one writer batch is unavailable. Its active
        cap must be enforced over short windows, not as post-burst average sleep.

    Raises:
        ValueError: Invalid metadata, unrepresentable derived values, or memory
            too small for the writer, diversity and a full unlocked batch.
        TypeError: Profile or runtime has the wrong configuration type.
    """
    if not isinstance(profile, ODSReuseProfileCfg) or not isinstance(
        runtime, ODSRuntimeCfg
    ):
        raise TypeError(
            "profile and runtime must use the typed ODS configuration classes"
        )
    profile.validate()
    runtime.validate()
    g = _number("accepted_episodes_per_second", accepted_episodes_per_second)
    t = _number("training_chunks_per_second", training_chunks_per_second)
    minimum = _integer("min_chunks_per_episode", profile.min_chunks_per_episode)
    maximum = _integer("max_chunks_per_episode", profile.max_chunks_per_episode)
    hard = _integer("hard_max_chunks_per_episode", profile.hard_max_chunks_per_episode)
    if not minimum <= maximum <= hard:
        raise ValueError(
            "Require min_chunks_per_episode <= max_chunks_per_episode <= hard cap"
        )
    if type(profile.qualified) is not bool:
        raise ValueError("qualified must be explicitly True or False")
    identity = _text("profile_id", profile.profile_id)
    scope = _text("profile scope", profile.scope)
    measurement_scope = _text("measurement_scope", runtime.measurement_scope)
    batch = _integer("batch_size", runtime.batch_size)
    producers = _integer("producer_envs", runtime.producer_envs)
    diversity = _integer("min_cache_episodes", runtime.min_cache_episodes)
    latency = _number(
        "first_publication_p95_seconds", runtime.first_publication_p95_seconds
    )
    safety = _number("latency_safety_factor", profile.latency_safety_factor)
    if safety < 1:
        raise ValueError("latency_safety_factor must be at least one")
    lead_seconds = latency * safety
    balance = t / g
    balance_float = _float(balance, "balance_chunks_per_episode")
    budget = min(max(math.floor(balance), minimum), maximum)
    reasons = []
    if not profile.qualified:
        reasons.append(
            "Provisional profile: sampling bounds require independent qualification."
        )
    if balance > maximum:
        reasons.append(
            "Supply cannot sustain learner capacity within the recommended selection budget."
        )
    elif balance < minimum:
        reasons.append(
            "Production is paced to preserve the profile's minimum planned selection budget."
        )
    # One complete batch must fit while P trajectories are locked by the writer.
    base_cache = _round_up(
        max(2 * producers, diversity, producers + math.ceil(Fraction(batch, budget))),
        producers,
    )
    desired_cache = _round_up(
        max(base_cache, producers + math.ceil(t * lead_seconds / budget)), producers
    )
    memory = runtime.cache_memory_bytes
    episode_bytes = runtime.bytes_per_episode
    if (memory is None) != (episode_bytes is None):
        raise ValueError(
            "cache_memory_bytes and bytes_per_episode must be supplied together"
        )
    cache = desired_cache
    if memory is not None:
        memory = _integer("cache_memory_bytes", memory)
        episode_bytes = _integer("bytes_per_episode", episode_bytes)
        capacity = (memory // episode_bytes // producers) * producers
        if capacity < base_cache:
            raise ValueError(
                "Infeasible cache memory: cannot retain writer space, diversity and one full unlocked batch"
            )
        cache = min(cache, capacity)
        if cache < desired_cache:
            reasons.append(
                "Cache memory limits short-window active consumption; enforce the returned active-rate cap."
            )
    unlocked_budget = _integer("unlocked chunk budget", (cache - producers) * budget)
    initial = _integer("initial chunk budget", cache * budget)
    active = min(t, Fraction(unlocked_budget) / lead_seconds)
    # Round a memory-constrained public rate down, never above its exact bound.
    active_float = _float(active, "active training cap", conservative=True)
    active = Fraction(active_float)
    target = min(active, g * budget)
    publication_target = target / budget
    threshold = _integer(
        "refill remaining chunks", max(batch, math.ceil(active * lead_seconds))
    )
    if threshold > unlocked_budget:
        raise ValueError("Refill threshold exceeds unlocked quota")
    reasons.append(
        "Refill sizing is a latency heuristic; validate subsequent publication spacing and remeasure after pacing changes."
    )
    if measurement_scope == "unspecified":
        reasons.append(
            "Measurement scope is unspecified; concurrent saturation capacity is not established."
        )
    return ODSConfiguration(
        episode_sample_budget=budget,
        episode_sample_hard_limit=hard,
        balance_chunks_per_episode=balance_float,
        batch_size=batch,
        producer_envs=producers,
        buffer_size=cache,
        cache_bytes=(
            None
            if episode_bytes is None
            else _integer("cache bytes", cache * episode_bytes)
        ),
        initial_chunk_budget=initial,
        initial_max_updates=initial // batch,
        initial_remainder_chunks=initial % batch,
        refill_remaining_chunks=threshold,
        active_chunks_per_second_cap=active_float,
        training_chunks_per_second_target=_float(
            target, "training target", conservative=True
        ),
        training_updates_per_second_target=_float(
            target / batch, "update target", conservative=True
        ),
        publication_episodes_per_second_target=_float(
            publication_target, "publication target", conservative=True
        ),
        training_duty_fraction=_float(target / t, "training duty"),
        producer_duty_fraction=_float(publication_target / g, "producer duty"),
        expected_training_throttle_fraction=_float(1 - target / t, "training throttle"),
        expected_producer_throttle_fraction=_float(
            1 - publication_target / g, "producer throttle"
        ),
        profile_id=identity,
        profile_scope=scope,
        profile_qualified=profile.qualified,
        measurement_scope=measurement_scope,
        reasons=tuple(reasons),
    )


@configclass
class ODSRateMeasurementCfg:
    """Count and active duration within an explicit absolute monotonic window."""

    unit: Literal["accepted_episodes", "completed_chunks"] = MISSING
    count: int = MISSING
    start_monotonic_seconds: float = MISSING
    end_monotonic_seconds: float = MISSING
    active_seconds: float = MISSING


@dataclass(frozen=True)
class ODSThroughputMeasurement:
    """Active capacities and realized wall rates, kept distinct by unit."""

    accepted_episodes_per_active_second: float
    accepted_episodes_per_wall_second: float
    training_chunks_per_active_second: float
    training_chunks_per_wall_second: float
    window_seconds: float


def measure_ods_throughput(
    production: ODSRateMeasurementCfg, training: ODSRateMeasurementCfg
) -> ODSThroughputMeasurement:
    """Calculate rates for matching monotonic windows with explicit count units.

    Counts may be zero; active durations must be positive and no longer than the
    window. A shared window alone does not prove simultaneous saturation: callers
    must record production/learner overlap and measurement scope separately.
    """
    rates = []
    windows = []
    for measurement, expected in (
        (production, "accepted_episodes"),
        (training, "completed_chunks"),
    ):
        if not isinstance(measurement, ODSRateMeasurementCfg):
            raise TypeError("measurement must be ODSRateMeasurementCfg")
        measurement.validate()
        if measurement.unit != expected:
            raise ValueError(f"Expected measurement unit {expected}")
        count = _integer("count", measurement.count, minimum=0)
        start = _number(
            "start_monotonic_seconds", measurement.start_monotonic_seconds, zero=True
        )
        end = _number(
            "end_monotonic_seconds", measurement.end_monotonic_seconds, zero=True
        )
        active = _number("active_seconds", measurement.active_seconds)
        if end <= start or active > end - start:
            raise ValueError(
                "Measurement window must be positive and contain its active duration"
            )
        windows.append((start, end))
        rates.extend(
            (
                _float(Fraction(count) / active, "active rate"),
                _float(Fraction(count) / (end - start), "wall rate"),
            )
        )
    if windows[0] != windows[1]:
        raise ValueError(
            "Production and training require the same absolute monotonic window"
        )
    return ODSThroughputMeasurement(
        *rates, window_seconds=_float(windows[0][1] - windows[0][0], "window seconds")
    )
