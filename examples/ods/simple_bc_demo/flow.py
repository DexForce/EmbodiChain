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

"""Read completed service intervals and aligned ODS publication observations."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Mapping, Sequence

__all__ = ["summarize_flow"]

_MAX_COUNT = (1 << 63) - 1


def _count(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_COUNT:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_MAX_COUNT}]")
    return value


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} is not finite") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _union(values: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    for start, end in sorted(values):
        if end == start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _duration(values: Sequence[tuple[float, float]]) -> float:
    return math.fsum(end - start for start, end in values)


def _intersect(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def _rate(count: int, seconds: float) -> float | None:
    if seconds == 0:
        return None
    result = count / seconds
    if not math.isfinite(result):
        raise ValueError("Derived throughput is not finite")
    return result


def summarize_flow(
    *,
    monotonic_origin_seconds: float,
    intervals: Mapping[str, Sequence[Sequence[float]]],
    refill_requests: Sequence[Mapping[str, Any]],
    episode_records: Sequence[Mapping[str, Any]],
    batch_size: int,
    updates: int,
    loop_start: float,
    loop_end: float,
    buffer_size: int,
) -> dict[str, Any]:
    """Summarize observed throughput without calling it saturated capacity.

    Args:
        monotonic_origin_seconds: Absolute monotonic origin shared by the parent
            and producer. Interval and request times are relative to this origin.
        intervals: Recorded intervals, including completed ``training_service``
            and distinct ``usage_throttle``, ``quota_wait`` and ``data_wait``.
        refill_requests: Events with kind ``refill_requested``, relative seconds
            and generation. At most one request per warm generation is allowed.
        episode_records: Usage summary episodes with absolute publication
            nanoseconds, generation and eligible-anchor count. Source identity
            fields, when supplied, must be complete and unique.
        batch_size: Delivered chunk selections per completed update.
        updates: Completed updates within this loop, excluding resumed history.
        loop_start: Relative beginning of the observed training loop.
        loop_end: Relative end of the observed training loop.
        buffer_size: Required accepted episode count for a complete refill.

    Returns:
        Counts, denominators, scope, active/wall learner rates, classified waits,
        complete productive refill rates and explicit excluded generations.
        Productive refill intervals are request-to-last-publication, so failed
        attempts and publication delays remain in their measured cost. Initial
        fill and refills outside the training window are excluded. These are
        observations under the actual scheduler, not saturation predictions.

    Raises:
        ValueError: Invalid counts/times, ambiguous requests, inconsistent clocks,
            incomplete identities or completed updates without completed service.
    """
    origin = _seconds(monotonic_origin_seconds, "monotonic_origin_seconds")
    start = _seconds(loop_start, "loop_start")
    end = _seconds(loop_end, "loop_end")
    if end < start:
        raise ValueError("loop_end cannot precede loop_start")
    batch = _count(batch_size, "batch_size", 1)
    update_count = _count(updates, "updates")
    capacity = _count(buffer_size, "buffer_size", 1)
    chunks = _count(batch * update_count, "completed chunks")
    if not isinstance(intervals, Mapping):
        raise ValueError("intervals must be a mapping")
    normalized = {}
    for name, values in intervals.items():
        rows = []
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name} intervals must be a sequence")
        for row in values:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError(f"{name} intervals must contain start/end pairs")
            begin = _seconds(row[0], f"{name} start")
            finish = _seconds(row[1], f"{name} end")
            if finish < begin:
                raise ValueError(f"{name} interval has reversed endpoints")
            if name == "training_service" and (begin < start or finish > end):
                raise ValueError(
                    "Completed training_service must lie inside the training window"
                )
            rows.append((begin, finish))
        normalized[name] = _union(rows)
    window = [(start, end)] if end > start else []
    selected = {
        name: _intersect(normalized.get(name, []), window)
        for name in ("training_service", "data_wait", "quota_wait", "usage_throttle")
    }
    service = selected["training_service"]
    service_seconds = _duration(service)
    service_wait = _duration(_intersect(service, selected["data_wait"]))
    active_seconds = max(0.0, service_seconds - service_wait)
    if update_count and (end == start or active_seconds <= 0):
        raise ValueError(
            "Completed updates require a positive completed-service denominator"
        )
    waits = {
        name: _duration(selected[name])
        for name in ("usage_throttle", "quota_wait", "data_wait")
    }
    wait_union = _union([row for name in waits for row in selected[name]])
    observed_union = _union(service + wait_union)

    if not isinstance(refill_requests, (list, tuple)) or not isinstance(
        episode_records, (list, tuple)
    ):
        raise ValueError("Refill requests and episode records must be sequences")
    requests = {}
    for event in refill_requests:
        if not isinstance(event, Mapping) or event.get("kind") != "refill_requested":
            raise ValueError("Refill events require kind=refill_requested")
        generation = _count(event.get("generation"), "request generation", 2)
        seconds = _seconds(event.get("seconds"), "request seconds")
        if generation in requests:
            raise ValueError("A generation has multiple refill request timestamps")
        requests[generation] = seconds

    generations: dict[int, list[tuple[float, int]]] = {}
    identities = set()
    for episode in episode_records:
        if not isinstance(episode, Mapping):
            raise ValueError("Episode records must be mappings")
        generation = _count(episode.get("generation"), "episode generation", 1)
        timestamp = _count(
            episode.get("published_at_monotonic_ns"), "publication nanoseconds"
        )
        anchors = _count(episode.get("eligible_anchors"), "eligible anchors")
        aligned = Fraction(timestamp, 1_000_000_000) - Fraction(origin)
        if aligned < 0:
            raise ValueError("Publication precedes the shared monotonic origin")
        seconds = float(aligned)
        if not math.isfinite(seconds):
            raise ValueError("Aligned publication time is not finite")
        if "run_id" in episode or "episode_id" in episode:
            identity = (
                _count(episode.get("run_id"), "run_id"),
                _count(episode.get("episode_id"), "episode_id"),
            )
            if identity in identities:
                raise ValueError("Duplicate source episode identity")
            identities.add(identity)
        generations.setdefault(generation, []).append((seconds, anchors))

    included = []
    excluded = []
    first_latencies = []
    production_intervals = []
    accepted = 0
    anchors_published = 0
    for generation in sorted(set(generations) | set(requests)):
        episodes = generations.get(generation, [])
        if len(episodes) > capacity:
            raise ValueError("Generation episode count exceeds configured buffer_size")
        request = requests.get(generation)
        first = min((time for time, _ in episodes), default=None)
        last = max((time for time, _ in episodes), default=None)
        latency = None
        if request is not None and first is not None:
            if first < request:
                raise ValueError(
                    "A refill publication precedes its request; check clock alignment"
                )
            latency = first - request
            if start <= request <= first <= end:
                first_latencies.append({"generation": generation, "seconds": latency})
        row = {
            "generation": generation,
            "accepted_episodes": len(episodes),
            "eligible_anchors": _count(
                sum(anchors for _, anchors in episodes), "generation eligible anchors"
            ),
            "request_seconds": request,
            "first_publication_seconds": first,
            "last_publication_seconds": last,
            "first_publication_latency_seconds": latency,
        }
        reasons = []
        if generation == 1:
            reasons.append("initial_fill")
        else:
            if request is None:
                reasons.append("unmatched_request")
            if len(episodes) != capacity:
                reasons.append("incomplete_generation")
            if request is not None and (
                request < start or request > end or (last is not None and last > end)
            ):
                reasons.append("outside_training_window")
        if reasons:
            excluded.append({**row, "reasons": reasons})
            continue
        if last <= request:
            raise ValueError(
                "Complete refill must have a positive request-to-publication duration"
            )
        production_intervals.append((request, last))
        included.append({**row, "productive_seconds": last - request})
        accepted += len(episodes)
        anchors_published += row["eligible_anchors"]
    accepted = _count(accepted, "productive accepted episodes")
    anchors_published = _count(anchors_published, "productive anchors")
    production_seconds = _duration(_union(production_intervals))
    wall_seconds = end - start
    return {
        "measurement": {
            "monotonic_origin_seconds": origin,
            "time_unit": "seconds on a shared system monotonic clock",
            "publication_timestamp_input_unit": "nanoseconds",
            "loop_start_seconds": start,
            "loop_end_seconds": end,
            "loop_wall_seconds": wall_seconds,
            "scope": "Observed under the actual learner/producer scheduler; not fully concurrent saturation capacity.",
            "limitations": [
                "Only complete matched warm refills inside the training window contribute to productive rates.",
                "Completed service excludes recorded data waits; bookkeeping outside training_service remains unattributed.",
                "Productive rates include request/publication overhead and retries but exclude idle gaps between refills.",
                "First-publication latency observations may include incomplete refills; small samples do not establish reliable tail latency.",
            ],
        },
        "learner": {
            "completed_updates": update_count,
            "chunk_unit": "one chunk consumed by a completed optimizer update",
            "update_unit": "one completed optimizer batch",
            "batch_chunks": batch,
            "completed_chunks": chunks,
            "service_seconds": service_seconds,
            "service_data_wait_seconds": service_wait,
            "active_seconds": active_seconds,
            "wall_seconds": wall_seconds,
            "active_chunks_per_second": _rate(chunks, active_seconds),
            "active_updates_per_second": _rate(update_count, active_seconds),
            "wall_chunks_per_second": _rate(chunks, wall_seconds),
            "wall_updates_per_second": _rate(update_count, wall_seconds),
            "unattributed_wall_seconds": max(
                0.0, wall_seconds - _duration(observed_union)
            ),
        },
        "waits": {
            **{name + "_seconds": seconds for name, seconds in waits.items()},
            "union_seconds": _duration(wait_union),
        },
        "producer": {
            "accepted_episodes": accepted,
            "episode_unit": "one accepted published trajectory",
            "anchor_unit": "one eligible chunk-start frame",
            "eligible_anchors": anchors_published,
            "productive_seconds": production_seconds,
            "accepted_episodes_per_productive_second": _rate(
                accepted, production_seconds
            ),
            "eligible_anchors_per_productive_second": _rate(
                anchors_published, production_seconds
            ),
            "included_generations": included,
            "excluded_generations": excluded,
            "first_publication_latencies": first_latencies,
            "first_publication_latency_observations": len(first_latencies),
        },
    }
