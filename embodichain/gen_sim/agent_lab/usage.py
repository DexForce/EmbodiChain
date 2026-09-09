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

"""Run-scoped host timing and reported Codex token counters, never text estimates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from collections.abc import Iterator
import json
import math
import os
from pathlib import Path
import threading
import time
from uuid import uuid4
import warnings

import psutil

from .catalog import write_json

__all__ = ["collect_usage", "refresh_usage", "UsageMeter"]

_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _events(path: Path) -> Iterator[dict]:
    try:
        stream = path.open()
    except OSError:
        return
    with stream:
        for line in stream:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
            except ValueError:
                # Interrupted writers may leave an incomplete trailing JSONL line.
                continue


def _header(path: Path) -> dict:
    try:
        with path.open() as stream:
            event = json.loads(stream.readline())
        value = event.get("payload") if isinstance(event, dict) else None
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _alive(record: dict) -> bool:
    try:
        process = psutil.Process(record["pid"])
        return (
            process.create_time() == record["created_at"]
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except (KeyError, psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _interval(record: dict, path: Path, now: float) -> tuple[float, float, str] | None:
    start = record.get("started_at", record.get("created_at"))
    if not _number(start):
        return None
    if _number(record.get("ended_at")):
        return start, max(start, record["ended_at"]), "reported"
    if _number(record.get("wall_seconds")):
        return start, start + max(0, record["wall_seconds"]), "reported"
    if _alive(record):
        return start, max(start, now), "partial"
    end = record.get("updated_at")
    return (start, max(start, min(end, now)), "partial") if _number(end) else None


def _union(intervals: list[tuple]) -> float:
    end = float("-inf")
    total = 0.0
    for start, stop, *_ in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return round(total, 6)


def _counts(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    result = {
        key: (
            value.get(key)
            if isinstance(value.get(key), int)
            and not isinstance(value[key], bool)
            and value[key] >= 0
            else None
        )
        for key in _TOKEN_FIELDS
    }
    if result["input_tokens"] is None or result["output_tokens"] is None:
        return None
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def _stamp(value: str | None) -> float | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def _rollout_turns(path: Path, until: float) -> list[dict]:
    turns = []
    current = None
    for event in _events(path):
        timestamp = _stamp(event.get("timestamp"))
        if timestamp is not None and timestamp > until:
            continue
        payload = event.get("payload", {})
        if event.get("type") == "turn_context":
            identity = payload.get("turn_id")
            if current is not None and current["turn_id"] == identity:
                continue
            current = {
                "turn_id": identity,
                "started_at": _stamp(event.get("timestamp")),
                "cwd": payload.get("cwd"),
                "observed_model": payload.get("model"),
                "observed_reasoning_effort": payload.get(
                    "effort",
                    (payload.get("collaboration_mode") or {})
                    .get("settings", {})
                    .get("reasoning_effort"),
                ),
                "first": None,
                "last": None,
                "values": None,
                "complete": False,
                "source": str(path),
            }
            turns.append(current)
        elif event.get("type") == "event_msg" and current is not None:
            if payload.get("type") == "token_count" and isinstance(
                payload.get("info"), dict
            ):
                info = payload["info"]
                total = _counts(info.get("total_token_usage"))
                if total:
                    if current["first"] is None:
                        current["first"] = total
                        current["first_increment"] = _counts(
                            info.get("last_token_usage")
                        )
                    previous_sample = current["last"]
                    if previous_sample and any(
                        total[k] < previous_sample[k]
                        for k in ("input_tokens", "output_tokens")
                    ):
                        current["counter_issue"] = (
                            "Cumulative counters decreased inside the turn; retained observed lower bound"
                        )
                    if (
                        previous_sample is None
                        or total["total_tokens"] >= previous_sample["total_tokens"]
                    ):
                        current["last"] = total
                    current["sampled_at"] = _stamp(event.get("timestamp"))
            elif (
                payload.get("type") == "task_complete"
                and payload.get("turn_id", current["turn_id"]) == current["turn_id"]
            ):
                current["complete"] = True
    previous = None
    for turn in turns:
        first, last = turn["first"], turn["last"]
        if first is None:
            continue
        increment = turn.get("first_increment")
        core = ("input_tokens", "output_tokens")
        if previous is None or (
            increment and all(first[k] == increment[k] for k in core)
        ):
            turn["values"] = last
        elif increment and all(first[k] == previous[k] + increment[k] for k in core):
            values = {
                k: (
                    last[k] - previous[k]
                    if last.get(k) is not None
                    and previous.get(k) is not None
                    and last[k] >= previous[k]
                    else None
                )
                for k in (*_TOKEN_FIELDS, "total_tokens")
            }
            if all(values[k] is not None for k in core):
                turn["values"] = values
        else:
            turn["counter_issue"] = (
                "Cannot determine whether cumulative counters reset or carried over"
            )
        previous = last
    return turns


def _find_rollout(
    home: Path, thread: str | None, root: Path, window: tuple
) -> Path | None:
    sessions = home / "sessions"
    if thread:
        # Filename-only lookup; never search unrelated conversation contents.
        return next(sessions.glob(f"**/*{thread}.jsonl"), None)
    first = datetime.fromtimestamp(window[0], timezone.utc).date() - timedelta(days=1)
    last = datetime.fromtimestamp(window[1], timezone.utc).date() + timedelta(days=1)
    while first <= last:
        for path in sorted((sessions / first.strftime("%Y/%m/%d")).glob("*.jsonl")):
            header = _header(path)
            timestamp = _stamp(header.get("timestamp"))
            if (
                header.get("cwd") == str(root / "workspace")
                and timestamp is not None
                and window[0] <= timestamp <= window[1]
            ):
                return path
        first += timedelta(days=1)
    return None


def _invocation_usage(
    root: Path, output: Path, window: tuple, home: Path
) -> list[dict]:
    thread = None
    current = -1
    completed = {}
    for event in _events(output / "stdout.log"):
        if event.get("type") == "thread.started":
            thread = event.get("thread_id")
        elif event.get("type") == "turn.started":
            current += 1
        elif event.get("type") == "turn.completed":
            values = _counts(event.get("usage"))
            if values:
                completed[max(0, current)] = values
    path = _find_rollout(home, thread, root, window)
    turns = []
    if path:
        header = _header(path)
        if thread is not None and header.get("id") != thread:
            path = None
        else:
            thread = header.get("id", thread)
    if path:
        turns = [
            turn
            for turn in _rollout_turns(path, window[1])
            if turn["cwd"] == str(root / "workspace")
            and turn["started_at"] is not None
            and window[0] <= turn["started_at"] <= window[1]
        ]
    result = []
    settings = _json(output / "agent_settings.json") or _json(output / "launch.json")
    for index in range(max(current + 1, len(completed), len(turns), 1)):
        turn = turns[index] if index < len(turns) else {}
        values = completed.get(index, turn.get("values"))
        reported = index in completed or (turn.get("complete") and values is not None)
        if turn.get("counter_issue"):
            reported = False
            if turn.get("values") is not None and (
                values is None
                or turn["values"]["total_tokens"] > values["total_tokens"]
            ):
                values = turn["values"]
        result.append(
            {
                "key": (
                    f"{thread or path}:{turn['turn_id']}"
                    if turn.get("turn_id")
                    else f"{output.name}:{index}"
                ),
                "invocation": output.name,
                "thread_id": thread,
                "turn_id": turn.get("turn_id"),
                "requested_model": settings.get("model"),
                "requested_reasoning_effort": settings.get("reasoning_effort"),
                "observed_model": turn.get("observed_model"),
                "observed_reasoning_effort": turn.get("observed_reasoning_effort"),
                "tokens": values,
                "status": (
                    "reported"
                    if reported
                    else "partial" if values is not None else "unavailable"
                ),
                "source": (
                    str(output / "stdout.log")
                    if index in completed
                    else str(path) if path else None
                ),
                "sampled_at": turn.get("sampled_at"),
                "note": turn.get("counter_issue")
                or (
                    "No complete CLI usage event or readable in-scope token snapshot"
                    if values is None
                    else None
                ),
            }
        )
    return result


def collect_usage(
    root: Path, *, codex_home: Path | None = None, now: float | None = None
) -> dict:
    """Collect this run's time intervals and linked Codex counters without inference.

    Cached input and reasoning output are subcounts, not additional total tokens.
    Incomplete invocations contribute observed lower bounds, never fabricated zeros.
    This excludes other chats, unlinked model calls and idle gaps between launches.
    """
    root = root.resolve()
    home = codex_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    now = time.time() if now is None else now
    intervals = {"codex": [], "simulation": [], "host": []}
    observations = {}
    sources = []
    missing_timing = False
    invocations = 0
    configurations = []
    for kind, pattern in (
        ("codex", "codex/*/process.json"),
        ("simulation", "attempts/*/process.json"),
        ("host", "usage_sessions/*.json"),
    ):
        seen = set()
        for path in sorted(root.glob(pattern)):
            if not path.resolve().is_relative_to(root):
                continue
            record = _json(path)
            identity = (
                (record["pid"], record["created_at"])
                if kind != "host" and "pid" in record and "created_at" in record
                else str(path)
            )
            if identity in seen:
                continue
            seen.add(identity)
            window = _interval(record, path, now)
            if window:
                intervals[kind].append(window)
                sources.append(
                    {
                        "kind": kind,
                        "path": str(path.relative_to(root)),
                        "start": window[0],
                        "end": window[1],
                        "status": window[2],
                    }
                )
            else:
                missing_timing = True
            if kind == "codex":
                invocations += 1
                rows = _invocation_usage(root, path.parent, window or (0, 0), home)
                settings = _json(path.parent / "agent_settings.json") or _json(
                    path.parent / "launch.json"
                )
                configurations.append(
                    {
                        "invocation": path.parent.name,
                        "requested_model": settings.get("model"),
                        "requested_reasoning_effort": settings.get("reasoning_effort"),
                        "wall_seconds": (
                            round(window[1] - window[0], 6) if window else None
                        ),
                    }
                )
                for row in rows:
                    old = observations.get(row["key"])
                    if old is None or (
                        row["tokens"] is not None
                        and (
                            old["tokens"] is None
                            or row["tokens"]["total_tokens"]
                            > old["tokens"]["total_tokens"]
                            or row["status"] == "reported"
                        )
                    ):
                        observations[row["key"]] = row
    rows = list(observations.values())
    for configuration in configurations:
        selected = [
            row for row in rows if row["invocation"] == configuration["invocation"]
        ]
        known_values = [
            row["tokens"]["total_tokens"]
            for row in selected
            if row["tokens"] is not None
        ]
        configuration.update(
            total_tokens=sum(known_values) if known_values else None,
            token_status=(
                "reported"
                if selected and all(row["status"] == "reported" for row in selected)
                else "partial" if known_values else "unavailable"
            ),
            observed_settings=[
                {"model": model, "reasoning_effort": effort}
                for model, effort in sorted(
                    {
                        (row["observed_model"], row["observed_reasoning_effort"])
                        for row in selected
                        if row["observed_model"] is not None
                        or row["observed_reasoning_effort"] is not None
                    },
                    key=str,
                )
            ],
        )
    known = [row for row in rows if row["tokens"] is not None]
    token_status = (
        "unavailable"
        if not known
        else (
            "reported"
            if all(row["status"] == "reported" for row in rows)
            else "partial"
        )
    )
    totals = {
        key: (
            sum(row["tokens"][key] for row in known)
            if known and all(row["tokens"].get(key) is not None for row in known)
            else None
        )
        for key in (*_TOKEN_FIELDS, "total_tokens")
    }
    all_intervals = [window for group in intervals.values() for window in group]
    partial_time = missing_timing or any(
        window[2] == "partial" for window in all_intervals
    )
    covered = bool(intervals["host"]) and _union(intervals["host"]) == _union(
        all_intervals
    )
    timing_status = (
        "unavailable"
        if not all_intervals
        else "partial" if partial_time else "reported" if covered else "reconstructed"
    )
    return {
        "schema_version": "agent-lab-usage/v1",
        "scope": "current_run_only",
        "configurations": configurations,
        "as_of": max(
            [window[1] for window in all_intervals]
            + [row["sampled_at"] for row in rows if row.get("sampled_at") is not None],
            default=None,
        ),
        "timing": {
            "total_wall_seconds": _union(all_intervals) if all_intervals else None,
            "codex_wall_seconds": (
                _union(intervals["codex"]) if intervals["codex"] else None
            ),
            "simulation_wall_seconds": (
                round(sum(end - start for start, end, _ in intervals["simulation"]), 6)
                if intervals["simulation"]
                else None
            ),
            "status": timing_status,
            "intervals": sources,
            "note": "Union of host/Codex/worker intervals. Includes tool waits; excludes between-launch pauses. Subtotals overlap and must not be added.",
        },
        "tokens": {
            **totals,
            "status": token_status,
            "invocations": invocations,
            "reported_turns": sum(row["status"] == "reported" for row in rows),
            "partial_turns": sum(row["status"] == "partial" for row in rows),
            "observed_turns": len(known),
            "missing_turns": sum(row["tokens"] is None for row in rows),
            "turns": rows,
            "note": "Total = input + output. Cached/reasoning counts are subsets. Partial totals are observed lower bounds, not billing or account-quota estimates.",
        },
        "limitations": [
            "No text-length token guessing; unavailable means unknown, not zero.",
            "Unlinked external model calls and other conversations are not counted.",
            "Legacy intervals may omit host setup/finalization gaps. No pure-model-thinking time is inferred.",
        ],
    }


def refresh_usage(root: Path, *, codex_home: Path | None = None) -> dict:
    """Atomically publish a numeric-only snapshot for the agent and final report."""
    result = collect_usage(root, codex_home=codex_home)
    staging = root / f".usage-{uuid4().hex}.tmp"
    write_json(staging, result)
    os.replace(staging, root / "usage.json")
    return result


class UsageMeter:
    """Own one host interval and refresh its usage snapshot until report publication."""

    def __init__(self, root: Path, entrypoint: str) -> None:
        self.root = root
        self.started = time.monotonic()
        self.record = {
            "entrypoint": entrypoint,
            "started_at": time.time(),
            "status": "running",
            "pid": os.getpid(),
            "created_at": psutil.Process().create_time(),
        }
        self.path = root / "usage_sessions" / f"{uuid4().hex}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stopped = threading.Event()
        self._save()
        refresh_usage(root)
        self.thread = threading.Thread(target=self._refresh, daemon=True)
        self.thread.start()

    def _save(self) -> None:
        self.record["updated_at"] = (
            self.record["started_at"] + time.monotonic() - self.started
        )
        staging = self.path.with_suffix(".tmp")
        write_json(staging, self.record)
        os.replace(staging, self.path)

    def _refresh(self) -> None:
        while not self.stopped.wait(5):
            try:
                self._save()
                refresh_usage(self.root)
            except (OSError, ValueError) as exc:
                warnings.warn(f"Usage snapshot unavailable: {exc}", RuntimeWarning)

    def close(self) -> None:
        """Freeze this interval; repeated closes never add time or duplicate tokens."""
        if self.stopped.is_set():
            return
        self.stopped.set()
        self.thread.join()
        self.record.update(
            status="closed",
            ended_at=self.record["started_at"] + time.monotonic() - self.started,
        )
        self._save()
