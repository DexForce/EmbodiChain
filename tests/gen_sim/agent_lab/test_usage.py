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

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from embodichain.gen_sim.agent_lab.catalog import write_json
from embodichain.gen_sim.agent_lab.usage import UsageMeter, collect_usage


def counts(value: int) -> dict:
    return {
        "input_tokens": value,
        "cached_input_tokens": value // 2,
        "output_tokens": value // 10,
        "reasoning_output_tokens": value // 20,
    }


def process(
    root: Path, relative: str, start: float, duration: float, events: list[dict] = ()
) -> Path:
    output = root / relative
    output.mkdir(parents=True)
    write_json(
        output / "process.json",
        {
            "started_at": start,
            "created_at": start,
            "wall_seconds": duration,
            "returncode": 0,
        },
    )
    (output / "stdout.log").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    return output


def rollout(
    home: Path, thread: str, cwd: Path, turns: list[tuple[str, float, list[int], bool]]
) -> None:
    path = home / "sessions/1970/01/01" / f"rollout-{thread}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [{"type": "session_meta", "payload": {"id": thread, "cwd": str(cwd)}}]
    for turn, start, samples, complete in turns:
        stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()
        events.append(
            {
                "type": "turn_context",
                "timestamp": stamp(start),
                "payload": {"turn_id": turn, "cwd": str(cwd)},
            }
        )
        for i, value in enumerate(samples):
            events.append(
                {
                    "type": "event_msg",
                    "timestamp": stamp(start + i + 1),
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": counts(value),
                            "last_token_usage": counts(
                                value if i == 0 else value - samples[i - 1]
                            ),
                        },
                    },
                }
            )
        if complete:
            events.append(
                {
                    "type": "event_msg",
                    "timestamp": stamp(start + len(samples) + 1),
                    "payload": {"type": "task_complete", "turn_id": turn},
                }
            )
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_nested_time_and_paused_gaps_are_not_added_twice(tmp_path: Path) -> None:
    process(tmp_path, "codex/one", 100, 100)
    process(tmp_path, "attempts/nested", 120, 30)
    process(tmp_path, "attempts/replay", 200, 20)
    process(tmp_path, "codex/resumed", 1000, 10)
    result = collect_usage(tmp_path, codex_home=tmp_path / "home")
    assert result["timing"]["total_wall_seconds"] == 130
    assert result["timing"]["codex_wall_seconds"] == 110
    assert result["timing"]["simulation_wall_seconds"] == 50
    assert result["timing"]["status"] == "reconstructed"


def test_host_meter_includes_finalization_but_not_nested_workers(
    tmp_path: Path,
) -> None:
    process(tmp_path, "codex/one", 100, 100)
    process(tmp_path, "attempts/work", 120, 30)
    meter = tmp_path / "usage_sessions/host.json"
    meter.parent.mkdir()
    write_json(meter, {"started_at": 95, "ended_at": 220, "status": "closed"})
    result = collect_usage(tmp_path, codex_home=tmp_path / "home")
    assert result["timing"]["total_wall_seconds"] == 125
    assert result["timing"]["status"] == "reported"


def test_completed_cli_usage_does_not_double_count_cached_or_reasoning(
    tmp_path: Path,
) -> None:
    events = [
        {"type": "thread.started", "thread_id": "thread"},
        {"type": "turn.started"},
        {"type": "turn.completed", "usage": counts(100)},
        {"type": "turn.completed", "usage": counts(100)},
    ]
    process(tmp_path, "codex/one", 100, 20, events)
    result = collect_usage(tmp_path, codex_home=tmp_path / "home")
    assert result["tokens"]["total_tokens"] == 110
    assert result["tokens"]["cached_input_tokens"] == 50
    assert result["tokens"]["reasoning_output_tokens"] == 5
    assert result["tokens"]["status"] == "reported"


def test_interrupted_usage_uses_last_snapshot_not_sum(tmp_path: Path) -> None:
    home = tmp_path / "home"
    process(
        tmp_path,
        "codex/one",
        100,
        20,
        [{"type": "thread.started", "thread_id": "thread"}],
    )
    rollout(
        home,
        "thread",
        tmp_path / "workspace",
        [("turn", 101, [100, 200, 200, 300], False)],
    )
    result = collect_usage(tmp_path, codex_home=home)
    assert result["tokens"]["total_tokens"] == 330
    assert result["tokens"]["status"] == "partial"


def test_resumed_turns_reset_and_cli_and_rollout_are_not_both_added(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    process(
        tmp_path,
        "codex/one",
        100,
        20,
        [
            {"type": "thread.started", "thread_id": "thread"},
            {"type": "turn.started"},
            {"type": "turn.completed", "usage": counts(200)},
        ],
    )
    process(
        tmp_path,
        "codex/two",
        200,
        20,
        [{"type": "thread.started", "thread_id": "thread"}],
    )
    rollout(
        home,
        "thread",
        tmp_path / "workspace",
        [("one", 101, [100, 200], True), ("two", 201, [50, 100], False)],
    )
    result = collect_usage(tmp_path, codex_home=home)
    assert result["tokens"]["total_tokens"] == 330
    assert result["tokens"]["status"] == "partial"


def test_unknown_usage_is_null_and_missing_tail_is_not_zero(tmp_path: Path) -> None:
    process(tmp_path, "codex/one", 100, 20)
    result = collect_usage(tmp_path, codex_home=tmp_path / "home")
    assert result["tokens"]["total_tokens"] is None
    assert result["tokens"]["status"] == "unavailable"


def test_other_threads_in_same_rollout_are_excluded_by_cwd_and_time(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    process(
        tmp_path,
        "codex/one",
        100,
        20,
        [{"type": "thread.started", "thread_id": "thread"}],
    )
    rollout(
        home,
        "thread",
        tmp_path / "workspace",
        [
            ("prior", 10, [10000], True),
            ("current", 101, [100], True),
            ("later", 200, [10000], True),
        ],
    )
    rollout(home, "unrelated", tmp_path / "elsewhere", [("other", 101, [90000], True)])
    result = collect_usage(tmp_path, codex_home=home)
    assert result["tokens"]["total_tokens"] == 110
    assert result["tokens"]["status"] == "reported"


def test_repeated_collection_is_stable_after_exit(tmp_path: Path) -> None:
    process(
        tmp_path,
        "codex/one",
        100,
        20,
        [{"type": "turn.started"}, {"type": "turn.completed", "usage": counts(100)}],
    )
    first = collect_usage(tmp_path, codex_home=tmp_path / "home", now=500)
    assert collect_usage(tmp_path, codex_home=tmp_path / "home", now=1000) == first


def test_carried_counters_are_differenced_across_turns(tmp_path: Path) -> None:
    home = tmp_path / "home"
    process(
        tmp_path,
        "codex/one",
        100,
        20,
        [{"type": "thread.started", "thread_id": "thread"}],
    )
    rollout(
        home,
        "thread",
        tmp_path / "workspace",
        [("old", 10, [100], True), ("new", 101, [150, 200], True)],
    )
    path = home / "sessions/1970/01/01/rollout-thread.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        info = event.get("payload", {}).get("info")
        if info and info["total_token_usage"]["input_tokens"] == 150:
            info["last_token_usage"] = counts(50)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = collect_usage(tmp_path, codex_home=home)
    assert result["tokens"]["total_tokens"] == 110


def test_samples_after_process_end_are_excluded(tmp_path: Path) -> None:
    home = tmp_path / "home"
    process(
        tmp_path,
        "codex/one",
        100,
        3,
        [{"type": "thread.started", "thread_id": "thread"}],
    )
    rollout(
        home, "thread", tmp_path / "workspace", [("turn", 101, [100, 200, 90000], True)]
    )
    result = collect_usage(tmp_path, codex_home=home)
    assert result["tokens"]["total_tokens"] == 220
    assert result["tokens"]["status"] == "partial"


def test_missing_end_does_not_turn_file_mtime_into_runtime(tmp_path: Path) -> None:
    output = tmp_path / "codex/one"
    output.mkdir(parents=True)
    write_json(
        output / "process.json", {"started_at": 100, "pid": 99999999, "created_at": 100}
    )
    result = collect_usage(tmp_path, codex_home=tmp_path / "home")
    assert result["timing"]["total_wall_seconds"] is None
    assert result["timing"]["codex_wall_seconds"] is None


def test_interactive_metadata_discovery_does_not_need_cli_json_events(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    process(tmp_path, "codex/one", 100, 20)
    rollout(home, "thread", tmp_path / "workspace", [("turn", 102, [100, 200], True)])
    path = home / "sessions/1970/01/01/rollout-thread.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[0]["payload"]["timestamp"] = datetime.fromtimestamp(
        101, timezone.utc
    ).isoformat()
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = collect_usage(tmp_path, codex_home=home)
    assert result["tokens"]["total_tokens"] == 220


def test_counter_decrease_is_not_claimed_as_complete(tmp_path: Path) -> None:
    home = tmp_path / "home"
    process(
        tmp_path,
        "codex/one",
        100,
        20,
        [{"type": "thread.started", "thread_id": "thread"}],
    )
    rollout(
        home, "thread", tmp_path / "workspace", [("turn", 101, [100, 300, 100], True)]
    )
    result = collect_usage(tmp_path, codex_home=home)
    assert result["tokens"]["total_tokens"] == 330
    assert result["tokens"]["status"] == "partial"


def test_meter_creates_readable_snapshot_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    meter = UsageMeter(tmp_path, "test")
    assert (
        json.loads((tmp_path / "usage.json").read_text())["scope"] == "current_run_only"
    )
    meter.close()
    before = meter.path.read_bytes()
    meter.close()
    assert meter.path.read_bytes() == before
    assert collect_usage(tmp_path)["timing"]["status"] == "reported"


def test_duplicate_process_record_is_counted_once(tmp_path: Path) -> None:
    events = [
        {"type": "turn.started"},
        {"type": "turn.completed", "usage": counts(100)},
    ]
    for name in ("one", "copy"):
        output = process(tmp_path, "codex/" + name, 100, 20, events)
        record = json.loads((output / "process.json").read_text())
        write_json(output / "process.json", {**record, "pid": 123})
    result = collect_usage(tmp_path, codex_home=tmp_path / "home")
    assert result["tokens"]["total_tokens"] == 110
    assert result["tokens"]["invocations"] == 1
    assert result["timing"]["codex_wall_seconds"] == 20
