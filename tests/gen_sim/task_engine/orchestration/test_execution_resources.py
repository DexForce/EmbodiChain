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

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from embodichain.gen_sim.task_engine.orchestration import (
    execution_resources as resources,
)


@pytest.fixture(autouse=True)
def _isolated_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(resources, "_LOCK_PATH", tmp_path / "e8.lock")


def _memory(available_gib: float) -> dict[str, int]:
    return {
        "available_bytes": int(available_gib * resources._GIB),
        "total_bytes": 32 * resources._GIB,
        "swap_free_bytes": 2 * resources._GIB,
        "swap_total_bytes": 2 * resources._GIB,
    }


def _fast_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resources,
        "wait_for_memory",
        lambda path: {"samples": 3, "last_sample": _memory(16)},
    )
    monkeypatch.setattr(resources, "_memory_sample", lambda: _memory(16))


def _receipt(log: Path) -> dict[str, object]:
    return json.loads(log.with_name(log.name + ".resources.receipt.json").read_text())


def test_mem_available_is_used_instead_of_mem_free(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text(
        "MemAvailable: 16000 kB\nMemFree: 100 kB\nMemTotal: 32000 kB\n"
        "SwapFree: 1000 kB\nSwapTotal: 2000 kB\n"
    )
    assert resources._memory_sample(path)["available_bytes"] == 16000 * 1024


@pytest.mark.parametrize(
    ("available_gib", "prior_low", "interrupt", "count"),
    [
        (5, 1, False, 0),
        (4.9, 0, True, 1),
        (4, 0, True, 1),
        (3, 1, True, 2),
        (2, 0, True, 1),
        (1.9, 0, True, 1),
    ],
)
def test_low_memory_thresholds(
    available_gib: float, prior_low: int, interrupt: bool, count: int
) -> None:
    assert resources._should_interrupt(
        int(available_gib * resources._GIB), prior_low
    ) == (interrupt, count)


def test_launch_gate_requires_consecutive_available_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observations = iter([8, 9, 7, 10, 11, 12])
    sleeps = []
    monkeypatch.setattr(
        resources, "_memory_sample", lambda: _memory(next(observations))
    )
    monkeypatch.setattr(resources, "_other_sim_processes", lambda pgid: [])
    monkeypatch.setattr(resources.time, "sleep", sleeps.append)
    path = tmp_path / "monitor.jsonl"
    result = resources.wait_for_memory(path)
    assert result["samples"] == 6
    assert sleeps == [2.0] * 5
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["consecutive_ready"] for event in events] == [1, 2, 0, 1, 2, 3]


def test_total_ram_below_launch_floor_is_rejected_without_starting_a_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = _memory(4)
    sample["total_bytes"] = resources._LAUNCH_AVAILABLE_BYTES - 1
    monkeypatch.setattr(resources, "_memory_sample", lambda: sample)

    def forbidden_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("A rejected resource admission must not launch a child")

    monkeypatch.setattr(resources.subprocess, "Popen", forbidden_launch)
    log = tmp_path / "cold.log"
    with pytest.raises(resources.ResourceAdmissionError) as raised:
        resources.run_e8_process([sys.executable, "-c", "pass"], log)
    assert raised.value.reason == "total_memory_below_launch_floor"
    assert raised.value.receipt_path.is_file()
    receipt = _receipt(log)
    assert receipt["status"] == "resource_not_admitted"
    assert receipt["pid"] is None and receipt["pgid"] is None
    assert receipt["exit_code"] is None
    assert receipt["interrupted_memory"] is False
    assert not log.exists()


def test_cold_low_memory_admission_times_out_without_a_running_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = [0.0]
    monkeypatch.setattr(resources, "_memory_sample", lambda: _memory(6))
    monkeypatch.setattr(resources, "_other_sim_processes", lambda pgid: [])
    monkeypatch.setattr(resources.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        resources.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    with pytest.raises(resources.ResourceAdmissionError) as raised:
        resources.wait_for_memory(tmp_path / "cold.jsonl")
    assert raised.value.reason == "no_running_simulation_and_memory_timeout"
    assert clock[0] == 300.0
    assert raised.value.available_bytes == 6 * resources._GIB


def test_cold_admission_timer_resets_while_actual_simulation_workers_are_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = [0.0]
    monkeypatch.setattr(resources, "_memory_sample", lambda: _memory(6))
    monkeypatch.setattr(
        resources,
        "_other_sim_processes",
        lambda pgid: (
            [{"pid": 123, "worker_kind": "agent_lab_episode_worker"}]
            if 200 <= clock[0] < 600
            else []
        ),
    )
    monkeypatch.setattr(resources.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        resources.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    with pytest.raises(resources.ResourceAdmissionError) as raised:
        resources.wait_for_memory(tmp_path / "waiting.jsonl")
    assert raised.value.reason == "no_running_simulation_and_memory_timeout"
    assert clock[0] == 900.0
    events = [
        json.loads(line)
        for line in (tmp_path / "waiting.jsonl").read_text().splitlines()
    ]
    assert (
        sum(event["event"] == "waiting_running_simulations" for event in events) == 200
    )


def test_resources_can_be_admitted_after_waiting_longer_than_cold_timeout_for_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        resources, "_memory_sample", lambda: _memory(6 if clock[0] < 600 else 16)
    )
    monkeypatch.setattr(
        resources,
        "_other_sim_processes",
        lambda pgid: [{"pid": 123, "worker_kind": "agent_lab_episode_worker"}],
    )
    monkeypatch.setattr(resources.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        resources.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    gate = resources.wait_for_memory(tmp_path / "waiting.jsonl")
    assert clock[0] == 604.0
    assert gate["last_sample"]["available_bytes"] == 16 * resources._GIB


def test_cold_admission_rejection_releases_both_execution_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = _memory(4)
    sample["total_bytes"] = resources._LAUNCH_AVAILABLE_BYTES - 1
    monkeypatch.setattr(resources, "_memory_sample", lambda: sample)
    with pytest.raises(resources.ResourceAdmissionError):
        resources.run_e8_process(
            [sys.executable, "-c", "pass"], tmp_path / "rejected.log"
        )
    _fast_gate(monkeypatch)
    results: list[subprocess.CompletedProcess[str]] = []
    retry = threading.Thread(
        target=lambda: results.append(
            resources.run_e8_process(
                [sys.executable, "-c", "pass"], tmp_path / "next.log"
            )
        ),
        daemon=True,
    )
    retry.start()
    retry.join(timeout=3)
    assert not retry.is_alive()
    assert results[0].returncode == 0


def test_stream_contract_combines_output_and_preserves_exact_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_gate(monkeypatch)
    chunks = []
    log = tmp_path / "child.log"
    command = [
        sys.executable,
        "-c",
        "import os; os.write(1, b'first\\n'); os.write(2, b'second\\xff\\n')",
    ]
    completed = resources.run_e8_process(command, log, emit=chunks.append)
    assert completed.args == command
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == "first\nsecond\ufffd\n"
    assert log.read_bytes() == b"first\nsecond\xff\n"
    assert b"".join(chunks) == log.read_bytes()
    receipt = _receipt(log)
    assert receipt["interrupted_memory"] is False
    assert receipt["pgid"] == receipt["pid"]
    assert receipt["start_new_session"] is True


def test_existing_artifacts_are_never_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_gate(monkeypatch)
    log = tmp_path / "child.log"
    log.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        resources.run_e8_process([sys.executable, "-c", "pass"], log)
    assert log.read_bytes() == b"preserved"


def test_silent_child_is_interrupted_without_waiting_for_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_gate(monkeypatch)
    monkeypatch.setattr(resources, "_memory_sample", lambda: _memory(1))
    log = tmp_path / "silent.log"
    with pytest.raises(resources.ResourceInterruptionError) as raised:
        resources.run_e8_process(
            [sys.executable, "-c", "import time; time.sleep(30)"], log
        )
    receipt = _receipt(log)
    assert receipt["interrupted_memory"] is True
    assert receipt["signals"] == ["SIGINT"]
    assert receipt["exit_code"] != 0
    assert raised.value.min_available_bytes == resources._GIB
    assert raised.value.receipt_path.is_file()
    assert resources._process_identity(int(receipt["pid"])) is None


def test_resource_stop_rejects_child_exit_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_gate(monkeypatch)
    monkeypatch.setattr(resources, "_SAMPLE_SECONDS", 0.05)
    samples = iter([_memory(16), _memory(1)])
    monkeypatch.setattr(resources, "_memory_sample", lambda: next(samples))
    log = tmp_path / "zero.log"
    command = [
        sys.executable,
        "-c",
        "import signal,time,sys; signal.signal(signal.SIGINT, lambda *args: sys.exit(0)); "
        "print('ready',flush=True); time.sleep(30)",
    ]
    with pytest.raises(resources.ResourceInterruptionError):
        resources.run_e8_process(command, log)
    assert _receipt(log)["exit_code"] == 0
    assert _receipt(log)["interrupted_memory"] is True


def test_callback_failure_stops_only_owned_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_gate(monkeypatch)
    log = tmp_path / "callback.log"

    def fail(chunk: bytes) -> None:
        raise RuntimeError("terminal failed")

    with pytest.raises(RuntimeError, match="terminal failed"):
        resources.run_e8_process(
            [
                sys.executable,
                "-c",
                "import time; print('ready',flush=True); time.sleep(30)",
            ],
            log,
            emit=fail,
        )
    receipt = _receipt(log)
    assert "terminal failed" in str(receipt["guard_error"])
    assert resources._process_identity(int(receipt["pid"])) is None


def test_competitor_wait_uses_pid_and_starttime_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monitor = tmp_path / "monitor.jsonl"
    interruption = resources.ResourceInterruptionError(
        tmp_path / "receipt.json",
        {
            "monitor_path": str(monitor),
            "min_running_available_bytes": resources._GIB,
            "other_sim_processes_at_interrupt": [{"pid": 123, "starttime_ticks": 50}],
        },
    )
    identities = iter(
        [
            {"pid": 123, "state": "S", "starttime_ticks": 50},
            {"pid": 123, "state": "S", "starttime_ticks": 51},
        ]
    )
    sleeps = []
    monkeypatch.setattr(resources, "_process_identity", lambda pid: next(identities))
    monkeypatch.setattr(resources.time, "sleep", sleeps.append)
    resources.wait_for_competitors(interruption)
    assert sleeps == [2.0]


@pytest.mark.parametrize(
    ("arguments", "kind"),
    [
        (
            ["-u", "-m", "embodichain.gen_sim.task_engine._bundle_runner"],
            "task_program_bundle_runner",
        ),
        (["-B", "-m", "embodichain.lab.scripts.run_env"], "environment_runner"),
        (
            [
                "-B",
                "-m",
                "embodichain.gen_sim.agent_lab.episode_worker",
                "--config",
                "lab.json",
            ],
            "agent_lab_episode_worker",
        ),
        (["-m", "embodichain", "run-env", "--config", "env.yaml"], "environment_cli"),
        (["/repo/paired_ablation.py", "--seed", "0"], "paired_ablation"),
        (["-u", "/repo/qualification_worker.py"], "qualification_worker"),
        (
            ["-X", "faulthandler", "/repo/qualification_worker_n04.py"],
            "qualification_worker",
        ),
        (["-u", "/repo/experiment.py", "--execute"], "experiment_executor"),
        (["/repo/experiment.py", "--output", "gen_sim/attempt"], None),
        (["/repo/run_paired_ablation.py", "paired_seed0"], None),
        (["-m", "embodichain.gen_sim.task_engine", "run-all"], None),
        (["-m", "pytest", "tests/gen_sim/task_engine/test_twist.py"], None),
        (["-c", "import gen_sim; print('qualification_worker.py')"], None),
        (
            [
                "-B",
                "-u",
                "-c",
                "from embodichain.gen_sim.agent_lab._run_task import run; run(...)",
            ],
            None,
        ),
        (["/repo/controller.py", "--worker", "/repo/paired_ablation.py"], None),
    ],
)
def test_worker_inventory_excludes_queued_controllers(
    arguments: list[str], kind: str | None
) -> None:
    assert resources._simulation_worker_kind([sys.executable, *arguments]) == kind


def test_worker_inventory_excludes_conda_and_shell_wrappers() -> None:
    worker = ["python", "-m", "embodichain.gen_sim.task_engine._bundle_runner"]
    assert (
        resources._simulation_worker_kind(["conda", "run", "-n", "sim", *worker])
        is None
    )
    assert resources._simulation_worker_kind(["bash", "-c", " ".join(worker)]) is None


def test_live_inventory_records_workers_but_not_controller_parents(
    tmp_path: Path,
) -> None:
    worker_path = tmp_path / "qualification_worker.py"
    controller_path = tmp_path / "run_paired_ablation.py"
    script = "import time; print('ready',flush=True); time.sleep(30)"
    worker_path.write_text(script)
    controller_path.write_text(script)
    worker = subprocess.Popen(
        [sys.executable, str(worker_path)],
        start_new_session=True,
        stdout=subprocess.PIPE,
    )
    controller = subprocess.Popen(
        [sys.executable, str(controller_path)],
        start_new_session=True,
        stdout=subprocess.PIPE,
    )
    try:
        assert worker.stdout is not None and controller.stdout is not None
        assert worker.stdout.readline() == b"ready\n"
        assert controller.stdout.readline() == b"ready\n"
        snapshots = {
            process["pid"]: process for process in resources._other_sim_processes(-1)
        }
        assert worker.pid in snapshots
        assert snapshots[worker.pid]["classification"] == "known_worker_heuristic"
        assert snapshots[worker.pid]["starttime_ticks"] > 0
        assert controller.pid not in snapshots
        assert worker.pid not in {
            process["pid"] for process in resources._other_sim_processes(worker.pid)
        }
    finally:
        for process in (worker, controller):
            process.terminate()
            process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()


def test_owned_group_stop_refuses_nonisolated_child(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(RuntimeError, match="owned session"):
            resources._stop_owned_group(process, tmp_path / "monitor.jsonl")
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_owned_signal_escalation_does_not_signal_other_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "_SIGINT_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(resources, "_SIGTERM_GRACE_SECONDS", 0.01)
    script = (
        "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"
    )
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    owned = subprocess.Popen(
        [sys.executable, "-c", script], start_new_session=True, stdout=subprocess.PIPE
    )
    try:
        assert owned.stdout is not None
        assert owned.stdout.readline() == b"ready\n"
        sent = resources._stop_owned_group(owned, tmp_path / "monitor.jsonl")
        assert sent == ["SIGINT", "SIGTERM", "SIGKILL"]
        assert owned.returncode == -signal.SIGKILL
        assert other.poll() is None
    finally:
        if owned.poll() is None:
            owned.kill()
            owned.wait(timeout=5)
        if owned.stdout is not None:
            owned.stdout.close()
        other.terminate()
        other.wait(timeout=5)


def test_threaded_e8_runs_are_serialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_gate(monkeypatch)
    active = maximum = 0
    lock = threading.Lock()

    def gate(path: Path) -> dict[str, object]:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {}

    monkeypatch.setattr(resources, "wait_for_memory", gate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        command = [
            sys.executable,
            "-c",
            "import time; print(time.monotonic()); time.sleep(0.08); print(time.monotonic())",
        ]
        futures = [
            pool.submit(
                resources.run_e8_process,
                command,
                tmp_path / f"{index}.log",
            )
            for index in range(2)
        ]
        completed = [future.result() for future in futures]
        assert all(process.returncode == 0 for process in completed)
    intervals = sorted(
        tuple(float(value) for value in process.stdout.splitlines())
        for process in completed
    )
    assert intervals[0][1] <= intervals[1][0]
    assert maximum == 1


def test_e8_file_lock_excludes_independently_launched_process(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import fcntl,sys; stream=open(sys.argv[1],'a'); "
        "\ntry: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)"
        "\nexcept BlockingIOError: print('busy')"
        "\nelse: print('acquired')",
        str(resources._LOCK_PATH),
    ]
    with resources._serialized_execution(tmp_path / "monitor.jsonl"):
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5)
        assert completed.returncode == 0
        assert completed.stdout == "busy\n"
    completed = subprocess.run(command, capture_output=True, text=True, timeout=5)
    assert completed.returncode == 0
    assert completed.stdout == "acquired\n"
