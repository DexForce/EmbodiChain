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

import json
from pathlib import Path
import signal
import subprocess
import sys
import pytest
from embodichain.gen_sim.task_engine.orchestration.execution_resources import (
    ResourceInterruptionError,
)
from embodichain.gen_sim.task_engine.workflow import SubprocessActionExecutor
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


def _resource_bundle(path: Path, *, twist: bool = True) -> Path:
    from embodichain.utils.utility import save_config

    (path / "task_program").mkdir(parents=True)
    save_config(
        path / "task_program/program.yaml",
        {
            "program": {
                "items": [
                    {
                        "steps": {
                            "call": {
                                "call_id": (
                                    "gen_sim.twist" if twist else "simulation.pick"
                                )
                            }
                        }
                    }
                ]
            }
        },
    )
    return path


def test_resource_retry_keeps_seed_and_never_selects_interrupted_attempt(
    tmp_path, monkeypatch
):
    _accept_scheduler_fixture(monkeypatch)
    from embodichain.gen_sim.task_engine.orchestration import (
        execution_resources as resources,
    )

    root = _resource_bundle(tmp_path / "bundle")
    executor = SubprocessActionExecutor()
    seen, waits = [], []

    def execute(path, output, **options):
        output.mkdir(parents=True)
        seen.append(options)
        if len(seen) == 1:
            raise ResourceInterruptionError(
                output / "receipt.json",
                {
                    "monitor_path": str(output / "memory.jsonl"),
                    "min_running_available_bytes": 100,
                    "other_sim_processes_at_interrupt": [],
                },
            )
        (output / "twist_evidence.json").write_text(
            '{"acceptance": {"accepted": true}}'
        )
        return {"status": "succeeded"}

    monkeypatch.setattr(executor, "_execute", execute)
    monkeypatch.setattr(
        resources, "wait_for_competitors", lambda error: waits.append(error)
    )
    report = executor(root, tmp_path / "execution", seed=3, num_envs=1)
    assert report["status"] == "succeeded" and seen[0] == seen[1] and len(waits) == 1
    selection = json.loads(
        (tmp_path / "execution/execution_selection.json").read_text()
    )
    assert selection["resource_attempt"] == 2
    assert (tmp_path / "execution/resource_attempts/attempt_0001").is_dir()
    assert (tmp_path / "execution/twist_evidence.json").is_file()


def test_non_e8_executor_does_not_add_resource_attempts(tmp_path, monkeypatch):
    _accept_scheduler_fixture(monkeypatch)
    root = _resource_bundle(tmp_path / "bundle", twist=False)
    executor = SubprocessActionExecutor()
    called = []
    monkeypatch.setattr(
        executor,
        "_execute",
        lambda path, output, **options: called.append(output)
        or {"status": "succeeded"},
    )
    executor(root, tmp_path / "execution", seed=0, num_envs=1)
    assert called == [tmp_path / "execution"]
    assert not (tmp_path / "execution").exists()


def test_resource_admission_failure_is_not_retried_as_a_physical_attempt(
    tmp_path, monkeypatch
):
    _accept_scheduler_fixture(monkeypatch)
    from embodichain.gen_sim.task_engine.orchestration import (
        execution_resources as resources,
    )

    root = _resource_bundle(tmp_path / "bundle")
    executor = SubprocessActionExecutor()
    calls = []

    def execute(path, output, **options):
        calls.append(output)
        raise resources.ResourceAdmissionError(
            output / "memory.jsonl",
            "no_running_simulation_and_memory_timeout",
            {"available_bytes": 9 * 1024**3, "total_bytes": 32 * 1024**3},
        )

    monkeypatch.setattr(executor, "_execute", execute)
    with pytest.raises(resources.ResourceAdmissionError):
        executor(root, tmp_path / "execution", seed=0, num_envs=1)
    assert len(calls) == 1
    attempts = json.loads((tmp_path / "execution/resource_attempts.json").read_text())
    assert attempts[0]["status"] == "resource_not_admitted"


def _accept_scheduler_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    # These cases isolate retry scheduling. Real package/file identity validation
    # and its no-launch rejection path are exercised by the bundle-loading tests.
    from embodichain.gen_sim.task_engine import bundle_loading

    monkeypatch.setattr(
        bundle_loading, "load_current_task_program_bundle", lambda _: None
    )


@pytest.mark.parametrize(
    "returncode,accepted,consistent",
    [
        (0, True, True),
        (2, False, True),
        (7, True, False),
        (-15, True, False),
        (2, True, False),
    ],
)
def test_executor_requires_worker_exit_to_match_its_verified_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    accepted: bool,
    consistent: bool,
) -> None:
    from copy import deepcopy
    import uuid
    import yaml
    from ..test_bundle_deployment import _files
    from embodichain.gen_sim.task_engine import workflow, bundle_loading
    from embodichain.gen_sim.task_engine.bundle_deployment import (
        prepare_configured_bundle,
        load_configured_bundle,
    )
    from embodichain.gen_sim.task_engine.bundle_io import write_bundle_atomic
    from embodichain.gen_sim.task_engine.bundle_spec import bundle_task_spec_payload
    from embodichain.gen_sim.task_engine.contracts import canonical_hash
    from embodichain.gen_sim.task_engine.reporting import (
        TASK_PROGRAM_EXECUTION_REPORT_SCHEMA,
        verify_bundle_execution_report,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot

    def forbidden(*args, **kwargs):
        pytest.fail("Worker completion check constructed a native runtime.")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(Robot, "__init__", forbidden)
    files = _files()
    files["task_spec.json"] = json.dumps(bundle_task_spec_payload(None)).encode()
    snapshot = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests={"fixture-assets": "a" * 64},
        implementation_revisions={"embodichain": "fixed-worker-test"},
    )
    directory = write_bundle_atomic(tmp_path / "bundle", snapshot)
    loaded = load_configured_bundle(
        directory,
        external_input_digests={"fixture-assets": "a" * 64},
        implementation_revisions={"embodichain": "fixed-worker-test"},
    )
    graph = json.loads(snapshot.files["semantic_task_graph.json"])
    program = yaml.safe_load(snapshot.files["task_program/program.yaml"])
    identity = snapshot.identity.to_dict()
    published = {}

    def child(command, log_path):
        output = Path(log_path).parent
        segments = [
            dict(
                name=node["id"],
                success=accepted,
                successes=[accepted],
                active=[True],
                metadata=dict(
                    segment_count=len(graph["nodes"]),
                    runtime=dict(failures=[]),
                    validation=dict(accepted_mask=[accepted]),
                ),
            )
            for node in graph["nodes"]
        ]
        strict = [
            dict(
                status="unavailable",
                certificate_ready=False,
                reason="No normative TaskSpec was declared.",
                init=[],
                goal=[],
                invariants=[],
                temporal=[],
                milestones=[],
            )
        ]
        physical = dict(
            schema_version="task_engine.physical_run_report/v1",
            bundle_digest=snapshot.identity.digest,
            run_id=str(uuid.uuid4()),
            episode_ids=[0],
            program_success=[accepted],
            program_completed=[accepted],
            terminal_reasons=["completed" if accepted else "failed"],
            lengths=[1],
            action_physical_effects=[],
            strict_task_spec=strict,
            acquisition_errors={},
        )
        runtime = dict(
            bundle_identity=snapshot.identity.digest,
            program_fingerprint=identity["program_fingerprint"],
            acceptance_mode="experiment",
            success=[accepted],
            completed=accepted,
            completed_by_env=[accepted],
            lengths=[1],
            segments=segments,
            physical_run_report=physical,
            task_spec_evaluation=dict(
                schema_version="task_spec_evaluation/v1",
                task_spec_digest=identity["task_spec_digest"],
                graph_digest=canonical_hash(graph),
                program_digest=canonical_hash(program),
                environments=[dict(env_id=0, **strict[0])],
            ),
        )
        report = dict(
            schema_version=TASK_PROGRAM_EXECUTION_REPORT_SCHEMA,
            status="succeeded" if accepted else "failed",
            task_id=graph["task_id"],
            semantic_call_count=len(graph["nodes"]),
            integration_fingerprint=graph["integration_fingerprint"],
            record_dir=str(output),
            environments=[
                dict(
                    env_id=0,
                    success=accepted,
                    terminal_reason="completed" if accepted else "failed",
                    semantic_success={
                        group["id"]: accepted for group in graph["task_groups"]
                    },
                )
            ],
            runtime_result=runtime,
            failure=None,
        )
        verify_bundle_execution_report(
            report, loaded, acceptance_mode="experiment", expected_record_dir=output
        )
        published.update(report=deepcopy(report))
        (output / "execution_report.json").write_text(json.dumps(report))
        Path(log_path).write_text("Synthetic worker finalization result.\n")
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    monkeypatch.setattr(
        bundle_loading, "load_current_task_program_bundle", lambda _: loaded
    )
    monkeypatch.setattr(workflow, "_run_streaming_process", child)
    if consistent:
        result = SubprocessActionExecutor()(
            directory,
            tmp_path / "run",
            seed=0,
            num_envs=1,
            acceptance_mode="experiment",
        )
        assert result == published["report"]
    else:
        with pytest.raises(RuntimeError, match="worker exit code"):
            SubprocessActionExecutor()(
                directory,
                tmp_path / "run",
                seed=0,
                num_envs=1,
                acceptance_mode="experiment",
            )
    # Process failure cannot rewrite a successful program or unavailable proof.
    assert (
        json.loads((tmp_path / "run/execution_report.json").read_text())
        == published["report"]
    )
    assert published["report"]["runtime_result"]["physical_run_report"][
        "program_success"
    ] == [accepted]
    assert (
        published["report"]["runtime_result"]["physical_run_report"][
            "strict_task_spec"
        ][0]["status"]
        == "unavailable"
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux process-session ownership"
)
@pytest.mark.parametrize("ignore_termination", [False, True])
def test_generic_cancellation_stops_owned_workers_and_preserves_other_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignore_termination: bool
) -> None:
    import os
    import time
    from embodichain.gen_sim.task_engine import workflow

    original_launch = subprocess.Popen
    other = original_launch([sys.executable, "-c", "import time; time.sleep(60)"])
    launched, worker_ids = [], []

    def launch(*args, **kwargs):
        process = original_launch(*args, **kwargs)
        launched.append(process)
        return process

    def cancel(chunk):
        worker_ids.append(json.loads(chunk.decode())["worker"])
        raise KeyboardInterrupt("cancel after worker launch")

    def running(pid):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return False
        return state not in {"Z", "X"}

    worker_source = "import signal,time; "
    if ignore_termination:
        worker_source += "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
    worker_source += "print('ready',flush=True); time.sleep(60)"
    controller_source = (
        "import json,subprocess,sys,time; "
        f"worker=subprocess.Popen([sys.executable,'-u','-c',{worker_source!r}],stdout=subprocess.PIPE,text=True); "
        "assert worker.stdout.readline().strip()=='ready'; "
        "print(json.dumps({'worker':worker.pid}),flush=True); time.sleep(60)"
    )
    monkeypatch.setattr(workflow.subprocess, "Popen", launch)
    monkeypatch.setattr(workflow, "_write_terminal_chunk", cancel)
    try:
        with pytest.raises(KeyboardInterrupt, match="worker launch"):
            workflow._run_streaming_process(
                [sys.executable, "-u", "-c", controller_source],
                tmp_path / "generic.log",
            )
        deadline = time.monotonic() + 1.0
        while running(worker_ids[0]) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not running(worker_ids[0])
        assert launched[0].poll() is not None
        assert other.poll() is None
        assert launched[0].stdout.closed
    finally:
        for pid in worker_ids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in launched:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
        other.terminate()
        other.wait(timeout=5)
