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

"""Private, Linux-only resource supervision for GenSim E8 subprocesses."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterator

__all__: list[str] = []

_GIB = 1024**3
_SAMPLE_SECONDS = 2.0
_LAUNCH_AVAILABLE_BYTES = 8 * _GIB
_LOW_AVAILABLE_BYTES = 5 * _GIB
_CRITICAL_AVAILABLE_BYTES = 4 * _GIB
_COLD_ADMISSION_TIMEOUT_SECONDS = 300.0
_SIGINT_GRACE_SECONDS = 45.0
_SIGTERM_GRACE_SECONDS = 20.0
_EXECUTION_LOCK = threading.Lock()
_LOCK_PATH = Path(tempfile.gettempdir()) / f"embodichain-gen-sim-e8-{os.getuid()}.lock"
_POLICY = {
    "launch_available_bytes": _LAUNCH_AVAILABLE_BYTES,
    "launch_consecutive_samples": 3,
    "sample_seconds": _SAMPLE_SECONDS,
    "interrupt_low_available_bytes": _LOW_AVAILABLE_BYTES,
    "interrupt_low_consecutive_samples": 1,
    "interrupt_critical_available_bytes": _CRITICAL_AVAILABLE_BYTES,
    "cold_admission_timeout_seconds": _COLD_ADMISSION_TIMEOUT_SECONDS,
    "sigint_grace_seconds": _SIGINT_GRACE_SECONDS,
    "sigterm_grace_seconds": _SIGTERM_GRACE_SECONDS,
}


class ResourceAdmissionError(RuntimeError):
    """A memory gate rejection before any simulation subprocess was launched."""

    def __init__(self, monitor_path: Path, reason: str, sample: dict[str, int]) -> None:
        self.monitor_path = monitor_path
        self.receipt_path = monitor_path.with_suffix(".receipt.json")
        self.reason = reason
        self.available_bytes = sample["available_bytes"]
        self.total_bytes = sample["total_bytes"]
        super().__init__(
            "E8 execution not admitted by the memory gate; "
            f"reason={reason}; available={self.available_bytes / _GIB:.2f} GiB; "
            f"total={self.total_bytes / _GIB:.2f} GiB; "
            f"required_available={_LAUNCH_AVAILABLE_BYTES / _GIB:.2f} GiB; "
            f"resource receipt={self.receipt_path}"
        )


class ResourceInterruptionError(RuntimeError):
    """A resource stop, not a physical task failure or a successful execution."""

    def __init__(self, receipt_path: Path, receipt: dict[str, Any]) -> None:
        self.receipt_path = receipt_path
        self.monitor_path = Path(receipt["monitor_path"])
        self.min_available_bytes = int(receipt["min_running_available_bytes"])
        self.other_sim_processes = tuple(receipt["other_sim_processes_at_interrupt"])
        super().__init__(
            "E8 execution interrupted for low available memory; "
            f"minimum={self.min_available_bytes / _GIB:.2f} GiB; "
            f"resource receipt={receipt_path}"
        )


def _memory_sample(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    values = {}
    for line in path.read_text().splitlines():
        name, raw = line.split(":", 1)
        fields = raw.split()
        if fields and fields[-1] == "kB":
            values[name] = int(fields[0]) * 1024
    return {
        "available_bytes": values["MemAvailable"],
        "total_bytes": values["MemTotal"],
        "swap_free_bytes": values["SwapFree"],
        "swap_total_bytes": values["SwapTotal"],
    }


def _process_identity(pid: int) -> dict[str, Any] | None:
    try:
        directory = Path(f"/proc/{pid}")
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "session": int(fields[3]),
            "starttime_ticks": int(fields[19]),
            "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
        }
    except (OSError, ValueError, IndexError):
        return None


def _live_processes() -> Iterator[dict[str, Any]]:
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            identity = _process_identity(int(directory.name))
            if identity is not None and identity["state"] not in {"Z", "X"}:
                yield identity


def _owned_processes(pgid: int) -> list[dict[str, Any]]:
    return [
        process
        for process in _live_processes()
        if process["pgid"] == pgid and process["session"] == pgid
    ]


def _simulation_worker_kind(command: list[str]) -> str | None:
    """Identify executable workers, not controllers waiting on the E8 lock."""
    if not command or not Path(command[0]).name.startswith("python"):
        return None
    index = 1
    while index < len(command):
        argument = command[index]
        if argument in {"-W", "-X"}:
            index += 2
            continue
        if argument in {"-u", "-B", "-E", "-I", "-O", "-OO", "-s", "-S", "-q"}:
            index += 1
            continue
        if argument == "-m" and index + 1 < len(command):
            module = command[index + 1]
            if module == "embodichain.gen_sim.task_engine._bundle_runner":
                return "task_program_bundle_runner"
            if module == "embodichain.gen_sim.agent_lab.episode_worker":
                return "agent_lab_episode_worker"
            if module == "embodichain.lab.scripts.run_env":
                return "environment_runner"
            if module == "embodichain" and command[index + 2 : index + 3] == [
                "run-env"
            ]:
                return "environment_cli"
            return None
        if argument.startswith("-"):
            return None
        name = Path(argument).name
        if name in {"_bundle_runner.py", "run_env.py", "paired_ablation.py"}:
            return name.removesuffix(".py")
        if name == "qualification_worker.py" or (
            name.startswith("qualification_worker_") and name.endswith(".py")
        ):
            return "qualification_worker"
        if name == "experiment.py" and "--execute" in command[index + 1 :]:
            return "experiment_executor"
        return None
    return None


def _other_sim_processes(owned_pgid: int) -> list[dict[str, Any]]:
    ancestors = set()
    identity = _process_identity(os.getpid())
    while identity is not None and identity["pid"] not in ancestors:
        ancestors.add(identity["pid"])
        identity = _process_identity(identity["ppid"])
    found = []
    for process in _live_processes():
        if process["pid"] in ancestors or process["session"] == owned_pgid:
            continue
        try:
            directory = Path(f"/proc/{process['pid']}")
            command = [
                argument.decode(errors="replace")
                for argument in (directory / "cmdline").read_bytes().split(b"\0")
                if argument
            ]
            worker_kind = _simulation_worker_kind(command)
            if worker_kind is None:
                continue
            found.append(
                {
                    **process,
                    "cmdline": command,
                    "cwd": os.readlink(directory / "cwd"),
                    "classification": "known_worker_heuristic",
                    "worker_kind": worker_kind,
                }
            )
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda process: process["pid"])


def _event(path: Path, event: str, **fields: Any) -> None:
    with path.open("a") as stream:
        stream.write(
            json.dumps({"event": event, "timestamp_unix": time.time(), **fields}) + "\n"
        )


@contextmanager
def _serialized_execution(
    monitor_path: Path, *, cleanup: Callable[[], None] | None = None
) -> Iterator[None]:
    # flock additionally prevents independently launched local E8 workflows
    # from allocating simulation buffers concurrently.
    with _EXECUTION_LOCK, _LOCK_PATH.open("a") as lock_stream:
        _event(monitor_path, "waiting_e8_lock", lock_path=str(_LOCK_PATH))
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        try:
            _event(monitor_path, "acquired_e8_lock", lock_path=str(_LOCK_PATH))
            yield
        finally:
            try:
                if cleanup is not None:
                    cleanup()
            finally:
                fcntl.flock(lock_stream, fcntl.LOCK_UN)


def wait_for_memory(monitor_path: str | Path) -> dict[str, Any]:
    """Require three >= 8 GiB observations, separated by two seconds.

    Args:
        monitor_path: Append-only resource observation artifact.

    Returns:
        Count and final sample proving the launch memory gate.

    Raises:
        ResourceAdmissionError: Physical RAM cannot meet the floor, or low
            available RAM persists for five minutes without an identifiable
            running simulation to wait for. This is not a runtime timeout.
    """
    path = Path(monitor_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    consecutive = samples = 0
    cold_wait_started: float | None = None
    while consecutive < 3:
        sample = _memory_sample()
        consecutive = (
            consecutive + 1
            if sample["available_bytes"] >= _LAUNCH_AVAILABLE_BYTES
            else 0
        )
        samples += 1
        _event(path, "waiting_memory", consecutive_ready=consecutive, **sample)
        reason = None
        if sample["total_bytes"] < _LAUNCH_AVAILABLE_BYTES:
            reason = "total_memory_below_launch_floor"
        elif sample["available_bytes"] < _LAUNCH_AVAILABLE_BYTES:
            running_simulations = _other_sim_processes(-1)
            if running_simulations:
                cold_wait_started = None
                _event(
                    path, "waiting_running_simulations", processes=running_simulations
                )
            else:
                now = time.monotonic()
                if cold_wait_started is None:
                    cold_wait_started = now
                if now - cold_wait_started >= _COLD_ADMISSION_TIMEOUT_SECONDS:
                    reason = "no_running_simulation_and_memory_timeout"
        else:
            cold_wait_started = None
        if reason is not None:
            _event(path, "resource_not_admitted", reason=reason, **sample)
            raise ResourceAdmissionError(path, reason, sample)
        if consecutive < 3:
            time.sleep(_SAMPLE_SECONDS)
    return {"samples": samples, "last_sample": sample}


def wait_for_competitors(interruption: ResourceInterruptionError) -> None:
    """Wait until identifiable other simulations present at interruption exit.

    Args:
        interruption: Receipt-backed resource stop from this E8 supervisor.

    .. attention::
        Known worker commands are not a complete simulation inventory.
        PID and start-time identity avoids waiting on a reused PID. No other
        process is signalled, suspended, or modified.
    """
    while True:
        live = []
        for candidate in interruption.other_sim_processes:
            current = _process_identity(candidate["pid"])
            if (
                current is not None
                and current["state"] not in {"Z", "X"}
                and current["starttime_ticks"] == candidate["starttime_ticks"]
            ):
                live.append(current)
        _event(interruption.monitor_path, "waiting_other_simulations", processes=live)
        if not live:
            return
        time.sleep(_SAMPLE_SECONDS)


def _should_interrupt(available_bytes: int, low_samples: int) -> tuple[bool, int]:
    count = low_samples + 1 if available_bytes < _LOW_AVAILABLE_BYTES else 0
    return available_bytes < _CRITICAL_AVAILABLE_BYTES or count >= 1, count


def _stop_owned_group(
    process: subprocess.Popen[bytes],
    monitor_path: Path,
    *,
    drain: Callable[[], None] | None = None,
) -> list[str]:
    pgid = process.pid
    if pgid == os.getpgrp():
        raise RuntimeError("Refusing to signal the supervisor's process group")
    if process.poll() is None:
        try:
            session = os.getsid(pgid)
        except ProcessLookupError:
            session = None
        if session is not None and session != pgid:
            raise RuntimeError("Refusing to signal a process without an owned session")
    sent = []
    for sig, grace in (
        (signal.SIGINT, _SIGINT_GRACE_SECONDS),
        (signal.SIGTERM, _SIGTERM_GRACE_SECONDS),
        (signal.SIGKILL, None),
    ):
        process.poll()
        if not _owned_processes(pgid):
            break
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        sent.append(sig.name)
        _event(monitor_path, "signal_owned_group", signal=sig.name, pgid=pgid)
        deadline = None if grace is None else time.monotonic() + grace
        while _owned_processes(pgid):
            process.poll()
            if drain is not None:
                drain()
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        else:
            break
    process.wait()
    return sent


def run_e8_process(
    command: list[str],
    log_path: str | Path,
    *,
    emit: Callable[[bytes], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the ordinary E8 child with memory supervision and exact output tee.

    Args:
        command: Ordinary runner argument vector, without a shell wrapper.
        log_path: Fresh combined-output artifact for this attempt.
        emit: Optional terminal byte callback owned by the caller.

    Returns:
        The original streaming runner's completed-process contract.

    Raises:
        ResourceAdmissionError: The child cannot safely enter the memory gate.
        ResourceInterruptionError: Low available RAM stopped the owned group.
        FileExistsError: An attempt would overwrite an existing artifact.

    .. attention::
        All retry and task-result decisions belong to the workflow. A resource
        interruption never becomes a physical failure or a passing result,
        even if the interrupted runner happened to write a success report.
    """
    log = Path(log_path).expanduser().resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    monitor = log.with_name(log.name + ".resources.jsonl")
    receipt_path = log.with_name(log.name + ".resources.receipt.json")
    if any(path.exists() for path in (log, monitor, receipt_path)):
        raise FileExistsError("Each E8 resource attempt requires fresh artifact paths")
    receipt: dict[str, Any] = {
        "command": command,
        "cwd": str(Path.cwd()),
        "log_path": str(log),
        "monitor_path": str(monitor),
        "policy": _POLICY,
        "started_unix": time.time(),
        "interrupted_memory": False,
        "status": "waiting_memory",
        "pid": None,
        "pgid": None,
        "other_sim_processes_at_interrupt": [],
        "other_process_scope": "known_worker_heuristic_not_complete_inventory",
        "signals": [],
        "exit_code": None,
        "start_new_session": True,
    }
    process: subprocess.Popen[bytes] | None = None
    captured = bytearray()
    with monitor.open("x"):
        pass

    def cleanup_owned() -> None:
        if process is not None and (
            process.poll() is None or _owned_processes(process.pid)
        ):
            receipt["signals"] = _stop_owned_group(process, monitor)
            receipt["exit_code"] = process.returncode

    try:
        with _serialized_execution(monitor, cleanup=cleanup_owned):
            receipt["launch_memory_gate"] = wait_for_memory(monitor)
            with log.open("xb") as output, selectors.DefaultSelector() as selector:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    start_new_session=True,
                )
                assert process.stdout is not None
                receipt["process_identity"] = _process_identity(process.pid)
                receipt.update(pid=process.pid, pgid=process.pid)
                _event(monitor, "launched", **receipt)
                os.set_blocking(process.stdout.fileno(), False)
                selector.register(process.stdout, selectors.EVENT_READ)

                def drain_output(timeout: float = 0.0) -> None:
                    for ready, _ in selector.select(timeout=timeout):
                        chunk = os.read(ready.fd, 64 * 1024)
                        if not chunk:
                            selector.unregister(ready.fileobj)
                            continue
                        captured.extend(chunk)
                        output.write(chunk)
                        output.flush()
                        if emit is not None:
                            emit(chunk)

                next_sample = time.monotonic()
                low_samples = 0
                while selector.get_map() or process.poll() is None:
                    if time.monotonic() >= next_sample and process.poll() is None:
                        sample = _memory_sample()
                        receipt["min_running_available_bytes"] = min(
                            receipt.get(
                                "min_running_available_bytes", sample["available_bytes"]
                            ),
                            sample["available_bytes"],
                        )
                        _event(monitor, "running", pgid=process.pid, **sample)
                        stop, low_samples = _should_interrupt(
                            sample["available_bytes"], low_samples
                        )
                        next_sample = time.monotonic() + _SAMPLE_SECONDS
                        if stop:
                            receipt["interrupted_memory"] = True
                            receipt["other_sim_processes_at_interrupt"] = (
                                _other_sim_processes(process.pid)
                            )
                            receipt["signals"] = _stop_owned_group(
                                process, monitor, drain=drain_output
                            )
                    drain_output(timeout=0.2)
                receipt["exit_code"] = process.wait()
                receipt["status"] = (
                    "resource_interrupted"
                    if receipt["interrupted_memory"]
                    else "finished"
                )
                process.stdout.close()
    except BaseException as error:
        receipt["guard_error"] = repr(error)
        if isinstance(error, ResourceAdmissionError):
            receipt["status"] = "resource_not_admitted"
            receipt["admission_failure"] = {
                "reason": error.reason,
                "available_bytes": error.available_bytes,
                "total_bytes": error.total_bytes,
            }
        else:
            receipt["status"] = "supervisor_error"
        if process is not None:
            cleanup_owned()
            receipt["exit_code"] = process.returncode
            if process.stdout is not None:
                process.stdout.close()
        raise
    finally:
        receipt["finished_unix"] = time.time()
        _event(monitor, "receipt", **receipt)
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
    if receipt["interrupted_memory"]:
        raise ResourceInterruptionError(receipt_path, receipt)
    return subprocess.CompletedProcess(
        args=command,
        returncode=receipt["exit_code"],
        stdout=captured.decode("utf-8", errors="replace"),
        stderr="",
    )
