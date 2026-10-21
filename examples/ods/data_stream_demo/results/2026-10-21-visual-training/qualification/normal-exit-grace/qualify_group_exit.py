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

"""Deferred, Linux CPU-only qualification of the candidate process-group grace.

Default prints a plan only. --execute starts tiny sleeping helper processes;
run only after the formal cohort is terminal and the root agent authorizes it.
It never imports the repository, Torch, a simulator, or a video decoder.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import psutil

__all__ = ["main"]

HERE = Path(__file__).resolve().parent


def _load_candidate() -> dict[str, Any]:
    names = {"_group_snapshot", "_await_group_exit", "_finish"}
    tree = ast.parse((HERE / "training_suite.candidate.py").read_text())
    selected = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(selected) == len(names)
    scope: dict[str, Any] = {
        "os": os,
        "time": time,
        "psutil": psutil,
        "Any": Any,
        "json": json,
        "signal_attempts": [],
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<candidate-only>", "exec"),
        scope,
    )

    def write(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def terminate(process: subprocess.Popen[Any]) -> None:
        # Same implementation as the frozen suite._terminate; do not grant
        # this fixture stronger cleanup behavior than the actual caller.
        for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 10)):
            scope["signal_attempts"].append(int(sig))
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
        if process.poll() is None:
            raise TimeoutError("Fixture process group did not terminate")

    scope.update(_write=write, _terminate=terminate)
    return scope


def _leader(
    output: Path, delay: float, *, ignore_sigterm: bool = False
) -> tuple[subprocess.Popen[Any], int]:
    code = """
import json, os, signal, sys, time
from pathlib import Path
ready = Path(sys.argv[1]).with_suffix('.ready')
child = os.fork()
if child == 0:
    if sys.argv[3] == '1':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.write_text(json.dumps({'ignores_sigterm': sys.argv[3] == '1'}))
    time.sleep(float(sys.argv[2]))
    os._exit(0)
deadline = time.monotonic() + 4
while not ready.exists():
    if time.monotonic() >= deadline:
        os.kill(child, signal.SIGKILL)
        os._exit(2)
    time.sleep(0.005)
Path(sys.argv[1]).write_text(json.dumps({'helper_pid': child}))
os._exit(0)
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(output / "helper.json"),
            str(delay),
            "1" if ignore_sigterm else "0",
        ],
        start_new_session=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.wait(timeout=5) == 0
    helper = json.loads((output / "helper.json").read_text())["helper_pid"]
    assert (
        json.loads((output / "helper.ready").read_text())["ignores_sigterm"]
        == ignore_sigterm
    )
    return process, helper


def _reap(pid: int, outcome: dict[str, Any]) -> None:
    # Driver is the adopted parent, so normal helpers can be reaped without
    # relying on the container's PID1 policy. Do not call this in zombie case.
    while True:
        try:
            waited, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if waited:
            outcome.update(
                reaped=True,
                raw_wait_status=status,
                exitcode=os.waitstatus_to_exitcode(status),
            )
            return
        time.sleep(0.01)


def _case(scope: dict[str, Any], output: Path, label: str) -> dict[str, Any]:
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps({"status": "complete", "qualification_passed": True})
    )
    (output / "post-audit.json").write_text(json.dumps({"passed": True}))
    process, helper = _leader(
        output,
        {"delayed_normal_exit": 0.25, "persistent_live": 30.0, "zombie_only": 0.0}[
            label
        ],
        ignore_sigterm=label == "persistent_live",
    )
    reap_outcome: dict[str, Any] = {"reaped": False}
    signal_start = len(scope["signal_attempts"])
    reaper: threading.Thread | None = None
    if label != "zombie_only":
        reaper = threading.Thread(
            target=_reap, args=(helper, reap_outcome), daemon=True
        )
        reaper.start()
    else:
        deadline = time.monotonic() + 2
        while psutil.Process(helper).status() != psutil.STATUS_ZOMBIE:
            if time.monotonic() >= deadline:
                raise TimeoutError("Controlled helper did not enter zombie state")
            time.sleep(0.01)
    stream = (output / "audit.log").open("w")
    item = {
        "process": process,
        "job": {"output": output, "evaluation_scenes": 0},
        "deadline": time.monotonic() + 30,
        "error": None,
        "started_ns": time.monotonic_ns(),
        "stream": stream,
        "stage": "audit",
        "evaluation": False,
        "guard": {},
    }
    try:
        summary, result = scope["_finish"](item, cancelled=False)
        shutdown = result["process_group_shutdown"]
        signals_sent = scope["signal_attempts"][signal_start:]
        if reaper is not None:
            reaper.join(timeout=2)
            assert not reaper.is_alive()
            assert reap_outcome["reaped"]
        assert any(row["pid"] == helper for row in shutdown["initial"]["members"])
        assert all(
            {"pid", "ppid", "create_time", "state"} <= row.keys()
            for row in shutdown["initial"]["members"]
        )
        if label == "delayed_normal_exit":
            assert summary["status"] == "complete"
            assert not result["leaked_process_group"]
            assert shutdown["group_disappeared"]
            assert shutdown["wait_seconds"] >= 0.1
            assert reap_outcome["exitcode"] == 0
            assert signals_sent == []
        else:
            assert summary["status"] == "failed"
            assert result["leaked_process_group"]
            assert not shutdown["group_disappeared"]
            if label == "persistent_live":
                assert shutdown["termination_attempted"]
                assert shutdown["after_termination"]["group_disappeared"]
                assert signals_sent == [int(signal.SIGTERM), int(signal.SIGKILL)]
                assert reap_outcome["exitcode"] == -int(signal.SIGKILL)
            else:
                assert shutdown["final"]["member_kind"] == "zombie_only"
                assert not shutdown["termination_attempted"]
                assert signals_sent == []
        return {
            "case": label,
            "passed": True,
            "summary_status": summary["status"],
            "shutdown": shutdown,
            "signals_sent": signals_sent,
            "helper_ignores_sigterm": label == "persistent_live",
            "reap_outcome_before_finally": dict(reap_outcome),
        }
    finally:
        stream.close()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if reaper is not None:
            reaper.join(timeout=5)
            if reaper.is_alive():
                raise TimeoutError("Driver could not reap its helper")
        else:
            try:
                os.waitpid(helper, 0)
            except ChildProcessError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = ("delayed_normal_exit", "persistent_live", "zombie_only")
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "inspect_only",
                    "cases": cases,
                    "execution": "Requires root authorization after formal cohort is terminal",
                },
                indent=2,
            )
        )
        return
    if args.output is None or args.output.exists():
        parser.error("--execute requires a fresh --output directory")
    # Linux PR_SET_CHILD_SUBREAPER lets this controlled driver reap helpers.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set child subreaper")
    args.output.mkdir(parents=True)
    scope = _load_candidate()
    results = [_case(scope, args.output / label, label) for label in cases]
    payload = {
        "passed": all(row["passed"] for row in results),
        "cases": results,
        "limits": "Controlled fixture only; does not identify the historical failure's remaining member or establish its cause",
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": payload["passed"]}))


if __name__ == "__main__":
    main()
