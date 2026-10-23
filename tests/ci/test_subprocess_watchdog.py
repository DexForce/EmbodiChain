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

"""Exercise diagnostic subprocesses without importing simulation dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.ci import subprocess_watchdog

pytestmark = pytest.mark.no_sim


def _run(
    script: Path, *arguments: str, interval: str = "0.02"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(subprocess_watchdog.__file__)),
            "--interval",
            interval,
            str(script),
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_periodic_snapshots_include_main_and_background_work(tmp_path: Path) -> None:
    script = tmp_path / "work.py"
    script.write_text(
        "import threading, time\n"
        "def background_work():\n    time.sleep(0.3)\n"
        "worker = threading.Thread(target=background_work, name='background-worker')\n"
        "worker.start()\nworker.join()\nprint('completed')\n"
    )
    result = _run(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "completed"
    assert result.stderr.count("Subprocess watchdog: Python stack snapshot") >= 2
    assert "MainThread" in result.stderr
    assert "background-worker" in result.stderr
    assert "background_work" in result.stderr
    assert str(script) in result.stderr


def test_script_arguments_and_exit_code_are_preserved(tmp_path: Path) -> None:
    script = tmp_path / "exit.py"
    script.write_text(
        "import json, sys\nprint(json.dumps(sys.argv))\nraise SystemExit(7)\n"
    )
    result = _run(script, "--help", "path with spaces", interval="150")
    assert result.returncode == 7
    assert json.loads(result.stdout) == [str(script), "--help", "path with spaces"]
    assert result.stderr == ""


def test_exception_stops_diagnostics_without_waiting_for_interval(
    tmp_path: Path,
) -> None:
    script = tmp_path / "failure.py"
    script.write_text("raise RuntimeError('expected failure')\n")
    result = _run(script, interval="150")
    assert result.returncode == 1
    assert "RuntimeError: expected failure" in result.stderr
    assert "Subprocess watchdog" not in result.stderr


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_interval_fails_before_running_script(
    tmp_path: Path, interval: str
) -> None:
    script = tmp_path / "unselected.py"
    script.write_text("raise AssertionError('script executed')\n")
    result = _run(script, interval=interval)
    assert result.returncode == 2
    assert "finite and positive" in result.stderr
    assert "script executed" not in result.stderr
