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

"""Exercise the actual process boundary without requiring Blender or DexSim."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from embodichain.gen_sim import _blender_runtime as runtime


def test_worker_uses_files_and_does_not_inherit_business_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "worker with spaces.py"
    worker.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "print('Native output is not a JSON response')\n"
        "result = {'value': request['arguments']['value'], 'root': str(Path(sys.argv[1]).parent), "
        "'pythonpath': os.getenv('PYTHONPATH'), 'libraries': os.getenv('LD_LIBRARY_PATH')}\n"
        "Path(sys.argv[2]).write_text(json.dumps({'result': result}))\n"
    )
    monkeypatch.setattr(runtime, "_WORKER", worker)
    monkeypatch.setenv("EMBODICHAIN_BLENDER_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", "/business/site-packages")
    monkeypatch.setenv("CONDA_PREFIX", "/business")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/business/lib:/usr/local/nvidia/lib64")
    result = runtime._run_blender_job("render", {"value": "asset with spaces.glb"})
    assert result["value"] == "asset with spaces.glb"
    assert result["pythonpath"] is None
    assert result["libraries"] == "/usr/local/nvidia/lib64"
    assert not Path(result["root"]).exists()


def test_worker_failure_contains_native_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "broken.py"
    worker.write_text("raise RuntimeError('synthetic render failure')\n")
    monkeypatch.setattr(runtime, "_WORKER", worker)
    monkeypatch.setenv("EMBODICHAIN_BLENDER_PYTHON", sys.executable)
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        runtime._run_blender_job("render", {})


def test_timed_out_worker_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "hanging.py"
    pid_file = tmp_path / "child.pid"
    worker.write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(request['arguments']['pid_file']).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(runtime, "_WORKER", worker)
    monkeypatch.setenv("EMBODICHAIN_BLENDER_PYTHON", sys.executable)
    with pytest.raises(subprocess.TimeoutExpired):
        runtime._run_blender_job("render", {"pid_file": str(pid_file)}, timeout=1.0)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_missing_blender_interpreter_has_an_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EMBODICHAIN_BLENDER_PYTHON", str(tmp_path / "missing-python"))
    with pytest.raises(RuntimeError, match="EMBODICHAIN_BLENDER_PYTHON"):
        runtime._run_blender_job("render", {})
