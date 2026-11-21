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


@pytest.fixture(autouse=True)
def _clear_blender_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EMBODICHAIN_BLENDER_EXECUTABLE", "EMBODICHAIN_BLENDER_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EMBODICHAIN_ENV_FILE", str(tmp_path / "missing.env"))


@pytest.mark.parametrize("system_prefix", [False, True])
def test_worker_uses_files_and_does_not_inherit_business_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system_prefix: bool
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
    if system_prefix:
        monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setenv("EMBODICHAIN_BLENDER_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", "/business/site-packages")
    monkeypatch.setenv("CONDA_PREFIX", "/business")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/business/lib:/usr/local/nvidia/lib64")
    result = runtime._run_blender_job("render", {"value": "asset with spaces.glb"})
    assert result["value"] == "asset with spaces.glb"
    assert result["pythonpath"] is None
    assert result["libraries"] == "/usr/local/nvidia/lib64"
    assert not Path(result["root"]).exists()


@pytest.mark.parametrize("selection", ["process", "dotenv"])
def test_blender_application_runs_the_same_worker_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    worker = tmp_path / "worker with spaces.py"
    worker.write_text(runtime._WORKER.read_text())
    (tmp_path / "_blender_operations.py").write_text(
        "def _render_object_front_top_views(*, value):\n"
        "    return {'value': value}\n"
    )
    application = tmp_path / "blender application"
    application.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        "assert '--background' in sys.argv\n"
        "assert '--factory-startup' in sys.argv\n"
        "assert '--disable-autoexec' in sys.argv\n"
        "assert sys.argv[sys.argv.index('--python-exit-code') + 1] == '1'\n"
        "runpy.run_path(sys.argv[sys.argv.index('--python') + 1], run_name='__main__')\n"
    )
    application.chmod(0o755)
    monkeypatch.setattr(runtime, "_WORKER", worker)
    dotenv = tmp_path / "blender.env"
    monkeypatch.setenv("EMBODICHAIN_ENV_FILE", str(dotenv))
    if selection == "process":
        dotenv.write_text("EMBODICHAIN_BLENDER_EXECUTABLE=/unused/blender\n")
        monkeypatch.setenv("EMBODICHAIN_BLENDER_EXECUTABLE", str(application))
    else:
        dotenv.write_text(f'EMBODICHAIN_BLENDER_EXECUTABLE="{application}"\n')
    monkeypatch.setenv("EMBODICHAIN_BLENDER_PYTHON", str(tmp_path / "unused-python"))
    assert runtime._run_blender_job(
        "front_top", {"value": "asset with spaces.glb"}
    ) == {"value": "asset with spaces.glb"}
    if selection == "dotenv":
        assert "EMBODICHAIN_BLENDER_EXECUTABLE" not in os.environ


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


@pytest.mark.parametrize(
    "setting", ["EMBODICHAIN_BLENDER_PYTHON", "EMBODICHAIN_BLENDER_EXECUTABLE"]
)
def test_missing_blender_runtime_has_an_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    monkeypatch.setenv(setting, str(tmp_path / "missing-runtime"))
    with pytest.raises(RuntimeError, match=setting):
        runtime._run_blender_job("render", {})
