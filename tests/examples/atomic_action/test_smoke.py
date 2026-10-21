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

"""Real CLI smoke coverage for single- and dual-arm Cartesian preparation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.subprocess_sim]
# Full scenes contain up to roughly 190 simulated seconds, plus cold planning.
# This is a bounded correctness smoke, not a wall-time performance assertion.
SMOKE_TIMEOUT_SECONDS = 600
SMOKE_TRACEBACK_INTERVAL_SECONDS = 150


@pytest.mark.parametrize("robot", ["ur5", "franka"])
@pytest.mark.parametrize("skill", ["place", "handover"])
def test_atomic_cli_plans_and_replays_nominal_scene(
    tmp_path: Path, robot: str, skill: str
) -> None:
    """Each preset produces finite, timed commands and finishes native cleanup."""
    output = tmp_path / f"{skill}_{robot}"
    # exec replaces the CLI interpreter. Arm the watchdog in the interpreter
    # that actually owns playback, while still exercising the real launcher.
    watchdog = (
        "import faulthandler, runpy, sys\n"
        f"faulthandler.dump_traceback_later({SMOKE_TRACEBACK_INTERVAL_SECONDS}, repeat=True)\n"
        "sys.argv = sys.argv[1:]\n"
        "try:\n"
        "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
        "finally:\n"
        "    faulthandler.cancel_dump_traceback_later()\n"
    )
    entrypoint = (
        "import os, runpy\n"
        "original_execv = os.execv\n"
        "def exec_with_watchdog(executable, arguments):\n"
        f"    original_execv(executable, [executable, '-u', '-c', {watchdog!r}, *arguments[1:]])\n"
        "os.execv = exec_with_watchdog\n"
        "runpy.run_module('embodichain', run_name='__main__')\n"
    )
    command = [
        sys.executable,
        "-u",
        "-c",
        entrypoint,
        "run-example",
        "sim/atomic_action",
        "--skill",
        skill,
        "--physics",
        "default",
        "--robot",
        robot,
        "--planner",
        "trapezoidal",
        "--headless",
        "--renderer",
        "no-render",
        "--num-envs",
        "1",
        "--no-expansion",
        "--no-save-video",
        "--output-dir",
        str(output),
    ]
    log_path = tmp_path / f"{skill}_{robot}.log"
    with log_path.open("w") as log:
        try:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=SMOKE_TIMEOUT_SECONDS,
                # One environment should not inherit the runner's potentially
                # hundreds of BLAS/OpenMP workers for small planning kernels.
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"CLI exceeded {SMOKE_TIMEOUT_SECONDS}s; log: {log_path}\n"
                + log_path.read_text()[-20000:],
                pytrace=False,
            )
    assert result.returncode == 0, log_path.read_text()[-20000:]
    manifest = json.loads((output / "expansion_manifest.json").read_text())
    assert manifest["status"] == "executed"
    assert manifest["variants"] == manifest["num_envs"] == 1
    assert manifest["replays"]
    for replay in manifest["replays"]:
        assert all(len(stage["rows"]) == 1 for stage in replay["stages"])
        trajectory = torch.load(replay["trajectory_path"], weights_only=True)
        positions, dt = trajectory["positions"], trajectory["dt"]
        assert positions.shape[0] == 1
        assert positions.shape[1] > 1
        assert dt.shape == positions.shape[:2]
        assert torch.isfinite(positions).all()
        assert torch.isfinite(dt).all()
        assert (dt >= 0).all()
        assert (dt.sum(dim=1) > 0).all()
