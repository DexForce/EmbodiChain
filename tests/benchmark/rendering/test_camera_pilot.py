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

"""CPU tests for the camera pilot's measurement and reporting contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest


def test_warmup_is_excluded_and_each_capture_counts_once() -> None:
    from scripts.benchmark.rendering.common import PilotCfg, measure

    now = 0.0
    calls = 0

    def capture() -> np.ndarray:
        nonlocal now, calls
        calls += 1
        now += 10.0 if calls <= 3 else 0.5
        return np.ones((4, 4, 3), dtype=np.uint8)

    result = measure(
        PilotCfg(width=4, height=4, warmup_frames=3, measured_frames=2),
        capture,
        clock=lambda: now,
    )
    assert result["measured_frames"] == 2
    assert result["window_s"] == 1.0
    assert result["camera_frames_per_s"] == 2.0
    assert result["capture_latency_s"] == [0.5, 0.5]


def test_invalid_image_is_not_counted_as_an_exposure() -> None:
    from scripts.benchmark.rendering.common import PilotCfg, measure

    cfg = PilotCfg(width=4, height=4, warmup_frames=0, measured_frames=1)
    with pytest.raises(ValueError, match="RGB"):
        measure(cfg, lambda: np.zeros((4, 4, 4), dtype=np.uint8))


@pytest.mark.parametrize(
    "values",
    [{"width": 0}, {"measured_frames": 0}, {"warmup_frames": -1}, {"width": True}],
)
def test_invalid_measurement_configuration_is_rejected(values: dict) -> None:
    from scripts.benchmark.rendering.common import PilotCfg

    with pytest.raises(ValueError):
        PilotCfg(**values)


def test_report_retains_failed_runs_and_does_not_claim_quality_speedup(
    tmp_path: Path,
) -> None:
    from scripts.benchmark.rendering.report import rebuild_report

    rows = [
        {
            "backend": "embodichain",
            "repeat": 0,
            "status": "completed",
            "config_sha256": "same-workload",
            "metrics": {
                "camera_frames_per_s": 2.0,
                "latency_p50_s": 0.5,
                "latency_p95_s": 0.6,
                "cpu_peak_rss_bytes": 1048576,
            },
            "quality_status": "not_qualified",
        },
        {
            "backend": "isaaclab",
            "repeat": 0,
            "status": "failed",
            "error": "worker crashed",
        },
    ]
    (tmp_path / "runs.json").write_text(json.dumps(rows))
    path = rebuild_report(tmp_path)
    report = path.read_text()
    assert "worker crashed" in report
    assert "not_qualified" in report
    assert "2.000" in report
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["speedup"] is None
    assert summary["completed_runs"] == 1
    assert summary["failed_runs"] == 1


def test_report_entrypoint_does_not_import_simulators(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "runs.json").write_text("[]")
    script = """
import importlib.abc
import sys
class BlockSim(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'dexsim', 'isaaclab', 'isaacsim'}:
            raise AssertionError('Unexpected simulator import: ' + fullname)
sys.meta_path.insert(0, BlockSim())
from scripts.benchmark.__main__ import main
main(['camera-pilot', '--report-only', sys.argv[1]])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "report.md").exists()


def test_legacy_failure_does_not_erase_a_known_platform_median(tmp_path: Path) -> None:
    from scripts.benchmark.rendering.report import rebuild_report

    rows = [
        {
            "backend": "a",
            "repeat": 0,
            "status": "completed",
            "config_sha256": "x",
            "metrics": {"camera_frames_per_s": 2.0, "latency_p50_s": 0.5},
        },
        {"backend": "a", "repeat": 1, "status": "failed", "error": "early crash"},
        {
            "backend": "b",
            "repeat": 0,
            "status": "completed",
            "config_sha256": "y",
            "metrics": {"camera_frames_per_s": 10.0, "latency_p50_s": 0.1},
        },
    ]
    (tmp_path / "runs.json").write_text(json.dumps(rows))
    rebuild_report(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["platforms"]["a"]["median_camera_frames_per_s"] == 2.0
    assert summary["platforms"]["a"]["scheduled"] == 2
    assert summary["speedup"] is None


@pytest.mark.parametrize(
    ("code", "timeout_s", "status"),
    [
        ("raise SystemExit(7)", 5.0, "failed"),
        ("import time; time.sleep(60)", 0.1, "timeout"),
    ],
)
def test_worker_failure_keeps_a_result_and_log(
    tmp_path: Path, code: str, timeout_s: float, status: str
) -> None:
    from scripts.benchmark.rendering.run_benchmark import execute_worker

    result = execute_worker(
        [sys.executable, "-c", code],
        tmp_path,
        backend="isaaclab",
        repeat=0,
        timeout_s=timeout_s,
    )
    assert result["status"] == status
    assert (tmp_path / "worker.log").exists()
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["status"] == status
    assert saved["error"]


@pytest.mark.parametrize("cancel_signal", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_reaps_worker_and_preserves_result(
    tmp_path: Path, cancel_signal: int
) -> None:
    root = Path(__file__).resolve().parents[3]
    pid_file = tmp_path / "worker.pid"
    script = """
import sys
from pathlib import Path
from scripts.benchmark.rendering.run_benchmark import execute_worker
child = 'import os,time; from pathlib import Path; Path(' + repr(sys.argv[1]) + ').write_text(str(os.getpid())); time.sleep(60)'
execute_worker([sys.executable, '-c', child], Path(sys.argv[2]), backend='isaaclab', repeat=0, timeout_s=60)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(pid_file), str(tmp_path / "run")],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists(), "Worker did not start"
        child_pid = int(pid_file.read_text())
        parent.send_signal(cancel_signal)
        parent.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        result = json.loads((tmp_path / "run/result.json").read_text())
        assert result["status"] == "interrupted"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cli_uses_the_requested_virtualenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import venv
    from scripts.benchmark.rendering import run_benchmark

    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    worker_dir = tmp_path / "source/scripts/benchmark/rendering"
    worker_dir.mkdir(parents=True)
    (worker_dir / "worker.py").write_text("""
import argparse, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a, _ = p.parse_known_args()
r = {'status': 'completed', 'interpreter_prefix': sys.prefix,
     'metrics': {'camera_frames_per_s': 1.0, 'latency_p50_s': 1.0, 'latency_p95_s': 1.0}}
Path(a.output, 'result.json').write_text(json.dumps(r))
""")
    (worker_dir / "camera_pilot.json").write_text("{}")
    monkeypatch.setattr(run_benchmark, "__file__", str(worker_dir / "run_benchmark.py"))
    run_benchmark.main(
        [
            "--backend",
            "embodichain",
            "--repeats",
            "1",
            "--embodichain-python",
            str(environment / "bin/python"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    result_file = next((tmp_path / "output").glob("*/r00_embodichain/result.json"))
    assert json.loads(result_file.read_text())["interpreter_prefix"] == str(environment)
