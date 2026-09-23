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

"""Pure core contracts, independent of camera images and simulator packages."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_generic_measurement_excludes_warmup_and_retains_samples() -> None:
    from scripts.benchmark.core.measurement import measure_loop

    now = 0.0
    count = 0

    def operation() -> int:
        nonlocal now, count
        count += 1
        now += 10.0 if count <= 2 else 0.25
        return count

    result = measure_loop(operation, warmup=2, iterations=4, clock=lambda: now)
    assert result.count == 4
    assert result.window_s == 1.0
    assert result.operations_per_s == 4.0
    assert result.latencies_s == (0.25, 0.25, 0.25, 0.25)


def test_atomic_json_rejects_nonfinite_without_damaging_previous_file(
    tmp_path: Path,
) -> None:
    from scripts.benchmark.core.artifacts import read_json_object, write_json

    path = tmp_path / "result.json"
    write_json(path, {"value": 1})
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert read_json_object(path) == {"value": 1}


def test_repeat_schedule_is_frozen_and_alternates_platform_order() -> None:
    from scripts.benchmark.core.execution import repeat_schedule

    assert repeat_schedule(("a", "b"), 3) == (
        ("a", 0),
        ("b", 0),
        ("b", 1),
        ("a", 1),
        ("a", 2),
        ("b", 2),
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("[]", "object"),
        ('{"status": []}', "status"),
        ('{"status":"completed","config_sha256":[]}', "config_sha256"),
        ('{"status":"completed","metrics":{"value":1e999}}', "finite"),
    ],
)
def test_malformed_worker_result_is_preserved_as_a_failure(
    tmp_path: Path, payload: str, reason: str
) -> None:
    from scripts.benchmark.core.execution import execute_worker

    code = (
        "from pathlib import Path; Path("
        + repr(str(tmp_path / "result.json"))
        + ").write_text("
        + repr(payload)
        + ")"
    )
    result = execute_worker(
        [sys.executable, "-S", "-c", code],
        tmp_path,
        backend="fixture",
        repeat=0,
        timeout_s=10,
    )
    assert result["status"] == "failed"
    assert reason in result["error"]
    assert (tmp_path / "invalid-worker-result.json").read_text() == payload


def test_experiment_retains_failed_runs_and_plan_metadata(tmp_path: Path) -> None:
    from scripts.benchmark.core.execution import run_experiment
    from scripts.benchmark.core.records import RunSpec

    success = tmp_path / "ok"
    failure = tmp_path / "failure"
    code = (
        "from pathlib import Path; Path("
        + repr(str(success / "result.json"))
        + ').write_text(\'{"status":"completed","metrics":{"value":2.0}}\')'
    )
    runs = (
        RunSpec("a", 0, (sys.executable, "-S", "-c", code), success, case_id="case1"),
        RunSpec(
            "b",
            0,
            (sys.executable, "-S", "-c", "raise SystemExit(3)"),
            failure,
            case_id="case1",
        ),
    )
    records = run_experiment(tmp_path, runs, experiment_id="scalar")
    assert [r["status"] for r in records] == ["completed", "failed"]
    assert {r["case_id"] for r in records} == {"case1"}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["experiment_id"] == "scalar"
    assert len(manifest["runs"]) == 2
    assert json.loads((tmp_path / "runs.json").read_text()) == records


def test_core_and_reporting_import_without_site_packages() -> None:
    root = Path(__file__).resolve().parents[3]
    code = """
from scripts.benchmark.core import artifacts, execution, measurement, provenance, records
from scripts.benchmark.reporting import aggregation, comparison
import sys
assert not any(name in sys.modules for name in ('torch','numpy','dexsim','isaaclab','isaacsim'))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
