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
"""CPU-only tests for the DexUni migration paths and pilot execution guard."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
SUITE = REPO / "scripts/benchmark/tech_reports/physics/experiments/dexuni"
spec = importlib.util.spec_from_file_location("dexuni_paths", SUITE / "paths.py")
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


def test_source_and_output_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEXUNI_INPUT_ROOT", raising=False)
    assert paths.REPO_ROOT == REPO
    assert paths.input_dir("s01c") == (
        Path.home() / ".cache/embodichain/report-assets/local/dexuni/pilot_v1/s01c"
    )
    assert (
        paths.output_dir("s01t")
        == REPO / "outputs/benchmarks/tech_reports/dexuni/development/s01t"
    )
    assert not any(name in sys.modules for name in ("dexsim", "isaaclab"))


def test_input_override_is_resolved_without_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEXUNI_INPUT_ROOT", str(tmp_path / "snapshot"))
    assert paths.input_dir("s02") == tmp_path / "snapshot/s02"
    assert not (tmp_path / "snapshot").exists()
    with pytest.raises(ValueError, match="Unknown"):
        paths.input_dir("../outside")


@pytest.mark.parametrize(
    "protected",
    [
        "scripts/benchmark/tech_reports/assets/physics/dexuni/s01c",
        "outputs/benchmarks/dexuni/pilot_v1/rewrite",
        "dexuni_benchmark/s02_soft_pick_place",
    ],
)
def test_output_rejects_sources_and_archives(protected: str) -> None:
    with pytest.raises(ValueError, match="must not overwrite"):
        paths.output_dir("s01c", REPO / protected)


def test_output_rejects_symlink_into_archive(tmp_path: Path) -> None:
    link = tmp_path / "archive"
    link.symlink_to(
        REPO / "outputs/benchmarks/dexuni/pilot_v1", target_is_directory=True
    )
    with pytest.raises(ValueError, match="must not overwrite"):
        paths.output_dir("s01c", link / "new")


def test_guard_requires_explicit_pilot_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEXUNI_ALLOW_PILOT", raising=False)
    with pytest.raises(SystemExit, match="Unqualified pilot"):
        paths.require_pilot()
    monkeypatch.setenv("DEXUNI_ALLOW_PILOT", "true")
    with pytest.raises(SystemExit):
        paths.require_pilot()
    monkeypatch.setenv("DEXUNI_ALLOW_PILOT", "1")
    paths.require_pilot()


@pytest.mark.parametrize(
    "task", ["s01c_compress", "s01t_stretch", "s02_soft_pick_place"]
)
def test_matrix_is_blocked_without_ack(task: str, tmp_path: Path) -> None:
    env = dict(os.environ, OUT_ROOT=str(tmp_path / "blocked"))
    env.pop("DEXUNI_ALLOW_PILOT", None)
    result = subprocess.run(
        ["bash", str(SUITE / task / "tools/run_matrix.sh"), "nominal"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "Blocked" in result.stderr
    assert not (tmp_path / "blocked").exists()


def test_matrix_keeps_failed_worker_exit(tmp_path: Path) -> None:
    worker = tmp_path / "fake-python"
    worker.write_text("#!/bin/sh\necho simulated-worker-failure >&2\nexit 7\n")
    worker.chmod(0o755)
    output = tmp_path / "run"
    env = dict(
        os.environ,
        DEXUNI_ALLOW_PILOT="1",
        PLATFORMS="embodichain",
        DEXUNI_EMBODICHAIN_PYTHON=str(worker),
        OUT_ROOT=str(output),
    )
    result = subprocess.run(
        ["bash", str(SUITE / "s01c_compress/tools/run_matrix.sh"), "nominal"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    journal = (output / "attempts.tsv").read_text()
    assert "started" in journal
    assert "finished\t7" in journal
    assert (
        "simulated-worker-failure"
        in next((output / "raw_data/_logs").glob("*.log")).read_text()
    )
    # A second invocation must not replace the first attempt journal.
    second = subprocess.run(
        ["bash", str(SUITE / "s01c_compress/tools/run_matrix.sh"), "nominal"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert second.returncode == 2
    assert (output / "attempts.tsv").read_text() == journal
