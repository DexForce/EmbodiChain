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

"""Protect complete-flow benchmarking from early-failure and partial-run claims."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark.testing.suite_runtime import _summarize


def _checkout(tmp_path: Path, fast_exit: int = 1) -> Path:
    root = tmp_path / "checkout"
    (root / "scripts/ci").mkdir(parents=True)
    (root / ".gitignore").write_text("*.executed\n__pycache__/\n")
    (root / "scripts/ci/run_test_plan.py").write_text("""
from pathlib import Path
def build_commands(plan, **kwargs): return []
def run_plan(plan, *, root, lanes, no_finalize_resource_lanes):
    build_commands(plan, lanes=lanes)
    lane = lanes[0]
    Path(root, lane + '.executed').write_text(lane)
    return FAST_EXIT if lane == 'fast' else 0
""".replace("FAST_EXIT", str(fast_exit)))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return root


@pytest.mark.parametrize("inside_ignored", [False, True])
def test_benchmark_continues_after_failure_and_preserves_exit(
    tmp_path: Path, inside_ignored: bool
) -> None:
    root = _checkout(tmp_path)
    if inside_ignored:
        with (root / ".gitignore").open("a") as stream:
            stream.write("results/\n")
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/testing/suite_runtime.py"
    )
    output = root / "results" if inside_ignored else tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--python",
            sys.executable,
            "--output",
            str(output),
            "--lane",
            "fast",
            "--lane",
            "sim",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    recorded = json.loads((output / "result.json").read_text())
    assert [(row["name"], row["exitcode"]) for row in recorded["steps"]] == [
        ("fast", 1),
        ("sim", 0),
    ]
    assert (root / "sim.executed").is_file()
    assert recorded["source_unchanged"]


@pytest.mark.parametrize("name", ["fast", "extra-unit"])
def test_extra_command_cannot_inherit_a_lane_exit(tmp_path: Path, name: str) -> None:
    root = _checkout(tmp_path, fast_exit=0)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/testing/suite_runtime.py"
    )
    plan = tmp_path / "extras.json"
    plan.write_text(
        json.dumps(
            [{"name": name, "argv": [sys.executable, "-c", "raise SystemExit(7)"]}]
        )
    )
    output = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--python",
            sys.executable,
            "--output",
            str(output),
            "--lane",
            "fast",
            "--extra-plan",
            str(plan),
        ],
        capture_output=True,
        text=True,
    )
    if name == "fast":
        assert result.returncode == 2
        assert "must be unique" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 7
        assert (
            json.loads((output / "result.json").read_text())["steps"][-1]["exitcode"]
            == 7
        )


def test_unignored_checkout_output_is_rejected_before_execution(tmp_path: Path) -> None:
    root = _checkout(tmp_path, fast_exit=0)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmark/testing/suite_runtime.py"
    )
    output = root / "unignored-results"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--python",
            sys.executable,
            "--output",
            str(output),
            "--lane",
            "fast",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "outside the checkout or ignored by Git" in result.stderr
    assert not output.exists()
    assert not (root / "fast.executed").exists()


def test_metrics_distinguish_unattempted_and_aborted_cases(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    selection = {
        "collection": [
            {"nodeid": name} for name in ["pass", "error", "crash", "unattempted"]
        ],
        "collection_reports": [{"nodeid": "optional_module", "outcome": "skipped"}],
    }
    for worker in (0, 1):
        (collection / f"gw{worker}.json").write_text(json.dumps(selection))
    phases = [
        {"nodeid": "pass", "phase": phase, "outcome": "passed"}
        for phase in ("setup", "call", "teardown")
    ] + [
        {"nodeid": "error", "phase": "setup", "outcome": "failed"},
        {"nodeid": "crash", "phase": "setup", "outcome": "passed"},
    ]
    (tmp_path / "phases-gw0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in phases)
    )
    stats = _summarize(tmp_path)
    assert stats == {
        "selected": 4,
        "attempted": 3,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "error": 1,
        "incomplete": 1,
        "unattempted": 1,
        "collection_skipped": 1,
        "collection_errors": 0,
    }
