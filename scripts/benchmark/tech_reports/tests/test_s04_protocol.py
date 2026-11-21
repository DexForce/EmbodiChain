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
"""CPU contracts for named S04 replay and portable elasticity settings."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TECH = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "s04_common", TECH / "physics/experiments/dexuni/s04_common.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def _fixture(root: Path) -> np.ndarray:
    """Create a small approved snapshot with deliberately distinct joint targets."""
    (root / "inputs").mkdir()
    (root / "preview").mkdir()
    video = b"accepted preview fixture"
    (root / "preview/dexsim.mp4").write_bytes(video)
    (root / "manifest.json").write_text(json.dumps({"user_scene_approved": True}))
    (root / "inputs/s04_config.json").write_text(
        json.dumps(
            {
                "steps": 2,
                "dt": 0.1,
                "native_preview_sha256": hashlib.sha256(video).hexdigest(),
            }
        )
    )
    np.savez(
        root / "inputs/soft_mesh.npz",
        particle_q=np.zeros((4, 3)),
        tet_indices=[[0, 1, 2, 3]],
    )
    targets = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]], dtype=np.float32)
    np.savez(
        root / "inputs/joint_commands.npz",
        joint_names=["right_joint", "left_joint"],
        time=[0.0, 0.1, 0.2],
        targets=targets,
    )
    return targets


def test_named_replay_preserves_column_identity(tmp_path: Path) -> None:
    targets = _fixture(tmp_path)
    loaded = common.load_inputs(tmp_path)
    assert loaded["names"] == ["right_joint", "left_joint"]
    np.testing.assert_array_equal(loaded["targets"], targets)


@pytest.mark.parametrize("problem", ["duplicate_names", "wrong_time", "nonfinite"])
def test_invalid_commands_fail_before_simulation(tmp_path: Path, problem: str) -> None:
    targets = _fixture(tmp_path)
    names = ["right_joint", "left_joint"]
    time = [0.0, 0.1, 0.2]
    if problem == "duplicate_names":
        names = ["left_joint", "left_joint"]
    elif problem == "wrong_time":
        time = [0.0, 0.1, 0.1]
    else:
        targets[1, 0] = np.nan
    np.savez(
        tmp_path / "inputs/joint_commands.npz",
        joint_names=names,
        time=time,
        targets=targets,
    )
    with pytest.raises(ValueError):
        common.load_inputs(tmp_path)


def test_approval_and_preview_identity_are_required(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"user_scene_approved": False}))
    with pytest.raises(ValueError, match="approval"):
        common.load_inputs(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"user_scene_approved": True}))
    (tmp_path / "preview/dexsim.mp4").write_bytes(b"changed scene")
    with pytest.raises(ValueError, match="changed"):
        common.load_inputs(tmp_path)


def test_elasticity_conversion_preserves_approved_lame_coefficients() -> None:
    mu, lam = 6000.0, 15000.0
    youngs, nu = common.lame_to_youngs(mu, lam)
    assert youngs / (2 * (1 + nu)) == pytest.approx(mu)
    assert youngs * nu / ((1 + nu) * (1 - 2 * nu)) == pytest.approx(lam)


# Measurement helpers are pure CPU logic; importing them must not initialize CUDA.
import sys

sys.path.insert(0, str(TECH.parents[2]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni import (
    s04_measurement as measurement,
)


def test_timer_waits_for_completed_gpu_work() -> None:
    """The second synchronization is inside the timed interval."""
    events = []
    ticks = iter([10.0, 10.5])

    def clock() -> float:
        events.append("clock")
        return next(ticks)

    seconds = measurement.completed_step_seconds(
        lambda: events.append("advance"), lambda: events.append("sync"), clock
    )
    assert events == ["sync", "clock", "advance", "sync", "clock"]
    assert seconds == pytest.approx(0.5)


def test_warmup_is_excluded_from_steady_mean_and_tail() -> None:
    """A slow cold interval does not contaminate steady statistics."""
    seconds = np.r_[np.full(30, 5.0), [0.010, 0.020, 0.030]]
    result = measurement.summarize_steps(seconds)
    assert result["cost_time_ms"] == pytest.approx(20.0)
    assert result["median_ms"] == pytest.approx(20.0)
    assert result["p95_ms"] == pytest.approx(29.0)
    assert result["measured_steps"] == 3
    assert result["warmup_s"] == pytest.approx(150.0)


@pytest.mark.parametrize(
    "seconds",
    [
        np.ones(30),
        np.r_[np.ones(30), 0.0],
        np.r_[np.ones(30), np.nan],
        np.r_[np.ones(30), -0.1],
    ],
)
def test_invalid_timing_is_never_reported_as_latency(seconds: np.ndarray) -> None:
    with pytest.raises(ValueError):
        measurement.summarize_steps(seconds)


def _complete_measurement() -> tuple[dict, dict, np.ndarray]:
    """Create finite complete data with deliberately extreme final geometry."""
    steps = 32
    manifest = {
        "status": "completed",
        "camera_count": 0,
        "renderer": "no-render",
        "config": {"steps": steps, "dt": 1 / 60, "substeps": 10, "vbd_iterations": 12},
        "effective": {
            "cuda_graph_status": "captured",
            "num_substeps": 10,
            "vbd_iterations": 12,
            "outer_steps_per_command": 1,
            "simulated_elapsed_s": steps / 60,
        },
    }
    states = {
        key: np.full((steps + 1, 1), 1e6)
        for key in ("particles", "qpos", "target_q", "rigid_pose", "tcp")
    }
    states["time"] = np.arange(steps + 1) / 60
    return manifest, states, np.full(steps, 0.01)


def test_performance_eligibility_does_not_filter_final_shape() -> None:
    manifest, states, seconds = _complete_measurement()
    assert all(measurement.completion_checks(manifest, states, seconds).values())


@pytest.mark.parametrize(
    "problem",
    [
        "short_state",
        "nonfinite",
        "missing_key",
        "short_timing",
        "graph_disabled",
        "camera",
        "budget",
    ],
)
def test_incomplete_or_wrong_protocol_samples_are_labeled(problem: str) -> None:
    manifest, states, seconds = _complete_measurement()
    if problem == "short_state":
        states["qpos"] = states["qpos"][:-1]
    elif problem == "nonfinite":
        states["particles"][5] = np.nan
    elif problem == "missing_key":
        del states["rigid_pose"]
    elif problem == "short_timing":
        seconds = seconds[:-1]
    elif problem == "graph_disabled":
        manifest["effective"]["cuda_graph_status"] = "disabled"
    elif problem == "camera":
        manifest["camera_count"] = 1
    elif problem == "budget":
        manifest["effective"]["vbd_iterations"] = 10
    assert not all(measurement.completion_checks(manifest, states, seconds).values())


def test_backend_elapsed_time_catches_hidden_extra_outer_steps() -> None:
    """Configured substeps and labeled time are insufficient to prove step cadence."""
    manifest, states, seconds = _complete_measurement()
    manifest["effective"]["simulated_elapsed_s"] *= 10
    checks = measurement.completion_checks(manifest, states, seconds)
    assert checks["solver_budget"]
    assert checks["time_grid"]
    assert not checks["outer_step_cadence"]


def test_report_exports_boolean_warmup_and_unchanged_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise CSV parsing through the report, including excluded cold steps."""
    import csv
    from scripts.benchmark.tech_reports.physics.experiments.dexuni import run_s04

    monkeypatch.setattr(run_s04, "OUTPUT", tmp_path)
    (tmp_path / "measurement").mkdir()
    (tmp_path / "measurement/protocol.json").write_text(
        json.dumps({"input_sha256": {}})
    )
    attempts = []
    steps = 32
    seconds = np.r_[np.full(30, 0.9), [0.002, 0.002]]
    for platform in ("embodichain", "isaaclab"):
        run = Path("measurement") / platform / "r01"
        directory = tmp_path / run
        directory.mkdir(parents=True)
        preview = tmp_path / "preview" / platform
        preview.mkdir(parents=True)
        np.savez(directory / "model.npz", particle_q=np.zeros((1, 3)))
        np.savez(preview / "model.npz", particle_q=np.zeros((1, 3)))
        manifest, _, _ = _complete_measurement()
        manifest.update(
            input_sha256={},
            init_s=0.1,
            environment={"versions": {}},
            summary=measurement.summarize_steps(seconds),
        )
        (directory / "manifest.json").write_text(json.dumps(manifest))
        np.savez(
            directory / "states.npz",
            time=np.arange(steps + 1) / 60,
            qpos=np.zeros((steps + 1, 2)),
            target_q=np.zeros((steps + 1, 2)),
            tcp=np.zeros((steps + 1, 2, 3)),
            particles=np.zeros((steps + 1, 1, 3)),
            rigid_pose=np.tile(np.eye(4), (steps + 1, 1, 1)),
        )
        with (directory / "timing.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "frame",
                    "time_s",
                    "steady",
                    "step_s",
                    "command_s",
                    "readback_record_s",
                ]
            )
            for frame, duration in enumerate(seconds, 1):
                writer.writerow(
                    [frame, frame / 60, frame > 30, duration, 0.0001, 0.0001]
                )
        attempts.append(
            {
                "platform": platform,
                "repeat": 1,
                "run": str(run),
                "returncode": 0,
                "process_wall_s": 1.0,
            }
        )
    run_s04.build_report(attempts)
    rows = list(csv.DictReader((tmp_path / "measurement/results/steps.csv").open()))
    assert len(rows) == 2 * steps
    assert sum(row["steady"] == "True" for row in rows) == 4
    assert all((row["steady"] == "True") == (int(row["frame"]) > 30) for row in rows)
    summary = json.loads((tmp_path / "measurement/results/summary.json").read_text())
    assert summary["embodichain"]["cost_time_ms"]["mean"] == pytest.approx(2.0)
    assert (
        sum(
            line.startswith("| --- |")
            for line in (tmp_path / "REPORT.md").read_text().splitlines()
        )
        == 3
    )
