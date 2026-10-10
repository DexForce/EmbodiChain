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

"""CPU-only tests for drive-verification metrics and inertia auditing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.robotics.cobotmagic_drives import (
    _append_physical_report,
    _held_motion_metrics,
    _held_window,
    _audit_urdf,
    _physical_task_metrics,
    _tracking_metrics,
    _write_report,
    run_all_benchmarks,
)


@pytest.mark.parametrize(
    ("mass", "diagonal", "expected"),
    [
        (1.0, (1, 1, 1), True),
        (0.0, (1, 1, 1), False),
        (1.0, (-1, 1, 1), False),
        (1.0, (1, 1, 3), False),
    ],
)
def test_inertia_audit_rejects_nonphysical_mass_or_tensor(
    tmp_path: Path, mass: float, diagonal: tuple[float, ...], expected: bool
) -> None:
    path = tmp_path / "robot.urdf"
    path.write_text(
        f'<robot><link name="test"><inertial><mass value="{mass}"/>'
        f'<inertia ixx="{diagonal[0]}" iyy="{diagonal[1]}" izz="{diagonal[2]}" '
        'ixy="0" ixz="0" iyz="0"/></inertial></link></robot>',
        encoding="utf-8",
    )
    assert _audit_urdf(path)["valid"] is expected


def test_imported_benchmark_drains_cleanup_after_preparation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A benchmark host must not retain queued native worlds after an error."""
    from embodichain import data
    from embodichain.lab import sim
    from embodichain.lab.sim.robots import CobotMagicCfg
    from scripts.benchmark.robotics import cobotmagic_drives

    events = []

    class FakeManager:
        def __init__(self, cfg: object) -> None:
            pass

        def add_robot(self, cfg: object) -> object:
            return object()

        def prepare(self) -> None:
            raise RuntimeError("injected prepare failure")

        def destroy(self, *, exit_process: bool) -> None:
            assert exit_process is False
            events.append("destroy")

        @staticmethod
        def flush_cleanup_queue() -> None:
            events.append("flush")

    monkeypatch.setattr(sim, "SimulationManager", FakeManager)
    monkeypatch.setattr(data, "get_data_path", lambda path: path)
    monkeypatch.setattr(cobotmagic_drives, "_audit_urdf", lambda path: {"valid": True})
    monkeypatch.setattr(
        CobotMagicCfg, "from_dict", classmethod(lambda cls, values: object())
    )
    with pytest.raises(RuntimeError, match="injected prepare failure"):
        run_all_benchmarks(tmp_path, ["configured"])
    assert events == ["destroy", "flush"]


def test_tracking_metrics_detect_velocity_violation_despite_perfect_positions() -> None:
    positions = np.zeros((100, 6))
    velocities = positions.copy()
    velocities[20, 5] = 3.3  # Wrist limit is 3 rad/s, not the shoulder's 5.
    result = _tracking_metrics(
        positions,
        velocities,
        positions,
        kp=np.ones(6),
        kd=np.ones(6),
        effort_limit=100,
        velocity_limits=np.array([5, 5, 5, 5, 5, 3]),
        case="sine",
        dt=0.01,
    )
    assert result["max_joint_rmse_deg"] == 0
    assert result["velocity_limit_ratio"] == pytest.approx(1.1)
    assert not result["success"]


def test_step_settling_requires_staying_inside_the_error_band() -> None:
    target = np.zeros((100, 6))
    target[:, 0] = 0.15
    measured = target.copy()
    measured[10:20, 0] -= 0.02  # An early crossing is not final settling.
    result = _tracking_metrics(
        measured,
        np.zeros_like(measured),
        target,
        kp=np.ones(6),
        kd=np.zeros(6),
        effort_limit=100,
        velocity_limits=np.ones(6),
        case="step_1",
        dt=0.01,
        moved_joint=0,
    )
    assert result["settling_s"] == pytest.approx(0.21)
    assert result["success"]


@pytest.mark.parametrize(("limit", "speed"), [(np.pi, 4.0), (0.25, 0.3)])
def test_tracking_uses_supplied_arm_or_gripper_speed_caps(
    limit: float, speed: float
) -> None:
    """Stricter native caps must reject motion allowed by the old defaults."""
    positions = np.zeros((100, 1))
    velocities = positions.copy()
    velocities[20, 0] = speed
    metrics = _tracking_metrics(
        positions,
        velocities,
        positions,
        kp=np.ones(1),
        kd=np.ones(1),
        effort_limit=100,
        velocity_limits=np.array([limit]),
        case="sine",
        dt=0.01,
    )
    assert metrics["velocity_limit_ratio"] == pytest.approx(speed / limit)
    assert not metrics["success"]


def test_report_has_three_tables_and_includes_every_profile(tmp_path: Path) -> None:
    rows = []
    for profile, success in (("legacy", False), ("source_limits", True)):
        metrics = _tracking_metrics(
            np.zeros((100, 6)),
            np.zeros((100, 6)),
            np.zeros((100, 6)),
            kp=np.ones(6),
            kd=np.zeros(6),
            effort_limit=100,
            velocity_limits=np.ones(6),
            case="sine",
            dt=0.01,
        )
        rows.append(
            {
                "case": "sine",
                "profile": profile,
                "cost_time_ms": 1,
                "cpu_delta_mb": 0,
                "gpu_delta_mb": 0,
                "peak_gpu_mb": 0,
                "tcp_error_mm": 0,
                **metrics,
                "success_rate": float(success),
            }
        )
    path = tmp_path / "report.md"
    _write_report(path, rows)
    report = path.read_text()
    assert sum(line.startswith("| ---") for line in report.splitlines()) == 3
    leaderboard = report.split("## Leaderboard")[1]
    assert "| 1 | source_limits |" in leaderboard
    assert "| 2 | legacy |" in leaderboard
    measured = _physical_task_metrics(
        np.empty((0, 1, 4, 4)), np.eye(4)[None], [False], np.zeros(1), np.zeros(1)
    )
    _append_physical_report(path, [{"bottle_masses_kg": [0.25], "rows": measured}])
    report = path.read_text()
    assert "250 g: 0/1 accepted" in report
    assert "maximum return error unavailable" in report
    assert sum(line.startswith("| ---") for line in report.splitlines()) == 3


def test_projected_task_success_requires_measured_lift_and_tilt() -> None:
    initial = np.eye(4)[None]
    poses = np.repeat(initial[None], 3, axis=0)
    result = _physical_task_metrics(
        poses, initial, [True], np.array([0.5]), np.array([0.1])
    )
    assert result[0]["runtime_success"]
    assert not result[0]["accepted"]
    poses[1, 0, 2, 3] = 0.1
    poses[1, 0, :3, :3] = [[1, 0, 0], [0, 0.5, -np.sqrt(0.75)], [0, np.sqrt(0.75), 0.5]]
    result = _physical_task_metrics(
        poses, initial, [True], np.array([0.5]), np.array([0.1])
    )
    assert result[0]["accepted"]


def test_failed_task_without_actions_cannot_pass_physical_checks() -> None:
    result = _physical_task_metrics(
        np.empty((0, 1, 4, 4)), np.eye(4)[None], [False], np.zeros(1), np.zeros(1)
    )
    assert not result[0]["accepted"]
    assert result[0]["return_error_m"] is None


def _grasp_trace() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eef = np.repeat(np.eye(4)[None, None], 10, axis=0)
    eef[:, 0, 0, 3] = np.linspace(0, 0.2, 10)
    bottle = eef.copy()
    bottle[:, 0, 2, 3] = 0.02
    cup = np.repeat(np.eye(4)[None, None], 10, axis=0)
    cup[:, 0, 1, 3] = 0.5
    return bottle, eef, cup, np.zeros((10, 1), dtype=int)


def _grasp_metrics(bottle, eef, cup, contacts, **kwargs):
    return _held_motion_metrics(
        bottle,
        eef,
        cup,
        contacts,
        held_start_s=0.1,
        release_s=kwargs.pop("release_s", 1.0),
        dt=0.1,
        bottle_z_bounds=(-0.086, 0.07),
        bottle_radius=0.03,
        cup_radius=0.055,
        **kwargs,
    )[0]


def test_rigid_held_motion_has_no_slip_despite_world_motion() -> None:
    metrics = _grasp_metrics(*_grasp_trace())
    assert metrics["grasp_and_clearance_pass"]
    assert metrics["max_grasp_drift_mm"] == pytest.approx(0)


def test_physical_report_exposes_contact_failure_despite_projected_success(
    tmp_path: Path,
) -> None:
    bottle, eef, cup, contacts = _grasp_trace()
    contacts[4, 0] = 1
    row = _physical_task_metrics(bottle, bottle[0], [True], np.zeros(1), np.zeros(1))[0]
    row.update(_grasp_metrics(bottle, eef, cup, contacts))
    assert row["runtime_success"] and not row["grasp_and_clearance_pass"]
    path = tmp_path / "report.md"
    _append_physical_report(path, [{"bottle_masses_kg": [0.25], "rows": [row]}])
    report = path.read_text()
    assert "maximum grasp drift 0.00 mm" in report
    assert "cup contact substeps 1" in report


def test_physical_report_marks_missing_trials_as_unmeasured(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    _append_physical_report(path, [])
    assert "No measured motion trials completed" in path.read_text()


def test_in_hand_drop_fails_even_when_the_bottle_remains_above_the_table() -> None:
    bottle, eef, cup, contacts = _grasp_trace()
    bottle[5:, 0, 2, 3] -= 0.01
    metrics = _grasp_metrics(bottle, eef, cup, contacts)
    assert not metrics["grasp_and_clearance_pass"]
    assert metrics["max_grasp_drift_mm"] == pytest.approx(10)


def test_in_hand_rotation_is_measured_relative_to_the_gripper() -> None:
    bottle, eef, cup, contacts = _grasp_trace()
    theta = np.deg2rad(5)
    bottle[5:, 0, :3, :3] = [
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ]
    metrics = _grasp_metrics(bottle, eef, cup, contacts)
    assert not metrics["grasp_and_clearance_pass"]
    assert metrics["max_grasp_rotation_deg"] == pytest.approx(5)


@pytest.mark.parametrize("failure", ["contact", "cup_motion", "clearance", "dropped"])
def test_cup_safety_rejects_contacts_motion_or_incomplete_evidence(
    failure: str,
) -> None:
    bottle, eef, cup, contacts = _grasp_trace()
    if failure == "contact":
        contacts[4, 0] = 1
    elif failure == "cup_motion":
        cup[5:, 0, 1, 3] += 0.003
    elif failure == "clearance":
        cup[:, 0, 1, 3] = 0.03
    metrics = _grasp_metrics(
        bottle, eef, cup, contacts, dropped_contacts=int(failure == "dropped")
    )
    assert not metrics["grasp_and_clearance_pass"]


def test_normal_release_is_outside_the_held_observation_window() -> None:
    bottle, eef, cup, contacts = _grasp_trace()
    bottle[5:, 0, 2, 3] -= 0.2
    metrics = _grasp_metrics(bottle, eef, cup, contacts, release_s=0.5)
    assert metrics["grasp_and_clearance_pass"]


def test_held_window_keeps_lift_and_release_across_program_segments() -> None:
    def segment(semantic: str, phase: str, timestamp: float) -> dict:
        return {
            "metadata": {
                "runtime": {
                    "calls": [
                        {
                            "semantic_id": semantic,
                            "events": [
                                {
                                    "kind": "trajectory_segment_entered",
                                    "segment_name": phase,
                                    "timestamp": timestamp,
                                }
                            ],
                        }
                    ]
                }
            }
        }

    metadata = {
        "segments": [segment("pick", "lift", 0.8), segment("place", "release", 2.0)]
    }
    assert _held_window(metadata, 3.0) == (0.8, 2.0)
