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
"""Numeric agent operations preserve caller ownership and semantic edit scope."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodichain.data_pipeline.real2sim.adaptation.inputs import RobotModel
from embodichain.data_pipeline.real2sim.adaptation.trajectory_tools import (
    TrajectoryTools,
    interpolate_joint_knots,
    read_replay_segment,
    validate_segment_revision,
)


@pytest.mark.parametrize("organized", [False, True])
def test_replay_segment_uses_clocks_and_preserves_contact_coverage(
    tmp_path: Path,
    organized: bool,
) -> None:
    diagnostics = tmp_path / "diagnostics" if organized else tmp_path
    diagnostics.mkdir(exist_ok=True)
    videos = tmp_path / "videos" if organized else tmp_path
    videos.mkdir(exist_ok=True)
    (videos / "simulation.mp4").touch()
    np.savez(
        diagnostics / "physics_rollout.npz",
        source_timestamps=[0, 0.11, 0.22, 0.34],
        timestamps=[0, 0.1, 0.2, 0.3, 0.4],
        command_timestamps=[0, 0.15, 0.30, 0.45],
        joint_names=["jaw"],
        qpos_observed=np.arange(5)[:, None],
        qpos_command=np.arange(10, 14)[:, None],
        object_names=["cup"],
        object_poses=np.tile(np.eye(4), (5, 1, 1, 1)),
    )
    contact = dict(
        actors={"1": "cup", "2": "finger"},
        samples=[dict(timestamp=t, user_ids=[]) for t in [0.1, 0.2, 0.3]],
        interval_samples=[
            dict(start_timestamp=0.1, timestamp=0.2, user_ids=[[1, 2]]),
            dict(start_timestamp=0.2, timestamp=0.3, user_ids=[]),
            dict(start_timestamp=0.3, timestamp=0.4, user_ids=[]),
        ],
    )
    path = diagnostics / "contacts.json"
    path.write_text(json.dumps(contact))
    segment = read_replay_segment(tmp_path, 1, 2)
    assert segment["time_range_s"] == [0.11, 0.22]
    np.testing.assert_array_equal(segment["observed_sample_indices"], [2])
    np.testing.assert_array_equal(segment["qpos_observed"], [[2]])
    np.testing.assert_array_equal(segment["qpos_command"], [[11]])
    assert [r["timestamp"] for r in segment["last_substep_contacts"]] == [0.2]
    assert segment["interval_contacts"] == contact["interval_samples"][:2]
    assert segment["videos"] == {"simulation": str(videos / "simulation.mp4")}
    # Old recordings have no interval coverage; empty latest samples cannot
    # establish absence of a short collision between observation boundaries.
    del contact["interval_samples"]
    path.write_text(json.dumps(contact))
    assert read_replay_segment(tmp_path, 1, 2)["interval_contacts"] is None


@pytest.fixture
def robot_model(tmp_path: Path) -> RobotModel:
    names = tuple(f"j{i}" for i in range(8))
    links = '<link name="base"/>' + "".join(
        f'<link name="link{i}"><collision><geometry><box size="0.02 0.02 0.02"/>'
        "</geometry></collision></link>"
        for i in range(8)
    )
    joints = "".join(
        f'<joint name="j{i}" type="{"revolute" if i < 6 else "prismatic"}">'
        f'<parent link="{"base" if i == 0 else "link" + str(min(i - 1, 5))}"/>'
        f'<child link="link{i}"/><origin xyz="0.05 0 0.02"/>'
        f'<axis xyz="0 0 1"/><limit lower="{-3 if i < 6 else 0}" upper="{3 if i < 6 else .05}" effort="10" velocity="1"/></joint>'
        for i in range(8)
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(f'<robot name="test">{links}{joints}</robot>')
    base = np.eye(4)
    base[:3, 3] = [0.4, 0.3, 0.2]
    model = RobotModel(
        urdf, names, ("arm",), ("link5",), np.eye(4), (), 0.0, base_pose_world=base
    )
    return model


def test_joint_limit_report_keeps_out_of_range_recordings_unchanged(
    robot_model: RobotModel,
) -> None:
    tools = TrajectoryTools(robot_model)
    recording = np.zeros((6, 8))
    recording[0:2, 0] = -3.02
    recording[4, 2] = 3.03
    original = recording.copy()
    report = tools.joint_limit_report(recording)["arm"]
    assert report["out_of_limits_sample_ranges"] == [[0, 2], [4, 5]]
    np.testing.assert_allclose(report["max_excess_rad"], [0.02, 0, 0.03, 0, 0, 0])
    np.testing.assert_array_equal(recording, original)


@pytest.mark.parametrize("robot_type", ["cobotmagic", "rh20t-franka"])
def test_prepared_loader_preserves_source_and_acquisition_clocks(
    robot_model: RobotModel, tmp_path: Path, robot_type: str
) -> None:
    from embodichain.data_pipeline.real2sim.adaptation.prepared import (
        load_prepared_source,
    )

    (tmp_path / "robot.json").write_text(
        json.dumps(
            dict(
                urdf=robot_model.urdf.name,
                robot_type=robot_type,
                side_names=list(robot_model.side_names),
                tcp_links=list(robot_model.tcp_links),
                flange_to_tcp=robot_model.flange_to_tcp.tolist(),
            )
        )
    )
    (tmp_path / "robot_visuals.json").write_text('{"visuals": []}')
    (tmp_path / "original_front.mp4").touch()
    (tmp_path / "first_front.png").touch()
    qpos = np.arange(24).reshape(3, 8) * 0.001
    observation = np.array([0.0, 0.1, 0.2])
    acquisition = np.array([10.0, 10.04, 10.11])
    path = tmp_path / "trajectory_original.npz"
    np.savez(
        path,
        qpos=qpos,
        timestamps=observation,
        acquisition_timestamps=acquisition,
        source_frame_indices=[0, 3, 6],
        joint_names=robot_model.joint_names,
        front_K=np.eye(3),
        front_camera=np.eye(4)[None],
    )
    original_bytes = path.read_bytes()
    loaded = load_prepared_source(tmp_path, "Move the cup")
    expected = (
        acquisition - acquisition[0] if robot_type.startswith("rh20t-") else observation
    )
    np.testing.assert_array_equal(loaded.demonstration.timestamps, expected)
    np.testing.assert_array_equal(loaded.demonstration.qpos, qpos)
    np.testing.assert_array_equal(loaded.cameras["front"].sample_frames, [0, 3, 6])
    assert path.read_bytes() == original_bytes


def test_world_fk_and_ik_preserve_seed_and_nonarm_coordinates(
    robot_model: RobotModel,
) -> None:
    tools = TrajectoryTools(robot_model)
    seed = np.zeros(8)
    seed[6:] = 0.02
    original = seed.copy()
    np.testing.assert_allclose(tools.fk(seed, 0)[:3, 3], [0.7, 0.3, 0.32])
    goal = seed.copy()
    goal[0] = 0.3
    target = tools.fk(goal, 0)
    result, errors = tools.ik(target, 0, seed)
    assert errors["solver"] == "real2sim_source_near_least_squares"
    assert errors["optimizer_converged"]
    assert errors["tcp_error_m"] < 0.002
    assert errors["orientation_error_deg"] < 0.1
    np.testing.assert_array_equal(result[6:], seed[6:])
    np.testing.assert_array_equal(seed, original)
    np.testing.assert_allclose(tools.fk(np.stack([seed, goal]), 0)[1], target)
    surfaces = tools.hand_surfaces(seed, 0)
    assert set(surfaces) == {"link5", "link6", "link7"}
    np.testing.assert_allclose(surfaces["link5"].min(0), [0.69, 0.29, 0.31])
    np.testing.assert_allclose(surfaces["link6"].min(0), [0.74, 0.29, 0.35])
    opened = seed.copy()
    opened[6] += 0.01
    moved = tools.hand_surfaces(opened, 0)
    np.testing.assert_allclose(
        moved["link6"] - surfaces["link6"],
        np.tile([0, 0, 0.01], (len(surfaces["link6"]), 1)),
        atol=1e-12,
    )
    np.testing.assert_array_equal(moved["link7"], surfaces["link7"])
    arm = tools.arm_surfaces(seed, 0)
    assert set(arm) == {f"link{i}" for i in range(6)}
    np.testing.assert_allclose(arm["link0"].min(0), [0.44, 0.29, 0.21])
    np.testing.assert_allclose(arm["link5"].max(0), [0.71, 0.31, 0.33])
    for link, points in tools.arm_surfaces(opened, 0).items():
        np.testing.assert_array_equal(points, arm[link])


def test_real2sim_ik_handles_calibration_and_reports_approximate_solution(
    robot_model: RobotModel,
) -> None:
    from dataclasses import replace
    from scipy.spatial.transform import Rotation

    base = robot_model.base_pose_world.copy()
    base[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_matrix()
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("x", 0.3).as_matrix()
    tcp[:3, 3] = [0.01, 0.02, 0.06]
    tools = TrajectoryTools(
        replace(robot_model, base_pose_world=base, flange_to_tcp=tcp)
    )
    seed = np.array([0.2, -0.5, 0.7, -0.2, 0.4, -0.3, 0.025, 0.025])
    goal = seed.copy()
    goal[0] += 0.05
    target = tools.fk(goal, 0)
    solved, _ = tools.ik(target, 0, seed)
    np.testing.assert_allclose(tools.fk(solved, 0), target, atol=0.001)
    np.testing.assert_array_equal(solved[6:], seed[6:])
    impossible = target.copy()
    impossible[0, 3] += 10
    approximate, errors = tools.ik(impossible, 0, seed)
    assert np.isfinite(approximate).all()
    assert errors["tcp_error_m"] > 9
    assert errors["max_joint_residual_rad"] <= 0.65
    np.testing.assert_array_equal(approximate[6:], seed[6:])
    np.testing.assert_array_equal(seed, [0.2, -0.5, 0.7, -0.2, 0.4, -0.3, 0.025, 0.025])

    # A source just beyond a nominal URDF limit remains a valid reference.
    seed[0] = 3.02
    target = tools.fk(seed, 0)
    solved, errors = tools.ik(target, 0, seed)
    np.testing.assert_allclose(solved, seed, atol=1e-10)
    assert errors["max_joint_limit_excess_rad"] == pytest.approx(0.02)
    assert errors["tcp_error_m"] < 1e-10


def test_interpolation_and_scope_allow_large_caller_chosen_revision() -> None:
    previous = np.zeros((10, 4))
    candidate = interpolate_joint_knots(
        previous, [2, 4, 7], [[4.0, -5.0], [8.0, -9.0], [12.0, -13.0]], [0, 2]
    )
    np.testing.assert_allclose(candidate[3, [0, 2]], [6.0, -7.0])
    np.testing.assert_array_equal(candidate[:2], previous[:2])
    np.testing.assert_array_equal(candidate[8:], previous[8:])
    np.testing.assert_array_equal(candidate[:, [1, 3]], previous[:, [1, 3]])
    phases = [dict(id="carry", start_frame=2, end_frame=7)]
    report = validate_segment_revision(previous, candidate, phases, ["carry"])
    assert report["changed_frames"] == [2, 3, 4, 5, 6, 7]
    np.testing.assert_array_equal(previous, np.zeros((10, 4)))


def test_revision_validation_rejects_scope_shape_and_nonfinite_errors_only() -> None:
    previous = np.zeros((10, 2))
    phases = [dict(id="grasp", start_frame=2, end_frame=5)]
    candidate = previous.copy()
    candidate[6, 0] = 1e-12
    with pytest.raises(ValueError, match="outside"):
        validate_segment_revision(previous, candidate, phases, ["grasp"])
    with pytest.raises(ValueError, match="grid"):
        validate_segment_revision(previous, previous[:-1], phases, ["grasp"])
    candidate[6, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_segment_revision(previous, candidate, phases, ["grasp"])
    with pytest.raises(ValueError, match="unknown"):
        validate_segment_revision(previous, previous, phases, ["other"])


def test_structured_residual_preserves_source_and_updates_gripper(
    robot_model: RobotModel,
) -> None:
    from embodichain.data_pipeline.real2sim.adaptation.trajectory_corrections import (
        apply_trajectory_corrections,
    )
    from scipy.spatial.transform import Rotation

    tools = TrajectoryTools(robot_model)
    original = np.zeros((12, 8))
    original[:, 0] = np.linspace(-0.7, 0.7, 12)
    original[:, 1] = 0.8
    original[:, 3] = -0.8
    original[:, 6:] = 0.025
    before = original.copy()
    times = np.arange(12) / 10
    phase = [dict(id="grasp", start_frame=2, end_frame=9)]
    goal = original[5].copy()
    goal[0] += 0.1
    target, source = tools.fk(goal, 0), tools.fk(original[5], 0)
    edit = dict(
        phase_ids=["grasp"],
        side="arm",
        anchor_sample=5,
        translation_m=(target[:3, 3] - source[:3, 3]).tolist(),
        rotation_vector_rad=Rotation.from_matrix(target[:3, :3] @ source[:3, :3].T)
        .as_rotvec()
        .tolist(),
        gripper_width_delta_m=-0.01,
    )
    result, report = apply_trajectory_corrections(
        robot_model, original, original, times, phase, [edit]
    )
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(
        result[[0, 1, 2, 9, 10, 11]], original[[0, 1, 2, 9, 10, 11]]
    )
    np.testing.assert_allclose(tools.fk(result[5], 0)[:3, 3], target[:3, 3], atol=0.002)
    # The arm turns during the edit. A world-space offset must not turn with it,
    # as happens when only the anchor's joint delta is interpolated.
    expected = tools.fk(original[7], 0)
    expected[:3, 3] += 0.5 * np.asarray(edit["translation_m"])
    expected[:3, :3] = (
        Rotation.from_rotvec(0.5 * np.asarray(edit["rotation_vector_rad"])).as_matrix()
        @ expected[:3, :3]
    )
    np.testing.assert_allclose(tools.fk(result[7], 0), expected, atol=0.001)
    np.testing.assert_allclose(result[5, 6:], 0.02)
    assert report["changed_phase_ids"] == ["grasp"]
    assert report["edits"][0]["max_source_translation_m"] > 0
    with pytest.raises(ValueError, match="forbidden"):
        apply_trajectory_corrections(
            robot_model,
            original,
            original,
            times,
            phase,
            [dict(edit, qpos=goal.tolist())],
        )
    with pytest.raises(ValueError, match="edit authority"):
        apply_trajectory_corrections(
            robot_model,
            original,
            original,
            times,
            phase,
            [dict(edit, translation_m=[0.2, 0, 0])],
        )
    with pytest.raises(ValueError, match="overlap"):
        apply_trajectory_corrections(
            robot_model, original, original, times, phase, [edit, edit]
        )


def test_source_deviation_is_reported_without_rejecting_a_local_edit(
    robot_model: RobotModel,
) -> None:
    from embodichain.data_pipeline.real2sim.adaptation.trajectory_corrections import (
        apply_trajectory_corrections,
    )

    original = np.zeros((11, 8))
    original[:, 6:] = 0.04
    current = original.copy()
    current[:, 0] = 0.6
    result, report = apply_trajectory_corrections(
        robot_model,
        current,
        original,
        np.arange(11) / 10,
        [dict(id="grasp", start_frame=0, end_frame=10)],
        [
            dict(
                phase_ids=["grasp"],
                side="arm",
                anchor_sample=5,
                gripper_width_delta_m=-0.01,
            )
        ],
    )
    np.testing.assert_array_equal(result[:, :6], current[:, :6])
    np.testing.assert_allclose(result[5, 6:], 0.035)
    assert report["edits"][0]["max_source_rotation_rad"] == pytest.approx(0.6)


def test_gripper_shift_uses_seconds_without_changing_arm_motion(
    robot_model: RobotModel,
) -> None:
    from embodichain.data_pipeline.real2sim.adaptation.trajectory_corrections import (
        apply_trajectory_corrections,
    )

    original = np.zeros((11, 8))
    original[:5, 6:] = 0.04
    original[5:, 6:] = 0.01
    times = np.arange(11) / 10
    result, _ = apply_trajectory_corrections(
        robot_model,
        original,
        original,
        times,
        [dict(id="close", start_frame=0, end_frame=10)],
        [dict(phase_ids=["close"], side="arm", anchor_sample=5, gripper_shift_s=0.1)],
    )
    np.testing.assert_array_equal(result[:, :6], original[:, :6])
    np.testing.assert_allclose(result[5, 6:], 0.04)
    np.testing.assert_array_equal(result[[0, 10]], original[[0, 10]])
    with pytest.raises(ValueError, match="temporal order"):
        apply_trajectory_corrections(
            robot_model,
            original,
            original,
            times,
            [dict(id="close", start_frame=0, end_frame=10)],
            [
                dict(
                    phase_ids=["close"],
                    side="arm",
                    anchor_sample=5,
                    gripper_shift_s=0.5,
                )
            ],
        )
