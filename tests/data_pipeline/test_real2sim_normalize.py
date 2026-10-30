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

"""Independent examples for real-to-simulation source conversion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from embodichain.data_pipeline.real2sim import (
    JointSpec,
    PhaseAnnotation,
    Real2SimTrajectory,
    build_motion_prototype,
    load_trajectory,
    load_prototype,
    normalize_trajectory,
    prepare_prototype,
    retarget_motion_prototype,
    save_trajectory,
)


def _source() -> Real2SimTrajectory:
    # A 90 degree link rotation plus a 10 cm local TCP offset gives +Y in world.
    poses = np.tile([100.0, 0.0, 0.0, 2**-0.5, 0.0, 0.0, 2**-0.5], (6, 1))
    frame = np.eye(4)
    frame[0, 3] = 1.0
    tool = np.eye(4)
    tool[0, 3] = 0.1
    return Real2SimTrajectory(
        source_id="fixture",
        source_revision="1",
        unit_id="pick-1",
        skill_id="pick_up",
        origin="synthetic_reference",
        joint_names=("finger", "arm"),
        joint_units=("mm", "deg"),
        qpos=np.array([[10, 0], [10, 45], [10, 90], [10, 90], [10, 90], [10, 90]]),
        timestamps=np.array([0, 20, 40, 60, 80, 100]),
        time_unit="ms",
        phases=(
            PhaseAnnotation("approach", 0, 2, "free", "object", ("joint_residual",)),
            PhaseAnnotation("close", 2, 4, "contact", "object"),
            PhaseAnnotation("lift", 4, 6, "hold", "gravity"),
        ),
        eef_poses=poses,
        pose_length_unit="mm",
        quaternion_order="wxyz",
        world_from_pose_frame=frame,
        observed_link_to_tcp=tool,
        object_pose=np.eye(4),
        pose_origin="fk_derived",
        qvel=np.zeros((6, 2)),
    )


def test_normalizes_joint_order_units_mimic_and_tcp_frames() -> None:
    result = normalize_trajectory(
        _source(),
        joints=(
            JointSpec("arm", "revolute"),
            JointSpec("finger", "prismatic"),
            JointSpec(
                "other_finger", "prismatic", mimic_driver="finger", mimic_multiplier=-1
            ),
        ),
        control_dt=0.01,
    )
    np.testing.assert_allclose(result.qpos[1], [np.pi / 8, 0.01, -0.01])
    np.testing.assert_allclose(result.world_tcp[0, :3, 3], [1.1, 0.1, 0.0], atol=1e-10)
    assert result.report.joint_source_indices == (1, 0, None)
    np.testing.assert_array_equal(
        result.report.time_map.execution_event_indices, [0, 4, 8, 10]
    )


def test_initial_observation_is_not_replaced_by_reference_command() -> None:
    source = replace(_source(), initial_observed_qpos=np.array([15.0, 0.0]))
    result = normalize_trajectory(
        source,
        joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
        control_dt=0.02,
    )
    np.testing.assert_allclose(result.initial_qpos, [0.0, 0.015])
    np.testing.assert_allclose(result.qpos[0], [0.0, 0.01])
    assert result.report.initial_state_origin == "observed"


@pytest.mark.parametrize("control_dt", [1e-9, 1e-320])
def test_normalization_rejects_oversized_grid_before_allocation(
    monkeypatch, control_dt
) -> None:
    source = _source()
    joints = (JointSpec("arm", "revolute"), JointSpec("finger", "prismatic"))
    monkeypatch.setattr(
        "embodichain.data_pipeline.real2sim.normalize.np.arange",
        lambda *a, **k: pytest.fail(
            "oversized normalization allocated its output grid"
        ),
    )
    with pytest.raises(ValueError, match="max_samples"):
        normalize_trajectory(source, joints=joints, control_dt=control_dt)


def test_normalization_enforces_byte_budget_and_exact_sample_boundary(
    monkeypatch,
) -> None:
    source = _source()
    joints = (JointSpec("arm", "revolute"), JointSpec("finger", "prismatic"))
    with monkeypatch.context() as guarded:
        guarded.setattr(
            "embodichain.data_pipeline.real2sim.normalize.np.arange",
            lambda *a, **k: pytest.fail(
                "byte-budget rejection allocated an output grid"
            ),
        )
        with pytest.raises(ValueError, match="max_output_bytes"):
            normalize_trajectory(
                source, joints=joints, control_dt=0.02, max_output_bytes=64
            )
    unit = normalize_trajectory(source, joints=joints, control_dt=0.02, max_samples=6)
    assert len(unit.timestamps) == 6
    with pytest.raises(ValueError, match="max_samples"):
        normalize_trajectory(source, joints=joints, control_dt=0.02, max_samples=5)


def test_prepare_and_load_apply_reader_owned_normalization_budgets(tmp_path) -> None:
    source = _source()
    joints = (JointSpec("arm", "revolute"), JointSpec("finger", "prismatic"))
    destination = tmp_path / "oversized"
    with pytest.raises(ValueError, match="max_samples"):
        prepare_prototype(
            source, joints=joints, control_dt=0.02, directory=destination, max_samples=5
        )
    assert not destination.exists()
    manifest = prepare_prototype(
        source, joints=joints, control_dt=0.02, directory=tmp_path / "valid"
    )
    with pytest.raises(ValueError, match="max_samples"):
        load_prototype(manifest, max_samples=5)
    with pytest.raises(ValueError, match="max_output_bytes"):
        load_prototype(manifest, max_output_bytes=64)
    assert len(load_prototype(manifest, max_samples=6).unit.timestamps) == 6


def test_real2sim_cli_reports_budget_rejection_before_publishing(
    tmp_path, capsys
) -> None:
    import json
    from dataclasses import asdict
    from embodichain.data_pipeline.real2sim.cli import main

    source = save_trajectory(_source(), tmp_path / "source")
    layout = tmp_path / "joints.json"
    layout.write_text(
        json.dumps(
            [
                asdict(JointSpec("arm", "revolute")),
                asdict(JointSpec("finger", "prismatic")),
            ]
        )
    )
    output = tmp_path / "prepared"
    args = [
        "prepare",
        str(source),
        "--joints",
        str(layout),
        "--control-dt",
        "0.02",
        "--output",
        str(output),
    ]
    with pytest.raises(SystemExit) as failure:
        main([*args, "--max-samples", "5"])
    assert failure.value.code == 2
    assert "max_samples" in capsys.readouterr().err
    assert not output.exists()
    main([*args, "--max-samples", "6"])
    assert json.loads(capsys.readouterr().out)["status"] == "valid_prototype"
    with pytest.raises(SystemExit) as failure:
        main(["validate", str(output / "prototype.json"), "--max-output-bytes", "64"])
    assert failure.value.code == 2
    assert "max_output_bytes" in capsys.readouterr().err


def test_rejects_missing_mimic_conflict_and_unrepresentable_event() -> None:
    source = _source()
    with pytest.raises(ValueError, match="missing independent"):
        normalize_trajectory(
            source,
            joints=(
                JointSpec("arm", "revolute"),
                JointSpec("finger", "prismatic"),
                JointSpec("absent", "revolute"),
            ),
            control_dt=0.02,
        )
    with pytest.raises(ValueError, match="conflicts"):
        normalize_trajectory(
            source,
            joints=(
                JointSpec("arm", "revolute"),
                JointSpec("driver", "prismatic", initial_hold=0.02),
                JointSpec("finger", "prismatic", mimic_driver="driver"),
            ),
            control_dt=0.02,
        )
    with pytest.raises(ValueError, match="control grid"):
        normalize_trajectory(
            source,
            joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
            control_dt=0.03,
        )


def test_rejects_quaternion_and_invalid_frame() -> None:
    source = _source()
    with pytest.raises(ValueError, match="explicit pose units"):
        replace(source, quaternion_order=None)
    joint_layout = (JointSpec("arm", "revolute"), JointSpec("finger", "prismatic"))
    bad = np.array(source.eef_poses, copy=True)
    bad[0, 3:] = 0
    with pytest.raises(ValueError, match="quaternions"):
        normalize_trajectory(
            replace(source, eef_poses=bad), joints=joint_layout, control_dt=0.02
        )
    transform = np.eye(4)
    transform[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        normalize_trajectory(
            replace(source, observed_link_to_tcp=transform),
            joints=joint_layout,
            control_dt=0.02,
        )


def test_pick_transfer_preserves_grasp_and_world_up_lift() -> None:
    source = _source()
    poses = np.array(source.eef_poses, copy=True)
    poses[4:, 2] = [0, 30]  # millimetres above the grasp at lift start
    unit = normalize_trajectory(
        replace(source, eef_poses=poses),
        joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
        control_dt=0.02,
    )
    prototype = build_motion_prototype(unit)
    moved = np.eye(4)
    moved[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    moved[:3, 3] = [0.03, -0.02, 0]
    retargeted = retarget_motion_prototype(prototype, object_pose=moved)
    grasp_index = unit.phases[2].start
    np.testing.assert_allclose(
        np.linalg.inv(moved) @ retargeted[grasp_index], prototype.object_to_tcp
    )
    np.testing.assert_allclose(
        retargeted[-1, :3, 3] - retargeted[grasp_index, :3, 3], [0, 0, 0.03], atol=1e-10
    )
    raised = moved.copy()
    raised[2, 3] = 0.01
    with pytest.raises(ValueError, match="height"):
        retarget_motion_prototype(prototype, object_pose=raised)
    tilt = np.deg2rad(1.0)
    raised[:3, :3] = moved[:3, :3] @ np.array(
        [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
    )
    supported = retarget_motion_prototype(
        prototype, object_pose=raised, object_supported_regrasp=True
    )
    np.testing.assert_allclose(
        np.linalg.inv(raised) @ supported[grasp_index],
        prototype.object_to_tcp,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        supported[-1, :3, 3] - supported[grasp_index, :3, 3],
        [0, 0, 0.03],
        atol=1e-10,
    )
    tilt = np.deg2rad(3.0)
    raised[:3, :3] = moved[:3, :3] @ np.array(
        [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
    )
    with pytest.raises(ValueError, match="roll or pitch"):
        retarget_motion_prototype(
            prototype, object_pose=raised, object_supported_regrasp=True
        )


def test_trajectory_bundle_round_trip_and_digest(tmp_path: Path) -> None:
    source = replace(
        _source(),
        robot_geometry_sha256="a" * 64,
        object_geometry_sha256="b" * 64,
    )
    manifest = save_trajectory(source, tmp_path)
    loaded = load_trajectory(manifest)
    assert loaded.origin == "synthetic_reference"
    assert loaded.phases == source.phases
    assert loaded.robot_geometry_sha256 == "a" * 64
    assert loaded.object_geometry_sha256 == "b" * 64
    np.testing.assert_array_equal(loaded.qpos, source.qpos)
    with pytest.raises(FileExistsError):
        save_trajectory(source, tmp_path)
    with (tmp_path / "trajectory.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        load_trajectory(manifest)


def test_prepared_prototype_is_portable_and_reverified(tmp_path: Path) -> None:
    source = _source()
    manifest = prepare_prototype(
        source,
        joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
        control_dt=0.02,
        directory=tmp_path / "prepared",
    )
    prototype = load_prototype(manifest)
    assert prototype.unit.source.unit_id == "pick-1"
    assert prototype.unit.source.origin == "synthetic_reference"
    np.testing.assert_allclose(
        retarget_motion_prototype(prototype, object_pose=np.eye(4)),
        prototype.unit.world_tcp,
    )
    with (manifest.parent / "normalized.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        load_prototype(manifest)


def test_qpos_only_prototype_is_portable_but_not_cartesian(tmp_path: Path) -> None:
    source = replace(_source(), eef_poses=None)
    manifest = prepare_prototype(
        source,
        joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
        control_dt=0.02,
        directory=tmp_path / "qpos_only",
    )
    prototype = load_prototype(manifest)
    assert prototype.object_to_tcp is None
    with pytest.raises(ValueError, match="requires normalized TCP"):
        retarget_motion_prototype(prototype, object_pose=np.eye(4))


def test_pick_prototype_rejects_an_editable_lift_phase() -> None:
    source = _source()
    lift = replace(source.phases[2], kind="free", allowed_operators=("joint_residual",))
    source = replace(source, phases=(*source.phases[:2], lift))
    unit = normalize_trajectory(
        source,
        joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
        control_dt=0.02,
    )
    with pytest.raises(ValueError, match="lift must remain a contact or hold"):
        build_motion_prototype(unit)


def test_command_alignment_is_explicit_and_preserved() -> None:
    source = _source()
    assert source.qpos_semantics == "design_reference"
    assert source.qvel_semantics == "design_reference"
    assert replace(source, origin="measured").qpos_semantics == "measured_state"
    commands = np.ones((len(source.timestamps) - 1, 2))
    with pytest.raises(ValueError, match="alignment"):
        replace(source, commands=commands, command_representation="joint_target")
    aligned = replace(
        source,
        commands=commands,
        command_representation="joint_target",
        command_alignment="between_states",
    )
    result = normalize_trajectory(
        aligned,
        joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
        control_dt=0.02,
    )
    np.testing.assert_array_equal(result.source.commands, commands)
    assert result.source.command_alignment == "between_states"
    assert result.source.command_semantics == "actual_submission"
    with pytest.raises(ValueError, match="command count"):
        replace(aligned, command_alignment="post_step")


def test_cli_prepares_and_validates_portable_prototype(tmp_path: Path, capsys) -> None:
    import json
    from embodichain.data_pipeline.real2sim.cli import main

    manifest = save_trajectory(_source(), tmp_path / "input")
    layout = tmp_path / "joints.json"
    layout.write_text(
        json.dumps(
            [
                {"name": "arm", "kind": "revolute"},
                {"name": "finger", "kind": "prismatic"},
            ]
        )
    )
    output = tmp_path / "output"
    main(
        [
            "prepare",
            str(manifest),
            "--joints",
            str(layout),
            "--control-dt",
            "0.02",
            "--output",
            str(output),
        ]
    )
    assert json.loads(capsys.readouterr().out)["status"] == "valid_prototype"
    main(["validate", str(output / "prototype.json")])
    assert json.loads(capsys.readouterr().out)["origin"] == "synthetic_reference"


def test_source_bundle_digest_is_stable_across_hash_seeds(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    manifest = save_trajectory(_source(), tmp_path / "input")
    command = (
        "from embodichain.data_pipeline.real2sim import load_trajectory, save_trajectory; "
        "import sys; save_trajectory(load_trajectory(sys.argv[1]), sys.argv[2])"
    )
    for seed in ("1", "2"):
        subprocess.run(
            [sys.executable, "-c", command, str(manifest), str(tmp_path / seed)],
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        )
    assert (tmp_path / "1" / "trajectory.npz").read_bytes() == (
        tmp_path / "2" / "trajectory.npz"
    ).read_bytes()


@pytest.mark.parametrize("skill", ["move_held_object", "place"])
def test_target_frame_transfer_preserves_the_current_held_relation(skill: str) -> None:
    source = _source()
    poses = np.array(source.eef_poses, copy=True)
    poses[:, 0:3] = 0
    poses[:, 3:] = [1, 0, 0, 0]  # source uses wxyz
    poses[:, 1] = np.linspace(0, 100, len(poses))
    if skill == "move_held_object":
        phases = (PhaseAnnotation("transport", 0, 6, "free", "target"),)
    else:
        phases = (
            PhaseAnnotation("approach", 0, 2, "free", "target"),
            PhaseAnnotation("release", 2, 4, "contact", "target"),
            PhaseAnnotation("retract", 4, 6, "free", "target"),
        )
    original = replace(
        source,
        skill_id=skill,
        phases=phases,
        eef_poses=poses,
        world_from_pose_frame=np.eye(4),
        observed_link_to_tcp=np.eye(4),
        object_pose=np.eye(4),
        target_pose=np.eye(4),
    )
    unit = normalize_trajectory(
        original,
        joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
        control_dt=0.02,
    )
    prototype = build_motion_prototype(unit)
    target = np.eye(4)
    target[0, 3] = 0.03
    held = np.eye(4)
    held[0, 3] = 0.01
    result = retarget_motion_prototype(
        prototype, target_pose=target, held_object_to_tcp=held
    )
    np.testing.assert_allclose(result[0, :3, 3], [0.04, 0, 0], atol=1e-12)
    np.testing.assert_allclose(result[-1, :3, 3], [0.04, 0.1, 0], atol=1e-12)
    target[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rotated = retarget_motion_prototype(
        prototype, target_pose=target, held_object_to_tcp=held
    )
    np.testing.assert_allclose(rotated[0, :3, 3], [0.03, 0.01, 0], atol=1e-12)
    np.testing.assert_allclose(rotated[-1, :3, 3], [-0.07, 0.01, 0], atol=1e-12)
