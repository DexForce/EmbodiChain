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

"""Task-specific source preparation, event preservation and portable assets."""

from __future__ import annotations

from dataclasses import replace
import json
import shutil

import numpy as np
import pytest

from embodichain.data_pipeline.real2sim import (
    RobotDemonstration,
    JointSpec,
    DemoSkillPhase,
    DemoSkillRole,
    DemoSkillEvent,
    DemoSkillConstraint,
    DemoSkillSpec,
    build_demo_skill,
    load_demo_skill,
    save_demo_skill,
    retarget_demo_skill,
    save_robot_demonstration,
)
from embodichain.data_pipeline.real2sim.cli import main


def _source():
    return RobotDemonstration(
        source_id="teleop",
        source_revision="1",
        unit_id="contour",
        joint_names=("arm", "finger"),
        joint_units=("deg", "mm"),
        qpos=np.column_stack((np.arange(12), np.full(12, 10.0))),
        timestamps=np.arange(12) * 0.02,
        time_unit="s",
        robot_geometry_sha256="a" * 64,
        recording_sha256="b" * 64,
        commands=np.full((11, 2), 9.0),
        command_representation="joint_position",
        command_alignment="between_states",
        eef_poses=np.tile([0, 0, 0, 0, 0, 0, 1], (12, 1)),
        world_from_pose_frame=np.eye(4),
        observed_link_to_tcp=np.eye(4),
        pose_length_unit="m",
        quaternion_order="xyzw",
        pose_origin="fk_derived",
        calibration_revision="tcp-v1",
    )


def _spec():
    return DemoSkillSpec(
        asset_id="task.contour",
        instruction="Trace a task-specific contour",
        roles=(DemoSkillRole("surface", "c" * 64, tuple(np.eye(4).flat)),),
        phases=(
            DemoSkillPhase(
                "enter", 0, 4, "free", "surface", ("joint_residual",), "eef_pose"
            ),
            DemoSkillPhase("trace", 4, 8, "contact", "surface", (), "eef_pose"),
            DemoSkillPhase("leave", 8, 12, "free", "surface", ("retime",), "eef_pose"),
        ),
        events=(
            DemoSkillEvent("touch", 4, "contact", "surface"),
            DemoSkillEvent("mid_trace", 6, "checkpoint"),
        ),
        constraints=(
            DemoSkillConstraint(
                "touch", "contact", "event", other="surface", event_id="touch"
            ),
            DemoSkillConstraint(
                "maintain",
                "contact",
                "throughout",
                other="surface",
                phase_names=("trace",),
            ),
            DemoSkillConstraint(
                "end", "relative_pose", "terminal", reference_pose=tuple(np.eye(4).flat)
            ),
        ),
        max_translation_m=0.1,
        max_yaw_rad=0.2,
    )


def _build(source=None, spec=None, **kwargs):
    return build_demo_skill(
        source or _source(),
        spec or _spec(),
        joints=(JointSpec("arm", "revolute"), JointSpec("finger", "prismatic")),
        control_dt=0.01,
        **kwargs,
    )


def test_unlabeled_demo_preserves_observed_commands_and_internal_hard_events():
    artifact = _build()
    np.testing.assert_allclose(artifact.qpos[2], [np.pi / 180, 0.01])
    assert artifact.demonstration.commands[0, 0] == 9
    assert [(e.event_id, e.sample_index) for e in artifact.spec.events] == [
        ("touch", 8),
        ("mid_trace", 12),
    ]
    assert artifact.spec.phases[1].start == 8
    assert 6 in artifact.report.time_map.source_event_indices
    assert not artifact.qpos.flags.writeable
    spec = artifact.spec
    spec.instruction = "changed"
    assert artifact.spec.instruction == "Trace a task-specific contour"


def test_portable_asset_rebuilds_exact_identity_and_rejects_tampering(tmp_path):
    artifact = _build()
    path = save_demo_skill(artifact, tmp_path / "original")
    shutil.copytree(path.parent, tmp_path / "moved")
    shutil.rmtree(path.parent)
    restored = load_demo_skill(tmp_path / "moved" / "demo_skill.json")
    assert restored.content_sha256 == artifact.content_sha256
    np.testing.assert_array_equal(restored.qpos, artifact.qpos)
    data = tmp_path / "moved" / "demonstration.npz"
    with data.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="digest"):
        load_demo_skill(tmp_path / "moved" / "demo_skill.json")


def test_preparation_enforces_caller_budgets_and_event_clock():
    with pytest.raises(ValueError, match="max_samples"):
        _build(max_samples=5)
    source = _source()
    times = np.array(source.timestamps)
    times[6] += 0.003
    with pytest.raises(ValueError, match="hard event"):
        _build(replace(source, timestamps=times))


def test_contact_permissions_and_evidence_cannot_be_inferred():
    with pytest.raises(ValueError, match="cannot authorize"):
        DemoSkillPhase("trace", 0, 4, "contact", allowed_operators=("joint_residual",))
    spec = _spec()
    spec.constraints = tuple(c for c in spec.constraints if c.check_id != "maintain")
    with pytest.raises(ValueError, match="throughout contact"):
        _build(spec=spec)


def test_role_transfer_preserves_local_contact_path_and_checks_geometry():
    artifact = _build()
    pose = np.eye(4)
    pose[0, 3] = 0.05
    targets = retarget_demo_skill(
        artifact, {"surface": pose}, geometry_sha256={"surface": "c" * 64}
    )
    np.testing.assert_allclose(targets[8:16, :3, 3], np.tile([0.05, 0, 0], (8, 1)))
    np.testing.assert_allclose(artifact.world_tcp[:, :3, 3], 0)
    with pytest.raises(ValueError, match="geometry"):
        retarget_demo_skill(
            artifact, {"surface": pose}, geometry_sha256={"surface": "d" * 64}
        )
    pose[0, 3] = 0.2
    with pytest.raises(ValueError, match="bounds"):
        retarget_demo_skill(
            artifact, {"surface": pose}, geometry_sha256={"surface": "c" * 64}
        )


def test_cli_build_and_validate_report_preparation_not_physical_success(
    tmp_path, capsys
):
    from dataclasses import asdict

    source = save_robot_demonstration(_source(), tmp_path / "source")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(asdict(_spec())))
    joints = tmp_path / "joints.json"
    joints.write_text(
        json.dumps(
            [
                {"name": "arm", "kind": "revolute"},
                {"name": "finger", "kind": "prismatic"},
            ]
        )
    )
    main(
        [
            "build-demo-skill",
            str(source),
            "--spec",
            str(spec),
            "--joints",
            str(joints),
            "--control-dt",
            "0.01",
            "--output",
            str(tmp_path / "skill"),
        ]
    )
    assert json.loads(capsys.readouterr().out)["physical_qualification"] == "not_run"
    main(["validate-demo-skill", str(tmp_path / "skill" / "demo_skill.json")])
    assert json.loads(capsys.readouterr().out)["asset_id"] == "task.contour"


def test_adapted_reference_keeps_raw_observation_and_command_evidence_separate(
    tmp_path,
):
    from embodichain.data_pipeline.real2sim import DemoSkillReference

    original = _build()
    reference = DemoSkillReference(
        original.qpos + 0.001,
        original.timestamps,
        original.spec.phases,
        original.spec.events,
        original.report.time_map,
        np.ones(len(original.timestamps), dtype=bool),
        json.dumps({"origin": "adapted_nominal", "source_recording_sha256": "b" * 64}),
        world_tcp=original.world_tcp,
        initial_qpos=original.initial_qpos,
    )
    adapted = replace(original, reference=reference)
    assert adapted.content_sha256 != original.content_sha256
    np.testing.assert_array_equal(
        adapted.demonstration.qpos, original.demonstration.qpos
    )
    np.testing.assert_array_equal(
        adapted.demonstration.commands, original.demonstration.commands
    )
    np.testing.assert_array_equal(adapted.execution_qpos, original.qpos + 0.001)
    manifest = save_demo_skill(adapted, tmp_path / "adapted")
    restored = load_demo_skill(manifest)
    assert restored.content_sha256 == adapted.content_sha256
    np.testing.assert_array_equal(
        restored.reference.initial_qpos, original.initial_qpos
    )
    (manifest.parent / "reference.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="reference digest"):
        load_demo_skill(manifest)
