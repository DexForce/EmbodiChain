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

"""Original DROID observation/command semantics and bounded input regression."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from embodichain.data_pipeline.real2sim.contracts import PhaseAnnotation
from embodichain.data_pipeline.real2sim.droid import (
    build_droid_trajectory,
    load_droid_episode,
    save_droid_unit_evidence,
    verify_droid_unit_evidence,
)


def _raw(path: Path, *, velocity: bool = True) -> Path:
    n = 6
    with h5py.File(path, "w") as f:
        f["observation/robot_state/joint_positions"] = (
            np.arange(n * 7).reshape(n, 7) * 0.01
        )
        if velocity:
            f["observation/robot_state/joint_velocities"] = np.full((n, 7), 0.125)
        pose = np.tile([0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2], (n, 1))
        f["observation/robot_state/cartesian_position"] = pose
        f["observation/robot_state/gripper_position"] = np.linspace(0, 0.5, n)
        f["action/joint_position"] = np.full((n, 7), 0.8)
        f["action/gripper_position"] = np.ones(n)
        f["observation/timestamp/robot_state/read_start"] = (
            1_700_000_000_000 + np.array([0, 70, 141, 212, 282, 354])
        )
        f["observation/timestamp/control/control_start"] = 1_700_000_000_001 + np.array(
            [0, 70, 141, 212, 282, 354]
        )
        f["observation/timestamp/skip_action"] = [
            True,
            False,
            False,
            False,
            False,
            True,
        ]
    return path


def _unit(episode, *, start: int = 1):
    return build_droid_trajectory(
        episode,
        source_id="DROID:independent_fixture",
        unit_id="pick",
        skill_id="pick_up",
        start=start,
        stop=6,
        arm_joint_names=tuple(f"joint{i}" for i in range(7)),
        gripper_joint_names=("driver",),
        gripper_joint_units=("rad",),
        observed_gripper_qpos=episode.gripper_closed_fraction[:, None] * 0.7,
        phases=(
            PhaseAnnotation("approach", 0, 2, "free", "object"),
            PhaseAnnotation("close", 2, 3, "contact", "object"),
            PhaseAnnotation("lift", 3, 6 - start, "hold", "gravity"),
        ),
        world_from_base=np.eye(4),
        reported_link_to_tcp=np.eye(4),
        object_pose=np.eye(4),
        target_pose=None,
        calibration_revision="measured-gripper-map-v1",
        annotation_revision="manual-v1",
    )


def test_droid_preserves_observation_command_clocks_and_derived_pose(
    tmp_path: Path,
) -> None:
    episode = load_droid_episode(_raw(tmp_path / "trajectory.h5"))
    np.testing.assert_allclose(
        episode.timestamps, [0, 0.070, 0.141, 0.212, 0.282, 0.354]
    )
    np.testing.assert_allclose(episode.command_timestamps - episode.timestamps, 0.001)
    np.testing.assert_allclose(
        episode.reported_poses[0], [0.1, 0.2, 0.3, 0, 0, 2**-0.5, 2**-0.5], atol=1e-12
    )
    np.testing.assert_array_equal(
        episode.submitted, [False, True, True, True, True, False]
    )
    unit = _unit(episode)
    assert unit.origin == "measured" and unit.pose_origin == "fk_derived"
    np.testing.assert_array_equal(unit.qpos[:, :7], episode.arm_qpos[1:])
    np.testing.assert_allclose(unit.qpos[:, -1], [0.07, 0.14, 0.21, 0.28, 0.35])
    np.testing.assert_allclose(unit.commands[:, :7], 0.8)
    np.testing.assert_allclose(unit.commands[:, -1], 1.0)
    assert unit.command_alignment == "between_states"
    assert unit.qvel is None
    np.testing.assert_allclose(episode.arm_qvel, 0.125)
    assert not episode.arm_qpos.flags.writeable
    with pytest.raises(ValueError, match="not submitted"):
        _unit(episode, start=0)


def test_droid_missing_velocity_is_not_synthesized(tmp_path: Path) -> None:
    episode = load_droid_episode(_raw(tmp_path / "no_velocity.h5", velocity=False))
    assert episode.arm_qvel is None
    assert _unit(episode).qvel is None


@pytest.mark.parametrize("budget", [{"max_samples": 5}, {"max_array_bytes": 32}])
def test_droid_rejects_expansion_before_reading_arrays(
    tmp_path: Path, monkeypatch, budget
) -> None:
    path = _raw(tmp_path / "large.h5")

    def reject(*args, **kwargs):
        pytest.fail("oversized DROID arrays were read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject)
    with pytest.raises(ValueError, match="max_samples|max_array_bytes"):
        load_droid_episode(path, **budget)


@pytest.mark.parametrize(
    "change", ["clock", "gripper", "layout", "link", "external_storage", "virtual"]
)
def test_droid_rejects_invalid_evidence(tmp_path: Path, change: str) -> None:
    path = _raw(tmp_path / "invalid.h5")
    with h5py.File(path, "a") as f:
        if change == "clock":
            f["observation/timestamp/robot_state/read_start"][2] = f[
                "observation/timestamp/robot_state/read_start"
            ][1]
        elif change == "gripper":
            f["observation/robot_state/gripper_position"][2] = 1.1
        elif change == "layout":
            del f["action/joint_position"]
            f["action/joint_position"] = np.zeros((6, 6))
        elif change == "link":
            del f["action/gripper_position"]
            f["action/gripper_position"] = h5py.ExternalLink("elsewhere.h5", "/targets")
        elif change == "external_storage":
            del f["action/gripper_position"]
            f.create_dataset(
                "action/gripper_position",
                (6,),
                dtype="f8",
                external=[(str(tmp_path / "outside.raw"), 0, h5py.h5f.UNLIMITED)],
            )
        else:
            del f["action/gripper_position"]
            layout = h5py.VirtualLayout((6,), dtype="f8")
            layout[:] = h5py.VirtualSource(
                str(tmp_path / "outside.h5"), "gripper", shape=(6,)
            )
            f.create_virtual_dataset("action/gripper_position", layout)
    with pytest.raises(ValueError):
        load_droid_episode(path)


def test_droid_original_evidence_retains_velocity_and_rejects_changed_unit(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    recording = _raw(tmp_path / "trajectory.h5")
    episode = load_droid_episode(recording)
    source = _unit(episode)
    evidence = save_droid_unit_evidence(
        episode,
        tmp_path / "evidence.npz",
        source_id=source.source_id,
        unit_id=source.unit_id,
        start=1,
        stop=6,
        arm_joint_names=source.joint_names[:7],
        calibrated_gripper_qpos=episode.gripper_closed_fraction[:, None] * 0.7,
        annotation={"origin": "manual_test_fixture"},
        calibration={
            "world_from_base": np.eye(4).tolist(),
            "reported_link_to_tcp": np.eye(4).tolist(),
        },
    )
    verify_droid_unit_evidence(evidence, recording, source)
    import io
    import zipfile

    malformed = tmp_path / "shape-bomb.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, dict(descr="|u1", fortran_order=False, shape=(10**12,))
    )
    with (
        zipfile.ZipFile(evidence) as original,
        zipfile.ZipFile(malformed, "w") as output,
    ):
        for entry in original.infolist():
            output.writestr(
                entry.filename,
                (
                    header.getvalue()
                    if entry.filename == "metadata.npy"
                    else original.read(entry)
                ),
            )
    with pytest.raises(ValueError, match="NPY shape"):
        verify_droid_unit_evidence(malformed, recording, source)
    with np.load(evidence, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["arm_qvel"], np.full((6, 7), 0.125))
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    changed = source.qpos.copy()
    changed[1, 0] += 0.001
    with pytest.raises(ValueError, match="differs from original"):
        verify_droid_unit_evidence(evidence, recording, replace(source, qpos=changed))
    arrays["arm_qvel"][0, 0] = 0.0
    broken = tmp_path / "tampered.npz"
    np.savez_compressed(broken, **arrays)
    with pytest.raises(ValueError, match="changed original arm_qvel"):
        verify_droid_unit_evidence(broken, recording, source)
    with pytest.raises(FileExistsError):
        save_droid_unit_evidence(
            episode,
            evidence,
            source_id=source.source_id,
            unit_id=source.unit_id,
            start=1,
            stop=6,
            arm_joint_names=source.joint_names[:7],
            calibrated_gripper_qpos=episode.gripper_closed_fraction[:, None] * 0.7,
            annotation={},
            calibration={},
        )
