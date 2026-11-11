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

from __future__ import annotations

from pathlib import Path
import zipfile
import io

import numpy as np
import pytest

from embodichain.data_pipeline.skill_fitting import (
    JointRecording,
    load_joint_recording,
    map_video_interval,
    save_skill_fit,
    load_skill_fit,
)


def test_inclusive_vfr_segment_uses_pts_offset_and_last_frame_duration() -> None:
    # Frames cover [10,10.04), [10.04,10.14), [10.14,10.20) in joint time.
    start, stop = map_video_interval(
        np.array([0, 0.04, 0.14]),
        np.array([0.04, 0.10, 0.06]),
        np.arange(1001) * 0.02,
        start_frame=1,
        end_frame=2,
        time_offset=10,
    )
    assert (start, stop) == (502, 510)


def test_joint_loader_reorders_named_columns_without_guessing_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "joints.npz"
    q = np.column_stack((np.arange(20), np.arange(20) + 100)).astype(float)
    np.savez(path, qpos=q, joint_names=np.array(["b", "a"]))
    with pytest.raises(ValueError, match="Missing joint timestamps"):
        load_joint_recording(path, joint_names=("a", "b"), semantics="measured_state")
    source = load_joint_recording(
        path, joint_names=("a", "b"), semantics="measured_state", sample_dt=0.1
    )
    np.testing.assert_array_equal(source.qpos, q[:, ::-1])
    assert not source.qpos.flags.writeable


def test_fit_bundle_loads_without_any_source_files(tmp_path: Path) -> None:
    output = tmp_path / "fit.npz"
    q = np.arange(60, dtype=float).reshape(20, 3)
    arrays = {
        "timestamps": np.arange(20) * 0.1,
        "original_qpos": q,
        "native_qpos": q + 1,
        "patched_qpos": q + 2,
    }
    save_skill_fit(output, arrays, {"schema_version": 1, "physical_success": "unknown"})
    loaded, metadata = load_skill_fit(output)
    np.testing.assert_array_equal(loaded["patched_qpos"], q + 2)
    assert metadata["physical_success"] == "unknown"


def test_npz_header_budget_is_checked_before_materializing_arrays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": (10**10,)}
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("qpos.npy", header.getvalue())
    with pytest.raises(ValueError, match="allocation budget"):
        load_joint_recording(path, joint_names=("x",), semantics="measured_state")


def test_joint_recording_rejects_ambiguous_instantaneous_jumps() -> None:
    with pytest.raises(ValueError, match="clock"):
        JointRecording(np.zeros(20), np.zeros((20, 1)), ("x",), "actual_submission")
