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

import json
import numpy as np
import pytest

from embodichain.data_analysis.recording import (
    load_trajectory,
    write_recording,
    physical_metrics,
)


def samples():
    return dict(
        timestamps=np.array([0.0, 0.1, 0.2]),
        qpos=np.zeros((3, 9)),
        tcp_position=np.zeros((3, 3)),
        object_position=np.array([[0, 0, 0.025], [0, 0, 0.15], [0.1, 0, 0.025]]),
    )


def test_recording_roundtrip_and_physical_evidence(tmp_path):
    paths = write_recording(
        tmp_path / "ep", samples(), {"schema_version": 5, "nodes": []}
    )
    data = load_trajectory(paths["trajectory"])
    assert data["qpos"].shape == (3, 9)
    assert (
        json.loads((tmp_path / "ep" / "scene.json").read_text())["schema_version"] == 5
    )
    metrics = physical_metrics(data, target_xy=(0.1, 0))
    assert metrics["lift_height_m"] == pytest.approx(0.125)
    assert metrics["physical_success"] is True
    assert "camera" not in paths


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamps", [0.0, 0.2, 0.1]),
        ("qpos", np.zeros((2, 9))),
        ("object_position", np.full((3, 3), np.nan)),
    ],
)
def test_rejects_misaligned_or_nonfinite_samples(tmp_path, field, value):
    arrays = samples()
    arrays[field] = np.array(value)
    with pytest.raises(ValueError):
        write_recording(tmp_path / "ep", arrays, {})
    assert not (tmp_path / "ep").exists()


def test_no_overwrite_and_failed_save_is_not_committed(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError):
        write_recording(tmp_path / "ep", samples(), {})
    assert not (tmp_path / "ep").exists()


def test_pickle_arrays_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, **(samples() | {"qpos": np.array([object()] * 3)}))
    with pytest.raises(ValueError):
        load_trajectory(path)
