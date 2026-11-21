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

"""Independent numerical controls for the P-02 offline measurements."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.benchmark.tech_reports.reporting.p02_analysis import (
    box_penetration,
    projected_overlap,
    analyze_episode,
)


def test_touch_gap_and_known_penetration() -> None:
    a = np.array(
        [
            [0, 0, 0.5000, 0, 0, 0, 1],
            [0, 0, 0.499, 0, 0, 0, 1],
            [0, 0, 0.501, 0, 0, 0, 1],
        ]
    )
    floor = np.broadcast_to([0, 0, -0.5, 0, 0, 0, 1], a.shape)
    np.testing.assert_allclose(
        box_penetration(a, np.ones(3), floor, np.array([10, 10, 1])),
        [0, 0.001, 0],
        atol=1e-14,
    )
    a[:, 3:] *= -1
    np.testing.assert_allclose(
        box_penetration(a, np.ones(3), floor, np.array([10, 10, 1])),
        [0, 0.001, 0],
        atol=1e-14,
    )


def test_rotated_corner_and_exact_horizontal_footprint() -> None:
    # The bottom vertex of a 45-degree tilted unit cube is sqrt(2)/2 below COM.
    q = Rotation.from_euler("x", 45, degrees=True).as_quat()
    a = np.array([[0, 0, 2**0.5 / 2 - 0.003, *q]])
    b = np.array([[0, 0, -0.5, 0, 0, 0, 1]])
    np.testing.assert_allclose(
        box_penetration(a, np.ones(3), b, np.array([10, 10, 1])), [0.003], atol=1e-14
    )
    # AABBs overlap here, but the rotated squares do not; require exact projection.
    q = Rotation.from_euler("z", 45, degrees=True).as_quat()
    a = np.array([[0, 0, 1, *q]])
    b = np.array([[1, 1, 0, *q]])
    assert not projected_overlap(a, np.ones(3), b, np.ones(3))[0]


def test_temporary_stack_event_is_retained(tmp_path: Path) -> None:
    t = np.arange(0, 3.01, 0.1)
    state = np.zeros((len(t), 2, 13))
    state[..., 6] = 1
    state[:, 0, 2] = 0.225
    state[:, 1, 2] = 0.2751
    state[2:5, 1, 0] = 0.1
    poses = state[0, :, :7].copy()
    path = tmp_path / "episode_00.npz"
    np.savez(
        path,
        t=t,
        state=state,
        body_size=np.array([0.05] * 3),
        support_pose=np.array([0, 0, 0.15, 0, 0, 0, 1]),
        support_size=np.array([3, 3, 0.1]),
        requested_initial_poses=poses,
    )
    meta = dict(
        run_id="synthetic",
        platform="embodichain",
        scene=dict(id="stack_02", kind="stack"),
        solver="test",
        phase="pilot",
        dt_s=0.1,
        protocol=dict(stack_settle_s=2),
    )
    result, _ = analyze_episode(path, meta)
    assert result["geometric_event_during_settle"]
    assert not result["geometric_event_present_final"]
    assert result["geometric_event_first_s"] == 0.2
    assert result["geometric_event_samples"] == 3
    assert result["xy_drift_full_max_mm"] == 100
    assert result["xy_drift_observation_max_mm"] == 0
