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

"""Hard-event and timing-provenance tests for source-neutral resampling."""

from __future__ import annotations

import numpy as np
import pytest

from embodichain.compute.trajectory import resample_with_anchors


def test_preserves_anchor_positions_and_bidirectional_mapping():
    t = np.array([0.0, 0.011, 0.021, 0.041, 0.070])
    q = np.c_[np.sin(t * 20), t**2]
    result, mapping = resample_with_anchors(q, t, 0.01, [2, 3])
    np.testing.assert_array_equal(
        result[mapping.anchor_execution_indices], q[mapping.anchor_source_indices]
    )
    np.testing.assert_allclose(np.diff(mapping.execution_times), 0.01)
    reconstructed = q[mapping.left_indices] + mapping.weights[:, None] * (
        q[mapping.right_indices] - q[mapping.left_indices]
    )
    np.testing.assert_allclose(result, reconstructed)
    np.testing.assert_allclose(
        mapping.source_query_times[mapping.anchor_execution_indices],
        t[mapping.anchor_source_indices],
    )
    assert mapping.execution_times[-1] >= t[-1]
    with pytest.raises(ValueError):
        mapping.weights.setflags(write=True)


def test_sub_tick_anchors_fail_instead_of_erasing_contact():
    t = np.array([0.0, 0.011, 0.012, 0.04])
    with pytest.raises(ValueError, match="anchors cannot fit"):
        resample_with_anchors(np.zeros((4, 2)), t, 0.01, [1, 2])


@pytest.mark.parametrize(
    "times",
    [[0.0, 0.1, 0.1], [0.0, 0.2, 0.1], [0.1, 0.2, 0.3], [0.0, float("nan"), 0.3]],
)
def test_invalid_source_clock_is_rejected(times):
    with pytest.raises(ValueError):
        resample_with_anchors(np.zeros((3, 2)), np.array(times), 0.01, [])


def test_allocation_bound_is_checked_before_grid_creation():
    with pytest.raises(ValueError, match="max_samples"):
        resample_with_anchors(
            np.zeros((2, 1)), np.array([0.0, 1000.0]), 1e-6, [], max_samples=100
        )
