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

import numpy as np
import pytest

from embodichain.compute.trajectory.phase_alignment import align_joint_phase
from embodichain.compute.trajectory.splicing import (
    splice_joint_segment,
    joint_path_diagnostics,
)


def test_splice_retains_external_values_and_boundary_derivative_stencils() -> None:
    times = np.linspace(0, 2, 101) ** 1.3
    source = np.column_stack((times**2, np.cos(times))).astype(np.float32)
    replacement = source[10:90].copy()
    replacement[:, 0] += 0.1
    output = splice_joint_segment(
        source,
        replacement,
        times,
        start=10,
        stop=90,
        joint_indices=(0,),
        blend_samples=12,
        anchors=(50,),
    )
    assert output.dtype == source.dtype
    np.testing.assert_array_equal(output[:13], source[:13])
    np.testing.assert_array_equal(output[87:], source[87:])
    np.testing.assert_array_equal(output[:, 1], source[:, 1])
    assert output[50, 0] == replacement[40, 0]
    # Independent physical-time derivative check across both outer joins.
    for values in (source, output):
        velocity = np.gradient(values.astype(float), times, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
        if values is source:
            expected = acceleration[[9, 10, 89, 90]]
        else:
            np.testing.assert_allclose(
                acceleration[[9, 10, 89, 90]], expected, atol=1e-10
            )


def test_splice_rejects_a_connection_that_overwrites_a_contact_anchor() -> None:
    source = np.zeros((40, 1))
    with pytest.raises(ValueError, match="overlaps a protected"):
        splice_joint_segment(
            source,
            np.ones((30, 1)),
            np.arange(40.0),
            start=5,
            stop=35,
            joint_indices=(0,),
            anchors=(9,),
        )


def test_derivative_gate_detects_a_single_step_jump() -> None:
    path = np.zeros((20, 1))
    path[10] = 1
    result = joint_path_diagnostics(
        path,
        np.arange(20) * 0.01,
        lower=np.array([-2]),
        upper=np.array([2]),
        velocity_limits=np.array([10]),
        acceleration_limits=np.array([1000]),
    )
    assert not result["accepted"]
    assert result["peak_velocity"] == pytest.approx([100])
    assert result["peak_acceleration"] == pytest.approx([20000])


def test_phase_clock_changes_timing_without_changing_the_native_path() -> None:
    native = np.column_stack((np.linspace(0, 1, 101), np.zeros(101)))
    progress = np.linspace(0, 1, 51) ** 2
    observed = np.column_stack((progress, np.full(51, 0.1)))
    fitted, clock = align_joint_phase(native, observed)
    np.testing.assert_allclose(fitted[:, 0], progress, atol=1e-12)
    np.testing.assert_array_equal(fitted[:, 1], 0)
    assert np.all(np.diff(clock) >= 0)
    # An incompatible geometric displacement remains observable.
    np.testing.assert_allclose(np.linalg.norm(fitted - observed, axis=1), 0.1)
