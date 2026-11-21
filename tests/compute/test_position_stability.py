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

"""Sampled position-window behavior independent of simulation."""

from __future__ import annotations

import pytest
import torch
from embodichain.compute.trajectory import PositionRangeMonitor


def test_position_window_requires_observed_intervals_and_does_not_latch_success():
    value = torch.tensor([0.1])
    predicate = torch.tensor([True])
    monitor = PositionRangeMonitor(value, required_steps=25, position_drift=0.001)
    assert not monitor.observe(value, predicate, elapsed_steps=0).any()
    assert not monitor.observe(value, predicate, elapsed_steps=0).any()
    with pytest.raises(ValueError, match="consecutive"):
        monitor.observe(value, predicate, elapsed_steps=2)
    for step in range(1, 25):
        assert not monitor.observe(value, predicate, elapsed_steps=step).any()
    accepted = monitor.observe(value, predicate, elapsed_steps=25)
    assert accepted.tolist() == [True]
    assert not monitor.observe(value + 0.02, predicate, elapsed_steps=26).any()
    assert accepted.tolist() == [True]  # Earlier returned results retain ownership.


def test_position_window_checks_total_drift_rather_than_neighbor_differences():
    value = torch.tensor([0.1], dtype=torch.float64)
    monitor = PositionRangeMonitor(value, required_steps=25, position_drift=0.0013)
    for step in range(26):
        # Each 0.08 mm increment is below the 1.3 mm cap, but cumulative drift
        # crosses it before a complete one-second (25 interval) window.
        result = monitor.observe(
            value + step * 0.00008, torch.tensor([True]), elapsed_steps=step
        )
    assert not result.any()


def test_position_window_validity_and_target_predicate_are_independent():
    value = torch.tensor([0.1, 0.2, float("nan")])
    monitor = PositionRangeMonitor(value, required_steps=2, position_drift=0.001)
    for step in range(3):
        result = monitor.observe(
            value, torch.tensor([True, False, True]), elapsed_steps=step
        )
    assert result.tolist() == [True, False, False]


def test_position_window_starts_at_zero_when_each_row_enters_target():
    monitor = PositionRangeMonitor(
        torch.tensor([0.11, 0.09]), required_steps=2, position_drift=0.1
    )
    samples = [
        ([0.11, 0.09], [False, False], [0, 0]),
        ([0.09, 0.09], [False, False], [0, 1]),
        ([0.09, 0.09], [False, True], [1, 2]),
        ([0.09, 0.11], [True, False], [2, 0]),
        ([0.09, 0.09], [True, False], [3, 0]),
        ([0.09, 0.09], [True, False], [4, 1]),
        ([0.09, 0.09], [True, True], [5, 2]),
    ]
    for step, (positions, expected, counts) in enumerate(samples):
        values = torch.tensor(positions)
        result = monitor.observe(values, values.abs() <= 0.1, elapsed_steps=step)
        assert result.tolist() == expected
        assert monitor.stable_counts.tolist() == counts
        if step in (1, 4):
            torch.testing.assert_close(monitor.minimum, values)
            torch.testing.assert_close(monitor.maximum, values)


def test_position_window_recovers_range_tracking_after_nonfinite_sample():
    monitor = PositionRangeMonitor(
        torch.tensor([float("nan")]), required_steps=2, position_drift=0.001
    )
    for step, position in enumerate([float("nan"), 0.1, 0.1008, 0.1016]):
        assert not monitor.observe(
            torch.tensor([position]), torch.tensor([True]), elapsed_steps=step
        ).any()
    assert not monitor.observe(
        torch.tensor([0.1016]), torch.tensor([True]), elapsed_steps=4
    ).any()
    assert monitor.observe(
        torch.tensor([0.1016]), torch.tensor([True]), elapsed_steps=5
    ).all()
