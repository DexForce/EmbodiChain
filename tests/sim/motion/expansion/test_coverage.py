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

import pytest
import torch

from embodichain.lab.sim.motion.expansion.contracts import TrajectoryPhase
from embodichain.lab.sim.motion.expansion.coverage import (
    CoverageIndex,
    TrajectoryDescriptor,
    describe_trajectory,
)


def descriptor(count: int = 3, duration: float = 1.0, offset: float = 0.0):
    return describe_trajectory(
        torch.linspace(0, 1, count)[:, None] + offset,
        torch.linspace(0, duration, count),
        torch.tensor([[0.0, 2.0]]),
        samples_per_phase=8,
    )


def test_extra_samples_and_timing_do_not_change_geometry() -> None:
    original, slow = descriptor(), descriptor(9, 2.0)
    torch.testing.assert_close(original.geometry, slow.geometry)
    torch.testing.assert_close(original.timing * 2, slow.timing)
    index = CoverageIndex()
    assert index.reserve("first", "family", original)
    assert index.reserve("slow", "family", slow)
    assert index.geometry_count == 0
    index.confirm("first")
    index.confirm("slow")
    assert index.geometry_count == 1


def test_actual_near_duplicates_use_one_family_and_pending_quota() -> None:
    index = CoverageIndex(target_per_geometry=2)
    assert index.reserve("first", "one", descriptor())
    assert not index.reserve("copy", "two", descriptor(10))
    assert index.reserve("slow", "two", descriptor(10, 2.0))
    assert not index.reserve("third", "three", descriptor(10, 3.0))
    index.release("first")
    assert index.reserve("third", "three", descriptor(10, 3.0))
    index.confirm("slow")
    index.confirm("third")
    assert index.geometry_count == 1


def test_family_alias_survives_tracking_noise_and_failed_write() -> None:
    index = CoverageIndex(geometry_tolerance=0.01)
    assert index.reserve("first", "one", descriptor())
    assert index.reserve("slow", "two", descriptor(duration=2.0))
    index.release("first")
    assert index.reserve("noisy", "two", descriptor(duration=3.0, offset=0.1))
    index.confirm("slow")
    index.confirm("noisy")
    assert index.geometry_count == 1


def test_confirmation_is_idempotent_and_cannot_be_released() -> None:
    index = CoverageIndex()
    assert index.reserve("commit", "family", descriptor())
    index.confirm("commit")
    index.confirm("commit")
    assert index.geometry_count == 1
    with pytest.raises(ValueError, match="Committed"):
        index.release("commit")


def test_retimed_phase_does_not_change_unlabelled_gap_identity() -> None:
    limits = torch.tensor([[0.0, 2.0]])
    original = describe_trajectory(
        torch.tensor([[0.0], [0.5], [1.0], [1.0]]),
        torch.arange(4.0),
        limits,
        phases=(TrajectoryPhase("move", 0, 3),),
    )
    slower = describe_trajectory(
        torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0], [1.0]]),
        torch.arange(6.0),
        limits,
        phases=(TrajectoryPhase("move", 0, 5),),
    )
    assert original.phase_ids == slower.phase_ids
    torch.testing.assert_close(original.geometry, slower.geometry)
    index = CoverageIndex(target_per_geometry=1)
    assert index.reserve("one", "first", original)
    assert not index.reserve("two", "second", slower)


def test_phase_labels_are_owned_through_reservation() -> None:
    labels = [["move", "free"]]
    value = TrajectoryDescriptor(labels, torch.zeros(1, 2, 1), torch.zeros(1, 2))
    labels[0][0] = "changed"
    assert value.phase_ids == (("move", "free"),)


def test_stationary_wait_counts_time_without_inventing_geometry() -> None:
    result = describe_trajectory(
        torch.zeros(4, 1),
        torch.arange(4.0),
        torch.tensor([[-1.0, 1.0]]),
    )
    assert torch.equal(result.geometry, torch.full_like(result.geometry, 0.5))
    assert float(result.timing[0, -1]) == 3.0


def test_phase_connections_and_unlabelled_samples_are_preserved() -> None:
    positions = torch.tensor([[0.0], [0.0], [1.0], [1.0], [2.0]])
    result = describe_trajectory(
        positions,
        torch.arange(5.0),
        torch.tensor([[0.0, 2.0]]),
        phases=(TrajectoryPhase("first", 0, 2), TrajectoryPhase("second", 2, 4)),
        samples_per_phase=4,
    )
    assert result.geometry.shape == (3, 4, 1)
    assert float(result.geometry[1, 0, 0]) == 0.0
    assert float(result.geometry[1, -1, 0]) == 0.5
    assert float(result.geometry[2, -1, 0]) == 1.0


@pytest.mark.parametrize(
    "times", [torch.tensor([0.0, 0.0]), torch.tensor([0.0, float("nan")])]
)
def test_invalid_actual_timestamps_are_rejected(times: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        describe_trajectory(torch.zeros(2, 1), times, torch.tensor([[0.0, 1.0]]))


def test_a_full_band_rejects_new_geometry_and_leaves_other_bands_open() -> None:
    index = CoverageIndex(target_per_band=1)
    assert index.reserve("well_conditioned", "one", descriptor(), band=2)
    # A new geometry cannot keep filling an exhausted manipulability band, but
    # the same geometry still fits a band that has capacity left.
    assert not index.reserve("second", "two", descriptor(offset=0.5), band=2)
    assert index.reserve("tight", "three", descriptor(offset=0.5), band=0)
    index.confirm("well_conditioned")
    index.confirm("tight")
    assert index.geometry_count == 2
    assert index.band_counts == {0: 1, 2: 1}


def test_released_band_capacity_is_reusable_and_pending_writes_are_uncounted() -> None:
    index = CoverageIndex(target_per_band=1)
    assert index.reserve("first", "one", descriptor(), band=1)
    assert index.band_counts == {}
    assert not index.reserve("second", "two", descriptor(10, 2.0), band=1)
    index.release("first")
    assert index.reserve("second", "two", descriptor(10, 2.0), band=1)
    index.confirm("second")
    assert index.band_counts == {1: 1}


@pytest.mark.parametrize(
    ("target_per_band", "band"),
    [(None, 0), (1, None), (1, -1), (1, True), (0, 0)],
)
def test_band_arguments_must_match_the_configured_quota(
    target_per_band: int | None, band: int | None
) -> None:
    with pytest.raises(ValueError):
        CoverageIndex(target_per_band=target_per_band).reserve(
            "commit", "family", descriptor(), band=band
        )


def test_unbanded_coverage_reports_no_band_counts() -> None:
    index = CoverageIndex()
    assert index.reserve("commit", "family", descriptor())
    index.confirm("commit")
    assert index.band_counts == {}
