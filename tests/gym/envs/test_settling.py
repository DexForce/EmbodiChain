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

from embodichain.lab.gym.envs.settling import (
    DynamicSettleMonitor,
    DynamicSettleMonitorCfg,
    DynamicSettleSample,
)


def _sample(
    linear: tuple[float, ...], angular: tuple[float, ...]
) -> DynamicSettleSample:
    return DynamicSettleSample(
        entity_id="cube",
        linear_speed=torch.tensor(linear, dtype=torch.float32).unsqueeze(1),
        angular_speed=torch.tensor(angular, dtype=torch.float32).unsqueeze(1),
    )


def test_settle_monitor_tracks_rows_independently_and_owns_metadata() -> None:
    env_ids = torch.tensor([4, 9], dtype=torch.long)
    monitor = DynamicSettleMonitor(
        DynamicSettleMonitorCfg(
            min_steps=1,
            max_steps=5,
            check_interval_steps=1,
            required_stable_checks=2,
        ),
        env_ids,
    )

    first = monitor.observe((_sample((0.0, 1.0), (0.0, 1.0)),), elapsed_steps=1)
    second = monitor.observe((_sample((0.0, 1.0), (0.0, 1.0)),), elapsed_steps=2)
    third = monitor.observe((_sample((0.0, 0.0), (0.0, 0.0)),), elapsed_steps=3)
    final = monitor.observe((_sample((0.0, 0.0), (0.0, 0.0)),), elapsed_steps=4)

    assert first.stable_counts.tolist() == [1, 0]
    assert second.settled_mask.tolist() == [True, False]
    assert third.stable_counts.tolist() == [2, 1]
    assert final.settled_mask.tolist() == [True, True]
    assert final.timeout_mask.tolist() == [False, False]
    assert final.complete is True
    metadata = final.to_metadata()
    assert metadata["env_ids"] == [4, 9]
    assert metadata["settled_mask"] == [True, True]

    env_ids[0] = 100
    assert monitor.env_ids.tolist() == [4, 9]


def test_settle_monitor_duplicate_observation_is_idempotent() -> None:
    monitor = DynamicSettleMonitor(
        DynamicSettleMonitorCfg(
            min_steps=0,
            max_steps=3,
            check_interval_steps=1,
            required_stable_checks=2,
        ),
        torch.tensor([0], dtype=torch.long),
    )

    first = monitor.observe((_sample((0.0,), (0.0,)),), elapsed_steps=0)
    duplicate = monitor.observe((_sample((0.0,), (0.0,)),), elapsed_steps=0)
    second = monitor.observe((_sample((0.0,), (0.0,)),), elapsed_steps=1)

    assert first.checked is True
    assert duplicate.checked is False
    assert duplicate.stable_counts.tolist() == [1]
    assert second.settled_mask.tolist() == [True]
    assert second.observation_count == 2


def test_settle_monitor_marks_only_unresolved_rows_timed_out() -> None:
    monitor = DynamicSettleMonitor(
        DynamicSettleMonitorCfg(
            min_steps=0,
            max_steps=1,
            check_interval_steps=1,
            required_stable_checks=1,
        ),
        torch.tensor([0, 1], dtype=torch.long),
    )

    state = monitor.observe((_sample((0.0, 1.0), (0.0, 1.0)),), elapsed_steps=1)

    assert state.settled_mask.tolist() == [True, False]
    assert state.timeout_mask.tolist() == [False, True]
    assert state.complete is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"min_steps": -1}, "min_steps"),
        ({"max_steps": 1, "min_steps": 2}, "max_steps"),
        ({"check_interval_steps": 0}, "check_interval_steps"),
        ({"linear_velocity_threshold": float("nan")}, "linear_velocity_threshold"),
        (
            {"min_steps": 0, "max_steps": 0, "required_stable_checks": 2},
            "cannot be reached",
        ),
    ),
)
def test_settle_monitor_cfg_rejects_invalid_values(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        DynamicSettleMonitorCfg(**kwargs)


def test_settle_monitor_rejects_regressing_steps_and_incomplete_samples() -> None:
    monitor = DynamicSettleMonitor(
        DynamicSettleMonitorCfg(
            min_steps=0,
            max_steps=2,
            required_stable_checks=1,
        ),
        torch.tensor([0], dtype=torch.long),
    )
    sample = _sample((1.0,), (1.0,))
    monitor.observe((sample,), elapsed_steps=1)

    with pytest.raises(ValueError, match="monotonic"):
        monitor.observe((sample,), elapsed_steps=0)
    with pytest.raises(ValueError, match="contain DynamicSettleSample"):
        monitor.observe((), elapsed_steps=2)
