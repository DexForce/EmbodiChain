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

"""Command targets remain separate from measured state and never refresh it."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.sensors.command_sampling import (
    capture_robot_commands,
    verify_robot_commands,
)


def inputs(dtype=torch.float32):
    values = {
        ("position", False): torch.tensor([[0.01, 0.02]], dtype=dtype),
        ("velocity", False): torch.tensor([[0.1, 0.2]], dtype=dtype),
        ("position", True): torch.tensor([[0.0, 0.0]], dtype=dtype),
        ("velocity", True): torch.tensor([[0.0, 0.0]], dtype=dtype),
    }
    calls = []

    def read(kind, target):
        calls.append((kind, target))
        return values[kind, target]

    robot = SimpleNamespace(
        uid="robot",
        joint_names=("a", "b"),
        get_qpos=lambda *, target: read("position", target),
        get_qvel=lambda *, target: read("velocity", target),
    )
    stamp = PhysicalObservationStamp(
        source_id="native-owner",
        source_kind="default.contacts",
        revision=12,
        simulation_time=0.2,
    )
    kwargs = dict(
        expected_qpos=values["position", False].clone(),
        expected_qvel=values["velocity", False].clone(),
        timestamp=0.04,
        source_stamp=stamp,
        read_source_stamp=lambda: stamp,
    )
    return robot, values, calls, stamp, kwargs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_command_capture_preserves_original_read_order_and_owned_values(dtype):
    robot, values, calls, stamp, kwargs = inputs(dtype)
    observed = capture_robot_commands(robot, **kwargs)
    assert calls == [
        ("position", False),
        ("velocity", False),
        ("position", True),
        ("velocity", True),
    ]
    assert torch.equal(observed.qpos, kwargs["expected_qpos"])
    assert observed.qpos.dtype == observed.target_qvel.dtype == dtype
    assert not torch.equal(observed.qpos, observed.target_qpos)
    assert observed.timestamp == 0.04 and observed.source_stamp.simulation_time == 0.2
    verify_robot_commands(robot, observed, read_source_stamp=lambda: stamp)
    values["position", True].fill_(0.5)
    assert torch.equal(observed.target_qpos, torch.zeros(1, 2))
    with pytest.raises(ValueError, match="changed_during_check"):
        verify_robot_commands(robot, observed, read_source_stamp=lambda: stamp)


def test_multiple_native_rows_are_rejected_before_reading_commands():
    robot, _, calls, _, kwargs = inputs()
    kwargs["expected_qpos"] = kwargs["expected_qpos"].repeat(2, 1)
    kwargs["expected_qvel"] = kwargs["expected_qvel"].repeat(2, 1)
    with pytest.raises(ValueError, match="explicit B1 observation"):
        capture_robot_commands(robot, **kwargs)
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_id", "other-native-owner"),
        ("source_kind", "other.native"),
        ("revision", 13),
        ("simulation_time", 0.3),
    ],
)
def test_complete_native_stamp_must_remain_constant_during_capture(field, value):
    robot, values, calls, stamp, kwargs = inputs()
    stamps = iter((stamp, replace(stamp, **{field: value})))
    kwargs["read_source_stamp"] = lambda: next(stamps)
    with pytest.raises(ValueError, match="stamp_changed"):
        capture_robot_commands(robot, **kwargs)
    assert len(calls) == 4


def test_stale_source_rejects_before_command_reads():
    robot, values, calls, stamp, kwargs = inputs()
    kwargs["read_source_stamp"] = lambda: replace(stamp, source_id="stale-owner")
    with pytest.raises(ValueError, match="stamp_changed"):
        capture_robot_commands(robot, **kwargs)
    assert calls == []


@pytest.mark.parametrize(
    "kind,target",
    [("position", False), ("velocity", False), ("position", True), ("velocity", True)],
)
def test_verification_rejects_every_changed_array_without_refreshing_observation(
    kind, target
):
    robot, values, calls, stamp, kwargs = inputs()
    observed = capture_robot_commands(robot, **kwargs)
    values[kind, target].add_(0.01)
    with pytest.raises(ValueError, match="changed_during_check"):
        verify_robot_commands(robot, observed, read_source_stamp=lambda: stamp)
    assert torch.equal(observed.qpos, kwargs["expected_qpos"])


def test_planning_projection_cannot_replace_the_expected_measured_state():
    robot, values, calls, stamp, kwargs = inputs()
    kwargs["expected_qpos"] += 0.01
    with pytest.raises(ValueError, match="retained_observation_changed"):
        capture_robot_commands(robot, **kwargs)
    assert len(calls) == 4
