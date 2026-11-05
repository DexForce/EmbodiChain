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

"""Frozen B0 command, candidate, query and RNG traces for prismatic recovery."""

from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch

from embodichain.lab.sim.atomic_actions.primitives import _articulation_recovery
from ._articulation_plan_fixture import (
    articulation_plan_fixture,
    articulation_plan_record,
)
from ._articulation_bounded_fixture import (
    bounded_fixture,
    bounded_record,
    park_fixture,
    park_record,
)

_DATA = Path(__file__).parent / "data"
_NOMINAL = json.loads((_DATA / "articulation_nominal_source.json").read_text())
_BOUNDED = json.loads((_DATA / "articulation_bounded_source.json").read_text())
_PARK = json.loads((_DATA / "articulation_park_source.json").read_text())


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


@pytest.mark.parametrize("case", _NOMINAL["cases"], ids=lambda value: value["mode"])
def test_nominal_prismatic_segments_preserve_full_source_commands(tmp_path, case):
    engine, _, context, invocation, calls = articulation_plan_fixture(
        tmp_path, case["mode"]
    )
    _seed(913)
    assert (
        articulation_plan_record(engine.plan(invocation, context), calls)
        == case["expected"]
    )


@pytest.mark.parametrize("case", _BOUNDED["cases"], ids=lambda value: value["mode"])
def test_bounded_prismatic_recovery_preserves_source_decisions_and_rng(
    tmp_path, monkeypatch, case
):
    fixture = bounded_fixture(tmp_path, case["mode"])
    monkeypatch.setattr(
        _articulation_recovery, "RobotMeshCollisionSnapshot", fixture.robot_query
    )
    _seed(913)
    assert (
        bounded_record(
            fixture, fixture.engine.plan(fixture.invocation, fixture.context)
        )
        == case["expected"]
    )


@pytest.mark.parametrize("case", _PARK["cases"], ids=lambda value: value["mode"])
def test_installed_withdrawal_park_preserves_source_vias_and_rng(tmp_path, case):
    fixture = park_fixture(tmp_path, case["mode"])
    _seed(917)
    assert (
        park_record(fixture, fixture.engine.plan(fixture.invocation, fixture.context))
        == case["expected"]
    )


def test_explicit_joint_offset_uses_normal_engine_and_full_free_motion_guard(tmp_path):
    from dataclasses import replace
    import torch
    from embodichain.lab.sim.atomic_actions import (
        JointPositionGoal,
        MoveJointsOptions,
        RecoveryPolicy,
    )
    from embodichain.lab.sim.atomic_actions.primitives.move_joints import (
        JointOffsetTarget,
    )

    fixture = park_fixture(tmp_path, "park_unowned")
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    ordinary = AtomicActionEngine(
        fixture.generator,
        free_motion_query=fixture.engine.planning_services.free_motion_query,
    )
    name = ordinary.robot.joint_names[0]
    invocation = ordinary.make_invocation(
        "move_joints",
        JointPositionGoal(JointOffsetTarget(name, 0.05)),
        control_parts={"primary": {"motion": "source_arm"}},
        motion_policy=fixture.invocation.motion_policy,
        skill_options=MoveJointsOptions(require_free_motion=True),
        recovery_policy=RecoveryPolicy(),
    )
    result = ordinary.plan(invocation, fixture.context)
    assert result.plan_success.tolist() == [True]
    expected = fixture.context.robot.qpos.clone()
    expected[:, 0] += 0.05
    torch.testing.assert_close(result.joint_trajectory.positions[:, -1], expected)
    assert result.diagnostics.metadata["idle_clearance"]["qualification"]["valid"]
    unsafe = replace(invocation, goal=JointPositionGoal(JointOffsetTarget(name, 0.6)))
    assert ordinary.plan(unsafe, fixture.context).plan_success.tolist() == [False]
    with pytest.raises(ValueError, match="qualification"):
        ordinary.plan(
            replace(invocation, skill_options=MoveJointsOptions()), fixture.context
        )
