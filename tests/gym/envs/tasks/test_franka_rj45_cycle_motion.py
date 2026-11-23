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
"""Verify that measured RJ45 corrections retain the authored command grid."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.extensions.tasks.manipulation.cable.rj45_integration import (
    FrankaRj45PhaseFactory,
    _FrankaRj45PhaseLowerer,
)
from embodichain.extensions.tasks.manipulation.cable.rj45_motion import Rj45Motion

pytestmark = pytest.mark.no_sim


def test_last_feedback_tick_uses_actual_target_at_seated_hold_boundary() -> None:
    positions = np.zeros((2401, 9), dtype=np.float32)
    positions[840] = 0.5
    positions[841:1141] = 0.375
    motion = Rj45Motion(
        positions,
        np.repeat(np.eye(4)[None], len(positions), axis=0),
        (),
        np.arange(len(positions)) / 60,
    )
    robot = Mock()
    robot.get_qpos.return_value = torch.full((1, 9), 0.125)
    feedback = Mock()
    feedback.correct.return_value = np.full(9, 0.25, dtype=np.float32)
    lowerer = _FrankaRj45PhaseLowerer(robot, Mock(), motion, feedback, 60)
    context = SimpleNamespace(
        batch_size=1,
        require_control_dt=lambda: 1 / 60,
        is_preview=False,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(
            timestamp=839 / 60,
            qpos=torch.zeros(1, 9),
        ),
    )
    insert = RegisteredSemanticCall(
        call_id=FrankaRj45PhaseFactory.call_id,
        arguments={"phase": "insert_and_release"},
    )

    corrected = lowerer.lower(
        insert, context=context, bound=None, option_template=None
    ).goal.target
    retried = lowerer.lower(
        insert, context=context, bound=None, option_template=None
    ).goal.target
    feedback.correct.assert_called_once()
    assert feedback.correct.call_args.args[0] == 14
    assert corrected.waypoint_count == 1
    assert corrected.duration.item() == 1 / 60
    torch.testing.assert_close(corrected.positions, torch.full((1, 1, 9), 0.25))
    torch.testing.assert_close(corrected.velocities, torch.full((1, 1, 9), 7.5))
    torch.testing.assert_close(retried.positions, corrected.positions)
    torch.testing.assert_close(retried.velocities, corrected.velocities)

    context.robot.timestamp = 14.0
    robot.get_qpos.return_value = corrected.positions[:, -1]
    hold = lowerer.lower(
        RegisteredSemanticCall(
            call_id=FrankaRj45PhaseFactory.call_id,
            arguments={"phase": "seated_hold"},
        ),
        context=context,
        bound=None,
        option_template=None,
    ).goal.target

    assert hold.waypoint_count == 300
    torch.testing.assert_close(hold.positions[:, 0], torch.full((1, 9), 0.375))
    torch.testing.assert_close(hold.velocities[:, 0], torch.full((1, 9), 7.5))
    assert torch.count_nonzero(hold.velocities[:, 1:]).item() == 0
