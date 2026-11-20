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
from dataclasses import replace
import torch
from embodichain.lab.expansion.contracts import TrajectoryTemplate
from embodichain.lab.sim.motion.expansion.operators import validate_motion_samples
from embodichain.lab.task_program.integrations._native_plan_expansion import (
    _fit_joint_residual,
)


def test_refinement_obeys_original_limits_and_keeps_owned_endpoints():
    reference = TrajectoryTemplate(
        "s",
        "r",
        "t",
        ("j",),
        torch.zeros(5, 1),
        torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]),
    )
    candidate = replace(
        reference, positions=torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]])
    )

    def check(value):
        return [
            validate_motion_samples(
                value.positions,
                value.dt,
                velocity_limits=torch.tensor([1.0]),
                acceleration_limits=torch.tensor([0.5]),
            )
        ]

    assert not check(candidate)[0].accepted
    fitted, checks, scale = _fit_joint_residual(
        reference,
        candidate,
        active=[0],
        joint_limits=torch.tensor([[-2.0, 2.0]]),
        minimum_delta=0.1,
        checks=check,
    )
    assert scale == 0.25
    assert fitted.positions[2, 0] == 0.25 and all(c.accepted for c in checks)
    assert torch.equal(
        fitted.positions[[0, 1, 3, 4]], reference.positions[[0, 1, 3, 4]]
    )
    fitted.positions.zero_()
    assert candidate.positions[2, 0] == 1.0
    assert (
        _fit_joint_residual(
            reference,
            candidate,
            active=[0],
            joint_limits=torch.tensor([[-2.0, 2.0]]),
            minimum_delta=0.3,
            checks=check,
        )
        is None
    )


def test_valid_draw_retains_original_amplitude():
    reference = TrajectoryTemplate(
        "s", "r", "t", ("j",), torch.zeros(3, 1), torch.tensor([0.0, 1.0, 1.0])
    )
    candidate = replace(reference, positions=torch.tensor([[0.0], [0.1], [0.0]]))

    def check(value):
        return [
            validate_motion_samples(
                value.positions,
                value.dt,
                velocity_limits=torch.tensor([1.0]),
                acceleration_limits=torch.tensor([0.5]),
            )
        ]

    fitted, _, scale = _fit_joint_residual(
        reference,
        candidate,
        active=[0],
        joint_limits=torch.tensor([[-1.0, 1.0]]),
        minimum_delta=0.01,
        checks=check,
    )
    assert scale == 1 and torch.equal(fitted.positions, candidate.positions)
    fitted.positions.zero_()
    assert candidate.positions[1, 0] == 0.1
