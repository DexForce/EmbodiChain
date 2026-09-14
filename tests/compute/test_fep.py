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
"""Numerical correctness and bounded-work regressions for FEP correction."""

from __future__ import annotations

import torch

from embodichain.compute.kinematics import _fep


def test_mixed_converged_and_stalled_rows_stop_without_false_success() -> None:
    calls = []

    def fk(qpos: torch.Tensor) -> torch.Tensor:
        calls.append(len(qpos))
        pose = torch.eye(4).repeat(len(qpos), 1, 1)
        pose[:, :3, 3] = qpos[:, :3]
        return pose

    def jacobian(qpos: torch.Tensor) -> torch.Tensor:
        return torch.eye(6, 7).repeat(len(qpos), 1, 1)

    target = torch.eye(4).repeat(3, 1, 1)
    target[:, 0, 3] = torch.tensor([0.1, 0.5, 1.0])
    seed = torch.zeros(3, 7)
    seed[0, 0] = 0.1  # Already converged; must remain exactly unchanged.
    lower, upper = -torch.ones(7), torch.ones(7)
    upper[0] = 0.6  # Last target is unreachable and must stay unsuccessful.
    valid, result = _fep.solve_seed(
        target,
        seed,
        lower,
        upper,
        fk,
        jacobian,
        max_iterations=200,
        damping=0.01,
        max_step=0.35,
        position_tolerance=1e-5,
        rotation_tolerance=1e-5,
    )
    assert valid.tolist() == [True, True, False]
    assert torch.equal(result[0], seed[0])
    torch.testing.assert_close(result[1, 0], torch.tensor(0.5), atol=1e-5, rtol=0)
    assert result[2, 0].item() == upper[0].item()
    # Detect accidentally spending the full budget on deterministic no-op steps.
    assert len(calls) < 10


def test_final_allowed_update_is_checked_using_its_fk_error() -> None:
    def fk(qpos: torch.Tensor) -> torch.Tensor:
        poses = torch.eye(4).repeat(len(qpos), 1, 1)
        poses[:, :3, 3] = qpos[:, :3]
        return poses

    target = torch.eye(4)[None]
    target[:, 0, 3] = 0.2
    valid, result = _fep.solve_seed(
        target,
        torch.zeros(1, 7),
        -torch.ones(7),
        torch.ones(7),
        fk,
        lambda q: torch.eye(6, 7).repeat(len(q), 1, 1),
        max_iterations=1,
        damping=0.001,
        max_step=0.35,
        position_tolerance=1e-5,
        rotation_tolerance=1e-5,
    )
    assert valid.item()
    torch.testing.assert_close(fk(result), target, atol=1e-5, rtol=0)


def test_backtracking_selects_lowest_error_scale() -> None:
    def fk(qpos: torch.Tensor) -> torch.Tensor:
        poses = torch.eye(4).repeat(len(qpos), 1, 1)
        poses[:, 0, 3] = qpos[:, 0].square()
        return poses

    def jacobian(qpos: torch.Tensor) -> torch.Tensor:
        jac = torch.zeros(len(qpos), 6, 7)
        jac[:, 0, 0] = 2 * qpos[:, 0]
        return jac

    target = torch.eye(4)[None]
    target[:, 0, 3] = 1.0
    seed = torch.zeros(1, 7)
    seed[:, 0] = 0.1
    valid, result = _fep.solve_seed(
        target,
        seed,
        -10 * torch.ones(7),
        10 * torch.ones(7),
        fk,
        jacobian,
        max_iterations=1,
        damping=0.01,
        max_step=4.0,
        position_tolerance=1e-5,
        rotation_tolerance=1e-5,
    )
    # Full/half steps overshoot. The quarter step gives x=1.21, closer to 1
    # than the eighth step's x=0.36; its q must be 0.1 + 4 / 4.
    torch.testing.assert_close(result[0, 0], torch.tensor(1.1))
    assert not valid.item()
