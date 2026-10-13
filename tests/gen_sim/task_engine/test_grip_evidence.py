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

from embodichain.gen_sim.task_engine._task_program.grip_evidence import (
    classify_contacts,
)


@pytest.mark.parametrize("swap", [False, True])
def test_wide_grip_uses_same_target_opposed_contacts_not_closing_angle(swap):
    pairs = torch.tensor([[1, 9], [2, 9]])
    normals = torch.tensor([[0.72, 0.69, 0.0], [-0.72, -0.69, 0.0]])
    if swap:
        pairs = pairs.flip(1)
        normals = -normals
    assert classify_contacts(
        pairs, normals, torch.tensor([0.02, 0.03]), target=9, pads=(1, 2)
    ) == (True, True)
    assert classify_contacts(
        pairs, normals, torch.tensor([0.02, 0.03]), target=10, pads=(1, 2)
    ) == (False, True)


@pytest.mark.parametrize(
    "normals,impulses,expected",
    [
        ([[1.0, 0, 0], [1.0, 0, 0]], [0.02, 0.03], (False, False)),
        ([[1.0, 0, 0], [-1.0, 0, 0]], [0.02, 0.0], (False, False)),
        ([[1.0, 0, 0], [-1.0, 0, 0]], [0.0, 0.0], (False, True)),
        ([[float("nan"), 0, 0], [-1.0, 0, 0]], [0.02, 0.03], (False, False)),
        ([[0.0, 0, 0], [-1.0, 0, 0]], [0.02, 0.03], (False, False)),
    ],
)
def test_unqualified_contact_is_never_positive(normals, impulses, expected):
    assert (
        classify_contacts(
            torch.tensor([[1, 9], [2, 9]]),
            torch.tensor(normals),
            torch.tensor(impulses),
            target=9,
            pads=(1, 2),
        )
        == expected
    )


def test_no_contacts_proves_clear_release():
    assert classify_contacts(
        torch.empty((0, 2), dtype=torch.long),
        torch.empty((0, 3)),
        torch.empty(0),
        target=9,
        pads=(1, 2),
    ) == (False, True)
