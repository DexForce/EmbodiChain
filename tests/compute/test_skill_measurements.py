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

"""Physical contact classification with independent positive and negative examples."""

from __future__ import annotations

import torch
from embodichain.compute.geometry.skill_measurements import manipulation_contacts


def test_contact_facts_preserve_rows_and_reject_missing_quantities():
    # Target 1, pads 2/3, support 4. Normal points from actor zero to actor one.
    pairs = torch.tensor([[[1, 2], [3, 1], [4, 1]]] * 5)
    normals = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]] * 5)
    impulses = torch.ones(5, 3)
    present = torch.ones(5, 3, dtype=torch.bool)
    present[1, 1] = False  # only one pad: not a grip
    present[2, :] = False  # empty complete sample: measured absence
    quantity = present.clone()
    quantity[3, 0] = False  # emitted contact lacks impulse
    normals[4, 1] = torch.tensor([-1.0, 0.0, 0.0])  # same-sided normals: not opposed
    grip, hand, support, valid = manipulation_contacts(
        pairs,
        normals,
        impulses,
        present,
        quantity,
        torch.ones(5, dtype=torch.long),
        torch.tensor([[2, 3]] * 5),
        torch.full((5,), 4),
    )
    assert grip.tolist() == [True, False, False, False, False]
    assert hand.tolist() == [True, True, False, False, True]
    assert support.tolist() == [True, True, False, False, True]
    assert valid.tolist() == [True, True, True, False, True]
