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


"""Carrier-envelope rows against the frozen source calculation."""

from __future__ import annotations
import torch
from embodichain.compute.geometry.cargo import inside_carrier_envelope


def test_carrier_envelope_matches_frozen_source_rows():
    # Preserve the source's float32 origin before its float64 envelope queries,
    # including cancellation at the original lower tolerance boundary.
    # Source: 89d0b64194aaada87bf2cf81ebb78aa8026d320c cargo.py:inside_envelope.
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.002],
            [0.0, 0.0, -0.0021],
            [0.0, 0.0, 0.202],
            [0.0, 0.0, 0.2021],
            [0.982, 0.0, 0.1],
            [0.9821, 0.0, 0.1],
            [0.0, 0.972, 0.1],
            [0.0, 0.9721, 0.1],
        ],
        dtype=torch.float64,
    )
    carrier = torch.eye(4, dtype=torch.float64).repeat(len(offsets), 1, 1)
    carrier[:, :3, 3] = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32).double()
    objects = carrier.clone()
    objects[:, :3, 3] += offsets
    actual = inside_carrier_envelope(
        torch.tensor([[-0.02, -0.03, 0.0], [0.02, 0.03, 0.5]], dtype=torch.float64),
        objects,
        carrier,
        torch.tensor([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.2]], dtype=torch.float64),
    )
    assert actual.tolist() == [
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
