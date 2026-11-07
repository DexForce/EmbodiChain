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
"""Signed envelope measurements retain source axes, precision and row ordering."""

from __future__ import annotations
import json
from pathlib import Path
import torch
import pytest
from embodichain.compute.geometry.envelope import signed_envelope_gaps

_SOURCE = json.loads(
    (Path(__file__).parent / "data/directional_source_reference.json").read_text()
)
_AXES = {
    "left_of": ((1, -1),),
    "right_of": ((1, 1),),
    "front_of": ((0, -1),),
    "behind": ((0, 1),),
    "front_left_of": ((0, -1), (1, -1)),
    "front_right_of": ((0, -1), (1, 1)),
    "back_left_of": ((0, 1), (1, -1)),
    "back_right_of": ((0, 1), (1, 1)),
}


@pytest.mark.parametrize("case", _SOURCE["cases"])
def test_signed_envelopes_retain_b0_values_dtype_and_axis_order(case):
    inputs = _SOURCE["inputs"][case["input"]]
    dtype = getattr(torch, inputs["dtype"].removeprefix("torch."))
    values = [
        torch.tensor(inputs[key], dtype=dtype)
        for key in (
            "object_vertices",
            "object_pose",
            "reference_vertices",
            "reference_pose",
        )
    ]
    gaps = signed_envelope_gaps(*values, _AXES[case["relation"]])
    assert gaps.dtype == dtype
    assert torch.equal(
        gaps, torch.tensor(case["metrics"]["envelope_gaps_m"], dtype=dtype)
    )
    assert ((gaps >= 0) & torch.isfinite(gaps)).all(0).tolist() == case["accepted"]


def test_signed_envelopes_do_not_infer_an_axis_or_accept_invalid_selection():
    pose = torch.eye(4)[None]
    vertices = torch.zeros(1, 1, 3)
    with pytest.raises(ValueError, match="axes"):
        signed_envelope_gaps(vertices, pose, vertices, pose, ((True, 1),))
