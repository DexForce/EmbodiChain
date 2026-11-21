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

"""Frozen actual Antipodal rows retain filtering, shifts and cross-axis fallback."""

from __future__ import annotations
from pathlib import Path
import json
import pytest
import torch
from embodichain.toolkits.graspkit.pg_grasp.preshape import (
    PreshapeHandClearance,
    generate_handle_grasp_candidates,
)

_DATA = json.loads((Path(__file__).parent / "data/preshape_grasp_b0.json").read_text())


@pytest.mark.parametrize(
    "case", _DATA["cases"], ids=lambda case: f"{case['fraction']}_{case['table_z']}"
)
def test_actual_sampler_rows_preserve_source_qualified_candidates(case):
    class RecordedSampler:
        def get_grasp_candidates(self, **kwargs):
            assert kwargs["approach_direction"].tolist() == case["approach"]
            return [
                tuple(torch.tensor(value) for value in row)
                for row in case["raw_candidates"]
            ]

    screen = PreshapeHandClearance(
        torch.tensor(case["points"]),
        case["aperture"],
        case["max_retreat"],
        case["table_z"],
    )
    result = generate_handle_grasp_candidates(
        RecordedSampler(),
        mesh_vertices=torch.tensor(_DATA["vertices"]),
        mesh_triangles=torch.tensor(_DATA["triangles"]),
        obj_poses=torch.tensor(_DATA["pose"]),
        approach_direction=torch.tensor(_DATA["direction"]),
        clearance=screen,
    )
    for row, expected in zip(result.rows, case["expected"], strict=True):
        for value, old in zip(row, expected, strict=True):
            tolerance = 4 * torch.finfo(value.dtype).eps
            torch.testing.assert_close(
                value,
                torch.tensor(old, dtype=value.dtype),
                rtol=tolerance,
                atol=tolerance,
            )
    assert result.required_width == pytest.approx(
        case["required_width"],
        rel=4 * torch.finfo(torch.float32).eps,
        abs=4 * torch.finfo(torch.float32).eps,
    )
    assert result.width_fit == case["width_fit"]
    assert (
        sum(int(torch.isfinite(row[2]).sum()) for row in result.rows)
        == case["qualified_count"]
    )
