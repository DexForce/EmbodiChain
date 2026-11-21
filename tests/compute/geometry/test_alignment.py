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

"""Pure alignment targets compared with frozen pre-migration caller outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from embodichain.compute.geometry.alignment import (
    alignment_target_axis,
    held_alignment_targets,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/alignment_source_reference.json").read_text()
)


def _numeric_inputs(case: dict[str, object]) -> dict[str, object]:
    """Reuse the original caller's fixed observed matrices without runtime objects."""
    pose = torch.eye(4)[None]
    pose[:, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor(case["rotation"], dtype=torch.float32)
    )[None]
    pose[:, :3, 3] = torch.tensor([[0.2, 0.1, 0.6]])
    held = torch.eye(4)[None]
    held[:, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([0.0, 1.5707963267948966, 0.0])
    )[None]
    held[:, 2, 3] = 0.05
    root = torch.eye(4)[None]
    root[:, :2, 3] = torch.tensor([case["root_xy"]])
    return dict(
        object_pose=pose,
        local_axis=(0.0, 0.0, 1.0),
        position=case["position"],
        direction_free=case["direction_free"],
        preserve_yaw=case["preserve"],
        object_vertices=torch.tensor([[-0.1, -0.1, -0.05], [0.1, 0.1, 0.3]]),
        root_pose=root,
        object_to_eef=held,
    )


@pytest.mark.parametrize(
    "case", _REFERENCE["cases"], ids=lambda case: str(case["source_case_index"])
)
def test_alignment_matches_frozen_source_target_leaves(case: dict[str, object]) -> None:
    inputs = _numeric_inputs(case["inputs"])
    before = inputs["object_pose"].clone()
    rng = torch.get_rng_state().clone()
    actual = held_alignment_targets(**inputs)
    assert actual.tolist() == case["targets"]
    assert torch.equal(inputs["object_pose"], before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert actual.data_ptr() != inputs["object_pose"].data_ptr()


def test_axis_pole_selection_preserves_dtype_and_owns_results() -> None:
    source = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    directed = alignment_target_axis(source, direction_free=False)
    undirected = alignment_target_axis(source, direction_free=True)
    assert directed.tolist() == [[0.0, 0.0, 1.0]] * 3
    assert undirected.tolist() == [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    assert directed.dtype == undirected.dtype == source.dtype
    undirected.zero_()
    assert source[0, 2] == -1.0
    assert directed[:, 2].eq(1).all()


@pytest.mark.parametrize("missing", ["root_pose", "object_to_eef"])
def test_yaw_free_alignment_requires_its_supplied_numeric_frames(missing: str) -> None:
    inputs = _numeric_inputs(_REFERENCE["cases"][0]["inputs"])
    inputs[missing] = None
    with pytest.raises(ValueError, match="explicit root and held frames"):
        held_alignment_targets(**inputs)
