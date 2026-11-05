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
"""Original numeric grasp order, sampled opening and clearance without a host."""

from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from embodichain.compute.geometry.gripper import (
    adaptive_grasp_approaches,
    full_object_opening_mask,
    interpolate_hand_samples,
)
from embodichain.compute.geometry.hand_clearance import (
    released_hand_height,
    released_hand_targets,
)
from embodichain.compute.geometry.staging import distal_staging_variants
from embodichain.compute.geometry.grasp_fit import compute_grasp_fit_scale

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/grasp_clearance_b0.json").read_text()
)
_FIT = json.loads((Path(__file__).parent / "data/grasp_fit_scale_b0.json").read_text())


def _dtype(case):
    return getattr(torch, case["dtype"].removeprefix("torch."))


@pytest.mark.parametrize("case", _REFERENCE["approaches"])
def test_grasp_candidate_direction_region_and_order_match_b0(case):
    inputs = _REFERENCE["approach_inputs"][case["input"]]
    expected = _REFERENCE["approach_results"][case["expected"]]
    dtype = _dtype(inputs)
    axis, pose, tcp = [
        torch.tensor(inputs[key], dtype=dtype) for key in ("axis", "pose", "tcp")
    ]
    positive = None if case["positive"] is None else torch.tensor(case["positive"])
    random_before = torch.get_rng_state().clone()
    actual = adaptive_grasp_approaches(
        axis, pose, tcp, ends=case["ends"], positive=positive, purpose=case["purpose"]
    )
    assert torch.equal(random_before, torch.get_rng_state())
    for value, original in zip(actual, expected, strict=True):
        assert value.direction.dtype == dtype
        assert value.direction.tolist() == original["direction"]
        assert (None if value.axis is None else value.axis.tolist()) == original["axis"]
        assert value.positive.tolist() == original["positive"]
    axis.zero_()
    if actual[0].axis is not None:
        assert actual[0].axis.tolist() == expected[0]["axis"]


@pytest.mark.parametrize("case", _REFERENCE["opening"])
def test_opening_mask_keeps_complete_mesh_and_64_candidate_chunk_edges(case):
    dtype = _dtype(case)
    vertices, object_pose = [
        torch.tensor(case[key], dtype=dtype) for key in ("vertices", "object_pose")
    ]
    poses = torch.eye(4, dtype=dtype).repeat(len(case["translations_x"]), 1, 1)
    poses[:, 0, 3] = torch.tensor(case["translations_x"], dtype=dtype)
    poses[tuple(case["nonfinite"])] = float("nan")
    before = vertices.clone()
    assert (
        full_object_opening_mask(vertices, poses, object_pose, case["opening"]).tolist()
        == case["expected"]
    )
    assert torch.equal(vertices, before)


@pytest.mark.parametrize("case", _REFERENCE["interpolation"])
def test_hand_interpolation_preserves_width_dtype_and_endpoint_clamps(case):
    body, pads, gaps = [
        torch.tensor(case[key], dtype=torch.float32) for key in ("body", "pads", "gaps")
    ]
    widths = torch.tensor(case["widths"], dtype=_dtype(case))
    result, surfaces = interpolate_hand_samples(body, pads, gaps, widths)
    assert result.dtype == surfaces.dtype == widths.dtype
    assert result.tolist() == case["expected_body"]
    assert surfaces.tolist() == case["expected_pads"]
    result.fill_(99)
    assert body.tolist() == case["body"]


@pytest.mark.parametrize("case", _REFERENCE["clearance"])
def test_released_nominal_envelope_and_ordered_targets_match_b0(case):
    dtype = _dtype(case)
    current, root, object_pose, vertices = [
        torch.tensor(case[key], dtype=dtype)
        for key in ("current", "root", "object_pose", "vertices")
    ]
    height = released_hand_height(
        current,
        object_pose,
        vertices,
        opening_width=case["opening_width"],
        **case["reserve"],
    )
    targets = released_hand_targets(
        current, root, height=height, retreat=case["retreat"]
    )
    assert height.tolist() == case["expected_height"]
    assert targets.tolist() == case["expected_targets"]
    assert targets.dtype == dtype


@pytest.mark.parametrize("case", _REFERENCE["staging"], ids=lambda case: case["name"])
def test_distal_staging_candidate_order_and_bound_failures_match_b0(case):
    snapshot = dict(case["snapshot"])
    for key in ("tcp", "roots", "points"):
        snapshot[key] = {
            name: np.asarray(values) for name, values in snapshot[key].items()
        }
    actual = distal_staging_variants(
        torch.tensor(case["targets"], dtype=torch.float64),
        torch.tensor(case["object_pose"], dtype=torch.float64),
        np.asarray(case["attachment"]),
        **snapshot,
    )
    for (label, target, report), expected in zip(actual, case["expected"], strict=True):
        assert label == expected["label"] and report == expected["report"]
        assert (None if target is None else target.tolist()) == expected["targets"]


@pytest.mark.parametrize("case", _FIT["cases"])
def test_uniform_grasp_fit_matches_original_scale_and_support_bottom(case):
    vertices = np.asarray(_FIT["vertices"])
    before = vertices.copy()
    result = json.loads(
        json.dumps(asdict(compute_grasp_fit_scale(vertices, **case["input"])))
    )
    assert {key: result[key] for key in case["expected"]} == case["expected"]
    np.testing.assert_array_equal(vertices, before)
