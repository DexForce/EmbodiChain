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

"""Frozen B0-F1 Pour geometry and pre-IK candidate projection contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from embodichain.compute.geometry.pouring import (
    PourGraspProjection,
    gravity_top_frame,
    jaw_tilt_axis,
    pour_frames,
    pour_staging_pose,
)

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/pour_projection_source_reference.json").read_text()
)


def _projection_inputs():
    source = _REFERENCE["inputs"]
    return dict(
        source_vertices=np.array(source["source_vertices"]),
        receiver_vertices=np.array(source["receiver_vertices"]),
        staging=np.array(_REFERENCE["stage"]),
        receiver_pose=np.array(source["receiver_pose"]),
        return_pose=np.array(source["return_pose"]),
        pad_line_tcp=torch.tensor(source["pad_line_tcp"]),
        gravity_world=tuple(source["gravity_world"]),
        angle_rad=source["angle_rad"],
        sample_count=source["sample_count"],
    )


def test_numeric_pour_arc_and_candidate_projection_match_frozen_source():
    source = _REFERENCE["inputs"]
    inputs = _projection_inputs()
    stage = pour_staging_pose(
        inputs["source_vertices"],
        inputs["receiver_vertices"],
        np.array(source["source_pose"]),
        inputs["receiver_pose"],
        clearance_m=source["clearance_m"],
        gravity_world=inputs["gravity_world"],
    )
    np.testing.assert_array_equal(stage, _REFERENCE["stage"])
    frames = pour_frames(
        inputs["source_vertices"],
        inputs["receiver_vertices"],
        stage,
        inputs["receiver_pose"],
        np.array([1.0, 0, 0]),
        inputs["angle_rad"],
        inputs["return_pose"],
        sample_count=inputs["sample_count"],
        gravity_world=inputs["gravity_world"],
    )
    for name, actual in frames.items():
        if isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, _REFERENCE["frames"][name])
        elif isinstance(actual, tuple):
            assert list(actual) == _REFERENCE["frames"][name]
        else:
            assert actual == _REFERENCE["frames"][name]
    up, pivot = gravity_top_frame(inputs["source_vertices"], stage[:3, :3])
    np.testing.assert_array_equal(up, _REFERENCE["frames"]["up_object"])
    np.testing.assert_array_equal(pivot, _REFERENCE["frames"]["pivot_object"])
    variants = torch.tensor(source["variants"])
    before = torch.get_rng_state().clone()
    targets, valid = PourGraspProjection(**inputs).targets(variants)
    torch.testing.assert_close(
        targets, torch.tensor(_REFERENCE["targets"]), atol=0, rtol=0
    )
    assert valid.tolist() == _REFERENCE["valid"]
    assert torch.equal(before, torch.get_rng_state())
    # The parallel-jaw candidate has five staging fallbacks in source order.
    fallback = torch.tensor(stage, dtype=variants.dtype) @ variants[0, 0, 1]
    torch.testing.assert_close(targets[:, 0, 0, 1], fallback.expand(5, -1, -1))


def test_pour_projection_owns_all_numeric_snapshot_inputs():
    inputs = _projection_inputs()
    projection = PourGraspProjection(**inputs)
    variants = torch.tensor(_REFERENCE["inputs"]["variants"])
    expected, expected_valid = projection.targets(variants)
    for value in inputs.values():
        if isinstance(value, np.ndarray):
            value.fill(99)
        elif isinstance(value, torch.Tensor):
            value.fill_(0)
    actual, valid = projection.targets(variants)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(valid, expected_valid)


def test_jaw_tilt_projection_preserves_order_and_invalid_parallel_case():
    axis, direction, valid = jaw_tilt_axis(
        torch.tensor([[1.0, 0, 1], [0.0, 0, 1]]),
        torch.tensor([0.0, 0, 1.0]).expand(2, -1),
    )
    torch.testing.assert_close(axis, torch.tensor([[1.0, 0, 0], [0.0, 0, 0]]))
    torch.testing.assert_close(direction, torch.tensor([[0.0, -1, 0], [0.0, 0, 0]]))
    assert valid.tolist() == [True, False]


def test_source_static_staging_preserves_b0_rotations_and_offset_order():
    from scipy.spatial.transform import Rotation
    from embodichain.compute.geometry.pouring import source_pour_staging_frame

    reference = json.loads(
        (Path(__file__).parent / "data/static_pour_staging_b0.json").read_text()
    )
    for case in reference["cases"]:
        relative, rotation, pivot = source_pour_staging_frame(
            np.asarray(reference["vertices"]),
            np.asarray(case["axis"]),
            Rotation.from_euler("XYZ", case["angles"], degrees=True).as_matrix(),
            reference["reference_top"],
            reference["finger_radius"],
        )
        np.testing.assert_array_equal(relative.reshape(-1), case["relative_pose"])
        np.testing.assert_array_equal(rotation.reshape(-1), case["world_orientation"])
        np.testing.assert_allclose(rotation @ case["axis"], [0, 0, 1], atol=1e-15)
        np.testing.assert_array_equal(pivot, [0.01, 0.015, 0.05])
