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

from itertools import product
import json
from pathlib import Path

import pytest
import torch
import trimesh

from embodichain.toolkits.graspkit.pg_grasp.contact_fit import (
    SampledHandGeometry,
    HandContactChecker,
    TriangleSurfaceQuery,
)


def _geometry() -> SampledHandGeometry:
    gaps = torch.tensor([0.12, 0.08, 0.04, 0.01])
    pads = []
    for gap in gaps:
        pair = []
        for sign in (-1, 1):
            pair.append(
                torch.tensor(
                    [
                        [sign * (float(gap) / 2 + depth), y, z]
                        for depth, y, z in product(
                            (0.0, 0.005), (-0.01, 0.01), (-0.02, 0.02)
                        )
                    ]
                )
            )
        pads.append(torch.stack(pair))
    return SampledHandGeometry(
        torch.tensor([[[0.0, 0.0, 0.5]]]).expand(4, -1, -1).clone(),
        torch.stack(pads),
        gaps,
    )


class _AnalyticBox:
    def query_batch_points(self, points, **kwargs):
        delta = points.abs() - points.new_tensor([0.03, 0.1, 0.1])
        distance = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1)
        distance += delta.amax(-1).clamp_max(0)
        return distance <= 0, distance

    def query_surface_normals(self, points):
        normals = torch.zeros_like(points)
        normals[:, 0] = points[:, 0].sign()
        return normals


def _checker(target=None):
    return HandContactChecker(
        _geometry(),
        _AnalyticBox() if target is None else target,
        contact_tolerance=0.0005,
        minimum_normal_cosine=0.5,
    )


def test_sample_interpolation_preserves_gap_and_returns_owned_outputs():
    geometry = _geometry()
    body, pads = geometry.at_width(torch.tensor([0.06, 0.2, 0.005]))
    torch.testing.assert_close(
        pads[:, 1, 0, 0] - pads[:, 0, 0, 0], torch.tensor([0.06, 0.12, 0.01])
    )
    assert torch.equal(body[:, 0, 2], torch.full((3,), 0.5))
    geometry.pads.zero_()
    assert pads[0, 1, 0, 0] == pytest.approx(0.03)


@pytest.mark.parametrize("translated", [False, True])
def test_finite_pad_fit_resolves_box_gap_then_qualifies_full_sweep(translated):
    checker = _checker()
    object_pose = torch.eye(4)
    if translated:
        object_pose[:3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        object_pose[:3, 3] = torch.tensor([0.2, -0.1, 0.3])
    relative = torch.eye(4)[None]
    relative[:, 0, 3] = 0.005
    original = object_pose @ relative
    poses, gaps = checker.fit_contacts(object_pose, original, torch.tensor([0.04]))
    # Two 0.1mm clearances around the independent 60mm box width.
    torch.testing.assert_close(gaps, torch.tensor([0.0602]), atol=1e-6, rtol=0)
    local = torch.linalg.inv(object_pose) @ poses
    torch.testing.assert_close(local[:, :3, 3], torch.zeros(1, 3), atol=1e-6, rtol=0)
    collision, depth = checker.query(object_pose, poses, gaps)
    assert collision.tolist() == [False]
    assert checker.last_counts["accepted"] == 1
    # Source diagnostics retain the last checked open sample (80mm),
    # not the already-checked fitted contact endpoint (60.2mm).
    assert depth[0] == pytest.approx((0.08 - 0.06) / 2, abs=1e-6)
    torch.testing.assert_close(original, object_pose @ relative)


def test_touching_both_pads_is_insufficient_without_opposed_material_normals():
    class SameFace(_AnalyticBox):
        def query_surface_normals(self, points):
            return points.new_tensor([1.0, 0.0, 0.0]).expand_as(points)

    checker = _checker(SameFace())
    collision, _ = checker.query(
        torch.eye(4), torch.eye(4)[None], torch.tensor([0.0602])
    )
    assert collision.tolist() == [True]
    assert checker.last_counts["contact_orientation"] == 1


def test_invalid_widths_are_rejected_without_surface_queries():
    class NoQuery(_AnalyticBox):
        def query_batch_points(self, points, **kwargs):
            raise AssertionError("invalid candidates must not query geometry")

    checker = _checker(NoQuery())
    collision, _ = checker.query(
        torch.eye(4),
        torch.eye(4).repeat(3, 1, 1),
        torch.tensor([0.0, 0.13, float("nan")]),
    )
    assert collision.tolist() == [True, True, True]


def test_signed_triangle_oracle_preserves_box_distances_and_outward_normals():
    mesh = trimesh.creation.box(extents=[0.06, 0.2, 0.2])
    query = TriangleSurfaceQuery(
        torch.tensor(mesh.vertices, dtype=torch.float32),
        torch.tensor(mesh.faces),
        normal_probe_distance=1e-5,
    )
    points = torch.tensor([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [-0.06, 0.0, 0.0]])
    _, distance = query.query_batch_points(points)
    torch.testing.assert_close(
        distance, torch.tensor([-0.03, 0.03, 0.03]), atol=1e-6, rtol=0
    )
    normals = query.query_surface_normals(
        torch.tensor([[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        normals, torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )


def test_signed_triangle_oracle_rejects_open_surface():
    with pytest.raises(ValueError, match="watertight"):
        TriangleSurfaceQuery(
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0, 1, 2]]),
            normal_probe_distance=1e-5,
        )


@pytest.mark.parametrize("index", [0, 1])
def test_contact_fit_matches_frozen_b0_f1_candidate_trace(index):
    record = json.loads(
        (Path(__file__).parent / "fixtures/contact_fit_b0_f1.json").read_text()
    )["records"][index]
    checker = _checker()
    obj = torch.tensor(record["object_pose"])
    rng = torch.random.get_rng_state().clone()
    fitted, gaps = checker.fit_contacts(
        obj, torch.tensor(record["poses"]), torch.tensor(record["widths"])
    )
    collision, depth = checker.query(obj, fitted, gaps)
    for actual, key in ((fitted, "fitted"), (gaps, "gaps"), (depth, "depth")):
        torch.testing.assert_close(actual, torch.tensor(record[key]), atol=0, rtol=0)
    assert collision.tolist() == record["collision"]
    assert checker.last_counts == record["counts"]
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("invalid", ["nonmonotonic", "nonfinite", "single_pad"])
def test_geometry_rejects_unusable_contact_samples(invalid):
    geometry = _geometry()
    if invalid == "nonmonotonic":
        geometry.gaps[1] = geometry.gaps[0]
    elif invalid == "nonfinite":
        geometry.pads[0, 0, 0, 0] = float("nan")
    pads = geometry.pads[:, :1] if invalid == "single_pad" else geometry.pads
    with pytest.raises(ValueError):
        SampledHandGeometry(geometry.body, pads, geometry.gaps)


def test_empty_candidate_set_is_not_an_accepted_grasp():
    checker = _checker()
    poses, widths = checker.fit_contacts(
        torch.eye(4), torch.empty(0, 4, 4), torch.empty(0)
    )
    collision, depth = checker.query(torch.eye(4), poses, widths)
    assert collision.numel() == depth.numel() == 0
    assert checker.last_counts["accepted"] == 0


def test_ambiguous_zero_normals_never_pass_a_zero_cosine_gate():
    class Ambiguous(_AnalyticBox):
        def query_surface_normals(self, points):
            return torch.zeros_like(points)

    checker = HandContactChecker(
        _geometry(), Ambiguous(), contact_tolerance=0.0005, minimum_normal_cosine=0.0
    )
    collision, _ = checker.query(
        torch.eye(4), torch.eye(4)[None], torch.tensor([0.0602])
    )
    assert collision.tolist() == [True]
    assert checker.last_counts["contact_orientation"] == 1
