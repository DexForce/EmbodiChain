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

"""Native planar observations preserve silhouettes, gaps and exact owners."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest
import torch
import trimesh

from embodichain.compute.geometry.landing import (
    planar_floor_region,
    select_container_landing,
)
from embodichain.lab.sim.scene.planar_observation import NativePlanarPlacement


def _native(mesh, count=1):
    vertices = torch.tensor(mesh.vertices.copy()).float()[None].repeat(count, 1, 1)
    faces = torch.tensor(mesh.faces.copy()).long()[None].repeat(count, 1, 1)
    return SimpleNamespace(
        get_vertices=lambda **kw: vertices.clone(), get_triangles=lambda: faces.clone()
    )


def _u_mesh(x, y):
    vertices = np.array([(a, b, 0.0) for b in y for a in x])
    faces = []
    for row, column in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)):
        first = row * 4 + column
        faces.extend(((first, first + 1, first + 5), (first, first + 5, first + 4)))
    return trimesh.Trimesh(vertices, faces, process=False)


def test_triangle_silhouette_preserves_a_payload_concavity():
    # Migrated source regression: a smaller U fits; its convex hull crosses the notch.
    natives = {
        "floor": _native(_u_mesh((-0.3, -0.1, 0.1, 0.3), (-0.3, -0.1, 0.3))),
        "object": _native(_u_mesh((-0.25, -0.13, 0.13, 0.25), (-0.25, -0.13, 0.25))),
    }
    observer = NativePlanarPlacement(
        SimpleNamespace(get_rigid_object=natives.get), "object", "floor", batch_size=1
    )
    pose = torch.eye(4)[None]
    assert observer.measure(pose, pose)["full_footprint_inside"] == [False]
    exact = observer.measure(pose, pose, exact_silhouette=True)
    assert exact["full_footprint_inside"] == [True]
    assert exact["outside_floor_area_m2"] == [0.0]
    assert exact["support_gap_m"] == [0.0]
    assert exact["footprint_kind"] == "native_triangle_projection/v1"
    assert "accepted" not in exact


def test_native_ownership_batch_and_measured_gaps_are_preserved():
    natives = {
        "floor": _native(trimesh.creation.box(extents=(1, 1, 0.1)), 2),
        "object": _native(trimesh.creation.box(extents=(0.1, 0.1, 0.1)), 2),
    }
    observer = NativePlanarPlacement(
        SimpleNamespace(get_rigid_object=natives.get), "object", "floor", batch_size=2
    )
    ref = torch.eye(4)[None].repeat(2, 1, 1)
    obj = ref.clone()
    obj[:, 2, 3] = torch.tensor([0.11, 0.08])
    measured = observer.measure(obj, ref, exact_silhouette=True)
    assert measured["full_footprint_inside"] == [True, True]
    assert measured["support_gap_m"] == pytest.approx([0.01, -0.02], abs=1e-7)
    # A row outside the support is still an acquired geometric observation.
    obj[1, 0, 3] = 2
    assert observer.measure(obj, ref)["full_footprint_inside"] == [True, False]
    natives["object"] = _native(trimesh.creation.box(extents=(0.1, 0.1, 0.1)), 2)
    with pytest.raises(ValueError, match="native bindings"):
        observer.measure(obj, ref)


@pytest.mark.parametrize("damage", ["scale", "reflection", "tilt"])
def test_invalid_or_unqualified_reference_is_unavailable(damage):
    natives = {
        "floor": _native(trimesh.creation.box()),
        "object": _native(trimesh.creation.box()),
    }
    observer = NativePlanarPlacement(
        SimpleNamespace(get_rigid_object=natives.get), "object", "floor", batch_size=1
    )
    pose = torch.eye(4)[None]
    ref = pose.clone()
    if damage == "scale":
        ref[0, 0, 0] = 2
    elif damage == "reflection":
        ref[0, 0, 0] = -1
    else:
        ref[0, :3, :3] = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
    with pytest.raises(ValueError):
        observer.measure(pose, ref)


def test_arm_side_landing_preserves_margin_and_payload_height():
    reference = trimesh.creation.box(extents=(1, 1, 0.1))
    payload = trimesh.creation.box(extents=(0.1, 0.1, 0.2))
    landing = select_container_landing(
        reference, payload.vertices, np.array([2.0, 0.0, 0.0])
    )
    assert landing == pytest.approx((0.43, 0.0, 0.151))
    floor, height = planar_floor_region(reference)
    assert floor.area == pytest.approx(1.0)
    assert height == pytest.approx(0.05)


def test_measurements_equal_frozen_b0_rows_and_transformed_frames():
    reference = json.loads(
        (Path(__file__).parent / "fixtures/planar_placement_b0.json").read_text()
    )
    assert reference["source_commit"] == "89d0b64194aaada87bf2cf81ebb78aa8026d320c"
    natives = {
        name: _native(
            trimesh.Trimesh(values["vertices"], values["faces"], process=False), 2
        )
        for name, values in reference["meshes"].items()
    }
    observer = NativePlanarPlacement(
        SimpleNamespace(get_rigid_object=natives.get), "object", "floor", batch_size=2
    )
    for case in reference["cases"]:
        measured = observer.measure(
            torch.tensor(case["object_pose"]),
            torch.tensor(case["reference_pose"]),
            exact_silhouette=case["exact_silhouette"],
        )
        expected = dict(case["measurements"])
        support = expected.pop("support_tolerance_m")
        penetration = expected.pop("penetration_tolerance_m")
        assert measured == expected, case["name"]
        # Applying the source acceptance thresholds stays a caller decision.
        assert [
            inside and -penetration <= gap <= support
            for inside, gap in zip(
                measured["full_footprint_inside"],
                measured["support_gap_m"],
                strict=True,
            )
        ] == case["accepted"]


def test_planar_source_snapshot_and_content_capture_keep_original_owners():
    from embodichain.lab.sim.scene.planar_observation import capture_planar_contents

    natives = {
        "floor": _native(trimesh.creation.box((0.6, 0.4, 0.02))),
        "object": _native(trimesh.creation.box((0.03, 0.03, 0.06))),
    }
    simulation = SimpleNamespace(num_envs=1, get_rigid_object=natives.get)
    observer = NativePlanarPlacement(simulation, "object", "floor", batch_size=1)
    vertices, floors = observer.source_geometry()
    vertices.fill_(100.0)
    second, again = observer.source_geometry()
    assert float(second.abs().max()) < 1.0 and floors[0][0].equals(again[0][0])
    content = capture_planar_contents(simulation, (("logical", "object"),))
    assert content[0][0] == "logical" and content[0][1].shape == (8, 3)
    natives["object"] = _native(trimesh.creation.box())
    with pytest.raises(ValueError, match="native bindings"):
        observer.source_geometry()
