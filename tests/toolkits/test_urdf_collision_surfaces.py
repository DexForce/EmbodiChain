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

"""Standalone URDF surface sampling with preserved source geometry oracles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from embodichain.toolkits.graspkit.pg_grasp.urdf_surfaces import (
    sample_urdf_collision_surfaces,
)


def test_mesh_scale_collision_origin_and_surface_density_match_source(tmp_path):
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("coordinated_urdf_surfaces.json")
        .read_text()
    )
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(fixture["urdf"])
    mesh = tmp_path / "pad.obj"
    mesh.write_text(fixture["mesh_obj"])
    before = (urdf.read_bytes(), mesh.read_bytes())

    actual = sample_urdf_collision_surfaces(
        urdf,
        link_names=tuple(fixture["expected"]),
        max_edge_by_link=fixture["max_edge_by_link"],
        max_subdivisions=fixture["max_subdivisions"],
    )

    assert tuple(actual) == tuple(fixture["expected"])
    for name, expected in fixture["expected"].items():
        assert list(actual[name].shape) == expected["shape"]
        assert actual[name].device.type == "cpu"
        assert actual[name].dtype == torch.float32
        torch.testing.assert_close(
            actual[name],
            torch.tensor(expected["points"], dtype=torch.float32).reshape(
                expected["shape"]
            ),
            rtol=0,
            atol=1e-7,
        )
    assert len(actual["body"]) > 8 and len(actual["pad"]) > 8
    assert actual["visual_only"].shape == (0, 3)
    assert before == (urdf.read_bytes(), mesh.read_bytes())


@pytest.mark.parametrize(
    "invalid", ["unknown_link", "unsupported_collision", "missing_bound"]
)
def test_explicit_geometry_selection_fails_without_shape_or_visual_fallback(
    tmp_path, invalid
):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        '<robot name="hand"><link name="pad"><collision><geometry>'
        '<cylinder radius="0.01" length="0.02"/>'
        "</geometry></collision></link></robot>"
    )
    name = "unknown" if invalid == "unknown_link" else "pad"
    bounds = {} if invalid == "missing_bound" else {name: 0.004}

    with pytest.raises(ValueError):
        sample_urdf_collision_surfaces(
            urdf,
            link_names=(name,),
            max_edge_by_link=bounds,
            max_subdivisions=10,
        )
