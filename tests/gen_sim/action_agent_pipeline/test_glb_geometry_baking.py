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

from pathlib import Path

import pytest

from embodichain.gen_sim.action_agent_pipeline.generation.glb_geometry_baking import (
    bake_body_scale_into_glbs,
    bake_glb_geometry,
)


_BODY_SCALE = [2.0, 3.0, 4.0]
_SOURCE_NODE_TRANSLATION = [1.0, -2.0, 0.5]
_EXPECTED_MINIMUM = [1.0, -7.5, 0.0]
_EXPECTED_MAXIMUM = [3.0, -4.5, 4.0]


def test_bake_glb_geometry_flattens_node_transform_and_body_scale(
    tmp_path: Path,
) -> None:
    source_path = _write_transformed_box_glb(tmp_path / "source.glb")
    baked_path = bake_glb_geometry(
        source_path,
        tmp_path / "baked.glb",
        body_scale=_BODY_SCALE,
    )

    baked_mesh = _load_glb_mesh(baked_path)

    assert baked_path.suffix == ".glb"
    assert baked_mesh.bounds[0].tolist() == pytest.approx(_EXPECTED_MINIMUM)
    assert baked_mesh.bounds[1].tolist() == pytest.approx(_EXPECTED_MAXIMUM)


def test_bake_body_scale_into_glbs_replaces_runtime_scale_with_identity(
    tmp_path: Path,
) -> None:
    source_path = _write_transformed_box_glb(tmp_path / "source.glb")
    gym_config = {
        "rigid_object": [
            {
                "uid": "scaled_box",
                "shape": {"shape_type": "Mesh", "fpath": source_path.as_posix()},
                "body_scale": _BODY_SCALE,
            }
        ]
    }

    reports = bake_body_scale_into_glbs(
        gym_config,
        output_dir=tmp_path / "baked_assets",
    )
    baked_path = Path(gym_config["rigid_object"][0]["shape"]["fpath"])
    baked_mesh = _load_glb_mesh(baked_path)

    assert gym_config["rigid_object"][0]["body_scale"] == [1.0, 1.0, 1.0]
    assert baked_path.suffix == ".glb"
    assert baked_mesh.bounds[0].tolist() == pytest.approx(_EXPECTED_MINIMUM)
    assert baked_mesh.bounds[1].tolist() == pytest.approx(_EXPECTED_MAXIMUM)
    assert reports[0]["baked_path"] == baked_path.as_posix()


def _write_transformed_box_glb(path: Path) -> Path:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh,
        node_name="source_box",
        geom_name="source_box_geometry",
        transform=trimesh.transformations.translation_matrix(_SOURCE_NODE_TRANSLATION),
    )
    scene.export(path.as_posix(), file_type="glb")
    return path


def _load_glb_mesh(path: Path):
    trimesh = pytest.importorskip("trimesh")
    scene = trimesh.load(path.as_posix(), force="scene")
    return scene.dump(concatenate=True)
