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

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPO_ROOT
    / "embodichain"
    / "gen_sim"
    / "simready_pipeline"
    / "configs"
    / "gen_config.json"
)
ALLOWED_SCENE_MESH_STRATEGIES = {"first", "concatenate"}
ALLOWED_SOURCE_PREPARATION_MODES = {"blender", "trimesh"}


@pytest.fixture(scope="module")
def gen_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_gen_config_uses_mesh_processing_schema(gen_config: dict[str, Any]) -> None:
    assert "ingest" in gen_config
    assert "mesh_processing" in gen_config
    assert "llm" in gen_config


def test_mesh_processing_declares_expected_stages(
    gen_config: dict[str, Any],
) -> None:
    mesh_processing = gen_config["mesh_processing"]

    assert "trimesh_ingest" in mesh_processing
    assert "blender_remesh_bake" in mesh_processing
    assert "blender_cleanup_decimate" in mesh_processing
    assert "simready_finalize" in mesh_processing


def test_ingest_config_declares_canonical_mesh_formats(
    gen_config: dict[str, Any],
) -> None:
    ingest_config = gen_config["ingest"]
    parseable_mesh_formats = ingest_config["parseable_mesh_formats"]
    source_preparation = ingest_config["source_preparation"]

    assert ingest_config["canonical_asset_name"].endswith(".obj")
    assert source_preparation["mode"] in ALLOWED_SOURCE_PREPARATION_MODES
    assert isinstance(parseable_mesh_formats, list)
    assert parseable_mesh_formats
    assert all(fmt.startswith(".") for fmt in parseable_mesh_formats)


def test_trimesh_ingest_config_values_are_valid(
    gen_config: dict[str, Any],
) -> None:
    trimesh_config = gen_config["mesh_processing"]["trimesh_ingest"]
    export_config = trimesh_config["export"]

    assert trimesh_config["scene_mesh_strategy"] in ALLOWED_SCENE_MESH_STRATEGIES
    assert trimesh_config["mtl_name"].endswith(".mtl")
    assert isinstance(trimesh_config["visual"]["default_face_color"], list)
    assert isinstance(trimesh_config["visual"]["pbr_base_color_only"], bool)
    assert isinstance(export_config["include_normals"], bool)
    assert isinstance(export_config["include_color"], bool)
    assert isinstance(export_config["include_texture"], bool)
    assert isinstance(export_config["write_texture"], bool)


def test_blender_mesh_processing_values_are_valid(
    gen_config: dict[str, Any],
) -> None:
    mesh_processing = gen_config["mesh_processing"]
    remesh_bake = mesh_processing["blender_remesh_bake"]
    cleanup_decimate = mesh_processing["blender_cleanup_decimate"]

    assert remesh_bake["remesh"]["voxel_size"] > 0.0
    assert remesh_bake["remesh"]["min_voxel_size_ratio"] > 0.0
    assert 0.0 < remesh_bake["decimate"]["ratio"] <= 1.0
    assert remesh_bake["bake"]["texture_size"] > 0
    assert isinstance(cleanup_decimate["enabled"], bool)
    assert cleanup_decimate["cleanup"]["merge_dist"] > 0.0
    assert isinstance(cleanup_decimate["cleanup"]["remove_non_manifold"], bool)
    assert isinstance(cleanup_decimate["cleanup"]["triangulate"], bool)
    assert 0.0 < cleanup_decimate["simplify"]["ratio"] <= 1.0
    assert cleanup_decimate["simplify"]["weld_distance"] > 0.0
    assert isinstance(cleanup_decimate["simplify"]["collapse_triangulate"], bool)


def test_simready_finalize_config_values_are_valid(
    gen_config: dict[str, Any],
) -> None:
    render_resolution = gen_config["mesh_processing"]["simready_finalize"][
        "render_resolution"
    ]

    assert isinstance(render_resolution, int)
    assert render_resolution > 0
