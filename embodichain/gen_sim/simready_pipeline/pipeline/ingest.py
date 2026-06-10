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
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Optional

from embodichain.gen_sim.simready_pipeline.core.asset import Asset
from embodichain.gen_sim.simready_pipeline.utils.ingest_utils import (
    compute_folder_sha256,
    new_uuid,
    trimesh_parse_ingest,
    blender_parser_ingest,
    copy_obj_ingest,
    inject_semantic_from_config,
    inject_user_extra_info,
)
from embodichain.gen_sim.simready_pipeline.io.json_store import JsonStore
from embodichain.gen_sim.simready_pipeline.parser.base import ParserManager


def _load_ingest_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "gen_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


GEN_CONFIG = _load_ingest_config()
INGEST_CONFIG = GEN_CONFIG.get("ingest", {})
MESH_PROCESSING_CONFIG = GEN_CONFIG.get("mesh_processing", {})
CANOCAIL_ASSET_NAME = INGEST_CONFIG.get("canonical_asset_name", "asset.obj")
SOURCE_PREPARATION_CONFIG = INGEST_CONFIG.get("source_preparation", {})
SOURCE_PREPARATION_MODE = SOURCE_PREPARATION_CONFIG.get("mode", "blender")
SOURCE_PREPARATION_MODES = {"blender", "trimesh", "copy"}
UNPROCESSED_FORMATS = INGEST_CONFIG.get(
    "unprocessed_formats", [".urdf", ".usd"]
)  # Copy these for now; parsing can be added later.
PARSEABLE_MESH_FORMATS = INGEST_CONFIG.get(
    "parseable_mesh_formats", [".glb", ".gltf", ".obj", ".ply", ".stl"]
)  # Common mesh formats that need processing.

TRIMESH_INGEST_CONFIG = MESH_PROCESSING_CONFIG.get("trimesh_ingest", {})
BLENDER_REMESH_BAKE_CONFIG = MESH_PROCESSING_CONFIG.get(
    "blender_remesh_bake", INGEST_CONFIG.get("blender_remesh_bake", {})
)


def _resolve_source_preparation_mode(source_preparation_mode: str | None) -> str:
    mode = (source_preparation_mode or SOURCE_PREPARATION_MODE).lower()
    if mode not in SOURCE_PREPARATION_MODES:
        allowed = ", ".join(sorted(SOURCE_PREPARATION_MODES))
        raise ValueError(
            f"Unsupported ingest.source_preparation.mode: {mode!r}. "
            f"Allowed values: {allowed}"
        )
    return mode


def ingest_one_asset(
    asset_dir: str | Path,
    category: str,
    output_root: Path,
    store: JsonStore,
    manager: ParserManager,
    source_preparation_mode: str | None = None,
) -> Optional[Asset]:

    asset_dir = Path(asset_dir)  # source path
    source_preparation_mode = _resolve_source_preparation_mode(source_preparation_mode)
    ingest_sha256 = compute_folder_sha256(asset_dir)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    asset_id = new_uuid()
    asset_root = output_root / asset_id
    asset_root.mkdir(parents=True, exist_ok=False)

    asset_source = asset_root / "asset_source"
    asset_archive = asset_root / "asset_archive"

    files = [p for p in asset_dir.iterdir() if p.is_file()]
    file_suffixes = {p.suffix.lower() for p in files}

    has_unprocessed_format = any(
        suffix in file_suffixes for suffix in UNPROCESSED_FORMATS
    )

    archive_dst = asset_archive / asset_dir.name
    if archive_dst.exists():
        raise RuntimeError(f"Archive destination already exists: {archive_dst}")
    shutil.copytree(asset_dir, archive_dst)

    def find_first_mesh_file(files, formats):
        for suffix in formats:
            candidates = sorted(p for p in files if p.suffix.lower() == suffix)
            if candidates:
                return candidates[0]
        raise RuntimeError("No Valid Mesh File")

    if has_unprocessed_format:
        source_file = None
        ingest_mode = "direct_copy"
        asset_name = asset_dir.stem
        visual_info = None
    else:
        mesh_formats = (
            [".obj"] if source_preparation_mode == "copy" else PARSEABLE_MESH_FORMATS
        )
        source_file = find_first_mesh_file(files, mesh_formats)
        asset_name = source_file.stem if source_file else None
        ingest_mode = "unified"
        if source_preparation_mode == "copy":
            visual_info = copy_obj_ingest(
                source_file,
                asset_source,
                obj_name=CANOCAIL_ASSET_NAME,
            )
        elif source_preparation_mode == "trimesh":
            visual_info = trimesh_parse_ingest(
                source_file,
                asset_source,
                obj_name=CANOCAIL_ASSET_NAME,
                mtl_name=Path(CANOCAIL_ASSET_NAME).with_suffix(".mtl").name,
                config=TRIMESH_INGEST_CONFIG,
            )
        elif source_preparation_mode == "blender":
            visual_info = blender_parser_ingest(
                source_file,
                asset_source,
                obj_name=CANOCAIL_ASSET_NAME,
                config=BLENDER_REMESH_BAKE_CONFIG,
                trimesh_config=TRIMESH_INGEST_CONFIG,
            )
        else:
            raise AssertionError("unreachable source preparation mode")

    asset = Asset(
        asset_id=asset_id,
        ingest_sha256=ingest_sha256,
        identity={
            "name": asset_name,
            "source_dir": asset_dir.name,
            "category": category,
            "ingest_mode": ingest_mode,
            "source_preparation_mode": source_preparation_mode,
        },
        parsed={"visual": visual_info},
    )
    asset.status["ingested"] = True
    asset.status.setdefault("parsed", False)
    asset.status.setdefault("validated", False)

    if ingest_mode == "direct_copy":
        shutil.copytree(asset_dir, asset_source)
        asset.identity["normalized_source"] = "raw_copy"
        asset.identity["source_file"] = None
        asset.identity["source_type"] = "direct_copy"
        store.save_asset(asset)
        return asset  # no parser
    else:
        asset.identity["source_file"] = source_file.name
        asset.identity["source_type"] = source_file.suffix.lower()
        asset.identity["normalized_source"] = CANOCAIL_ASSET_NAME

    inject_semantic_from_config(asset_dir, asset)
    inject_user_extra_info(asset_dir, asset)
    manager.parse(asset, asset_root)
    store.save_asset(asset)

    return asset
