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

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Optional

from embodichain.toolkits.simready_pipeline.core.asset import Asset
from embodichain.toolkits.simready_pipeline.utils.ingest_utils import (
    new_uuid,
    trimesh_parse_ingest,
    blender_parser_ingest,
    inject_semantic_from_config,
    inject_user_extra_info,
)
from embodichain.toolkits.simready_pipeline.io.json_store import JsonStore
from embodichain.toolkits.simready_pipeline.parser.base import ParserManager


def _load_ingest_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "gen_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("ingest", {})


INGEST_CONFIG = _load_ingest_config()
CANOCAIL_ASSET_NAME = INGEST_CONFIG.get("canonical_asset_name", "asset.obj")
CANOCAIL_TEXTURE_NAME = INGEST_CONFIG.get("canonical_texture_name", "")
UNPROCESSED_FORMATS = INGEST_CONFIG.get(
    "unprocessed_formats", [".urdf", ".usd"]
)  # 当前先复制，后续可以考虑解析
PARSEABLE_MESH_FORMATS = INGEST_CONFIG.get(
    "parseable_mesh_formats", [".glb", ".gltf", ".obj", ".ply", ".stl"]
)  # 主流的需要处理的格式

tex_size: int = int(INGEST_CONFIG.get("blender_texture_size", 2048))
png_name: str = INGEST_CONFIG.get("blender_texture_name", "surface_texture.png")


def ingest_one_asset(
    asset_dir: str | Path,
    category: str,
    output_root: Path,
    store: JsonStore,
    manager: ParserManager,
    simple_ingest: bool = True,
) -> Optional[Asset]:

    asset_dir = Path(asset_dir)  # source path

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    asset_id = "assets"
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
        return RuntimeError("No Vailed Mesh File")

    if has_unprocessed_format:
        source_file = None
        ingest_mode = "direct_copy"
        asset_name = asset_dir.stem
        visual_info = None
    else:
        source_file = find_first_mesh_file(files, PARSEABLE_MESH_FORMATS)
        asset_name = source_file.stem if source_file else None
        ingest_mode = "unified"
        if simple_ingest:
            visual_info = trimesh_parse_ingest(
                source_file,
                asset_source,
                obj_name=CANOCAIL_ASSET_NAME,
                mtl_name=Path(CANOCAIL_ASSET_NAME).with_suffix(".mtl").name,
            )
        else:
            visual_info = blender_parser_ingest(
                source_file,
                asset_source,
                texture_size=tex_size,
                png_name=png_name,
                obj_name=CANOCAIL_ASSET_NAME,
            )

    asset = Asset(
        asset_id=asset_id,
        identity={
            "name": asset_name,
            "source_dir": asset_dir.name,
            "category": category,
            "ingest_mode": ingest_mode,
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
