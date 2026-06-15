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

from embodichain.gen_sim.simready_pipeline.core.asset import Asset
from embodichain.gen_sim.simready_pipeline.io.json_store import JsonStore


def test_asset_json_promotes_databank_server_metadata() -> None:
    asset = Asset(
        asset_id="asset-1",
        ingest_sha256="abc123",
        identity={"name": "source_mesh", "category": "cup"},
        semantics={"object_name_generated": "coffee cup"},
    )

    asset_json = asset.to_dict()

    assert asset_json["asset_id"] == "asset-1"
    assert asset_json["ingest_sha256"] == "abc123"
    assert asset_json["name"] == "coffee cup"
    assert asset_json["display_name"] == "coffee cup"
    assert asset_json["object_name_generated"] == "coffee cup"
    assert asset_json["category"] == "cup"
    assert asset_json["type"] == "cup"


def test_asset_json_name_falls_back_to_identity_name() -> None:
    asset = Asset(
        asset_id="asset-1",
        ingest_sha256="abc123",
        identity={"name": "source_mesh", "category": "cup"},
    )

    asset_json = asset.to_dict()

    assert asset_json["name"] == "source_mesh"
    assert asset_json["display_name"] == "source_mesh"
    assert asset_json["object_name_generated"] == ""


def test_json_store_registry_uses_top_level_category(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    asset = Asset(
        asset_id="asset-1",
        ingest_sha256="abc123",
        identity={"name": "source_mesh", "category": "cup"},
    )

    store.save_asset(asset)

    registry = json.loads((tmp_path / "registry.json").read_text())
    assert registry["assets"]["asset-1"]["category"] == "cup"
