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

import json
from pathlib import Path
from typing import Any, Optional

from embodichain.gen_sim.simready_pipeline.core.asset import Asset


class JsonStore:
    """
    Simple JSON-based store for Assets and a global registry.
    """

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.registry_path = self.root / "registry.json"

    def _get_asset_json_path(self, asset_id: str) -> Path:
        return self.root / asset_id / "asset.json"

    def load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"assets": {}}

        registry = json.loads(self.registry_path.read_text())
        registry.setdefault("assets", {})
        return registry

    def _write_registry(self, registry: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(registry, indent=2))

    def _register_asset(self, asset_id: str, asset_json: dict[str, Any]) -> None:
        registry = self.load_registry()
        registry["assets"][asset_id] = {
            "path": str(self.root / asset_id),
            "category": asset_json.get("identity", {}).get("category"),
        }
        self._write_registry(registry)

    def save_asset(self, asset: Asset) -> None:
        asset_path = self._get_asset_json_path(asset.asset_id)
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_json = asset.to_dict()
        asset_path.write_text(json.dumps(asset_json, indent=2))
        self._register_asset(asset.asset_id, asset_json)

    def load_asset(self, asset_id: str) -> Optional[Asset]:
        asset_path = self._get_asset_json_path(asset_id)
        if not asset_path.exists():
            return None
        data = json.loads(asset_path.read_text())
        return Asset.from_dict(data)

    def write_asset(self, asset_id: str, asset_json: dict[str, Any]) -> None:
        asset_root = self.root / asset_id
        asset_root.mkdir(parents=True, exist_ok=True)

        asset_path = asset_root / "asset.json"
        asset_path.write_text(json.dumps(asset_json, indent=2))
        self._register_asset(asset_id, asset_json)

    def list_asset_ids(self) -> list[str]:
        registry = self.load_registry()
        return list(registry.get("assets", {}).keys())
