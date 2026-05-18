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
from typing import Any, Dict

import numpy as np
import trimesh
from embodichain.toolkits.simready_pipeline.parser.base import AssetParser
from embodichain.toolkits.simready_pipeline.core.asset import Asset
from embodichain.toolkits.simready_pipeline.utils.usd_utils import (
    convert_model_to_usd,
    DEFAULT_PHYSICS_PARAMS,
)


class UsdParser(AssetParser):

    name = "usd"

    def __init__(self):
        super().__init__()
        self.physics_properties = {}

    def build_physics(self, asset: Asset) -> Dict[str, Any]:

        if not isinstance(asset.physics, dict):
            raise ValueError("asset.physics must be a dict")

        physics_block = asset.physics

        if "properties" not in physics_block:
            raise KeyError("asset.physics missing 'properties'")

        props_block = physics_block["properties"]

        if not isinstance(props_block, dict):
            raise ValueError("asset.physics['properties'] must be dict")

        if "data" not in props_block:
            raise KeyError("asset.physics['properties'] missing 'data'")

        data_block = props_block["data"]

        if not isinstance(data_block, dict):
            raise ValueError("asset.physics['properties']['data'] must be dict")

        # Required numeric physics keys used by USD pipeline
        required_keys = [
            "mass",
            "density",
            "static_friction",
            "dynamic_friction",
            "restitution",
            "linear_damping",
            "angular_damping",
        ]

        # Merge provided data with defaults so missing keys are filled with safe defaults
        merged_data = DEFAULT_PHYSICS_PARAMS.copy()
        # data_block may contain a subset of params; update defaults with provided values
        merged_data.update({k: v for k, v in data_block.items() if v is not None})

        # Report any keys that were missing and therefore filled from defaults
        missing = [k for k in required_keys if k not in data_block]
        if missing:
            print(
                f"[Warning] Missing physics keys {missing}; using DEFAULT_PHYSICS_PARAMS for those values."
            )

        # Validate numeric types for required numeric keys
        for k in required_keys:
            if k not in merged_data:
                # This should not happen because DEFAULT_PHYSICS_PARAMS contains these keys
                raise KeyError(
                    f"Missing required physics parameter even after merging defaults: {k}"
                )
            if not isinstance(merged_data[k], (int, float)):
                raise TypeError(
                    f"Physics param '{k}' must be numeric, got {type(merged_data[k])}"
                )

        # Use merged_data going forward
        data_block = merged_data

        self.physics_properties = {
            "mode": physics_block["mode"],
            "source": physics_block.get("source"),
            "confidence": physics_block.get("confidence"),
            "properties": {
                "mode": props_block["mode"],
                "data": data_block,
            },
        }

        return self.physics_properties

    def parse(self, asset: Asset, asset_root: Path) -> None:
        asset.usd.setdefault("is_usd", False)
        asset.usd.setdefault("usd_path", "")
        if asset.asset_data.get("type") != "mesh":
            asset.usd.update({"asset dont have a mesh": "skipped"})
            return

        mesh_path_ori = asset_root / asset.asset_data.get("path")
        mesh_path_sr = asset_root / "asset_simready" / "asset_simready.obj"
        mesh_path = (
            mesh_path_sr
            if mesh_path_sr.exists()
            else mesh_path_ori if mesh_path_ori.exists() else None
        )
        out_path = asset_root / "asset_usd"
        self.build_physics(asset)
        convert_model_to_usd(
            mesh_path,
            out_path,
            physics_params=self.physics_properties["properties"]["data"],
        )
        usd_file = out_path / "asset_simready_inst.usdc"
        usd_path_str = ""
        if usd_file.exists():
            try:
                usd_path_str = str(usd_file.relative_to(asset_root))
            except Exception:
                usd_path_str = str(usd_file)

        asset.usd.update(
            {
                "is_usd": True,
                "usd_path": usd_path_str,
            }
        )
        return
