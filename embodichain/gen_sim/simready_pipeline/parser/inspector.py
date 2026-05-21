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
from embodichain.gen_sim.simready_pipeline.core.asset import Asset
from embodichain.gen_sim.simready_pipeline.parser.base import AssetParser


class AssetInspector(AssetParser):
    name = "inspector"

    def _find_first_file(self, root: Path, suffixes: tuple[str, ...]) -> Path | None:
        candidates: list[Path] = []
        for suffix in suffixes:
            candidates.extend(sorted(root.rglob(f"*{suffix}")))
        return candidates[0] if candidates else None

    def parse(self, asset: Asset, asset_root: Path) -> None:
        asset_source_dir = asset_root / "asset_source"

        asset.asset_data.clear()
        asset.simulation.setdefault("articulation", {})

        if not asset_source_dir.exists():
            print(f"Warning: asset_source not found: {asset_source_dir}")
            return

        asset_id = asset.asset_id
        canonical_mesh = asset_source_dir / "asset.obj"

        urdf_file = self._find_first_file(asset_source_dir, (".urdf",))
        if urdf_file is not None:
            asset.simulation["articulation"] = {
                "type": "articulation",
                "format": "urdf",
                "file_path": str(urdf_file.relative_to(asset_root)),
            }
            asset.asset_data = {
                "id": asset_id,
                "type": "articulation",
                "format": "urdf",
                "path": str(urdf_file.relative_to(asset_root)),
            }
            return

        if canonical_mesh.exists():
            asset.asset_data = {
                "id": asset_id,
                "type": "mesh",
                "format": "obj",
                "path": str(canonical_mesh.relative_to(asset_root)),
            }
            return

        mesh_file = self._find_first_file(
            asset_source_dir, (".obj", ".gltf", ".glb", ".ply", ".stl")
        )
        if mesh_file is not None:
            asset.asset_data = {
                "id": asset_id,
                "type": "mesh",
                "format": mesh_file.suffix.lstrip(".").lower(),
                "path": str(mesh_file.relative_to(asset_root)),
            }
            return

        usd_file = self._find_first_file(asset_source_dir, (".usd",))

        if usd_file is not None:
            asset.asset_data = {
                "id": asset_id,
                "type": "scene",
                "format": "usd",
                "path": str(usd_file.relative_to(asset_root)),
            }
            return

        print(f"Warning: No supported files found in {asset_source_dir}")
