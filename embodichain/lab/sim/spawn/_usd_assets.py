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

"""Scene-owned concrete files for USD package textures on released DexSim."""

from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexsim.spawn import MaterialDesc

__all__: list[str] = []


class _UsdTextureFiles:
    """Keep package texture files available through scene topology rebuilds."""

    def __init__(self) -> None:
        self._directory: TemporaryDirectory[str] | None = None
        self._paths: dict[tuple[str, int, int], str] = {}

    def resolve(self, materials: dict[str, MaterialDesc]) -> dict[str, MaterialDesc]:
        from pxr import Ar

        resolved = {}
        for name, material in materials.items():
            changes = {}
            for item in fields(material):
                value = getattr(material, item.name)
                if (
                    item.name.endswith("_map")
                    and isinstance(value, str)
                    and Ar.IsPackageRelativePath(value)
                ):
                    changes[item.name] = self._resolve_path(value)
            resolved[name] = replace(material, **changes) if changes else material
        return resolved

    def _resolve_path(self, asset_path: str) -> str:
        from pxr import Ar

        outer, _ = Ar.SplitPackageRelativePathOuter(asset_path)
        stamp = Path(outer).stat()
        key = (asset_path, stamp.st_mtime_ns, stamp.st_size)
        if key in self._paths:
            return self._paths[key]
        asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(asset_path))
        if asset is None:
            raise FileNotFoundError(f"Cannot read USD package texture: {asset_path}")
        if self._directory is None:
            self._directory = TemporaryDirectory(prefix="embodichain-usdz-textures-")
        _, inner = Ar.SplitPackageRelativePathInner(asset_path)
        # Use a generated filename, never an archive member path, so package
        # names cannot escape the scene's private temporary directory.
        digest = sha256(repr(key).encode("utf-8")).hexdigest()
        destination = Path(self._directory.name) / f"{digest}{Path(inner).suffix}"
        destination.write_bytes(asset.GetBuffer())
        self._paths[key] = str(destination)
        return str(destination)

    def close(self) -> None:
        self._paths.clear()
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
