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
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import pytest
from dexsim.spawn import MaterialDesc

from embodichain.lab.sim.spawn._usd_assets import _UsdTextureFiles
from embodichain.lab.sim.spawn.scene import SpawnScene


def _package(path: Path, payload: bytes) -> str:
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("textures/map.png", payload)
    return f"{path}[textures/map.png]"


def test_package_textures_share_scene_files_and_preserve_source(tmp_path: Path) -> None:
    source = _package(tmp_path / "asset.usdz", b"texture data")
    material = MaterialDesc(name="source", base_color_map=source, normal_map=source)
    cache = _UsdTextureFiles()
    try:
        resolved = cache.resolve({"material": material})["material"]
        target = Path(resolved.base_color_map)
        assert target.read_bytes() == b"texture data"
        assert resolved.normal_map == str(target)
        assert cache.resolve({"material": material})["material"].base_color_map == str(
            target
        )
        assert material.base_color_map == source
        assert target.suffix == ".png"
    finally:
        cache.close()
    assert not target.exists()
    cache.close()


def test_local_textures_are_not_copied_or_changed(tmp_path: Path) -> None:
    source = tmp_path / "local.png"
    source.write_bytes(b"local data")
    material = MaterialDesc(name="local", base_color_map=str(source))
    cache = _UsdTextureFiles()
    try:
        assert cache.resolve({"local": material})["local"] is material
    finally:
        cache.close()
    assert source.read_bytes() == b"local data"


def test_changed_package_retains_old_files_and_reads_new_contents(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    uri = _package(package, b"first")
    material = MaterialDesc(name="image", base_color_map=uri)
    cache = _UsdTextureFiles()
    try:
        first = Path(cache.resolve({"image": material})["image"].base_color_map)
        _package(package, b"second, longer")
        second = Path(cache.resolve({"image": material})["image"].base_color_map)
        assert first != second
        assert first.read_bytes() == b"first"
        assert second.read_bytes() == b"second, longer"
    finally:
        cache.close()


def test_missing_package_texture_fails_before_native_materialization(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    _package(package, b"present")
    material = MaterialDesc(name="missing", base_color_map=f"{package}[absent.png]")
    cache = _UsdTextureFiles()
    try:
        with pytest.raises(FileNotFoundError, match="Cannot read USD package texture"):
            cache.resolve({"missing": material})
    finally:
        cache.close()


@pytest.mark.parametrize("close_error", [False, True])
def test_spawn_close_releases_texture_files_after_native_scene(
    tmp_path: Path, close_error: bool
) -> None:
    uri = _package(tmp_path / "asset.usdz", b"texture")
    cache = _UsdTextureFiles()
    resolved = cache.resolve({"image": MaterialDesc(name="image", base_color_map=uri)})
    target = Path(resolved["image"].base_color_map)

    def close_native() -> None:
        assert target.is_file()
        if close_error:
            raise RuntimeError("native close failed")

    scene = object.__new__(SpawnScene)
    scene._assets = {"image": object()}
    scene._usd_texture_files = cache
    scene.builder = SimpleNamespace(result=SimpleNamespace(close=close_native))
    if close_error:
        with pytest.raises(RuntimeError, match="native close failed"):
            scene.close()
    else:
        scene.close()
    assert not target.exists()
    assert not scene._assets
    assert scene.builder.result is None
