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

"""Qualified cloth input installation and preservation of local variants."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import shutil
import zipfile

import pytest

pytestmark = pytest.mark.no_sim


def _installer():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/tools/prepare_cloth_fold_assets.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_cloth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_qualified_inputs_and_preserve_local_meshes(tmp_path, monkeypatch):
    module = _installer()
    archive = tmp_path / "qualified.zip"
    files = {
        "w1_v030/robot.urdf": b"qualified robot",
        "cloth_mesh.npz": b"qualified cloth",
        "w1_v030/collision/pad.obj": b"qualified pad",
    }
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in files.items():
            bundle.writestr("m02_cloth_single_fold/inputs/" + name, data)
    monkeypatch.setattr(
        module, "_ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        module,
        "_FILES",
        {
            name: hashlib.sha256(files[name]).hexdigest()
            for name in ("w1_v030/robot.urdf", "cloth_mesh.npz")
        },
    )
    downloads = []

    def download(url, destination):
        downloads.append(url)
        shutil.copyfile(archive, destination)

    monkeypatch.setattr(module.urllib.request, "urlretrieve", download)
    root = tmp_path / "cache"
    module._prepare(root)
    module._prepare(root)
    assert downloads == [module._URL]
    for name, data in files.items():
        assert (root / name).read_bytes() == data
    local = root / "w1_v030/collision/pad.obj"
    local.write_bytes(b"local variant")
    with pytest.raises(ValueError, match="differs from the qualified bundle"):
        module._prepare(root)
    assert local.read_bytes() == b"local variant"


def test_reject_corrupt_download_before_installation(tmp_path, monkeypatch):
    module = _installer()
    monkeypatch.setattr(
        module.urllib.request,
        "urlretrieve",
        lambda url, target: Path(target).write_bytes(b"corrupt"),
    )
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        module._prepare(tmp_path)
    assert not (tmp_path / "w1_v030").exists()
    assert not (tmp_path / "cloth_mesh.npz").exists()
