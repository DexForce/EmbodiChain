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

"""The observed-replay robot uses a separate, checked asset cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from embodichain.data import constants, dataset
from embodichain.data.download import find_asset_class, get_registry
from embodichain_robots.agibot.g1 import assets

pytestmark = pytest.mark.no_sim


def test_observed_replay_asset_is_distinct_from_the_original_usd() -> None:
    assert dataset.get_data_class("AgibotG1") is assets.AgibotG1
    assert (
        dataset.get_data_class("AgibotG1ObservedReplay")
        is assets.AgibotG1ObservedReplay
    )
    assert find_asset_class("AgibotG1ObservedReplay", get_registry()) == (
        "robot",
        assets.AgibotG1ObservedReplay,
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "https://hf-mirror.com/datasets/DexForceAI/embodichain_data/resolve/main/",
        "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/",
        "https://assets.example.test/",
    ],
)
def test_observed_replay_descriptor_pins_revision_and_respects_mirrors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    captured = {}

    class Intercepted(Exception):
        pass

    def initialize(self, cache_prefix, descriptor, root):
        captured.update(cache_prefix=cache_prefix, descriptor=descriptor, root=root)
        raise Intercepted

    monkeypatch.setattr(dataset.EmbodiChainDataset, "__init__", initialize)
    monkeypatch.setattr(assets, "EMBODICHAIN_DOWNLOAD_PREFIX", prefix)
    with pytest.raises(Intercepted):
        assets.AgibotG1ObservedReplay(str(tmp_path))
    revision = assets.AgibotG1ObservedReplay._revision
    filename = "robot_assets/AgibotG1ObservedReplayV1.zip"
    origin = (
        "https://huggingface.co/datasets/DexForceAI/embodichain_data/"
        f"resolve/{revision}/{filename}"
    )
    selected = prefix.replace("/resolve/main/", f"/resolve/{revision}/") + filename
    assert captured["descriptor"].urls == list(dict.fromkeys([selected, origin]))
    assert captured["descriptor"].md5 == assets.AgibotG1ObservedReplay._archive_md5
    assert captured["cache_prefix"] == "AgibotG1ObservedReplayV1"
    assert captured["root"] == str(tmp_path)


@pytest.fixture
def observed_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source"
    archive = source / "robot_assets/AgibotG1ObservedReplayV1.zip"
    archive.parent.mkdir(parents=True)
    contents = {
        "robot_physical/robot.urdf": b'<robot name="fixture"><link name="base"/></robot>',
        "robot_physical/pad.obj": b"fixture collision resource",
        "robot_physical/LICENSE.agibot": b"fixture retained license",
    }
    checksums = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in contents.items()
    ).encode()
    with ZipFile(archive, "w") as output:
        output.writestr("robot_physical/", b"")
        for name, content in contents.items():
            output.writestr(name, content)
        output.writestr("SHA256SUMS", checksums)
    cls = assets.AgibotG1ObservedReplay
    monkeypatch.setattr(
        cls, "_archive_md5", hashlib.md5(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        cls, "_archive_sha256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(cls, "_checksums_sha256", hashlib.sha256(checksums).hexdigest())
    monkeypatch.setattr(assets, "EMBODICHAIN_DOWNLOAD_PREFIX", source.as_uri() + "/")
    return archive


def test_observed_replay_resolves_flat_archive_from_empty_cache(
    observed_archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = str(tmp_path / "cache")
    monkeypatch.setattr(constants, "EMBODICHAIN_DEFAULT_DATA_ROOT", root)
    monkeypatch.setattr(assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", root)
    path = Path(
        dataset.get_data_path("AgibotG1ObservedReplay/robot_physical/robot.urdf")
    )
    assert (
        path
        == Path(root) / "extract/AgibotG1ObservedReplayV1/robot_physical/robot.urdf"
    )
    assert path.read_bytes() == b'<robot name="fixture"><link name="base"/></robot>'
    observed_archive.unlink()
    assert dataset.get_data_path(
        "AgibotG1ObservedReplay/robot_physical/robot.urdf"
    ) == str(path)


@pytest.mark.parametrize("relative", ["SHA256SUMS", "robot_physical/pad.obj"])
def test_observed_replay_rejects_changed_cached_resources(
    observed_archive: Path, tmp_path: Path, relative: str
) -> None:
    root = str(tmp_path / "cache")
    asset = assets.AgibotG1ObservedReplay(root)
    observed_archive.unlink()
    (Path(asset.extract_dir) / relative).write_bytes(b"modified cached resource")
    with pytest.raises(ValueError, match="missing or changed"):
        assets.AgibotG1ObservedReplay(root)
