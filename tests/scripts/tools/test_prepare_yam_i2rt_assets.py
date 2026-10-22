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

"""Verify pinned downloads and preserve existing assets on preparation failure."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.tools import prepare_yam_i2rt_assets as preparation


@pytest.fixture
def source_payload(monkeypatch: pytest.MonkeyPatch) -> bytes:
    payload = b"pinned MIT asset"
    monkeypatch.setattr(preparation, "_SOURCE_FILES", ("LICENSE",))
    monkeypatch.setattr(
        preparation,
        "_EXPECTED_SHA256",
        {"LICENSE": hashlib.sha256(payload).hexdigest()},
    )
    return payload


def test_download_uses_fixed_revision_and_verifies_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_payload: bytes
) -> None:
    def open_asset(request, timeout):
        assert request.full_url == (
            "https://raw.githubusercontent.com/robosimstudio/robosimstudio/"
            "73739abe212f963e8a9f5862b27bca370df091a7/"
            "assets/embodiments/yam_i2rt/LICENSE"
        )
        assert timeout == 60
        return io.BytesIO(source_payload)

    monkeypatch.setattr(preparation.urllib.request, "urlopen", open_asset)
    preparation._fetch_sources(tmp_path, None)
    assert (tmp_path / "LICENSE").read_bytes() == source_payload


def test_bad_download_never_installs_a_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_payload: bytes
) -> None:
    monkeypatch.setattr(
        preparation.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong")
    )
    generate = Mock()
    monkeypatch.setattr(preparation, "_generate_robot", generate)
    output = tmp_path / "YamI2rt"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preparation._prepare(output, None)
    generate.assert_not_called()
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_offline_sources_require_the_same_hash_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_payload: bytes
) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "LICENSE").write_bytes(source_payload)
    download = Mock(side_effect=AssertionError("Offline mode used the network"))
    monkeypatch.setattr(preparation.urllib.request, "urlopen", download)
    preparation._fetch_sources(tmp_path / "staged", upstream)
    download.assert_not_called()
    assert (tmp_path / "staged/LICENSE").read_bytes() == source_payload
    (upstream / "LICENSE").write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preparation._fetch_sources(tmp_path / "other", upstream)


def test_generated_mismatch_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_payload: bytes
) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "LICENSE").write_bytes(source_payload)

    def wrong_output(staged):
        (staged / "LICENSE").write_bytes(b"conversion damaged the license")

    monkeypatch.setattr(preparation, "_generate_robot", wrong_output)
    output = tmp_path / "cache/YamI2rt"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preparation._prepare(output, upstream)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_existing_verified_bundle_is_reused_without_download_or_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "yam_i2rt.urdf").write_text('<robot name="fixture"/>')
    monkeypatch.setattr(
        preparation,
        "_EXPECTED_SHA256",
        {"yam_i2rt.urdf": preparation._sha256(tmp_path / "yam_i2rt.urdf")},
    )
    fetch, generate = Mock(), Mock()
    monkeypatch.setattr(preparation, "_fetch_sources", fetch)
    monkeypatch.setattr(preparation, "_generate_robot", generate)
    assert preparation._prepare(tmp_path, None) == tmp_path / "yam_i2rt.urdf"
    fetch.assert_not_called()
    generate.assert_not_called()


def test_modified_existing_bundle_is_preserved(
    tmp_path: Path, source_payload: bytes
) -> None:
    (tmp_path / "LICENSE").write_bytes(b"local modification")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preparation._prepare(tmp_path, None)
    assert (tmp_path / "LICENSE").read_bytes() == b"local modification"


def test_cli_returns_failure_for_an_invalid_asset_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("EMBODICHAIN_DATA_ROOT", str(tmp_path))
    (tmp_path / "YamI2rt").mkdir()
    with pytest.raises(SystemExit) as error:
        preparation.main([])
    assert error.value.code == 1
    assert "Asset preparation failed" in capsys.readouterr().err


def test_robot_uses_cache_and_accepts_an_explicit_offline_urdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain_robots import yam_i2rt

    cached = tmp_path / "YamI2rt/yam_i2rt.urdf"
    requested = []

    def resolve(path):
        requested.append(path)
        return str(cached)

    monkeypatch.setattr(yam_i2rt, "get_data_path", resolve)
    with pytest.raises(FileNotFoundError, match="data download --name YamI2rt"):
        yam_i2rt.YamI2rtCfg.from_dict({})
    custom = tmp_path / "offline.urdf"
    custom.write_text('<robot name="fixture"/>')
    cfg = yam_i2rt.YamI2rtCfg.from_dict({"fpath": str(custom)})
    assert cfg.fpath == str(custom)
    assert cfg._pk_urdf_path == str(custom)
    assert cfg.urdf_cfg.components["arm"]["urdf_path"] == str(custom)
    assert requested == ["YamI2rt/yam_i2rt.urdf"]
    cached.parent.mkdir()
    cached.write_text(custom.read_text())
    cfg = yam_i2rt.YamI2rtCfg.from_dict({})
    assert cfg.fpath is None
    assert cfg._pk_urdf_path == str(cached)
    assert cfg.urdf_cfg.components["arm"]["urdf_path"] == str(cached)


def test_archive_is_reproducible_and_preserves_license_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "robot"
    (source / "assets").mkdir(parents=True)
    files = {
        "LICENSE": b"MIT Copyright I2RT Robotics",
        "yam_i2rt.urdf": b'<robot name="fixture"><link name="base"/></robot>',
        "assets/tip.obj": b"v 0 0 0\n",
    }
    for name, data in files.items():
        (source / name).write_bytes(data)
    (source / "unrelated.log").write_text("do not publish")
    monkeypatch.setattr(
        preparation,
        "_EXPECTED_SHA256",
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    )
    first = preparation._archive(source, tmp_path / "first.zip")
    second = preparation._archive(source, tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    assert preparation._archive(source, first) == first
    with zipfile.ZipFile(first) as stream:
        assert stream.testzip() is None
        assert stream.namelist().index("assets/") < stream.namelist().index(
            "assets/tip.obj"
        )
        assert set(stream.namelist()) == {*files, "assets/", "NOTICE"}
        for name, data in files.items():
            assert stream.read(name) == data
        assert b"I2RT Robotics" in stream.read("NOTICE")
        assert b"MIT" in stream.read("NOTICE")
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in stream.infolist()
        )
    (source / "LICENSE").write_text("changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preparation._archive(source, tmp_path / "invalid.zip")
    assert not (tmp_path / "invalid.zip").exists()


def test_archive_does_not_replace_an_existing_different_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "robot"
    source.mkdir()
    (source / "yam_i2rt.urdf").write_text('<robot name="fixture"/>')
    monkeypatch.setattr(
        preparation,
        "_EXPECTED_SHA256",
        {"yam_i2rt.urdf": preparation._sha256(source / "yam_i2rt.urdf")},
    )
    archive = tmp_path / "published.zip"
    archive.write_bytes(b"already exists")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        preparation._archive(source, archive)
    assert archive.read_bytes() == b"already exists"
    with pytest.raises(ValueError, match="outside"):
        preparation._archive(source, source / "invalid.zip")
