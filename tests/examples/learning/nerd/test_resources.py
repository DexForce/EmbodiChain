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

"""Verify the explicitly requested showcase resource-download capability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import huggingface_hub
import pytest

from examples.learning.nerd import _resources as resources


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def asset_server(tmp_path, monkeypatch):
    payloads = {
        "ATTRIBUTION.md": b"reference attribution",
        "LICENSE.md": b"Apache-2.0 license fixture",
        "ant.xml": b"<mujoco/>\n",
        "cartpole_single.urdf": b"<robot name='cartpole'/>\n",
    }
    server = tmp_path / "server"
    server.mkdir()
    archive = server / "reference_assets.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nerd_reference_assets/", b"")
        for name, value in payloads.items():
            bundle.writestr(f"nerd_reference_assets/{name}", value)
    archive_hash = digest(archive.read_bytes())
    member_hashes = {name: digest(value) for name, value in payloads.items()}
    manifest = {
        "schema_version": 1,
        "license": "Apache-2.0",
        "upstream": {"revision": resources._UPSTREAM_REVISION},
        "archive": {
            "root": "nerd_reference_assets",
            "sha256": archive_hash,
            "size": archive.stat().st_size,
        },
        "files": {
            name: {
                "path": f"nerd_reference_assets/{name}",
                "sha256": digest(value),
                "size": len(value),
            }
            for name, value in payloads.items()
        },
    }
    metadata = server / "manifest.json"
    metadata.write_text(json.dumps(manifest))
    monkeypatch.setattr(resources, "_ASSET_FILES", member_hashes)
    monkeypatch.setattr(resources, "_ARCHIVE_SHA256", archive_hash)
    monkeypatch.setattr(resources, "_MANIFEST_SHA256", digest(metadata.read_bytes()))
    calls = []

    def download(**options):
        assert options["repo_id"] == "DexForceAI/embodichain_data"
        assert options["repo_type"] == "dataset"
        assert options["revision"] == "97e202e41fe7fb504656a6b163291305e400780d"
        calls.append(options)
        return str(server / Path(options["filename"]).name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return payloads, calls, archive


def test_pinned_asset_cache_repairs_corruption_and_reuses_verified_files(
    tmp_path, asset_server
):
    payloads, calls, _ = asset_server
    cache = tmp_path / "cache"
    target = resources.prepare_reference_assets(cache)
    assert {path.name for path in target.iterdir()} == set(payloads)
    assert all(
        (target / name).read_bytes() == value for name, value in payloads.items()
    )
    assert len(calls) == 2
    assert resources.prepare_reference_assets(cache, local_files_only=True) == target
    assert len(calls) == 2  # A verified cache needs neither Hub nor network access.
    (target / "ant.xml").write_bytes(b"damaged local cache")
    assert resources.prepare_reference_assets(cache, local_files_only=True) == target
    assert len(calls) == 4 and calls[-1]["local_files_only"] is True
    assert (target / "ant.xml").read_bytes() == payloads["ant.xml"]
    assert not list(target.parent.glob(".staging-*"))


def test_invalid_download_archive_is_rejected_before_extraction(tmp_path, asset_server):
    _, _, archive = asset_server
    archive.write_bytes(archive.read_bytes() + b"changed")
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resources.prepare_reference_assets(cache)
    assert not (cache / "reference_assets").exists()


def test_example_resource_environment_and_explicit_cache_precedence(
    tmp_path, asset_server, monkeypatch
):
    monkeypatch.setenv("EMBODICHAIN_NERD_RESOURCE_ROOT", str(tmp_path / "configured"))
    environmental = resources.prepare_reference_assets()
    assert environmental.is_relative_to(tmp_path / "configured/assets")
    explicit = resources.prepare_reference_assets(tmp_path / "explicit")
    assert explicit.is_relative_to(tmp_path / "explicit")


def test_assets_only_does_not_start_model_or_dataset_downloads(
    tmp_path, asset_server, monkeypatch
):
    def unexpected_snapshot(**kwargs):
        raise AssertionError("Assets-only preparation must not download large bundles")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_snapshot)
    prepared = resources.prepare_resources(
        tmp_path / "resources", "cartpole", assets_only=True
    )
    assert set(prepared) == {"assets"}
    assert prepared["assets"].is_dir()
    assert not (tmp_path / "resources/datasets").exists()
    assert not (tmp_path / "resources/models").exists()


def test_published_payload_revisions_paths_and_checksums(
    tmp_path, asset_server, monkeypatch
):
    calls = []
    manifests = {"dataset": {}, "model": {}}
    monkeypatch.setattr(resources, "_PAYLOAD_MANIFEST_SHA256", manifests)
    metadata = {
        "codec": {"robot": "cartpole"},
        "dt": 1 / 60,
        "asset_sha256": resources._ASSET_FILES["cartpole_single.urdf"],
    }

    def snapshot(**options):
        calls.append(options)
        is_dataset = options["repo_type"] == "dataset"
        assert options["revision"] == (
            "b9ec58f36da99cc99a93f97adfe74ebc9fd66586"
            if is_dataset
            else "78e581526570069410bf269b38c0c52c800b24a7"
        )
        directory = Path(options["local_dir"]) / "neural_dynamics/nerd/cartpole/v1"
        directory.mkdir(parents=True, exist_ok=True)
        if is_dataset:
            value = b"immutable dataset shard"
            (directory / "shard-000001.h5").write_bytes(value)
            manifest = {
                "schema_version": 1,
                "metadata": metadata,
                "shards": [{"file": "shard-000001.h5", "sha256": digest(value)}],
            }
        else:
            value = b"immutable portable weights"
            (directory / "weights.pt").write_bytes(value)
            manifest = {
                "format_version": 1,
                **metadata,
                "weights_sha256": digest(value),
            }
        (directory / "manifest.json").write_text(json.dumps(manifest))
        manifests["dataset" if is_dataset else "model"]["cartpole"] = digest(
            (directory / "manifest.json").read_bytes()
        )

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    prepared = resources.prepare_resources(tmp_path / "resources", "cartpole")
    assert (
        prepared["dataset"]
        == tmp_path / "resources/datasets/neural_dynamics/nerd/cartpole/v1"
    )
    assert (
        prepared["model"]
        == tmp_path / "resources/models/neural_dynamics/nerd/cartpole/v1"
    )
    assert len(calls) == 2
    resources.prepare_resources(
        tmp_path / "resources", "cartpole", local_files_only=True
    )
    assert len(calls) == 2
    (prepared["model"] / "weights.pt").write_bytes(b"bad weights")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: None)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resources.prepare_resources(tmp_path / "resources", "cartpole")
    # A self-consistent replacement still cannot impersonate the pinned release.
    manifest_path = prepared["model"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["weights_sha256"] = digest(b"bad weights")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest.json"):
        resources.prepare_resources(tmp_path / "resources", "cartpole")
