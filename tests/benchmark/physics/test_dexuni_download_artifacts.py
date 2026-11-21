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
"""Publication restore contracts: selection, integrity and existing evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.benchmark.tech_reports.physics.experiments.dexuni import (
    download_artifacts as download,
)


@pytest.fixture
def publication(tmp_path: Path) -> tuple[Path, Path, dict]:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("case/inputs/state.json", '{"position": [1, 2, 3]}')
    artifact = {
        "path": "reports/report/inputs/case/input.zip",
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "kind": "inputs",
        "task_folder": "case",
    }
    index = {
        "schema_version": 1,
        "repo_id": "DexForceAI/embodichain_reports",
        "report_id": "report",
        "revision": "a" * 40,
        "tasks": {"S01-C": "case", "S01-T": "case"},
        "artifacts": [artifact],
    }
    index_path = tmp_path / "publication.json"
    index_path.write_text(json.dumps(index))
    return index_path, archive, index


def test_selected_bundle_restores_same_bytes_offline_without_overwrite(
    publication, tmp_path, monkeypatch
):
    index_path, archive, _ = publication
    requests = []

    def fetch(**kwargs):
        requests.append(kwargs)
        return str(archive)

    monkeypatch.setattr(download, "hf_hub_download", fetch)
    root = tmp_path / "out"
    for _ in range(2):
        download.restore_artifacts(
            tasks=["S01-C", "S01-T"],
            output_root=root,
            offline=True,
            index_path=index_path,
        )
    output = root / "case/inputs/state.json"
    assert output.read_text() == '{"position": [1, 2, 3]}'
    assert (
        len(requests) == 2
    )  # Shared S01 input bundle is not downloaded twice per call.
    assert all(
        r["revision"] == "a" * 40 and r["local_files_only"] and r["token"] is False
        for r in requests
    )
    output.write_text("local measurement")
    with pytest.raises(FileExistsError, match="Existing file differs"):
        download.restore_artifacts(
            tasks=["S01-C"], output_root=root, offline=True, index_path=index_path
        )
    assert output.read_text() == "local measurement"


@pytest.mark.parametrize("offline,expected_calls", [(True, 1), (False, 2)])
def test_corruption_is_not_materialized_and_online_repairs_once(
    publication, tmp_path, monkeypatch, offline, expected_calls
):
    index_path, archive, _ = publication
    archive.write_bytes(b"corrupt")
    requests = []

    def fetch(**kwargs):
        requests.append(kwargs)
        return str(archive)

    monkeypatch.setattr(download, "hf_hub_download", fetch)
    with pytest.raises(ValueError, match="size/SHA256 mismatch"):
        download.restore_artifacts(
            output_root=tmp_path / "out", offline=offline, index_path=index_path
        )
    assert len(requests) == expected_calls
    assert [r["force_download"] for r in requests] == (
        [False] if offline else [False, True]
    )
    assert not (tmp_path / "out/case").exists()


def test_successful_online_repair(publication, tmp_path, monkeypatch):
    index_path, archive, _ = publication
    corrupt = tmp_path / "corrupt"
    corrupt.write_bytes(b"bad")
    monkeypatch.setattr(
        download,
        "hf_hub_download",
        lambda **kw: str(archive if kw["force_download"] else corrupt),
    )
    download.restore_artifacts(output_root=tmp_path / "out", index_path=index_path)
    assert (
        tmp_path / "out/case/inputs/state.json"
    ).read_text() == '{"position": [1, 2, 3]}'


@pytest.mark.parametrize(
    "member", ["../escape", "/absolute", "case/../../escape", "other/state.json"]
)
def test_archive_escape_rejected_before_materializing(publication, tmp_path, member):
    _, archive, index = publication
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("case/valid.json", "valid")
        target.writestr(member, "bad")
    root = tmp_path / "out"
    root.mkdir()
    with pytest.raises(ValueError):
        download._materialize(archive, index["artifacts"][0], root)
    assert not (root / "case").exists()
    assert not (tmp_path / "escape").exists()


def test_symlink_output_does_not_modify_external_file(publication, tmp_path):
    _, archive, index = publication
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "out"
    root.mkdir()
    (root / "case").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        download._materialize(archive, index["artifacts"][0], root)
    assert list(external.iterdir()) == []


def test_unknown_task_and_unpinned_release_make_no_request(
    publication, tmp_path, monkeypatch
):
    index_path, _, index = publication

    def unexpected(**kwargs):
        pytest.fail("Invalid selection must not make a network request")

    monkeypatch.setattr(download, "hf_hub_download", unexpected)
    with pytest.raises(ValueError, match="Unknown tasks"):
        download.restore_artifacts(
            tasks=["C03"], output_root=tmp_path / "out", index_path=index_path
        )
    index["revision"] = "main"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="full verified dataset commit"):
        download.restore_artifacts(output_root=tmp_path / "out", index_path=index_path)


def test_unpublished_media_is_rejected_before_download(
    publication, tmp_path, monkeypatch
):
    index_path, _, _ = publication
    monkeypatch.setattr(
        download, "hf_hub_download", lambda **kw: pytest.fail("Unexpected download")
    )
    with pytest.raises(ValueError, match="Only input assets"):
        download.restore_artifacts(
            kinds=["videos"], output_root=tmp_path / "out", index_path=index_path
        )


def test_runtime_inputs_do_not_require_video_but_detect_changed_geometry(
    tmp_path, monkeypatch
):
    from scripts.benchmark.tech_reports.physics.experiments.dexuni import (
        reference_inputs,
    )

    root = tmp_path / "case"
    (root / "inputs").mkdir(parents=True)
    mesh = root / "inputs/mesh.bin"
    mesh.write_bytes(b"frozen geometry")
    index = tmp_path / "checks.json"
    index.write_text(
        json.dumps(
            {
                "case": {
                    "files": {
                        "inputs/mesh.bin": hashlib.sha256(mesh.read_bytes()).hexdigest()
                    }
                }
            }
        )
    )
    monkeypatch.setattr(reference_inputs, "_INDEX", index)
    reference_inputs.verify_reference(root, "historical-video-not-downloaded")
    assert not (root / "preview").exists()
    mesh.write_bytes(b"modified geometry")
    with pytest.raises(ValueError, match="runtime input changed"):
        reference_inputs.verify_reference(root, "historical-video-not-downloaded")
    mesh.unlink()
    with pytest.raises(FileNotFoundError):
        reference_inputs.verify_reference_inputs(root)


def test_relocated_robot_uses_frozen_asset_without_rewriting_record(tmp_path):
    from scripts.benchmark.tech_reports.physics.experiments.dexuni.paths import (
        frozen_robot_urdf,
    )

    inputs = tmp_path / "inputs/s01c"
    original = "/old/machine/franka/robot.urdf"
    assert frozen_robot_urdf(inputs, original) == original
    local = inputs.parent / "franka_emika_panda/robot.urdf"
    local.parent.mkdir(parents=True)
    local.write_text("<robot/>")
    assert frozen_robot_urdf(inputs, original) == str(local)


def test_cold_task_restores_once_and_warm_task_stays_offline(
    publication, tmp_path, monkeypatch
):
    index_path, archive, _ = publication
    assets = tmp_path / "source/assets/physics/dexuni"
    assets.mkdir(parents=True)
    (assets / "runtime_checks.json").write_text(
        json.dumps({"case": {"files": {"inputs/state.json": "checked by loader"}}})
    )
    monkeypatch.setattr(download, "_INDEX", index_path)
    monkeypatch.setattr(download, "ASSETS", assets)
    fetched = []

    def fetch(**kwargs):
        fetched.append(kwargs)
        return str(archive)

    monkeypatch.setattr(download, "hf_hub_download", fetch)
    root = tmp_path / "restored/case"
    download.ensure_task_inputs("S01-C", root)
    download.ensure_task_inputs("S01-C", root)
    assert len(fetched) == 1
    assert (root / "inputs/state.json").read_text() == '{"position": [1, 2, 3]}'
