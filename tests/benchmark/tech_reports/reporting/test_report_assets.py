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

"""Exercise report downloads without a simulator or network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmark.tech_reports.assets import download

_COMMIT = "1234567890abcdef" * 2 + "12345678"
_BUNDLE = "report-p-01"
_INPUT = "reports/report/inputs/p-01/initial-states.json"
_VALID_BYTES = b'{"initial_z_m": 1.0}\n'


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the same snapshot layout as the Hub package."""
    root = tmp_path / "snapshots" / _COMMIT
    target = root / _INPUT
    target.parent.mkdir(parents=True)
    target.write_bytes(b"incomplete")
    index = {
        "schema_version": 1,
        "bundles": {
            _BUNDLE: {
                "report_id": "report",
                "benchmark_id": "p-01",
                "files": [
                    {
                        "path": _INPUT,
                        "size_bytes": len(_VALID_BYTES),
                        "sha256": hashlib.sha256(_VALID_BYTES).hexdigest(),
                    }
                ],
            }
        },
    }
    (root / "index.json").write_text(json.dumps(index))
    calls = []

    def fetch(
        *, filename: str, revision: str, force_download: bool = False, **options
    ) -> str:
        calls.append((filename, revision, force_download, options["local_files_only"]))
        assert not options["token"]
        if force_download:
            target.write_bytes(_VALID_BYTES)
        return str(root / filename)

    monkeypatch.setattr(download, "hf_hub_download", fetch)
    return root, target, index, calls


def test_resolves_branch_once_and_repairs_online_checksum(fake_hub) -> None:
    root, target, _, calls = fake_hub
    directory, provenance = download.download_report_assets(_BUNDLE, revision="main")
    assert directory == target.parent
    assert (directory / target.name).read_bytes() == _VALID_BYTES
    assert provenance["revision"] == _COMMIT
    assert (
        provenance["index_sha256"]
        == hashlib.sha256((root / "index.json").read_bytes()).hexdigest()
    )
    assert calls == [
        ("index.json", "main", False, False),
        (_INPUT, _COMMIT, False, False),
        (_INPUT, _COMMIT, True, False),
    ]


def test_offline_rejects_corruption_without_retry(fake_hub) -> None:
    _, _, _, calls = fake_hub
    with pytest.raises(ValueError, match="checksum mismatch"):
        download.download_report_assets(
            _BUNDLE, revision=_COMMIT, local_files_only=True
        )
    assert calls == [
        ("index.json", _COMMIT, False, True),
        (_INPUT, _COMMIT, False, True),
    ]


def test_invalid_bundle_location_fails_before_download(fake_hub) -> None:
    root, _, index, calls = fake_hub
    index["bundles"][_BUNDLE]["files"][0][
        "path"
    ] = "reports/other/inputs/p-01/input.json"
    (root / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Invalid file entry"):
        download.download_report_assets(_BUNDLE, revision=_COMMIT)
    assert calls == [("index.json", _COMMIT, False, False)]
