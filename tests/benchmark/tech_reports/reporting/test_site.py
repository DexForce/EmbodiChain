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

"""Preserve readable historical reports after explicit media retirement."""

from __future__ import annotations
import json
from pathlib import Path
from bs4 import BeautifulSoup
import pytest
from scripts.benchmark.tech_reports.reporting.site import build


def test_retired_media_are_labelled_without_hiding_unknown_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "media-cleanup.md").write_text(
        "# Media cleanup\nMeasurements remain available."
    )
    (reports / "media-retention.json").write_text(
        json.dumps(
            {
                "retired_rules": [
                    {
                        "directory": "reports/old",
                        "suffixes": [".png", ".mp4"],
                        "reason": "superseded recording",
                    }
                ],
                "redirects": [
                    {
                        "path": "reports/duplicate.mp4",
                        "replacement": "reports/current.mp4",
                        "note": "identical recording",
                    }
                ],
                "compressed_files": [
                    {"path": "reports/scene.usda", "archive": "reports/scene.usda.gz"}
                ],
            }
        )
    )
    (reports / "current.mp4").write_bytes(b"video")
    (reports / "scene.usda.gz").write_bytes(b"archive")
    (reports / "old").mkdir()
    (reports / "old/capture.json").write_text("{}")
    source = reports / "history.md"
    text = '# Historic report\n![old frame](old/frame.png)\n\n[old clip](old/native.mp4)\n\n<video><source src="duplicate.mp4"></video>\n\n[scene](scene.usda)\n\n[metadata](old/capture.json)'
    source.write_text(text)
    builder = build.SiteBuilder(tmp_path / "public")
    owner = builder.catalog[0]
    builder.register_document(source, owner)
    builder.render_document(source)
    soup = BeautifulSoup(builder.documents[source]["body"], "html.parser")
    assert not soup.select("img, source[src*='old/']")
    retired = soup.select("a[href='/notes/reports-media-cleanup.html']")
    assert len([n for n in retired if "已清理" in n.get_text()]) == 2
    assert soup.select_one("video source")["src"] == "/files/reports/current.mp4"
    archive = soup.select_one("a[href='/files/reports/scene.usda.gz']")
    assert "gzip" in archive.get_text()
    assert soup.select_one("a[href='/files/reports/old/capture.json']")
    assert source.read_text() == text
    with pytest.raises(FileNotFoundError):
        builder.local_url("unrecorded/missing.mp4", source, owner)
