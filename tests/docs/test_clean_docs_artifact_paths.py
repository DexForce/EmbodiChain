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

"""Tests for docs artifact path cleanup."""

from __future__ import annotations

from pathlib import Path

from .conftest import clean_docs_artifact_paths


def test_clean_removes_wget_query_string_files(tmp_path: Path) -> None:
    build_dir = tmp_path / "html"
    static_dir = build_dir / "main" / "_static"
    static_dir.mkdir(parents=True)
    valid_asset = static_dir / "clipboard.min.js"
    invalid_asset = static_dir / "clipboard.min.js?v=a7894cd8"
    valid_asset.write_text("valid", encoding="utf-8")
    invalid_asset.write_text("duplicate", encoding="utf-8")

    removed = clean_docs_artifact_paths(build_dir)

    assert removed == [Path("main/_static/clipboard.min.js?v=a7894cd8")]
    assert valid_asset.is_file()
    assert not invalid_asset.exists()
