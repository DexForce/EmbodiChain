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

"""URDF subtree cache identity includes immutable source provenance."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from embodichain.data.urdf import materialize_urdf_subtree


def test_subtree_cache_separates_provenance_and_detects_tampering(
    tmp_path, monkeypatch
):
    import embodichain.data.constants as constants

    monkeypatch.setattr(
        constants, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(tmp_path / "cache")
    )
    source = tmp_path / "first.urdf"
    source.write_text(
        '<robot name="first"><link name="arm"/><link name="hand"/>'
        '<joint name="mount" type="fixed"><parent link="arm"/><child link="hand"/></joint></robot>'
    )
    first = materialize_urdf_subtree(source, "hand")
    assert materialize_urdf_subtree(source, "hand") == first
    assert [node.get("name") for node in ET.parse(first).findall("link")] == ["hand"]
    source.write_text(source.read_text().replace('name="first"', 'name="second"'))
    changed = materialize_urdf_subtree(source, "hand")
    relocated = tmp_path / "relocated.urdf"
    relocated.write_bytes(source.read_bytes())
    copied = materialize_urdf_subtree(relocated, "hand")
    assert len({first, changed, copied}) == 3
    assert first.read_bytes() == changed.read_bytes() == copied.read_bytes()
    for derived, original in ((changed, source), (copied, relocated)):
        report = json.loads(derived.with_name("derivation.json").read_text())
        assert report["source"] == str(original.resolve())
        assert (
            report["source_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
        )
        assert (
            report["derived_sha256"] == hashlib.sha256(derived.read_bytes()).hexdigest()
        )
    copied.write_text("modified")
    with pytest.raises(ValueError, match="Cached URDF derivative was modified"):
        materialize_urdf_subtree(relocated, "hand")
