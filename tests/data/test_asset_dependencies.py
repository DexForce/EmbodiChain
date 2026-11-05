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
"""Local input inventories preserve URI resolution and source bytes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from embodichain.data.asset_dependencies import (
    asset_dependency_files,
    collect_declared_asset_files,
)


def test_xml_dependency_closure_retains_local_uri_and_package_resolution(tmp_path):
    package = tmp_path / "fixture_package"
    includes = package / "includes"
    includes.mkdir(parents=True)
    model = package / "model.urdf"
    mesh = package / "mesh input.stl"
    texture = package / "texture.png"
    child = includes / "child.xml"
    mesh.write_bytes(b"original opaque mesh input")
    texture.write_bytes(b"original texture input")
    child.write_text('<mujoco><include file="../model.urdf"/></mujoco>')
    model.write_text(
        '<robot xmlns:x="urn:fixture">'
        '<x:mesh filename="package://fixture_package/mesh%20input.stl"/>'
        '<mesh filename="mesh%20input.stl"/>'
        f'<texture file="{texture.as_uri()}"/>'
        '<include url="includes/child.xml"/>'
        "</robot>"
    )
    expected = {model, mesh, texture, child}
    original = {path: path.read_bytes() for path in expected}
    assert asset_dependency_files(model) == tuple(sorted(expected))
    assert {path: path.read_bytes() for path in expected} == original


def test_declaration_inventory_uses_owner_local_paths_before_registry(
    tmp_path, monkeypatch
):
    import embodichain.data.asset_dependencies as owner

    local = tmp_path / "local"
    local.mkdir()
    source = local / "model.urdf"
    source.write_text("<robot/>")
    registered = tmp_path / "registered.urdf"
    registered.write_text("<robot/>")
    requests = []

    def lookup(reference):
        requests.append(reference)
        assert reference == "PinnedAsset/model.urdf"
        return str(registered)

    monkeypatch.setattr(owner, "get_data_path", lookup)
    declarations = {
        "scene": [{"fpath": "local/model.urdf"}, {"urdf_path": str(source)}],
        "robot": (
            {"xml_path": "PinnedAsset/model.urdf"},
            {"mjcf_path": str(registered)},
        ),
        "metadata": {"description": "not a file reference", "fpath": None},
    }
    before = deepcopy(declarations)
    assert collect_declared_asset_files(declarations, base_dir=tmp_path) == {
        source,
        registered,
    }
    assert requests == ["PinnedAsset/model.urdf"]
    assert declarations == before


def test_usd_inventory_retains_used_layers_and_adjacent_source_annotation(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    child = tmp_path / "child.usda"
    root = tmp_path / "model.usda"
    stage = Usd.Stage.CreateNew(str(child))
    UsdGeom.Xform.Define(stage, "/Child")
    stage.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(root))
    stage.GetRootLayer().subLayerPaths = [child.name]
    UsdGeom.Xform.Define(stage, "/Model")
    stage.GetRootLayer().Save()
    annotation = root.with_suffix(".twist.json")
    annotation.write_text('{"authored": "retained without semantic evaluation"}')
    before = {p: p.read_bytes() for p in (root, child, annotation)}
    assert asset_dependency_files(root) == tuple(sorted((root, annotation)))
    assert collect_declared_asset_files({"fpath": root.name}, base_dir=tmp_path) == {
        root,
        child,
        annotation,
    }
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize(
    "reference,error",
    [
        ("missing.stl", FileNotFoundError),
        ("https://example.invalid/model.stl", ValueError),
        ("ftp://example.invalid/model.stl", ValueError),
    ],
)
def test_declared_dependency_failures_remain_explicit(tmp_path, reference, error):
    model = tmp_path / "model.urdf"
    model.write_text(f'<robot><mesh filename="{reference}"/></robot>')
    with pytest.raises(error):
        asset_dependency_files(model)


def test_opaque_xml_looking_source_remains_a_direct_dependency(tmp_path):
    source = tmp_path / "opaque.urdf"
    source.write_bytes(b"opaque original source")
    assert asset_dependency_files(source) == (source,)
