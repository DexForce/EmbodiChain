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

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import sys
from threading import Thread
import zipfile

from pxr import Sdf, Usd, UsdGeom
import pytest

from embodichain.extensions.benchmarks.grasp_pose_generator._objects import (
    suite_objects,
)
from scripts.tools import prepare_dextrove_subset as tool

pytestmark = pytest.mark.no_sim


@pytest.fixture
def archive_api():
    asset_id = "00000000-0000-4000-8000-000000000001"
    asset_url = f"/v1/assets/{asset_id}"
    archive_url = f"{asset_url}/files/model.zip"
    geometry = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Mesh.Define(geometry, "/Mug").GetPrim()
    profiles = {}
    for name in ("PhysX", "Newton"):
        variants = prim.GetVariantSets().AddVariantSet(name)
        variants.AddVariant("Enabled")
        variants.AddVariant("Disabled")
        profiles[name] = {
            "prim": "/Mug",
            "variantSetName": name,
            "activateOption": "Enabled",
        }
    model = Sdf.Layer.CreateAnonymous(".usda")
    model.defaultPrim = "Mug"
    model.subLayerPaths = ["geometry/base.usda"]
    model.customLayerData = {"SimReady_Metadata": {"Variants": {"Physics": profiles}}}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model.usda", model.ExportToString())
        archive.writestr("geometry/base.usda", geometry.GetRootLayer().ExportToString())
    payload = buffer.getvalue()
    entry = {
        "relative_path": "payload/model.zip",
        "size_bytes": len(payload),
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }
    metadata = {
        "asset_id": asset_id,
        "resource_kind": "archive",
        "metadata": {"entrypoint": "model.usda"},
        "entries": [entry],
        "entry_count": 1,
    }
    reads = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            reads.append(self.path)
            if self.path == asset_url:
                body = json.dumps(metadata).encode()
            elif self.path == archive_url:
                body = payload
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", asset_id, entry, reads
        finally:
            server.shutdown()
            thread.join()


def test_shortlist_installs_one_composable_usd_across_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_api
) -> None:
    api, asset_id, _, reads = archive_api
    shortlist = tmp_path / "selected.jsonl"
    tags = [{"domain": "feature.category", "key": key} for key in ("mug", "cup")]
    shortlist.write_text(
        json.dumps({"asset_id": asset_id, "name": "Tea mug", "tags": tags})
    )
    output = tmp_path / "subset"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dextrove_subset",
            "--api-root",
            api,
            "--shortlist",
            str(shortlist),
            "--output-dir",
            str(output),
        ],
    )

    assert tool.main() == 0
    assets = output / "assets"
    canonical = assets / "cup" / asset_id / f"{asset_id}.usda"
    alias = assets / "mug" / asset_id / canonical.name
    assert alias.samefile(canonical)
    stage = Usd.Stage.Open(str(alias))
    assert stage.GetDefaultPrim().IsA(UsdGeom.Mesh)
    del stage
    assert not list(assets.rglob("*.zip"))
    assert sum(path.endswith("/files/model.zip") for path in reads) == 1
    first_reads = reads.copy()

    def check_backends() -> None:
        suite = {"dataset_root": "assets", "objects": [{"category": "mug"}]}
        for backend, enabled, disabled in (
            ("default", "PhysX", "Newton"),
            ("newton", "Newton", "PhysX"),
        ):
            (obj,) = suite_objects(suite, output, physics=backend)
            assert obj["rigid"]["source"]["variants"] == {
                "/Mug": {enabled: "Enabled", disabled: "Disabled"}
            }

    check_backends()
    # Older installs kept only the source path and authored no wrapper customLayerData.
    canonical.write_text(
        '#usda 1.0\n(defaultPrim = "Mug"\nsubLayers = [@source/model.usda@])\n'
    )
    (canonical.parent / ".dextrove.json").write_text(
        json.dumps({"asset_id": asset_id, "source_entrypoint": "source/model.usda"})
    )
    assert tool.main() == 0
    check_backends()
    assert reads == first_reads


def test_mismatched_digest_leaves_no_installed_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_api
) -> None:
    api, asset_id, entry, _ = archive_api
    entry["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(tool.time, "sleep", lambda seconds: None)
    row = {"asset_id": asset_id, "tags": [{"domain": "feature.category", "key": "mug"}]}
    with tool.requests.Session() as session, pytest.raises(ValueError, match="SHA-256"):
        tool._install(session, api, row, tmp_path)
    assert not (tmp_path / "mug" / asset_id).exists()


def test_inventory_screens_tagged_tabletop_assets_and_resumes_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cup = dict(tag_id="cup", key="cup", domain="feature.category", display_name="cup")
    fixture = dict(
        cup, tag_id="wall_fixture", key="wall_fixture", display_name="wall fixture"
    )
    assets = [
        dict(asset_id="1", name="Cup", domain_id="objects", tags=[cup]),
        dict(
            asset_id="2", name="Fixed wall switch", domain_id="objects", tags=[fixture]
        ),
        dict(asset_id="3", name="Untagged", domain_id="objects", tags=[]),
    ]
    metadata = {
        "assets": dict(total=3, offset=0, items=assets),
        "assets/facets": dict(
            total=3, items=[dict(cup, count=1), dict(fixture, count=1)]
        ),
        "tags": dict(items=[cup, fixture]),
    }
    for asset in assets:
        uid = asset["asset_id"]
        job = dict(
            job_id=uid,
            source_id=uid,
            preview={
                "bounds_aabb": {
                    "min": [0, 0, 0],
                    "max": [0.05, 0.06, 0.07],
                    "unit": "m",
                }
            },
        )
        metadata[f"assets/{uid}/ingestions"] = dict(total=1, offset=0, items=[job])
    monkeypatch.setattr(
        tool, "_get_json", lambda root, route, **params: metadata[route]
    )
    monkeypatch.setenv("CODEX_APP_RUNTIME_CLI_PATH", "mock-codex")
    # Both tagged assets fit the jaws; only the cup passes tabletop screening.
    answers = iter(["G000 B B", "G000 F X"])

    def model_response(cli, model, lines, prompt):
        assert all("Untagged" not in line for line in lines)
        return next(answers)

    monkeypatch.setattr(tool, "_ask", model_response)
    api = "https://catalog.example/v1"
    shortlist = tool._prepare_shortlist(tmp_path, api, "test-model")
    (selected,) = map(json.loads, shortlist.read_text().splitlines())
    assert (selected["asset_id"], selected["reference_width_cm"]) == ("1", 5.0)
    assert selected["tabletop_category"] == "food_and_kitchenware"
    inventory = (tmp_path / "asset_tags.jsonl").read_text().splitlines()
    assert [r["asset_id"] for r in map(json.loads, inventory)] == ["1", "2", "3"]
    metadata.clear()
    assert tool._prepare_shortlist(tmp_path, api, "test-model") == shortlist
