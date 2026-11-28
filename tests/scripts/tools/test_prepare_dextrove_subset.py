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
from uuid import UUID
import zipfile

from pxr import Sdf, Usd, UsdGeom
import pytest

from embodichain.extensions.benchmarks.grasp_pose_generator._objects import (
    suite_objects,
)
from scripts.tools import _dextrove_categories as categories
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
        archive.writestr("simready_usd/model.usda", model.ExportToString())
        archive.writestr(
            "simready_usd/geometry/base.usda", geometry.GetRootLayer().ExportToString()
        )
    payload = buffer.getvalue()
    entry = {
        "relative_path": "payload/model.zip",
        "size_bytes": len(payload),
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }
    metadata = {
        "asset_id": asset_id,
        "resource_kind": "archive",
        "metadata": {"entrypoint": "simready_usd/model.usda"},
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


def test_catalog_download_and_legacy_replay_share_one_categorized_usd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_api
) -> None:
    api, asset_id, _, reads = archive_api
    tags = [
        dict(tag_id=key, domain="feature.category", key=key, display_name=key)
        for key in ("mug", "cup")
    ]
    catalog = tmp_path / "catalog.json"
    # A catalog from the earlier taxonomy keeps its original "drinkware" label.
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "classification_model": "gpt-6.2-astra",
                "categories": {"drinkware": "Cups and glasses"},
                "tag_dictionary": {
                    "domains": [{"domain": "feature.category"}],
                    "items": tags,
                },
                "assets": [
                    dict(
                        asset_id=asset_id,
                        name="Tea mug",
                        tag_ids=["mug", "cup"],
                        category="drinkware",
                    )
                ],
            }
        )
    )
    monkeypatch.setattr(tool, "DEFAULT_CATALOG", catalog)
    monkeypatch.setenv(
        "CODEX_APP_RUNTIME_CLI_PATH", str(tmp_path / "nonexistent-codex")
    )
    monkeypatch.setattr(
        tool, "_ask", lambda *args: pytest.fail("Unexpected model call")
    )

    def download(output: Path, *flags: str) -> None:
        argv = [
            "prepare_dextrove_subset",
            "--api-root",
            api,
            "--output-dir",
            str(output),
        ]
        monkeypatch.setattr(sys, "argv", [*argv, *flags])
        assert tool.main() == 0

    def check_category(directory: Path, category: str) -> Path:
        category_dir = directory / "categories" / category / asset_id
        assert list((directory / "categories").glob(f"*/{asset_id}")) == [category_dir]
        wrapper = category_dir / f"{asset_id}.usda"
        stage = Usd.Stage.Open(str(wrapper))
        assert stage.GetDefaultPrim().IsA(UsdGeom.Mesh)
        suite = {"dataset_root": "categories", "objects": [{"category": category}]}
        for backend, enabled, disabled in (
            ("default", "PhysX", "Newton"),
            ("newton", "Newton", "PhysX"),
        ):
            # Its nested source/simready_usd package belongs to this same UUID.
            (obj,) = suite_objects(suite, directory, physics=backend)
            assert obj["id"] == asset_id
            assert obj["rigid"]["source"]["variants"] == {
                "/Mug": {enabled: "Enabled", disabled: "Disabled"}
            }
        return wrapper

    output = tmp_path / "download"
    download(output)
    wrapper = check_category(output, "drinkware")
    assert not list(output.rglob("*.zip"))
    assert reads == [f"/v1/assets/{asset_id}", f"/v1/assets/{asset_id}/files/model.zip"]
    first_reads = reads.copy()

    # Older installs stored packages by first tag and created aliases for the rest.
    legacy_output = tmp_path / "legacy"
    canonical = legacy_output / "assets" / "cup" / asset_id
    canonical.parent.mkdir(parents=True)
    wrapper.resolve().parent.rename(canonical)
    alias = legacy_output / "assets" / "mug" / asset_id
    alias.parent.mkdir()
    alias.symlink_to(Path("..") / "cup" / asset_id, target_is_directory=True)
    legacy_wrapper = canonical / wrapper.name
    old_source = "source/simready_usd/model.usda"
    legacy_wrapper.write_text(
        f'#usda 1.0\n(defaultPrim = "Mug"\nsubLayers = [@{old_source}@])\n'
    )
    (canonical / ".dextrove.json").write_text(
        json.dumps({"asset_id": asset_id, "source_entrypoint": old_source})
    )

    # A tag-only shortlist gets its category from the catalog without model access.
    shortlist = tmp_path / "selected.jsonl"
    row = {"asset_id": asset_id, "name": "Tea mug", "tags": tags}
    shortlist.write_text(json.dumps(row))
    download(legacy_output, "--shortlist", str(shortlist))
    assert check_category(legacy_output, "drinkware").samefile(legacy_wrapper)
    assert (alias / wrapper.name).samefile(legacy_wrapper)

    # A prior catalog also supplied a UUID which is absent from this shortlist.
    retired = str(UUID(int=9))
    retired_cache = legacy_output / "objects" / retired
    retired_cache.mkdir(parents=True)
    retired_link = legacy_output / "categories" / "retired" / retired
    retired_link.parent.mkdir()
    retired_link.symlink_to(Path("../../objects") / retired, target_is_directory=True)

    # Replaying an explicit correction moves the label and reuses the same package.
    shortlist.write_text(json.dumps(dict(row, category="tableware")))
    download(legacy_output, "--shortlist", str(shortlist))
    assert check_category(legacy_output, "tableware").samefile(legacy_wrapper)
    assert retired_link.is_dir()

    download(legacy_output, "--download-limit", "1")
    assert retired_link.is_dir()
    download(legacy_output)
    assert not retired_link.is_symlink()
    assert retired_cache.is_dir()
    assert reads == first_reads


def test_mismatched_digest_leaves_no_installed_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_api
) -> None:
    api, asset_id, entry, _ = archive_api
    entry["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(tool.time, "sleep", lambda seconds: None)
    with tool.requests.Session() as session, pytest.raises(ValueError, match="SHA-256"):
        tool._install_source(session, api, asset_id, tmp_path / "objects" / asset_id)
    assert not (tmp_path / "objects" / asset_id).exists()


def test_refresh_screens_and_categorizes_then_replays_saved_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cup = dict(tag_id="cup", key="cup", domain="feature.category", display_name="cup")
    fixture = dict(
        cup, tag_id="wall_fixture", key="wall_fixture", display_name="wall fixture"
    )
    unused = dict(cup, tag_id="retired", key="retired", enabled=False)
    numbered_id, switch_id, untagged_id = [str(UUID(int=i)) for i in (1, 2, 3)]
    assets = [
        dict(asset_id=numbered_id, name="D00162", domain_id="objects", tags=[cup]),
        dict(
            asset_id=switch_id,
            name="Fixed wall switch",
            domain_id="objects",
            tags=[fixture],
        ),
        dict(asset_id=untagged_id, name="Untagged", domain_id="objects", tags=[]),
    ]
    metadata = {
        "assets": dict(total=3, offset=0, items=assets),
        "assets/facets": dict(
            total=3, items=[dict(cup, count=1), dict(fixture, count=1)]
        ),
        "tags": dict(
            domains=[{"domain": "feature.category"}], items=[cup, fixture, unused]
        ),
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
    # A model must be able to select the last of fifty classes by its object name.
    taxonomy = {f"category_{i}": f"Test object {i}" for i in range(48)}
    taxonomy["cup"] = "A drinking cup"
    taxonomy["spoon"] = "A spoon"
    monkeypatch.setattr(categories, "_CATEGORIES", taxonomy)

    # Both tagged assets fit the jaws, but only D00162 passes placement screening.
    def model_response(cli, model, lines, prompt):
        assert cli == "mock-codex"
        assert model == "gpt-6.2-astra"
        assert all("Untagged" not in line for line in lines)
        if prompt == tool.PROMPT:
            return "G000 B B"
        if prompt == tool.PLACEMENT_PROMPT:
            return "G000 F X"
        (item,) = [json.loads(line) for line in lines]
        assert item[1:3] == ["D00162", ["cup"]]
        if item[4:] == [None, None]:
            return f"{item[0]} cup"
        assert item[4:] == ["Measuring Spoons", "spoon"]
        return f"{item[0]} spoon"

    monkeypatch.setattr(tool, "_ask", model_response)
    catalog = tmp_path / "pinned.json"
    args = [
        "prepare_dextrove_subset",
        "--refresh",
        "--catalog",
        str(catalog),
        "--output-dir",
        str(tmp_path),
        "--api-root",
        "https://catalog.example/v1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert tool.main() == 0
    snapshot = json.loads(catalog.read_text())
    assert snapshot["tag_dictionary"] == metadata["tags"]
    assert {r["asset_id"]: r["tag_ids"] for r in snapshot["assets"]} == {
        numbered_id: ["cup"],
        switch_id: ["wall_fixture"],
        untagged_id: [],
    }
    (selected,) = tool.load_catalog(catalog)
    assert (
        selected["asset_id"],
        selected["category"],
        selected["reference_width_cm"],
    ) == (
        numbered_id,
        "cup",
        5.0,
    )
    assert selected["tabletop_category"] == "food_and_kitchenware"
    assert "usd_name" not in selected and "usd_category" not in selected

    # A later offline refresh can use USD evidence that was unavailable at collection.
    cached = tmp_path / "objects" / numbered_id / f"{numbered_id}.usda"
    cached.parent.mkdir(parents=True)
    layer = Sdf.Layer.CreateNew(str(cached))
    layer.customLayerData = {
        "SimReady_Metadata": {"asset_name": "Measuring Spoons", "category": "spoon"}
    }
    layer.Save()
    metadata.clear()
    args[args.index("--output-dir") + 1] = str(tmp_path / "reclassified")
    args.extend(["--snapshot-dir", str(tmp_path)])
    assert tool.main() == 0
    (enriched,) = tool.load_catalog(catalog)
    assert enriched == {
        **selected,
        "name": "D00162",
        "tags": [cup],
        "category": "spoon",
        "usd_name": "Measuring Spoons",
        "usd_category": "spoon",
    }

    # Invalid decisions must leave the published catalog usable by download clients.
    previous = catalog.read_bytes()
    monkeypatch.setattr(tool, "_ask", lambda *args: "0 cup\n0 category_1")
    with pytest.raises(ValueError, match="ambiguous|invalid"):
        tool.main()
    assert catalog.read_bytes() == previous

    # The ceiling also applies to declared classes that have no assignments.
    snapshot["categories"]["extra"] = "One class too many"
    catalog.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="categor"):
        tool.load_catalog(catalog)


def test_bundled_catalog_has_exclusive_categories_and_resolvable_tags() -> None:
    catalog = json.loads(tool.DEFAULT_CATALOG.read_text())
    selected = tool.load_catalog(tool.DEFAULT_CATALOG)
    expected_ids = {
        item["asset_id"] for item in catalog["assets"] if "category" in item
    }
    assert expected_ids
    assert len(selected) == len(expected_ids)
    assert {item["asset_id"] for item in selected} == expected_ids
    assert 1 < len({item["category"] for item in selected}) <= 50
    assert all(item["name"] and item["tags"] for item in selected)


def test_explicit_shortlist_accepts_fifty_categories_and_rejects_the_next(
    tmp_path: Path,
) -> None:
    shortlist = tmp_path / "selected.jsonl"
    catalog = tmp_path / "unused.json"
    with shortlist.open("w") as stream:
        for i in range(50):
            row = {"asset_id": str(UUID(int=i)), "category": f"object_{i}"}
            stream.write(json.dumps(row) + "\n")
    assert len(tool._shortlist_rows(shortlist, catalog)) == 50
    with shortlist.open("a") as stream:
        row = {"asset_id": str(UUID(int=50)), "category": "extra"}
        stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="categor"):
        tool._shortlist_rows(shortlist, catalog)
