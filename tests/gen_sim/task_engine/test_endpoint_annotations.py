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

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from embodichain.gen_sim.task_engine import cli
from embodichain.gen_sim.task_engine.endpoint_annotations import (
    MANIFEST_SCHEMA,
    migrate_endpoint_manifest,
    write_endpoint_report,
)


def _asset(tmp_path: Path, *, authored: float | None = None) -> tuple[Path, Path]:
    root = tmp_path / "assets"
    path = root / "cabinet/drawer.usda"
    path.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdPhysics.ArticulationRootAPI.Apply(world)
    for name in ("base", "left", "right"):
        UsdPhysics.RigidBodyAPI.Apply(
            UsdGeom.Cube.Define(stage, f"/World/{name}").GetPrim()
        )
    for name, body, limits in (
        ("left_slide", "left", (0.0, 0.1)),
        ("right_slide", "right", (-0.2, 0.0)),
    ):
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/World/{name}")
        joint.CreateBody0Rel().SetTargets(["/World/base"])
        joint.CreateBody1Rel().SetTargets([f"/World/{body}"])
        joint.CreateAxisAttr("Y")
        joint.CreateLowerLimitAttr(limits[0])
        joint.CreateUpperLimitAttr(limits[1])
        if authored is not None and name == "left_slide":
            joint.GetPrim().CreateAttribute(
                "gen_sim:closedPosition", Sdf.ValueTypeNames.Double
            ).Set(authored)
    stage.GetRootLayer().Save()
    return root, path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, root: Path) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "assets": [
            {
                "path": path.relative_to(root).as_posix(),
                "expected_sha256": _sha256(path),
                "joints": [
                    {
                        "joint": "left_slide",
                        "closed_position": 0.1,
                        "evidence": ["Reviewed closed-pose QA image."],
                    },
                    {
                        "joint": "right_slide",
                        "closed_position": 0.0,
                        "evidence": ["Reviewed motion-strip endpoint."],
                    },
                ],
            }
        ],
    }


def test_endpoint_migration_defaults_to_non_mutating_dry_run(tmp_path: Path) -> None:
    root, path = _asset(tmp_path)
    before = path.read_bytes()
    report = migrate_endpoint_manifest(
        asset_root=root,
        manifest=_manifest(path, root),
    )
    assert report["mode"] == "dry_run"
    assert report["assets"][0]["status"] == "would_annotate"
    assert path.read_bytes() == before
    stage = Usd.Stage.Open(str(path))
    assert not stage.GetPrimAtPath("/World/left_slide").GetAttribute(
        "gen_sim:closedPosition"
    )


def test_endpoint_migration_applies_with_backup_and_exact_scope(
    tmp_path: Path,
) -> None:
    root, path = _asset(tmp_path)
    manifest = _manifest(path, root)
    before = path.read_bytes()
    backup = tmp_path / "backups"
    report = migrate_endpoint_manifest(
        asset_root=root,
        manifest=manifest,
        apply=True,
        backup_root=backup,
    )
    assert report["mode"] == "apply"
    assert report["assets"][0]["status"] == "annotated"
    assert (backup / "cabinet/drawer.usda").read_bytes() == before
    assert path.read_bytes() != before
    stage = Usd.Stage.Open(str(path))
    assert stage.GetPrimAtPath("/World/left_slide").GetAttribute(
        "gen_sim:closedPosition"
    ).Get() == pytest.approx(0.1)
    assert stage.GetPrimAtPath("/World/right_slide").GetAttribute(
        "gen_sim:closedPosition"
    ).Get() == pytest.approx(0.0)
    assert not list(path.parent.glob(".*.endpoint-*.part.usda"))

    verified = _manifest(path, root)
    second = migrate_endpoint_manifest(asset_root=root, manifest=verified)
    assert second["assets"][0]["status"] == "already_annotated"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("hash", "hash changed"),
        ("incomplete", "cover every prismatic joint"),
        ("interior", "not a joint-limit endpoint"),
        ("evidence", "requires evidence"),
        ("escape", "escapes asset_root"),
    ],
)
def test_endpoint_migration_rejects_unreviewed_or_stale_inputs(
    tmp_path: Path, mutation: str, match: str
) -> None:
    root, path = _asset(tmp_path)
    manifest = deepcopy(_manifest(path, root))
    asset = manifest["assets"][0]
    if mutation == "hash":
        asset["expected_sha256"] = "0" * 64
    elif mutation == "incomplete":
        asset["joints"].pop()
    elif mutation == "interior":
        asset["joints"][0]["closed_position"] = 0.05
    elif mutation == "evidence":
        asset["joints"][0]["evidence"] = []
    else:
        asset["path"] = "../outside.usda"
    with pytest.raises(ValueError, match=match):
        migrate_endpoint_manifest(asset_root=root, manifest=manifest)


def test_endpoint_migration_rejects_conflicting_authored_value(
    tmp_path: Path,
) -> None:
    root, path = _asset(tmp_path, authored=0.0)
    with pytest.raises(ValueError, match="conflicts"):
        migrate_endpoint_manifest(asset_root=root, manifest=_manifest(path, root))


def test_endpoint_apply_requires_fresh_backup_root(tmp_path: Path) -> None:
    root, path = _asset(tmp_path)
    manifest = _manifest(path, root)
    with pytest.raises(ValueError, match="requires backup_root"):
        migrate_endpoint_manifest(asset_root=root, manifest=manifest, apply=True)
    backup = tmp_path / "existing"
    backup.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        migrate_endpoint_manifest(
            asset_root=root,
            manifest=manifest,
            apply=True,
            backup_root=backup,
        )


def test_endpoint_report_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_endpoint_report(path, {"status": "first"})
    with pytest.raises(FileExistsError, match="already exists"):
        write_endpoint_report(path, {"status": "replacement"})
    assert json.loads(path.read_text()) == {"status": "first"}


def test_endpoint_annotation_cli_writes_dry_run_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, path = _asset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(path, root)))
    report_path = tmp_path / "report.json"
    assert (
        cli.main(
            [
                "annotate-endpoints",
                "--asset-root",
                str(root),
                "--manifest",
                str(manifest_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    assert json.loads(report_path.read_text())["mode"] == "dry_run"
    assert '"mode": "dry_run"' in capsys.readouterr().out
