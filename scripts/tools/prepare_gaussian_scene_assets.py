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

"""Build versioned visual-only USD/AssetRevision packages from pinned GS scans.

Requires the qualified DexSim scheme-one packaging implementation. This is an
explicit offline publication tool; runtime consumers download completed bundles.
No collision proxy, point cleanup, scale estimate or readiness certification is
generated. Archive metadata gets a fresh revision identity on each build.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
from uuid import NAMESPACE_URL, uuid5
import zipfile

__all__ = ["main"]

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "examples/sim/assets/gaussian_backgrounds"


def _digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _source(recipe: dict, directory: Path, supplied: Path | None) -> Path:
    candidate = (supplied or directory) / recipe["local_source_name"]
    if not candidate.exists() and supplied is None:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(recipe["source_url"], timeout=120) as source,
            candidate.open("xb") as target,
        ):
            shutil.copyfileobj(source, target, length=1024 * 1024)
    if _digest(candidate) != recipe["sha256"]:
        raise ValueError(f"Upstream bytes differ: {candidate}")
    return candidate


def _build(name: str, recipe: dict, output: Path, source_dir: Path | None) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics
    from dexsim.kit.simready import (
        audit_package,
        open_asset_package,
        validate_asset_revision,
    )

    # The SDK's public converter also generates enclosing collision. Its existing
    # scheme-one exporter is intentionally used only by this offline build tool.
    from dexsim.kit.simready._usd_package import export_processed_usd

    source = _source(recipe, output / "upstream", source_dir)
    root = output / "processed" / recipe["slug"]
    root.mkdir(parents=True)
    (root / "source").mkdir()
    original = root / "source/original.ply"
    shutil.copyfile(source, original)
    render_path = original
    operations = ["preserve_original_bytes"]
    if recipe["compressed_super_splat"]:
        (root / "visual").mkdir()
        render_path = root / "visual/scene.ply"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("_decode_gaussian_super_splat.py")),
                str(original),
                str(render_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        operations.append("decode_super_splat_to_inria_ply")
    entry = root / "asset.usda"
    stage = Usd.Stage.CreateNew(str(entry))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    visual = UsdGeom.Xform.Define(stage, "/Asset/Visual")
    # Coordinate normalization is authored as a rigid USD transform. In
    # particular, SH coefficients and per-Gaussian covariance are never baked.
    angles = recipe["rotation_xyz_degrees"]
    if any(angles):
        visual.AddRotateXYZOp().Set(tuple(float(v) for v in angles))
        operations.append("author_source_to_z_up_rigid_transform")
    visual.GetPrim().GetReferences().AddReference(
        render_path.relative_to(root).as_posix()
    )
    stage.GetRootLayer().Save()
    del stage
    provenance = dict(
        recipe,
        render_path=render_path.relative_to(root).as_posix(),
        render_sha256=_digest(render_path),
        original_sha256=_digest(original),
        package_convention={"meters_per_unit": 1.0, "up_axis": "Z", "scale": 1.0},
        operations=operations,
        collision={"present": False, "scope": "visual_only"},
        license_scope="upstream dataset card; project code licenses are not inferred for model data",
    )
    _json(root / "source/provenance.json", provenance)
    license_name = {
        "office": "robosplatter.Apache-2.0.txt",
        "robosimgs": "robosimgs.MIT.txt",
    }.get(name)
    if license_name:
        shutil.copyfile(RESOURCES / "licenses" / license_name, root / "LICENSE.txt")
    else:
        (root / "LICENSE.txt").write_text(
            "The upstream DISCOVERSE model repository does not declare an asset license.\nThe project's MIT software license is not inferred to cover these model bytes.\nNo additional asset license is granted by packaging. See source/provenance.json.\n"
        )
    (root / "NOTICE.md").write_text(
        f"# {recipe['name']}\n\nOriginal provider: {recipe['repository']}\n\nPinned upstream revision: `{recipe['revision']}`. Source SHA256: `{recipe['sha256']}`.\n\nData license declaration: {recipe['license'] or 'not supplied by the upstream model repository'}.\n\nOriginal source bytes are unchanged. Operations: {', '.join(operations)}.\nThe USD reference carries coordinate normalization; no geometry is rescaled, reconstructed, or collisionized.\n"
    )
    (root / "README.md").write_text(
        f"# {recipe['name']} — Gaussian background v1\n\nEntry: `asset.usda`, default prim `/Asset`; visual subtree `/Asset/Visual`.\nThe entry declares metres and Z-up. Source bytes and covariance/SH data are preserved.\n\nThis is a visual-only captured scene. It contains no collider, task object, robot, camera or light.\nTask environments must supply their own independently qualified collision geometry.\nUse Gaussian rendering with `apply_display_srgb=False` for the captured colors.\n\n`asset.json` follows AssetRevision 2.0-draft. Packaging does not confer READY or metric-depth certification.\nSource, derivation and licensing evidence are recorded in `source/provenance.json`, `NOTICE.md` and `LICENSE.txt`.\n"
    )
    _json(
        root / "reports/normalization.json",
        {
            "version": 1,
            "source_sha256": _digest(original),
            "render_sha256": _digest(render_path),
            "source_to_asset_rotation_xyz_degrees": angles,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "scale": 1.0,
            "generated_collision": False,
            "geometry_cleaning": False,
            "operations": operations,
        },
    )
    release = output / "release"
    release.mkdir(exist_ok=True)
    archive = release / f"{recipe['slug']}_v1.zip"
    now = datetime.now(timezone.utc).isoformat()
    catalog = {
        "name": recipe["name"],
        "category": "captured_gaussian_background",
        "contributor": "DexForceAI",
        "contributor_origin": "declared",
        "annotation": "Visual-only coordinate-normalized package; source content and licensing remain upstream. No industrial readiness or depth claim.",
        "version_number": 1,
        "version": "v1",
        "created_at": now,
        "published_at": None,
        "license": recipe["license"],
        "permission": "public",
        "tags": ["3dgs", "scene", "visual-only", "z-up", "metres"],
    }
    registry = json.loads((RESOURCES / "registry.json").read_text())
    export_processed_usd(
        entry,
        archive,
        source_root=root,
        catalog=catalog,
        registry=registry,
        asset_id=str(
            uuid5(
                NAMESPACE_URL, "urn:embodichain:gaussian-background:" + recipe["slug"]
            )
        ),
    )
    with open_asset_package(archive, workspace=output / "audit") as packaged:
        revision = validate_asset_revision(packaged)
        admitted = audit_package(packaged)
        if not admitted.ready:
            raise ValueError(f"USD admission failed: {admitted.to_dict()}")
        stage = Usd.Stage.Open(str(packaged / "asset.usda"))
        if (
            UsdGeom.GetStageUpAxis(stage) != "Z"
            or UsdGeom.GetStageMetersPerUnit(stage) != 1
        ):
            raise ValueError("Package coordinate convention differs")
        if any(
            p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)
            for p in stage.Traverse()
        ):
            raise ValueError("Background package contains generated physics")
        if revision.definition != "AssetRevision":
            raise ValueError("Package does not declare AssetRevision metadata")
        asset_document = json.loads((packaged / "asset.json").read_text())
        del stage
    with zipfile.ZipFile(archive) as zipped:
        asset_json_sha256 = hashlib.sha256(zipped.read("asset.json")).hexdigest()
    return {
        "name": recipe["name"],
        "scene": name,
        "archive": archive.name,
        "sha256": _digest(archive),
        "md5": _digest(archive, "md5"),
        "bytes": archive.stat().st_size,
        "entry": "asset.usda",
        "asset_json_sha256": asset_json_sha256,
        "asset_id": asset_document["asset_id"],
        "revision_id": asset_document["revision_id"],
        "source_sha256": recipe["sha256"],
        "license": recipe["license"],
        "validation": {
            "asset_revision": "passed",
            "usd_admission": "passed",
            "up_axis": "Z",
            "meters_per_unit": 1,
            "collision_count": 0,
            "quality_certification": None,
        },
    }


def main() -> None:
    """Build new archives without mutating input files or existing releases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Optional verified local cache; otherwise fetch pinned upstream files",
    )
    parser.add_argument(
        "--scene", choices=["all", "office", "discoverse", "robosimgs"], default="all"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((RESOURCES / "sources.lock.json").read_text())
    records = []
    for name, recipe in lock["scenes"].items():
        if args.scene in ("all", name):
            record = _build(
                name,
                recipe,
                output,
                None if args.source_dir is None else args.source_dir.resolve(),
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    _json(
        output / "release/manifest.json",
        {
            "schema_version": 1,
            "scenes": records,
            "publication_scope": "visual-only GS scene packages; no metric-depth or industrial certification",
        },
    )


if __name__ == "__main__":
    main()
