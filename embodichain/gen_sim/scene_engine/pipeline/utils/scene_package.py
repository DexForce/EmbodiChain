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

"""Adapt Scene Engine USD delivery to DexSim's AssetRevision package contract."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from zipfile import ZIP_STORED, ZipFile

__all__: list[str] = []


def _package_registry() -> dict[str, Any]:
    """Describe local packaging scope without claiming a SimReady Profile."""
    version = "1.0.0"
    rule_id = "GENSIM-PACKAGE-DEPENDENCIES"
    return {
        "schema_version": "2.0-draft",
        "registry_id": "embodichain-gensim-packaging",
        "version": version,
        "rules": [
            {
                "rule_id": rule_id,
                "version": version,
                "domain": "ci",
                "title": "Scene package dependency closure",
                "requirement": "Keep USD layers and asset dependencies inside the package with content digests.",
                "capability_id": "scene_package_inventory",
                "source": {
                    "namespace": "embodichain",
                    "document_uri": "urn:embodichain:gensim:scene-package:1",
                    "version": version,
                    "retrieved_at": "2026-11-11T00:00:00Z",
                    "document_digest": None,
                    "requirement_id": rule_id,
                    "mapping": "none",
                },
                "usd_binding": None,
                "applicability": {
                    "asset_kinds": [],
                    "formats": ["usd", "usda", "usdc"],
                    "required_declarations": [],
                    "subject_scope": "asset",
                    "condition_description": "Generated Scene Engine USD delivery packages.",
                },
                "checker": {
                    "id": "dexsim-processed-usd-package",
                    "version": version,
                    "method": "static",
                    "parameters_schema": {"type": "object"},
                    "implementation": "partial",
                },
                "acceptance": "Package inventory and relative dependencies resolve; runtime readiness remains unevaluated.",
                "default_severity": "error",
                "evidence_kinds": ["log"],
            }
        ],
        "capabilities": [
            {
                "capability_id": "scene_package_inventory",
                "version": version,
                "domain": "ci",
                "requirement_ids": [rule_id],
            }
        ],
        "features": [
            {
                "feature_id": "scene_package_inventory",
                "version": version,
                "required_rules": [{"rule_id": rule_id, "version": version}],
                "runtime_test_ids": [],
                "dependencies": [],
            }
        ],
        "profiles": [
            {
                "profile_id": "embodichain-scene-packaging",
                "version": version,
                "namespace": "embodichain",
                "source": None,
                "features": [
                    {"feature_id": "scene_package_inventory", "version": version}
                ],
                "required_runtime_tests": [],
                "asset_kinds": [],
                "target_runtimes": [],
                "policy": {
                    "policy_version": version,
                    "blocking_rule_ids": [rule_id],
                    "allow_manual_evidence": False,
                    "required_evaluation_methods": ["static"],
                    "parameters": {},
                },
            }
        ],
    }


def _export_scene_package(
    *, scene_usd_path: Path, destination: Path, scene_export_root: Path | None = None
) -> Path:
    """Publish the USD closure and, when available, its editable scene layout.

    DexSim owns the schema, namespace remapping, inventories and provenance.
    Its packaging adapter is isolated here because it is currently a private
    SDK entry point. The editable scene and its USDZ retain their own layout.
    A whole scene can contain independent rigid bodies, so packaging does not
    guess an asset classification or grant runtime capabilities.
    """
    from pxr import UsdUtils

    try:
        from dexsim.kit.simready._usd_package import export_processed_usd
    except ImportError as error:
        raise RuntimeError(
            "Scene package export requires a DexSim build with "
            "AssetRevision 2.0-draft processed-USD packaging."
        ) from error

    source_root = scene_usd_path.parent.resolve()
    destination = destination.absolute()
    destination.unlink(missing_ok=True)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene_usd_path))
    if unresolved:
        raise ValueError(f"Scene USD has unresolved dependencies: {unresolved}")
    dependencies = {Path(layer.realPath) for layer in layers} | {
        Path(asset) for asset in assets
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    catalog = {
        "name": "GenSim scene",
        "contributor": "EmbodiChain Scene Engine",
        "contributor_origin": "declared",
        "annotation": "Generated scene package; quality and runtime certification are not evaluated.",
        "version_number": 0,
        "version": "v0",
        "created_at": timestamp,
        "published_at": None,
        "permission": "private",
        "tags": ["gensim", "scene"],
    }
    with tempfile.TemporaryDirectory(
        prefix="scene-package-", dir=destination.parent
    ) as directory:
        workspace = Path(directory)
        staging = workspace / "source"
        staging.mkdir()
        for dependency in sorted(dependencies):
            resolved = dependency.resolve(strict=True)
            if not resolved.is_relative_to(source_root):
                raise ValueError(
                    f"Scene USD dependency escapes its package: {dependency}"
                )
            relative = resolved.relative_to(source_root)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, target)
        if scene_export_root is not None:
            from ._scene_package_metadata import _stage_scene_export

            _stage_scene_export(scene_export_root.resolve(), staging / "scene_export")
        packaged = export_processed_usd(
            staging / scene_usd_path.name,
            workspace / "scene.zip",
            source_root=staging,
            catalog=catalog,
            registry=_package_registry(),
        )
        if scene_export_root is not None:
            packaged = _finish_editable_package(packaged, workspace)
        packaged.replace(destination)
    return destination


def _finish_editable_package(packaged: Path, workspace: Path) -> Path:
    """Add scene metadata to the staged SDK archive and validate before publication."""
    from dexsim.kit.simready import open_asset_package, validate_asset_revision
    from ._scene_package_metadata import _SCENE_EXTENSION, _scene_package_metadata

    # The pinned SDK writes the file inventory but has no scene-metadata input.
    # Its digests cover the staged editable files. Only asset.json changes here.
    enriched = workspace / "editable.zip"
    with open_asset_package(packaged, workspace=workspace) as root:
        asset_path = root / "asset.json"
        asset = json.loads(asset_path.read_text(encoding="utf-8"))
        metadata = _scene_package_metadata(root, asset)
        asset.setdefault("extensions", {})[_SCENE_EXTENSION] = metadata.pop(
            _SCENE_EXTENSION
        )
        asset.update(metadata)
        asset_path.write_text(
            json.dumps(asset, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if validate_asset_revision(root) is None:
            raise ValueError("Scene package has no AssetRevision")
        with ZipFile(enriched, "x", compression=ZIP_STORED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
    return enriched


def _restore_scene_export(output_root: Path) -> None:
    """Materialize the editable files from a package-only scene workspace."""
    destination = output_root / "scene_export"
    packaged = output_root / "scene_package.zip"
    if destination.exists() or destination.is_symlink() or not packaged.is_file():
        return

    from dexsim.kit.simready import open_asset_package, validate_asset_revision
    from ._scene_package_metadata import (
        _SCENE_EXTENSION,
        _same_scene_layout,
        _scene_package_metadata,
    )

    with open_asset_package(packaged, workspace=output_root) as root:
        if validate_asset_revision(root) is None:
            raise ValueError("Scene package has no AssetRevision")
        # Read the validated document because the SDK's typed view omits optional
        # scene fields on some releases.
        asset = json.loads((root / "asset.json").read_text(encoding="utf-8"))
        extension = asset.get("extensions", {}).get(_SCENE_EXTENSION)
        if not isinstance(extension, dict) or extension.get("version") != 1:
            raise ValueError(
                "Scene package has no supported editable Scene Engine export"
            )
        expected = _scene_package_metadata(root, asset)
        semantics = asset.get("semantics")
        if (
            extension != expected[_SCENE_EXTENSION]
            or not _same_scene_layout(
                asset.get("scene_layout"), expected["scene_layout"]
            )
            or asset["representations"] != expected["representations"]
            or (
                semantics != expected["semantics"]
                # Older v1 packages copied floor roots' table edges.
                and semantics
                != _scene_package_metadata(root, asset, legacy_floor_semantics=True)[
                    "semantics"
                ]
            )
        ):
            raise ValueError(
                "Scene package metadata disagrees with its editable export"
            )
        # The SDK checks bytes before this same-filesystem rename. A conflicting
        # nonempty destination cannot be replaced by rename.
        (root / "scene_export").rename(destination)
