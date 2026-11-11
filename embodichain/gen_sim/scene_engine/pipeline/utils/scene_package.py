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
from pathlib import Path
import shutil
import tempfile
from typing import Any

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


def _export_scene_package(*, scene_usd_path: Path, destination: Path) -> Path:
    """Publish a schema-validated ZIP from only the USD dependency closure.

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
        packaged = export_processed_usd(
            staging / scene_usd_path.name,
            workspace / "scene.zip",
            source_root=staging,
            catalog=catalog,
            registry=_package_registry(),
        )
        packaged.replace(destination)
    return destination
