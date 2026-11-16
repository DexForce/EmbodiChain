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

"""Auditable stage boundaries for Scene Engine generation and editing."""

from __future__ import annotations

from embodichain.gen_sim.services.runtime import GenSimServices, service_entry
from embodichain.gen_sim.services.compat import (
    VisionFacade,
    GeometryFacade,
    ImageFacade,
    SegmentationFacade,
    ArticulationFacade,
)


from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Final

from embodichain.gen_sim.services.compat import ArticulationClient
from embodichain.gen_sim.services.compat import GeometryClient
from embodichain.gen_sim.services.compat import ImageClient
from embodichain.gen_sim.services.compat import SegmentationClient
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_edit_plan import SceneEditPlan
from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraph
from embodichain.gen_sim.services.compat import VisionClient
from embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_asset_preparation import (
    prepare_scene_edit_assets,
)
from embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_layout_generation import (
    edit_layout,
)
from embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_understanding import (
    understand_scene_edit,
)
from embodichain.gen_sim.scene_engine.pipeline.generation.scene_generation import (
    generate_scene_and_refine,
)
from embodichain.gen_sim.scene_engine.pipeline.generation.scene_understanding import (
    understand_scene,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import SceneExporter
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    SceneExportImporter,
)
from embodichain.utils.logger import log_info

__all__ = [
    "SCENE_BLUEPRINT_SCHEMA",
    "SCENE_EDIT_BLUEPRINT_SCHEMA",
    "SceneBlueprintPackage",
    "SceneEditBlueprintPackage",
    "SceneMaterialization",
    "analyze_edit",
    "analyze_image",
    "materialize_blueprint",
    "materialize_edit",
]

SCENE_BLUEPRINT_SCHEMA: Final = "embodichain.scene-blueprint/v2"
SCENE_EDIT_BLUEPRINT_SCHEMA: Final = "embodichain.scene-edit-blueprint/v2"


@dataclass(frozen=True)
class SceneBlueprintPackage:
    """In-process scene semantics plus their persisted audit document."""

    schema_version: str
    blueprint_id: str
    image_path: Path
    output_root: Path
    manifest_path: Path
    scene: Scene
    scene_graph: SceneGraph

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_BLUEPRINT_SCHEMA:
            raise ValueError(
                "SceneBlueprintPackage schema_version must be "
                f"{SCENE_BLUEPRINT_SCHEMA!r}."
            )


@dataclass(frozen=True)
class SceneEditBlueprintPackage:
    """Validated edit intent before added assets and layout are materialized."""

    schema_version: str
    blueprint_id: str
    edit_prompt: str
    output_root: Path
    manifest_path: Path
    scene_edit_plan: SceneEditPlan
    updated_scene_graph: SceneGraph

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_EDIT_BLUEPRINT_SCHEMA:
            raise ValueError(
                "SceneEditBlueprintPackage schema_version must be "
                f"{SCENE_EDIT_BLUEPRINT_SCHEMA!r}."
            )


@dataclass(frozen=True)
class SceneMaterialization:
    """One exported materialized scene revision."""

    scene: Scene
    scene_graph: SceneGraph
    output_root: Path
    scene_config_path: Path


@service_entry
def analyze_image(
    image_path: str | Path,
    output_root: str | Path,
    *,
    vlm_client: VisionClient | None = None,
    image_segmentation_client: SegmentationClient | None = None,
    services: GenSimServices | None = None,
) -> SceneBlueprintPackage:
    """Understand an image and persist the pre-generation semantic blueprint.

    Args:
        image_path: Source scene image.
        output_root: Directory receiving understanding artifacts and the blueprint.
        vlm_client: Optional caller-owned scene-understanding client.
        image_segmentation_client: Optional caller-owned segmentation client.

    Returns:
        In-process scene and graph with the persisted blueprint identity.
    """
    resolved_image = Path(image_path).expanduser().resolve()
    resolved_output = Path(output_root).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    effective_vlm = vlm_client or VisionFacade(services, "scene_understanding")
    segmentation = image_segmentation_client or SegmentationFacade(services)
    owns_segmentation = image_segmentation_client is None
    log_info("Starting Scene Understanding")
    try:
        segmentation.check_health()
        scene, scene_graph = understand_scene(
            scene=Scene(),
            image_path=resolved_image,
            output_root=resolved_output,
            vlm_client=effective_vlm,
            image_segmentation_client=segmentation,
        )
    finally:
        if owns_segmentation:
            segmentation.close()
    log_info("Completed Scene Understanding")

    payload = {
        "schema_version": SCENE_BLUEPRINT_SCHEMA,
        "image_path": resolved_image.as_posix(),
        "scene": scene.to_dict(),
        "scene_graph": scene_graph.to_dict(),
        "artifacts": [
            *_artifact_records(resolved_output / "scene_understanding"),
            _artifact_record(resolved_image),
        ],
    }
    blueprint_id = _canonical_hash(payload)
    document = {**payload, "blueprint_id": blueprint_id}
    manifest_path = resolved_output / "scene_blueprint.json"
    _write_json(manifest_path, document)
    return SceneBlueprintPackage(
        schema_version=SCENE_BLUEPRINT_SCHEMA,
        blueprint_id=blueprint_id,
        image_path=resolved_image,
        output_root=resolved_output,
        manifest_path=manifest_path,
        scene=scene,
        scene_graph=scene_graph,
    )


@service_entry
def materialize_blueprint(
    blueprint: SceneBlueprintPackage,
    *,
    vlm_client: VisionClient | None = None,
    geometry_generation_client: GeometryClient | None = None,
    articulated_generation_client: ArticulationClient | None = None,
    services: GenSimServices | None = None,
) -> SceneMaterialization:
    """Generate assets and layout for one image-derived blueprint.

    Args:
        blueprint: Audited scene/graph package returned by :func:`analyze_image`.
        vlm_client: Optional caller-owned layout/refinement client.
        geometry_generation_client: Optional caller-owned geometry client.
        articulated_generation_client: Optional caller-owned articulation client.

    Returns:
        Exported scene revision without mutating the supplied blueprint objects.

    Raises:
        ValueError: If the saved blueprint or its source/understanding artifacts
            changed. Re-analyze the scene before making generation requests.
    """
    _verify_blueprint_artifacts(blueprint)
    scene = deepcopy(blueprint.scene)
    scene_graph = deepcopy(blueprint.scene_graph)
    effective_vlm = vlm_client or VisionFacade(services, "scene_understanding")
    geometry = geometry_generation_client or GeometryFacade(services)
    has_articulated_objects = any(item.is_articulated for item in scene.objects)
    articulated = articulated_generation_client if has_articulated_objects else None
    owns_articulated = False
    log_info("Starting Objects + Coarse Layout Generation")
    try:
        if has_articulated_objects and articulated is None:
            articulated = ArticulationFacade(services)
            owns_articulated = True
        geometry.check_health()
        if articulated is not None:
            articulated.check_health()
        scene = generate_scene_and_refine(
            image_path=blueprint.image_path,
            output_root=blueprint.output_root,
            scene=scene,
            scene_graph=scene_graph,
            geometry_generation_client=geometry,
            vlm_client=effective_vlm,
            articulated_generation_client=articulated,
        )
    finally:
        if owns_articulated and articulated is not None:
            articulated.close()
        if geometry_generation_client is None:
            geometry.close()
    log_info("Completed Objects + Coarse Layout Generation")
    return _export_materialization(
        scene=scene,
        scene_graph=scene_graph,
        output_root=blueprint.output_root,
    )


@service_entry
def analyze_edit(
    *,
    output_root: str | Path,
    edit_prompt: str,
    vlm_client: VisionClient | None = None,
    services: GenSimServices | None = None,
) -> SceneEditBlueprintPackage:
    """Interpret and persist one edit against an already generated scene.

    Args:
        output_root: Existing generation directory containing ``scene_export``.
        edit_prompt: Nonempty natural-language scene edit.
        vlm_client: Optional caller-owned edit-understanding client.

    Returns:
        Validated edit plan and updated graph with their persisted audit identity.

    Raises:
        ValueError: If the edit prompt is empty or the exported scene is invalid.
    """
    resolved_output = Path(output_root).expanduser().resolve()
    normalized_prompt = str(edit_prompt).strip()
    if not normalized_prompt:
        raise ValueError("Edit prompt must not be empty.")
    scene, scene_graph = SceneExportImporter(
        output_root=resolved_output
    ).import_scene_and_graph()
    effective_vlm = vlm_client or VisionFacade(services, "scene_edit")
    log_info("Starting Edit Understanding")
    scene_edit_plan, updated_scene_graph = understand_scene_edit(
        scene=scene,
        scene_graph=scene_graph,
        edit_prompt=normalized_prompt,
        vlm_client=effective_vlm,
    )
    log_info("Completed Edit Understanding")
    payload = {
        "schema_version": SCENE_EDIT_BLUEPRINT_SCHEMA,
        "edit_prompt": normalized_prompt,
        "scene_edit_plan": scene_edit_plan.to_dict(),
        "updated_scene_graph": updated_scene_graph.to_dict(),
    }
    blueprint_id = _canonical_hash(payload)
    manifest_path = resolved_output / "scene_edit" / "scene_edit_blueprint.json"
    _write_json(manifest_path, {**payload, "blueprint_id": blueprint_id})
    return SceneEditBlueprintPackage(
        schema_version=SCENE_EDIT_BLUEPRINT_SCHEMA,
        blueprint_id=blueprint_id,
        edit_prompt=normalized_prompt,
        output_root=resolved_output,
        manifest_path=manifest_path,
        scene_edit_plan=scene_edit_plan,
        updated_scene_graph=updated_scene_graph,
    )


@service_entry
def materialize_edit(
    blueprint: SceneEditBlueprintPackage,
    *,
    vlm_client: VisionClient | None = None,
    image_generation_client: ImageClient | None = None,
    geometry_generation_client: GeometryClient | None = None,
    image_segmentation_client: SegmentationClient | None = None,
    services: GenSimServices | None = None,
) -> SceneMaterialization:
    """Generate added assets, apply layout edits, and export the new revision.

    Args:
        blueprint: Audited edit package returned by :func:`analyze_edit`.
        vlm_client: Optional caller-owned asset-understanding client.
        image_generation_client: Optional caller-owned image generation client.
        geometry_generation_client: Optional caller-owned geometry client.
        image_segmentation_client: Optional caller-owned segmentation client.

    Returns:
        Exported edited scene without mutating the supplied blueprint objects.
    """
    scene_edit_plan = deepcopy(blueprint.scene_edit_plan)
    updated_scene_graph = deepcopy(blueprint.updated_scene_graph)
    owned_clients = []
    added_assets = []
    if any(operation.op == "add" for operation in scene_edit_plan.operations):
        effective_vlm = vlm_client or VisionFacade(services, "scene_edit")
        log_info("Starting Objects Preparation")
        try:
            image_generation = image_generation_client or ImageFacade(services)
            owned_clients.append((image_generation, image_generation_client is None))
            geometry = geometry_generation_client or GeometryFacade(services)
            owned_clients.append((geometry, geometry_generation_client is None))
            segmentation = image_segmentation_client or SegmentationFacade(services)
            owned_clients.append((segmentation, image_segmentation_client is None))
            for client, _ in owned_clients:
                client.check_health()
            added_assets = prepare_scene_edit_assets(
                scene_edit_plan=scene_edit_plan,
                output_root=blueprint.output_root,
                image_generation_client=image_generation,
                geometry_generation_client=geometry,
                image_segmentation_client=segmentation,
                vlm_client=effective_vlm,
            )
        finally:
            for client, owned in owned_clients:
                if owned:
                    client.close()
        log_info("Completed Objects Preparation")
    log_info("Starting Layout Generation")
    scene = edit_layout(
        scene=scene_edit_plan.scene,
        scene_edit_plan=scene_edit_plan,
        updated_scene_graph=updated_scene_graph,
        added_assets=added_assets,
        output_root=blueprint.output_root,
    )
    log_info("Completed Layout Generation")
    return _export_materialization(
        scene=scene,
        scene_graph=updated_scene_graph,
        output_root=blueprint.output_root,
    )


def _export_materialization(
    *,
    scene: Scene,
    scene_graph: SceneGraph,
    output_root: Path,
) -> SceneMaterialization:
    log_info("Starting Scene Export")
    scene_config_path = SceneExporter(
        scene=scene,
        scene_graph=scene_graph,
        output_root=output_root,
    ).export()
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
        build_scene_usd,
        build_scene_usdz,
    )

    scene_usd_path = build_scene_usd(output_root=output_root)
    build_scene_usdz(scene_usd_path=scene_usd_path)
    log_info("Completed Scene Export")
    return SceneMaterialization(
        scene=scene,
        scene_graph=scene_graph,
        output_root=output_root,
        scene_config_path=scene_config_path,
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    return [
        _artifact_record(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _verify_blueprint_artifacts(blueprint: SceneBlueprintPackage) -> None:
    """Reject overwritten observations or declarations before generation calls."""
    try:
        document = json.loads(blueprint.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("blueprint manifest must be an object")
        identity = document.pop("blueprint_id")
        if identity != blueprint.blueprint_id or _canonical_hash(document) != identity:
            raise ValueError("blueprint identity changed")
        if (
            document["scene"] != blueprint.scene.to_dict()
            or document["scene_graph"] != blueprint.scene_graph.to_dict()
            or document["image_path"] != blueprint.image_path.as_posix()
        ):
            raise ValueError("blueprint declarations changed")
        records = document["artifacts"]
        if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records
        ):
            raise ValueError("invalid artifact records")
        if not any(
            record.get("path") == blueprint.image_path.as_posix() for record in records
        ):
            raise ValueError("source image is not audited")
        for record in records:
            if _artifact_record(Path(record["path"])) != record:
                raise ValueError(f"artifact changed: {record['path']}")
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"Scene blueprint is stale or invalid; analyze the scene again: {error}"
        ) from error


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
