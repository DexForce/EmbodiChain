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

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Final

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.image_generation import (
    ImageGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_edit_plan import SceneEditPlan
from embodichain.gen_sim.scene_engine.core.scene_graph import GeneratedSceneGraph
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
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

SCENE_BLUEPRINT_SCHEMA: Final = "embodichain.scene-blueprint/v1"
SCENE_EDIT_BLUEPRINT_SCHEMA: Final = "embodichain.scene-edit-blueprint/v1"


@dataclass(frozen=True)
class SceneBlueprintPackage:
    """In-process scene semantics plus their persisted audit document."""

    blueprint_id: str
    image_path: Path
    output_root: Path
    manifest_path: Path
    scene: Scene
    scene_graph: GeneratedSceneGraph


@dataclass(frozen=True)
class SceneEditBlueprintPackage:
    """Validated edit intent before added assets and layout are materialized."""

    blueprint_id: str
    edit_prompt: str
    output_root: Path
    manifest_path: Path
    scene_edit_plan: SceneEditPlan
    updated_scene_graph: GeneratedSceneGraph


@dataclass(frozen=True)
class SceneMaterialization:
    """One exported materialized scene revision."""

    scene: Scene
    scene_graph: GeneratedSceneGraph
    output_root: Path
    scene_config_path: Path


def analyze_image(
    image_path: str | Path,
    output_root: str | Path,
    *,
    vlm_client: OpenAICompatibleVLM | None = None,
    image_segmentation_client: ImageSegmentationClient | None = None,
) -> SceneBlueprintPackage:
    """Understand an image and persist the pre-generation semantic blueprint."""
    resolved_image = Path(image_path).expanduser().resolve()
    resolved_output = Path(output_root).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    effective_vlm = vlm_client or OpenAICompatibleVLM.from_dotenv()
    segmentation = image_segmentation_client or ImageSegmentationClient.from_dotenv()
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
        "artifacts": _artifact_records(resolved_output / "scene_understanding"),
    }
    blueprint_id = _canonical_hash(payload)
    document = {**payload, "blueprint_id": blueprint_id}
    manifest_path = resolved_output / "scene_blueprint.json"
    _write_json(manifest_path, document)
    return SceneBlueprintPackage(
        blueprint_id=blueprint_id,
        image_path=resolved_image,
        output_root=resolved_output,
        manifest_path=manifest_path,
        scene=scene,
        scene_graph=scene_graph,
    )


def materialize_blueprint(
    blueprint: SceneBlueprintPackage,
    *,
    vlm_client: OpenAICompatibleVLM | None = None,
    geometry_generation_client: GeometryGenerationClient | None = None,
    seed: int | None = None,
) -> SceneMaterialization:
    """Generate assets and layout for one image-derived blueprint."""
    scene = deepcopy(blueprint.scene)
    scene_graph = deepcopy(blueprint.scene_graph)
    effective_vlm = vlm_client or OpenAICompatibleVLM.from_dotenv()
    geometry = geometry_generation_client or GeometryGenerationClient.from_dotenv()
    owns_geometry = geometry_generation_client is None
    log_info("Starting Objects + Coarse Layout Generation")
    try:
        geometry.check_health()
        scene = generate_scene_and_refine(
            image_path=blueprint.image_path,
            output_root=blueprint.output_root,
            scene=scene,
            scene_graph=scene_graph,
            geometry_generation_client=geometry,
            vlm_client=effective_vlm,
            seed=seed,
        )
    finally:
        if owns_geometry:
            geometry.close()
    log_info("Completed Objects + Coarse Layout Generation")
    return _export_materialization(
        scene=scene,
        scene_graph=scene_graph,
        output_root=blueprint.output_root,
    )


def analyze_edit(
    *,
    output_root: str | Path,
    edit_prompt: str,
    vlm_client: OpenAICompatibleVLM | None = None,
) -> SceneEditBlueprintPackage:
    """Interpret and persist one edit against an already generated scene."""
    resolved_output = Path(output_root).expanduser().resolve()
    normalized_prompt = str(edit_prompt).strip()
    if not normalized_prompt:
        raise ValueError("Edit prompt must not be empty.")
    scene, scene_graph = SceneExportImporter(
        output_root=resolved_output
    ).import_scene_and_graph()
    effective_vlm = vlm_client or OpenAICompatibleVLM.from_dotenv()
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
        blueprint_id=blueprint_id,
        edit_prompt=normalized_prompt,
        output_root=resolved_output,
        manifest_path=manifest_path,
        scene_edit_plan=scene_edit_plan,
        updated_scene_graph=updated_scene_graph,
    )


def materialize_edit(
    blueprint: SceneEditBlueprintPackage,
    *,
    vlm_client: OpenAICompatibleVLM | None = None,
    image_generation_client: ImageGenerationClient | None = None,
    geometry_generation_client: GeometryGenerationClient | None = None,
    image_segmentation_client: ImageSegmentationClient | None = None,
    seed: int | None = None,
) -> SceneMaterialization:
    """Generate added assets, apply layout edits, and export the new revision."""
    scene_edit_plan = deepcopy(blueprint.scene_edit_plan)
    updated_scene_graph = deepcopy(blueprint.updated_scene_graph)
    effective_vlm = vlm_client or OpenAICompatibleVLM.from_dotenv()
    image_generation = image_generation_client or ImageGenerationClient.from_dotenv()
    geometry = geometry_generation_client or GeometryGenerationClient.from_dotenv()
    segmentation = image_segmentation_client or ImageSegmentationClient.from_dotenv()
    owned_clients = (
        (image_generation, image_generation_client is None),
        (geometry, geometry_generation_client is None),
        (segmentation, image_segmentation_client is None),
    )
    log_info("Starting Objects Preparation")
    try:
        for client, _ in owned_clients:
            client.check_health()
        added_assets = prepare_scene_edit_assets(
            scene_edit_plan=scene_edit_plan,
            output_root=blueprint.output_root,
            image_generation_client=image_generation,
            geometry_generation_client=geometry,
            image_segmentation_client=segmentation,
            vlm_client=effective_vlm,
            seed=seed,
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
    scene_graph: GeneratedSceneGraph,
    output_root: Path,
) -> SceneMaterialization:
    log_info("Starting Scene Export")
    scene_config_path = SceneExporter(
        scene=scene,
        scene_graph=scene_graph,
        output_root=output_root,
    ).export()
    log_info("Completed Scene Export")
    return SceneMaterialization(
        scene=scene,
        scene_graph=scene_graph,
        output_root=output_root,
        scene_config_path=scene_config_path,
    )


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.resolve().as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    return records


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
