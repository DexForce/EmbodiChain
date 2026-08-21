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

"""Task-owned adapter for Scene Engine analysis, revisions, and edits."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import json
import shutil
from typing import Any

from embodichain.gen_sim.action_engine.generation.source_scene import (
    resolve_source_scene,
)
from embodichain.gen_sim.scene_engine.pipeline import (
    SceneBlueprintPackage,
    SceneMaterialization,
    analyze_edit,
    analyze_image,
    materialize_blueprint,
    materialize_edit,
)

from .orchestration.legacy_scene import (
    convert_legacy_gym_project,
    restore_locked_scene_entities,
)
from .orchestration.scene_adapter import CandidateSelection, SceneAdapter
from .orchestration.scene_source import (
    SceneSourceFingerprint,
    fingerprint_scene_source,
    scene_revision_id,
    verify_scene_source_fingerprint,
)
from .scene.final_inspection import FinalSceneInspection, inspect_final_scene
from .workflow_contracts import TaskRunRequest, scene_input_kind

__all__ = [
    "SceneRemediableError",
    "SceneAnalysis",
    "SceneEngineBackend",
    "SceneRevision",
    "scene_blueprint_objects",
]

_LOCKED_SCENE_MANIFEST = "locked_scene_entities.json"


class SceneRemediableError(RuntimeError):
    """A Scene output failure that permits a fresh materialization attempt."""


@dataclass(frozen=True)
class SceneAnalysis:
    """Scene semantics available before asset materialization."""

    input_kind: str
    source: Path
    blueprint: SceneBlueprintPackage | None
    source_fingerprint: SceneSourceFingerprint | None


@dataclass(frozen=True)
class SceneRevision:
    """One immutable scene source selected for final Action preparation."""

    source: Path
    output_root: Path | None
    revision_id: str
    seed: int
    edit_plan: dict[str, Any] | None
    source_fingerprint: SceneSourceFingerprint | None


class SceneEngineBackend:
    """Expose Scene Engine stages without giving it workflow ownership."""

    def analyze(
        self,
        request: TaskRunRequest,
        output_root: str | Path,
    ) -> SceneAnalysis:
        """Analyze an image or fingerprint an existing read-only project.

        Args:
            request: Validated Task Engine run request.
            output_root: Directory for image-understanding artifacts.

        Returns:
            Scene semantics and immutable source provenance.
        """
        root = Path(output_root).expanduser().resolve()
        if scene_input_kind(request) == "image":
            image_path = Path(str(request["image_path"])).resolve()
            blueprint = analyze_image(image_path, root)
            return SceneAnalysis(
                input_kind="image",
                source=image_path,
                blueprint=blueprint,
                source_fingerprint=None,
            )
        source = Path(str(request["gym_project"])).resolve()
        return SceneAnalysis(
            input_kind="gym_project",
            source=source,
            blueprint=None,
            source_fingerprint=fingerprint_scene_source(source),
        )

    def select(
        self,
        analysis: SceneAnalysis,
        candidate_set: Mapping[str, Any],
        scene_adapter: SceneAdapter,
        *,
        force_most_likely: bool,
    ) -> CandidateSelection:
        """Select a task candidate from blueprint or existing-scene semantics.

        Args:
            analysis: Pre-materialization scene analysis.
            candidate_set: Task candidates to ground and vote.
            scene_adapter: Task-owned semantic binding adapter.
            force_most_likely: Whether ranked UID hypotheses must be resolved.

        Returns:
            Audited initial candidate selection.
        """
        if analysis.blueprint is not None:
            return scene_adapter.select_objects(
                candidate_set,
                scene_blueprint_objects(analysis.blueprint),
                force_most_likely=force_most_likely,
            )
        adaptation = scene_adapter.adapt(
            candidate_set,
            analysis.source,
            force_most_likely=force_most_likely,
        )
        return CandidateSelection(
            scene_manifest=adaptation.scene_manifest,
            role_bindings=adaptation.role_bindings,
            binding_report=adaptation.binding_report,
            selected_candidate=adaptation.selected_candidate,
            candidate_bindings=adaptation.candidate_bindings,
        )

    def materialize(
        self,
        analysis: SceneAnalysis,
        request: TaskRunRequest,
        output_root: str | Path,
        *,
        seed: int,
    ) -> SceneRevision:
        """Produce a new revision, or return the untouched existing source.

        Args:
            analysis: Pre-materialization scene analysis.
            request: Validated Task Engine run request.
            output_root: Fresh directory for this scene attempt.
            seed: Attempt seed recorded for recovery audit.

        Returns:
            Final scene source for binding and Action Engine generation.
        """
        root = Path(output_root).expanduser().resolve()
        edit_prompt = request["scene_edit_prompt"]
        if analysis.input_kind == "image":
            assert analysis.blueprint is not None
            root.mkdir(parents=True, exist_ok=False)
            blueprint = replace(analysis.blueprint, output_root=root)
            materialization = materialize_blueprint(blueprint, seed=seed)
            edit_plan = None
            if edit_prompt is not None:
                edit_blueprint = analyze_edit(
                    output_root=root,
                    edit_prompt=str(edit_prompt),
                )
                edit_plan = edit_blueprint.scene_edit_plan.to_dict()
                materialization = materialize_edit(edit_blueprint, seed=seed)
            revision = _revision(materialization, seed=seed, edit_plan=edit_plan)
            _write_revision_audit(
                root,
                revision_id=revision.revision_id,
                seed=seed,
                edit_plan=edit_plan,
            )
            return revision

        fingerprint = analysis.source_fingerprint
        assert fingerprint is not None
        if edit_prompt is None:
            verify_scene_source_fingerprint(fingerprint.to_dict())
            return SceneRevision(
                source=analysis.source,
                output_root=None,
                revision_id=scene_revision_id(analysis.source),
                seed=seed,
                edit_plan=None,
                source_fingerprint=fingerprint,
            )

        resolved = resolve_source_scene(analysis.source)
        if resolved.source_format == "legacy_gym_config":
            converted = convert_legacy_gym_project(analysis.source, root)
            editable_root = converted.output_root
        else:
            editable_root = _copy_scene_export_revision(resolved.path, root)
        edit_blueprint = analyze_edit(
            output_root=editable_root,
            edit_prompt=str(edit_prompt),
        )
        edit_plan = edit_blueprint.scene_edit_plan.to_dict()
        materialization = materialize_edit(edit_blueprint, seed=seed)
        if resolved.source_format == "legacy_gym_config":
            restore_locked_scene_entities(editable_root)
        else:
            _restore_scene_export_locked_entities(editable_root)
        verify_scene_source_fingerprint(fingerprint.to_dict())
        _write_revision_audit(
            editable_root,
            revision_id=scene_revision_id(materialization.scene_config_path),
            seed=seed,
            edit_plan=edit_plan,
            source_fingerprint=fingerprint,
        )
        return SceneRevision(
            source=materialization.scene_config_path,
            output_root=editable_root,
            revision_id=scene_revision_id(materialization.scene_config_path),
            seed=seed,
            edit_plan=edit_plan,
            source_fingerprint=fingerprint,
        )

    def inspect(
        self,
        revision: SceneRevision,
        output_path: str | Path,
    ) -> FinalSceneInspection:
        """Inspect final geometry and publish support/orientation evidence.

        Args:
            revision: Completed immutable scene revision.
            output_path: JSON path receiving the inspection document.

        Returns:
            Validated final scene inspection.
        """
        try:
            actual_revision_id = scene_revision_id(revision.source)
        except (OSError, TypeError, ValueError) as exc:
            raise SceneRemediableError(
                f"Final scene content could not be hashed: {exc}"
            ) from exc
        if actual_revision_id != revision.revision_id:
            raise RuntimeError("Final scene changed before geometry inspection.")
        try:
            inspection = inspect_final_scene(
                revision.source,
                revision_id=actual_revision_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SceneRemediableError(
                f"Final scene assets could not be inspected: {exc}"
            ) from exc
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(inspection, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return inspection


def scene_blueprint_objects(blueprint: SceneBlueprintPackage) -> list[dict[str, Any]]:
    """Convert image semantics into the redacted grounding inventory shape.

    Args:
        blueprint: Scene Engine image-understanding package.

    Returns:
        Semantic objects with unknown physical fields represented conservatively.
    """
    nodes = blueprint.scene_graph.node_by_id()
    result = []
    for item in blueprint.scene.objects:
        node = nodes.get(item.id)
        orientation = None if node is None else node.orientation_state
        initial_state = {}
        if orientation == "lying":
            initial_state["orientation"] = "fallen"
        elif orientation == "standing":
            initial_state["orientation"] = "upright"
        result.append(
            {
                "uid": item.id,
                "source_uid": item.id,
                "role": "table" if item.kind == "table" else "rigid_object",
                "name": item.name,
                "description": item.description,
                "category": item.category,
                "color": None,
                "init_pos": [0.0, 0.0, 0.0],
                "affordances": [],
                "initial_state": initial_state,
                "attributes": {},
            }
        )
    return result


def _copy_scene_export_revision(source_config: Path, output_root: Path) -> Path:
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise ValueError("Scene revision output_root must be empty.")
    source_root = source_config.parent
    destination = output_root / "scene_export"
    shutil.copytree(source_root, destination)
    config_path = destination / "scene_config.json"
    config = _read_json_mapping(config_path)
    background = list(config.get("background", ()))
    rigid_objects = list(config.get("rigid_object", ()))
    articulations = list(config.get("articulation", ()))
    editable_rigid = [
        item
        for item in rigid_objects
        if isinstance(item, Mapping) and _scene_editable_rigid(item)
    ]
    locked_rigid = [item for item in rigid_objects if item not in editable_rigid]
    table = [
        item
        for item in background
        if isinstance(item, Mapping) and item.get("uid") == "table"
    ]
    if len(table) != 1:
        raise ValueError("Scene export revision requires exactly one table.")
    locked = {
        "schema_version": "embodichain.locked-scene-entities/v1",
        "background": [item for item in background if item not in table],
        "rigid_object": locked_rigid,
        "articulation": articulations,
    }
    config["background"] = table
    config["rigid_object"] = editable_rigid
    config["articulation"] = []
    _write_json_mapping(config_path, config)
    _write_json_mapping(output_root / _LOCKED_SCENE_MANIFEST, locked)
    graph_path = destination / "scene_graph.json"
    if graph_path.is_file():
        graph = _read_json_mapping(graph_path)
        editable_uids = {str(item.get("uid")) for item in [*table, *editable_rigid]}
        graph["nodes"] = [
            item
            for item in graph.get("nodes", ())
            if isinstance(item, Mapping) and item.get("object_id") in editable_uids
        ]
        graph["relations"] = [
            item
            for item in graph.get("relations", ())
            if isinstance(item, Mapping)
            and item.get("source_id") in editable_uids
            and item.get("target_id") in editable_uids
        ]
        _write_json_mapping(graph_path, graph)
    return output_root


def _restore_scene_export_locked_entities(output_root: Path) -> None:
    manifest = _read_json_mapping(output_root / _LOCKED_SCENE_MANIFEST)
    if manifest.get("schema_version") != "embodichain.locked-scene-entities/v1":
        raise ValueError("Locked scene entity manifest schema is invalid.")
    config_path = output_root / "scene_export" / "scene_config.json"
    config = _read_json_mapping(config_path)
    existing = {
        str(item.get("uid"))
        for section in ("background", "rigid_object", "articulation")
        for item in config.get(section, ())
        if isinstance(item, Mapping) and item.get("uid")
    }
    for section in ("background", "rigid_object", "articulation"):
        target = list(config.get(section, ()))
        for raw in manifest.get(section, ()):
            item = deepcopy(dict(raw))
            uid = str(item.get("uid", ""))
            if not uid or uid in existing:
                raise ValueError(f"Scene edit reused locked entity UID {uid!r}.")
            target.append(item)
            existing.add(uid)
        config[section] = target
    _write_json_mapping(config_path, config)


def _scene_editable_rigid(value: Mapping[str, Any]) -> bool:
    shape = value.get("shape")
    return (
        isinstance(shape, Mapping)
        and shape.get("shape_type") == "Mesh"
        and isinstance(shape.get("fpath"), str)
        and bool(shape["fpath"])
    )


def _read_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return dict(value)


def _write_json_mapping(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _revision(
    value: SceneMaterialization,
    *,
    seed: int,
    edit_plan: dict[str, Any] | None,
) -> SceneRevision:
    return SceneRevision(
        source=value.scene_config_path,
        output_root=value.output_root,
        revision_id=scene_revision_id(value.scene_config_path),
        seed=seed,
        edit_plan=edit_plan,
        source_fingerprint=None,
    )


def _write_revision_audit(
    output_root: Path,
    *,
    revision_id: str,
    seed: int,
    edit_plan: Mapping[str, Any] | None,
    source_fingerprint: SceneSourceFingerprint | None = None,
) -> None:
    payload = {
        "schema_version": "embodichain.scene-revision-attempt/v1",
        "revision_id": revision_id,
        "seed": int(seed),
        "edit_plan": None if edit_plan is None else dict(edit_plan),
        "source_fingerprint": (
            None if source_fingerprint is None else source_fingerprint.to_dict()
        ),
    }
    (output_root / "scene_revision_attempt.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
