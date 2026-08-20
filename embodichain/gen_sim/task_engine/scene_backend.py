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
    verify_scene_source_fingerprint,
)
from .workflow_contracts import TaskRunRequest, scene_input_kind

__all__ = [
    "SceneAnalysis",
    "SceneEngineBackend",
    "SceneRevision",
    "scene_blueprint_objects",
]


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
            materialization = materialize_blueprint(blueprint)
            edit_plan = None
            if edit_prompt is not None:
                edit_blueprint = analyze_edit(
                    output_root=root,
                    edit_prompt=str(edit_prompt),
                )
                edit_plan = edit_blueprint.scene_edit_plan.to_dict()
                materialization = materialize_edit(edit_blueprint)
            _write_revision_audit(root, seed=seed, edit_plan=edit_plan)
            return _revision(materialization, seed=seed, edit_plan=edit_plan)

        fingerprint = analysis.source_fingerprint
        assert fingerprint is not None
        if edit_prompt is None:
            verify_scene_source_fingerprint(fingerprint.to_dict())
            return SceneRevision(
                source=analysis.source,
                output_root=None,
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
        materialization = materialize_edit(edit_blueprint)
        if resolved.source_format == "legacy_gym_config":
            restore_locked_scene_entities(editable_root)
        verify_scene_source_fingerprint(fingerprint.to_dict())
        _write_revision_audit(
            editable_root,
            seed=seed,
            edit_plan=edit_plan,
            source_fingerprint=fingerprint,
        )
        return SceneRevision(
            source=materialization.scene_config_path,
            output_root=editable_root,
            seed=seed,
            edit_plan=edit_plan,
            source_fingerprint=fingerprint,
        )


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
    return output_root


def _revision(
    value: SceneMaterialization,
    *,
    seed: int,
    edit_plan: dict[str, Any] | None,
) -> SceneRevision:
    return SceneRevision(
        source=value.scene_config_path,
        output_root=value.output_root,
        seed=seed,
        edit_plan=edit_plan,
        source_fingerprint=None,
    )


def _write_revision_audit(
    output_root: Path,
    *,
    seed: int,
    edit_plan: Mapping[str, Any] | None,
    source_fingerprint: SceneSourceFingerprint | None = None,
) -> None:
    payload = {
        "schema_version": "embodichain.scene-revision-attempt/v1",
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
