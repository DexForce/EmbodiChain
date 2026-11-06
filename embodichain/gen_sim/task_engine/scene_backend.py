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

"""Read-only existing-scene preparation for Task Engine deployments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .orchestration.grounding import GroundingCaller
from .orchestration.scene_adapter import CandidateSelection, SceneAdapter
from .orchestration.scene_source import (
    SceneSourceFingerprint,
    fingerprint_scene_source,
    scene_revision_id,
    verify_scene_source_fingerprint,
)
from .scene.final_inspection import FinalSceneInspection, inspect_final_scene
from .orchestration.workflow_artifacts import write_run_json
from .workflow_contracts import TaskRunRequest

__all__ = ["SceneAnalysis", "SceneEngineBackend", "SceneRevision"]


@dataclass(frozen=True)
class SceneAnalysis:
    """Source identity captured before semantic candidate selection."""

    input_kind: str
    source: Path
    blueprint: None
    source_fingerprint: SceneSourceFingerprint


@dataclass(frozen=True)
class SceneRevision:
    """An immutable existing scene; Task Engine never edits its source."""

    source: Path
    output_root: Path | None
    revision_id: str
    seed: int
    edit_plan: None
    source_fingerprint: SceneSourceFingerprint


class SceneEngineBackend:
    """Bind existing scenes without importing the scene-generation pipeline."""

    def analyze(
        self, request: TaskRunRequest, output_root: str | Path
    ) -> SceneAnalysis:
        """Fingerprint an existing scene before any model or simulator call.

        Args:
            request: Validated existing-scene request.
            output_root: Reserved audit location; analysis does not alter the source.

        Returns:
            Scene source and independently captured content identity.
        """
        if (
            request.get("image_path") is not None
            or request.get("scene_edit_prompt") is not None
        ):
            raise ValueError(
                "Task Engine accepts existing scenes only; image/edit inputs are unsupported."
            )
        if request.get("gym_project") is None:
            raise ValueError("Task Engine requires an existing scene.")
        source = Path(str(request["gym_project"])).resolve()
        return SceneAnalysis(
            "gym_project", source, None, fingerprint_scene_source(source)
        )

    def select(
        self,
        analysis: SceneAnalysis,
        candidate_set: Mapping[str, Any],
        scene_adapter: SceneAdapter,
        *,
        force_most_likely: bool,
        grounding_caller: GroundingCaller | None = None,
    ) -> CandidateSelection:
        """Resolve candidates against the unchanged native scene inventory.

        Args:
            analysis: Fingerprinted scene selection.
            candidate_set: Validated task candidates in declared order.
            scene_adapter: Configured semantic binding adapter.
            force_most_likely: Explicit source candidate-selection option.
            grounding_caller: Optional structured grounding provider override.

        Returns:
            Candidate bindings and the selected semantic task, if any.
        """
        adaptation = scene_adapter.adapt(
            candidate_set,
            analysis.source,
            force_most_likely=force_most_likely,
            grounding_caller=grounding_caller,
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
        """Revalidate and select the existing source without copying or editing it.

        Args:
            analysis: Previously fingerprinted source.
            request: Validated existing-scene request.
            output_root: Reserved revision audit location.
            seed: Audited source selection seed.

        Returns:
            Source revision after verifying that its declared inputs did not change.
        """
        if (
            request.get("image_path") is not None
            or request.get("scene_edit_prompt") is not None
        ):
            raise ValueError(
                "Task Engine accepts existing scenes only; image/edit inputs are unsupported."
            )
        verify_scene_source_fingerprint(analysis.source_fingerprint.to_dict())
        return SceneRevision(
            source=analysis.source,
            output_root=None,
            revision_id=scene_revision_id(analysis.source),
            seed=int(seed),
            edit_plan=None,
            source_fingerprint=analysis.source_fingerprint,
        )

    def inspect(
        self, revision: SceneRevision, output_path: str | Path
    ) -> FinalSceneInspection:
        """Inspect current geometry only after verifying the selected revision.

        Args:
            revision: Previously selected source identity.
            output_path: Run-owned destination for semantic inspection facts.

        Returns:
            Detached scene-inspection declaration from the shared geometry owner.
        """
        actual_revision_id = scene_revision_id(revision.source)
        if actual_revision_id != revision.revision_id:
            raise RuntimeError("Final scene changed before geometry inspection.")
        inspection = inspect_final_scene(
            revision.source, revision_id=actual_revision_id
        )
        write_run_json(output_path, inspection)
        return inspection
