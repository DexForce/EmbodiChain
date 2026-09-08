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

"""Task-owned preparation ending at the canonical Task Program boundary."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from embodichain.gen_sim.task_engine import (
    TaskAgent,
    TaskCandidateSet,
    validate_scene_output_separation,
    validate_task_candidate_set,
)
from embodichain.gen_sim.task_engine.semantic_graph import (
    SemanticTaskGraph,
    semantic_task_graph_hash,
)
from embodichain.gen_sim.task_engine.semantic_planner import (
    SemanticTaskPlanner,
    UnsupportedSemanticCapabilityError,
)
from embodichain.gen_sim.task_engine.task_program_bundle import (
    TaskProgramBundlePaths,
    generate_task_program_bundle,
)

from .artifacts import (
    ArtifactTransaction,
    TaskEngineArtifactPaths,
    task_engine_artifact_paths,
    write_preparation_failure,
    write_task_engine_artifacts,
)
from .scene_adapter import SceneAdaptation, SceneAdapter
from .scene_source import SceneSourceRef

__all__ = ["PreparationResult", "TaskEngineCoordinator"]

_PREPARATION_FAILURE_SCHEMA = "semantic_task_preparation_failure/v1"


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Published result of one Task -> Scene -> Semantic Skill preparation."""

    status: str
    output_dir: Path
    candidate_set: TaskCandidateSet
    adaptation: SceneAdaptation
    artifacts: TaskEngineArtifactPaths
    semantic_task_graph: SemanticTaskGraph | None = None
    generated_paths: TaskProgramBundlePaths | None = None
    feasibility_report: dict[str, Any] | None = None
    planning_attempts: tuple[dict[str, Any], ...] = ()
    unbound_action_plan: dict[str, Any] | None = None

    @property
    def bound(self) -> bool:
        """Return whether a fingerprint-bound Task Program bundle was published."""
        return self.status == "bound"

    @property
    def selected_candidate_id(self) -> str | None:
        """Return the selected candidate identity, if one was bound."""
        return self.adaptation.selected_candidate_id


class TaskEngineCoordinator:
    """Interpret and bind tasks without owning physical action construction."""

    def __init__(
        self,
        *,
        task_agent: TaskAgent | None = None,
        scene_adapter: SceneAdapter | None = None,
        semantic_planner: SemanticTaskPlanner | None = None,
    ) -> None:
        self.task_agent = task_agent or TaskAgent()
        self.scene_adapter = scene_adapter or SceneAdapter()
        self.semantic_planner = semantic_planner or SemanticTaskPlanner()

    def prepare(
        self,
        task_id: str,
        instruction: str,
        source: SceneSourceRef | str | Path,
        output_dir: str | Path,
        *,
        model: str | None = None,
        candidate_count: int = 3,
        overwrite: bool = False,
        planning_mode: str = "offline",
        gripper_model: str = "pgi",
        ik_solver: str = "auto",
        vlm_model: str | None = None,
        max_episodes: int | None = None,
        max_episode_steps: int | None = None,
        planner_policy: Mapping[str, Any] | None = None,
        randomize_scene: bool = False,
        randomize_table_material: bool = False,
        candidate_set: TaskCandidateSet | Mapping[str, Any] | None = None,
        force_most_likely: bool = False,
        final_inspection: Mapping[str, Any] | None = None,
        unbound_action_plan: Mapping[str, Any] | None = None,
    ) -> PreparationResult:
        """Publish a graph and configured Task Program as one transaction.

        Robot part routing, action options, grounding, physical effects, and
        command execution remain owned by the selected Task Program integration.
        Legacy CLI keywords are accepted for one migration window but cannot
        alter those lower-layer contracts.
        """
        del (
            gripper_model,
            ik_solver,
            vlm_model,
            planner_policy,
            randomize_scene,
            randomize_table_material,
        )
        normalized_source = self._coerce_source(source)
        validate_scene_output_separation(normalized_source.path, output_dir)
        with ArtifactTransaction(output_dir, overwrite=overwrite) as transaction:
            staging = transaction.staging_dir
            assert staging is not None
            if candidate_set is None:
                normalized_candidates = self.task_agent.generate(
                    task_id,
                    instruction,
                    model=model,
                    candidate_count=candidate_count,
                )
            else:
                normalized_candidates = validate_task_candidate_set(candidate_set)
                if normalized_candidates["task_id"] != str(task_id).strip():
                    raise ValueError("TaskCandidateSet.task_id must match task_id.")
                if normalized_candidates["instruction"] != str(instruction).strip():
                    raise ValueError(
                        "TaskCandidateSet.instruction must match instruction."
                    )

            adaptation_kwargs: dict[str, Any] = {"force_most_likely": force_most_likely}
            if final_inspection is not None:
                adaptation_kwargs["final_inspection"] = final_inspection
            adaptation = self.scene_adapter.adapt(
                normalized_candidates,
                normalized_source,
                **adaptation_kwargs,
            )
            status = str(adaptation.binding_report["status"])
            self._write_audit_artifacts(
                staging,
                candidate_set=normalized_candidates,
                adaptation=adaptation,
                final_inspection=final_inspection,
            )
            if status != "bound":
                published = transaction.commit()
                return PreparationResult(
                    status=status,
                    output_dir=published,
                    candidate_set=deepcopy(normalized_candidates),
                    adaptation=adaptation,
                    artifacts=task_engine_artifact_paths(published),
                )

            selected = adaptation.selected_candidate
            role_bindings = adaptation.role_bindings
            if selected is None or role_bindings is None:
                raise ValueError(
                    "A bound SceneAdaptation must include a selected candidate and "
                    "RoleBindings."
                )
            planning_attempt = {
                "candidate_id": str(selected["candidate_id"]),
                "planner_route": str(planning_mode),
                "status": "running",
            }
            try:
                graph = self.semantic_planner.plan(
                    selected,
                    role_bindings,
                    adaptation.prepared_scene.planner_objects,
                    planner_route=planning_mode,
                )
                graph, generated = generate_task_program_bundle(
                    graph,
                    adaptation.prepared_scene,
                    staging,
                    robot_profile=str(adaptation.scene_manifest["robot_profile"]),
                    max_episodes=max_episodes,
                    max_episode_steps=max_episode_steps,
                )
            except (TypeError, ValueError, UnsupportedSemanticCapabilityError) as exc:
                planning_attempt["status"] = "failed"
                planning_attempt["error"] = _error_record(exc)
                write_preparation_failure(
                    staging,
                    {
                        "schema_version": _PREPARATION_FAILURE_SCHEMA,
                        "task_id": str(normalized_candidates["task_id"]),
                        "status": "unsupported_semantic_capability",
                        "selected_candidate_id": str(selected["candidate_id"]),
                        "attempts": [deepcopy(planning_attempt)],
                    },
                )
                published = transaction.commit()
                return PreparationResult(
                    status="planning_failed",
                    output_dir=published,
                    candidate_set=deepcopy(normalized_candidates),
                    adaptation=adaptation,
                    artifacts=task_engine_artifact_paths(published),
                    planning_attempts=(deepcopy(planning_attempt),),
                )

            planning_attempt.update(
                {
                    "status": "preflight_succeeded",
                    "semantic_task_graph_hash": semantic_task_graph_hash(graph),
                    "semantic_call_count": len(graph["nodes"]),
                    "integration_fingerprint": graph["integration_fingerprint"],
                }
            )
            _write_json(staging / "planner_report.json", planning_attempt)
            self._write_selected_candidate_artifacts(staging, selected)
            published = transaction.commit()
            return PreparationResult(
                status="bound",
                output_dir=published,
                candidate_set=deepcopy(normalized_candidates),
                adaptation=adaptation,
                artifacts=task_engine_artifact_paths(published),
                semantic_task_graph=deepcopy(graph),
                generated_paths=_published_paths(generated, published),
                planning_attempts=(deepcopy(planning_attempt),),
                unbound_action_plan=(
                    None
                    if unbound_action_plan is None
                    else deepcopy(dict(unbound_action_plan))
                ),
            )

    def _coerce_source(self, source: SceneSourceRef | str | Path) -> SceneSourceRef:
        if isinstance(source, SceneSourceRef):
            return source
        return SceneSourceRef(
            source,
            robot_profile=self.scene_adapter.robot_profile,
        )

    @staticmethod
    def _write_audit_artifacts(
        output_dir: Path,
        *,
        candidate_set: TaskCandidateSet,
        adaptation: SceneAdaptation,
        final_inspection: Mapping[str, Any] | None,
    ) -> None:
        write_task_engine_artifacts(
            output_dir,
            candidate_set=candidate_set,
            scene_manifest=(
                adaptation.scene_manifest
                if adaptation.binding_report["status"] == "bound"
                else None
            ),
            role_bindings=adaptation.role_bindings,
            binding_report=adaptation.binding_report,
            static_scene_manifest=adaptation.static_scene_manifest,
            conservative_scene_graph=adaptation.conservative_scene_graph,
            final_scene_inspection=final_inspection,
        )

    @staticmethod
    def _write_selected_candidate_artifacts(
        output_dir: Path,
        selected: Mapping[str, Any],
    ) -> None:
        _write_json(output_dir / "task_draft.json", selected["draft"])
        _write_json(output_dir / "scene_request.json", selected["scene_request"])
        _write_json(output_dir / "success_spec.json", selected["success_spec"])


def _published_paths(
    paths: TaskProgramBundlePaths,
    published: Path,
) -> TaskProgramBundlePaths:
    def target(path: Path) -> Path:
        return published / path.relative_to(paths.root)

    return TaskProgramBundlePaths(
        root=published,
        deployment=target(paths.deployment),
        program=target(paths.program),
        integration=target(paths.integration),
        scene=target(paths.scene),
        embodiment=target(paths.embodiment),
        execution_policy=target(paths.execution_policy),
        semantic_task_graph=target(paths.semantic_task_graph),
        integration_fingerprint=target(paths.integration_fingerprint),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _error_record(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}
