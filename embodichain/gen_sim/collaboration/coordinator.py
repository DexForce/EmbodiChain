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

"""End-to-end orchestration for the first three-agent collaboration phase."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
from typing import Any

from embodichain.gen_sim.action_engine.generation import (
    GeneratedConfigPaths,
    generate_action_engine_config,
)
from embodichain.gen_sim.action_engine.generation.artifacts import artifact_paths
from embodichain.gen_sim.action_engine.agent import ActionAgent
from embodichain.gen_sim.action_engine.domain.task_contracts import (
    TASK_CONTRACTS as ACTION_TASK_CONTRACTS,
)
from embodichain.gen_sim.action_engine.tasks import (
    GroundedTaskSpec,
    ground_instruction_draft,
)
from embodichain.gen_sim.task_engine import (
    TaskAgent,
    TaskCandidate,
    TaskCandidateSet,
    validate_task_candidate,
)
from embodichain.gen_sim.scene_bridge import FeasibilityBroker, FeasibilityReport

from .artifacts import (
    ArtifactTransaction,
    CollaborationArtifactPaths,
    collaboration_artifact_paths,
    write_collaboration_artifacts,
)
from .contracts import (
    GROUNDED_TASK_PLAN_SCHEMA,
    GroundedTaskPlan,
    RoleBindings,
    canonical_hash,
    validate_grounded_task_plan,
    validate_binding_report,
    validate_role_bindings,
)
from .scene_adapter import SceneAdaptation, SceneAdapter
from .scene_store import ScenePackageRef, SceneSourceRef

__all__ = [
    "CollaborationCoordinator",
    "Coordinator",
    "PreparationResult",
    "build_grounded_task_plan",
    "lower_task_candidate",
]


BundleGenerator = Callable[..., GeneratedConfigPaths]


def lower_task_candidate(
    candidate: Mapping[str, Any],
    reference_bindings: Mapping[str, Any],
    scene_objects: Sequence[Mapping[str, Any]],
    robot_profile: str,
) -> GroundedTaskSpec:
    """Lower a selected TaskCandidate across the Task/Action boundary."""
    normalized = validate_task_candidate(candidate)
    if reference_bindings.get("schema_version") is not None:
        role_bindings = validate_role_bindings(reference_bindings)
        if role_bindings["task_id"] != normalized["draft"]["task_id"]:
            raise ValueError("RoleBindings.task_id must match the TaskCandidate.")
        if role_bindings["candidate_id"] != normalized["candidate_id"]:
            raise ValueError("RoleBindings.candidate_id must match the TaskCandidate.")
        raw_bindings = role_bindings["reference_bindings"]
    else:
        raw_bindings = reference_bindings
    bindings = {
        str(reference_id): [str(uid) for uid in uids]
        for reference_id, uids in raw_bindings.items()
    }
    grounded = ground_instruction_draft(
        normalized["draft"]["task_id"],
        normalized["draft"]["instruction"],
        {"steps": normalized["draft"]["steps"]},
        scene_objects,
        robot_profile=robot_profile,
        reference_bindings=bindings,
    )
    _validate_lowered_success(normalized, bindings, grounded)
    return grounded


@dataclass(frozen=True)
class PreparationResult:
    """Published result of one Task -> Scene -> Action preparation attempt."""

    status: str
    output_dir: Path
    candidate_set: TaskCandidateSet
    adaptation: SceneAdaptation
    collaboration_artifacts: CollaborationArtifactPaths
    grounded_task_plan: GroundedTaskPlan | None = None
    action_graph: dict[str, Any] | None = None
    generated_paths: GeneratedConfigPaths | None = None
    feasibility_report: FeasibilityReport | None = None

    @property
    def bound(self) -> bool:
        return self.status == "bound"

    @property
    def selected_candidate_id(self) -> str | None:
        return self.adaptation.selected_candidate_id


class CollaborationCoordinator:
    """Run Task Agent, Scene Adapter, and Action Agent as one transaction."""

    def __init__(
        self,
        *,
        task_agent: TaskAgent | None = None,
        scene_adapter: SceneAdapter | None = None,
        action_agent: ActionAgent | None = None,
        bundle_generator: BundleGenerator = generate_action_engine_config,
        feasibility_broker: FeasibilityBroker | None = None,
    ) -> None:
        self.task_agent = task_agent or TaskAgent()
        self.scene_adapter = scene_adapter or SceneAdapter()
        self.action_agent = action_agent or ActionAgent()
        self.bundle_generator = bundle_generator
        self.feasibility_broker = feasibility_broker or FeasibilityBroker()

    def prepare(
        self,
        task_id: str,
        instruction: str,
        source: SceneSourceRef | ScenePackageRef | str | Path,
        output_dir: str | Path,
        *,
        model: str | None = None,
        candidate_count: int = 3,
        overwrite: bool = False,
        planning_mode: str = "offline",
        vlm_model: str | None = None,
        max_episodes: int | None = None,
        max_episode_steps: int | None = None,
        randomize_scene: bool = False,
        randomize_table_material: bool = False,
    ) -> PreparationResult:
        """Prepare and atomically publish a collaboration-compatible bundle.

        Ambiguous and unsatisfied scene adaptations are valid terminal results.
        They publish the complete audit hand-off but never publish a TaskSpec,
        SeedGraph, Gym configuration, or GroundedTaskPlan.
        """
        normalized_source = self._coerce_source(source)
        with ArtifactTransaction(output_dir, overwrite=overwrite) as transaction:
            staging_dir = transaction.staging_dir
            assert staging_dir is not None
            candidate_set = self.task_agent.generate(
                task_id,
                instruction,
                model=model,
                candidate_count=candidate_count,
            )
            adaptation = self.scene_adapter.adapt(candidate_set, normalized_source)
            status = str(adaptation.binding_report["status"])

            if status != "bound":
                write_collaboration_artifacts(
                    staging_dir,
                    candidate_set=candidate_set,
                    scene_manifest=None,
                    role_bindings=None,
                    binding_report=adaptation.binding_report,
                    static_scene_manifest=adaptation.static_scene_manifest,
                )
                published = transaction.commit()
                return PreparationResult(
                    status=status,
                    output_dir=published,
                    candidate_set=deepcopy(candidate_set),
                    adaptation=adaptation,
                    collaboration_artifacts=collaboration_artifact_paths(published),
                )

            selected = adaptation.selected_candidate
            raw_role_bindings = adaptation.role_bindings
            if selected is None or raw_role_bindings is None:
                raise ValueError(
                    "A bound SceneAdaptation must include a selected candidate "
                    "and RoleBindings."
                )
            feasibility_report = self._assess_feasibility(
                selected,
                raw_role_bindings,
                adaptation,
            )
            if (
                feasibility_report is not None
                and feasibility_report["status"] == "contradicted"
            ):
                adaptation, selected, raw_role_bindings, feasibility_report = (
                    self._fallback_feasible_candidate(
                        candidate_set,
                        adaptation,
                        selected,
                        raw_role_bindings,
                        feasibility_report,
                    )
                )
            if (
                feasibility_report is not None
                and feasibility_report["status"] == "contradicted"
            ):
                write_collaboration_artifacts(
                    staging_dir,
                    candidate_set=candidate_set,
                    scene_manifest=adaptation.scene_manifest,
                    role_bindings=raw_role_bindings,
                    binding_report=adaptation.binding_report,
                    static_scene_manifest=adaptation.static_scene_manifest,
                    feasibility_report=feasibility_report,
                )
                published = transaction.commit()
                return PreparationResult(
                    status="infeasible",
                    output_dir=published,
                    candidate_set=deepcopy(candidate_set),
                    adaptation=adaptation,
                    collaboration_artifacts=collaboration_artifact_paths(published),
                    feasibility_report=deepcopy(feasibility_report),
                )
            robot_profile = str(adaptation.scene_manifest["robot_profile"])
            grounded = lower_task_candidate(
                selected,
                raw_role_bindings,
                adaptation.prepared_scene.planner_objects,
                robot_profile,
            )
            role_bindings = validate_role_bindings(
                {
                    **deepcopy(raw_role_bindings),
                    "role_bindings": deepcopy(grounded.role_bindings),
                }
            )
            grounded_plan = build_grounded_task_plan(
                candidate=selected,
                task_spec=grounded.task_spec,
                scene_requirements=grounded.scene_requirements,
                scene_manifest=adaptation.scene_manifest,
                role_bindings=role_bindings,
                binding_report=adaptation.binding_report,
            )
            action_graph = self.action_agent.plan(grounded_plan)
            preflight = getattr(self.action_agent, "preflight", None)
            if callable(preflight):
                preflight(
                    action_graph,
                    scene_manifest=adaptation.scene_manifest,
                )

            generator_kwargs: dict[str, Any] = {
                "task_name": grounded_plan["task_id"],
                "task_spec": grounded_plan["task_spec"],
                "robot_profile": robot_profile,
                "source_scene_z_rotation_degrees": (
                    adaptation.prepared_scene.z_rotation_degrees
                ),
                "body_scale_policy": adaptation.prepared_scene.body_scale_policy,
                "body_scale": adaptation.prepared_scene.body_scale,
                "overwrite": False,
                "randomize_scene": randomize_scene,
                "randomize_table_material": randomize_table_material,
                "planning_mode": planning_mode,
                "vlm_model": vlm_model,
            }
            if max_episodes is not None:
                generator_kwargs["max_episodes"] = max_episodes
            if max_episode_steps is not None:
                generator_kwargs["max_episode_steps"] = max_episode_steps
            compatibility_input = staging_dir / ".collaboration_input"
            compatibility_input.mkdir()
            task_spec_path = compatibility_input / "task_spec.json"
            requirements_path = compatibility_input / "scene_requirements.json"
            _write_compatibility_input(task_spec_path, grounded.task_spec)
            _write_compatibility_input(
                requirements_path,
                grounded.scene_requirements,
            )
            generator_kwargs["task_spec"] = task_spec_path
            try:
                generated = self.bundle_generator(
                    adaptation.source_config_path,
                    staging_dir,
                    **generator_kwargs,
                )
            finally:
                shutil.rmtree(compatibility_input, ignore_errors=True)
            _require_matching_generated_graph(generated, action_graph)
            write_collaboration_artifacts(
                staging_dir,
                candidate_set=candidate_set,
                scene_manifest=adaptation.scene_manifest,
                role_bindings=role_bindings,
                binding_report=adaptation.binding_report,
                grounded_task_plan=grounded_plan,
                static_scene_manifest=adaptation.static_scene_manifest,
                feasibility_report=feasibility_report,
            )
            published = transaction.commit()
            return PreparationResult(
                status="bound",
                output_dir=published,
                candidate_set=deepcopy(candidate_set),
                adaptation=adaptation,
                collaboration_artifacts=collaboration_artifact_paths(published),
                grounded_task_plan=grounded_plan,
                action_graph=deepcopy(action_graph),
                generated_paths=artifact_paths(
                    published,
                    planning_mode=planning_mode,
                ),
                feasibility_report=deepcopy(feasibility_report),
            )

    def _fallback_feasible_candidate(
        self,
        candidate_set: Mapping[str, Any],
        adaptation: SceneAdaptation,
        selected: TaskCandidate,
        role_bindings: RoleBindings,
        report: FeasibilityReport,
    ) -> tuple[
        SceneAdaptation,
        TaskCandidate,
        RoleBindings,
        FeasibilityReport | None,
    ]:
        """Try other resolved semantic candidates after a static contradiction."""
        candidates = {
            str(candidate["candidate_id"]): candidate
            for candidate in candidate_set.get("candidates", ())
            if isinstance(candidate, Mapping) and candidate.get("candidate_id")
        }
        selected_id = str(selected["candidate_id"])
        for audit in adaptation.binding_report["candidates"]:
            candidate_id = str(audit["candidate_id"])
            if candidate_id == selected_id or audit["status"] != "resolved":
                continue
            candidate = candidates.get(candidate_id)
            alternative_bindings = adaptation.candidate_bindings.get(candidate_id)
            if candidate is None or alternative_bindings is None:
                continue
            alternative_report = self._assess_feasibility(
                candidate,
                alternative_bindings,
                adaptation,
            )
            if (
                alternative_report is not None
                and alternative_report["status"] == "contradicted"
            ):
                continue
            binding_report = validate_binding_report(
                {
                    **deepcopy(adaptation.binding_report),
                    "selected_candidate_id": candidate_id,
                    "selection_reason": (
                        "Selected the next resolved candidate after static "
                        f"feasibility contradicted {selected_id}."
                    ),
                }
            )
            chosen = deepcopy(candidate)
            updated = replace(
                adaptation,
                selected_candidate=chosen,
                role_bindings=deepcopy(alternative_bindings),
                binding_report=binding_report,
            )
            return (
                updated,
                chosen,
                deepcopy(alternative_bindings),
                alternative_report,
            )
        return adaptation, selected, role_bindings, report

    def _assess_feasibility(
        self,
        candidate: Mapping[str, Any],
        role_bindings: Mapping[str, Any],
        adaptation: SceneAdaptation,
    ) -> FeasibilityReport | None:
        """Intersect task requirements with scene and Action Engine capabilities."""
        manifest = adaptation.static_scene_manifest
        registry = getattr(self.action_agent, "registry", None)
        if manifest is None or registry is None:
            return None
        catalog = getattr(registry, "catalog", None)
        if not callable(catalog):
            return None
        return self.feasibility_broker.assess(
            candidate,
            role_bindings,
            manifest,
            capability_catalog=catalog(),
            task_actions={
                task_type: contract.core_actions
                for task_type, contract in ACTION_TASK_CONTRACTS.items()
            },
        )

    @staticmethod
    def _coerce_source(
        source: SceneSourceRef | ScenePackageRef | str | Path,
    ) -> SceneSourceRef | ScenePackageRef:
        if isinstance(source, (SceneSourceRef, ScenePackageRef)):
            return source
        path = Path(source).expanduser()
        text = str(source).strip().lower()
        if (
            not path.exists()
            and len(text) == 64
            and all(char in "0123456789abcdef" for char in text)
        ):
            return ScenePackageRef(text)
        return SceneSourceRef(path)


# Short public name used in the phase-one design document.
Coordinator = CollaborationCoordinator


def build_grounded_task_plan(
    *,
    candidate: Mapping[str, Any],
    task_spec: Mapping[str, Any],
    scene_requirements: Mapping[str, Any],
    scene_manifest: Mapping[str, Any],
    role_bindings: RoleBindings,
    binding_report: Mapping[str, Any],
) -> GroundedTaskPlan:
    """Assemble a validated plan with hashes over every authoritative hand-off."""
    draft = deepcopy(candidate["draft"])
    # A scene-ref quantifier may expand one draft step into several concrete
    # task instances. The grounded plan records the executable success terms,
    # while the selected TaskCandidate retains the pre-grounding SuccessSpec.
    success_spec = {
        **deepcopy(candidate["success_spec"]),
        "terms": [
            {
                "step_id": str(term["task_instance_id"]),
                "type": str(term["type"]),
            }
            for term in task_spec["success"]["terms"]
        ],
    }
    task = deepcopy(dict(task_spec))
    requirements = deepcopy(dict(scene_requirements))
    manifest = deepcopy(dict(scene_manifest))
    bindings = deepcopy(dict(role_bindings))
    report = deepcopy(dict(binding_report))
    base = {
        "schema_version": GROUNDED_TASK_PLAN_SCHEMA,
        "task_id": draft["task_id"],
        "instruction": draft["instruction"],
        "selected_candidate_id": candidate["candidate_id"],
        "task_draft": draft,
        "task_spec": task,
        "scene_requirements": requirements,
        "success_spec": success_spec,
        "scene_manifest": manifest,
        "role_bindings": bindings,
        "binding_report": report,
    }
    plan = {
        **base,
        "hashes": {
            "task_draft": canonical_hash(draft),
            "task_spec": canonical_hash(task),
            "scene_manifest": canonical_hash(manifest),
            "role_bindings": canonical_hash(bindings),
            "plan": canonical_hash(base),
        },
    }
    return validate_grounded_task_plan(plan)


def _validate_lowered_success(
    candidate: TaskCandidate,
    bindings: Mapping[str, list[str]],
    grounded: GroundedTaskSpec,
) -> None:
    success_by_step = {
        term["step_id"]: term["type"] for term in candidate["success_spec"]["terms"]
    }
    expected: list[str] = []
    multiplicity_by_step: dict[str, int] = {}
    for step in _topological_steps(candidate["draft"]["steps"]):
        selector = step["object"]
        multiplicity = 1
        if selector["kind"] == "scene_ref":
            multiplicity = len(bindings.get(f"{step['id']}.object", ()))
        elif selector["kind"] == "step_result":
            multiplicity = multiplicity_by_step[str(selector["step_id"])]
        multiplicity_by_step[str(step["id"])] = multiplicity
        expected.extend([success_by_step[step["id"]]] * multiplicity)
    actual = [term.get("type") for term in grounded.task_spec["success"]["terms"]]
    if actual != expected:
        raise ValueError(
            "Lowered TaskSpec success terms do not match the expanded SuccessSpec."
        )


def _topological_steps(
    steps: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    positions = {str(step["id"]): index for index, step in enumerate(steps)}
    pending = {str(step["id"]): set(step["depends_on"]) for step in steps}
    result: list[Mapping[str, Any]] = []
    emitted: set[str] = set()
    while len(result) < len(steps):
        ready = [
            step
            for step in steps
            if step["id"] not in emitted and pending[str(step["id"])] <= emitted
        ]
        if not ready:
            raise ValueError("TaskDraft step dependencies contain a cycle.")
        ready.sort(key=lambda step: positions[str(step["id"])])
        for step in ready:
            result.append(step)
            emitted.add(str(step["id"]))
    return result


def _require_matching_generated_graph(
    generated: GeneratedConfigPaths,
    expected: Mapping[str, Any],
) -> None:
    """Catch a compatibility-generator drift before publishing the bundle."""
    graph_path = getattr(generated, "seed_task_graph", None)
    if graph_path is None or not Path(graph_path).is_file():
        # Injected generators used by API consumers may publish by other means.
        # The Coordinator's independently planned graph remains authoritative.
        return
    try:
        actual = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Generated SeedGraph is unreadable: {graph_path}") from exc
    if canonical_hash(actual) != canonical_hash(expected):
        raise ValueError(
            "Legacy bundle generation produced a SeedGraph different from "
            "ActionAgent.plan."
        )


def _write_compatibility_input(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
