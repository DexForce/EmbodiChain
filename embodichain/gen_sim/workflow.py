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

"""Public, artifact-based GenSim operations shared by CLI and future adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .scene_providers import (
    CancellationReport,
    SceneGenerationProvider,
    SceneGenerationRequest,
    SceneJobRef,
    SceneJobStatus,
)
from .services.runtime import GenSimServices, service_entry
from .task_engine._image_run import plan_image_task

__all__ = [
    "cancel_scene_generation",
    "inspect_scene_generation",
    "load_generated_scene",
    "load_task_plan",
    "materialize_scene_generation",
    "plan_image_task",
    "prepare_task_deployment",
    "qualify_task_scene",
    "resume_scene_generation",
    "revise_task_plan",
    "submit_scene_generation",
]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _artifact_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact path must be nonempty")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Invalid artifact path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"Invalid artifact path: {relative}")
    return path


def _verify_inventory(root: Path, inventory: Mapping[str, Any]) -> None:
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("Artifact inventory must be a nonempty mapping")
    for relative, expected in inventory.items():
        path = _artifact_path(root, relative)
        if (
            not isinstance(expected, str)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Artifact hash mismatch: {relative}")


def load_task_plan(path: str | Path) -> dict[str, Any]:
    """Load a published Task Engine plan after verifying every artifact.

    Args:
        path: Directory containing a versioned task-plan manifest.

    Returns:
        Verified manifest, resolved root and structured scene requirements.
    """
    root = Path(path).expanduser().resolve(strict=True)
    manifest = _load_json(root / "task_plan_manifest.json", "task plan manifest")
    version = manifest.get("schema_version")
    if version not in {
        "embodichain.task-plan-artifact/v1",
        "embodichain.task-plan-artifact/v2",
    }:
        raise ValueError("Unsupported task plan artifact schema")
    _verify_inventory(root, manifest.get("artifacts"))
    requirements = _load_json(root / "scene_requirements.json", "scene requirements")
    if version == "embodichain.task-plan-artifact/v2":
        required_files = {
            "task_spec_v1.json",
            "task_spec.json",
            "semantic_task_graph.json",
            "plan_coverage.json",
            "image_grounding.json",
            "semantic_graph_validation.json",
            "scene_requirements.json",
        }
        if required_files - manifest["artifacts"].keys():
            raise ValueError("Incomplete task-plan inventory")
        from .task_spec import TaskSpec, canonical_hash
        from .task_spec.publication import (
            decode_task_document,
            project_scene_requirements,
        )

        from .task_engine.agent import derive_scene_request
        from .task_engine.contracts import TASK_DRAFT_SCHEMA, validate_task_draft
        from .task_engine.semantic_plan_validation import (
            _validate_interpreted_plan,
            _validate_clause_coverage,
        )

        normative = TaskSpec.from_dict(
            _load_json(root / "task_spec.json", "normative source")
        )
        document = decode_task_document(
            _load_json(root / "task_spec_v1.json", "task document"),
            normative,
        )
        graph = _load_json(root / "semantic_task_graph.json", "semantic graph")
        coverage = _load_json(root / "plan_coverage.json", "plan coverage")
        validation = _load_json(
            root / "semantic_graph_validation.json", "coverage validation"
        )
        draft = validate_task_draft(
            {
                "schema_version": TASK_DRAFT_SCHEMA,
                "task_id": document["task_id"],
                "instruction": document["inputs"]["instruction"],
                "steps": document["interpretation"]["steps"],
            }
        )
        source = validation["source"]
        if source["grounding"] != _load_json(
            root / "image_grounding.json", "image grounding"
        ):
            raise ValueError("Grounding differs from the validated plan")
        _validate_clause_coverage(document, source["requirements"])
        declared_coverage = _validate_interpreted_plan(
            graph,
            draft,
            derive_scene_request(draft)["references"],
            source["grounding"],
            source["annotations"],
            source["requirements"],
            source["bindings"],
        )
        if (
            any(
                document[k] != manifest[k]
                for k in ("spec_digest", "task_semantic_hash", "generalization_hash")
            )
            or coverage["spec_digest"] != document["spec_digest"]
            or manifest["task_spec_sha256"] != validation["task_spec_digest"]
            or validation["graph_digest"] != coverage["graph_digest"]
            or declared_coverage != validation["coverage"]
            or validation["task_spec_digest"] != document["source_spec_digest"]
            or coverage["milestone_groups"] != normative.metadata["milestone_sources"]
            or coverage["goal_groups"] != normative.metadata["goal_sources"]
            or coverage["role_entities"]
            != {
                role: sorted(
                    {
                        uid
                        for ref in refs
                        for uid in validation["source"]["grounding"]["bindings"][ref]
                    }
                )
                for role, refs in normative.metadata["role_references"].items()
            }
            or document["inputs"]["image"]["ref"] not in manifest["artifacts"]
            or document["inputs"]["instruction"] != manifest["instruction"]
            or document["task_id"] != manifest["task_id"]
            or document["inputs"]["image"]["sha256"] != manifest["image_sha256"]
            or _file_hash(_artifact_path(root, document["inputs"]["image"]["ref"]))
            != manifest["image_sha256"]
            or canonical_hash(graph) != coverage["graph_digest"]
            or coverage["graph_digest"] != manifest["semantic_graph_sha256"]
            or project_scene_requirements(document) != requirements
        ):
            raise ValueError("Task plan spec, graph or scene projection mismatch")
        return {
            **manifest,
            "root": str(root),
            "scene_requirements": requirements,
            "task_document": document,
        }
    if (
        requirements.get("schema_version") != "embodichain.scene-requirements/v1"
        or requirements.get("task_id") != manifest.get("task_id")
        or requirements.get("candidate_id") != manifest.get("candidate_id")
        or requirements.get("image_sha256") != manifest.get("image_sha256")
    ):
        raise ValueError("Scene requirements do not match the task plan")
    return {**manifest, "root": str(root), "scene_requirements": requirements}


def revise_task_plan(
    parent: str | Path,
    proposal: Mapping[str, Any],
    output: str | Path,
    *,
    expected_spec_digest: str,
    expected_graph_digest: str,
) -> Path:
    """Publish a fresh graph revision against the parent's frozen task and grounding.

    The proposal uses the ordinary planner response schema (nodes and quoted
    requirements). Both expected identities are checked before publication.

    Args:
        parent: Directory containing the source task plan.
        proposal: Canonical planner response with nodes and quoted requirements.
        output: Fresh directory for the graph revision.
        expected_spec_digest: Frozen task identity selected by the caller.
        expected_graph_digest: Source graph identity selected by the caller.

    Returns:
        Published revision directory retaining the parent task specification.

    Raises:
        ValueError: The revisions are stale or the proposal fails validation.
    """
    import shutil
    from .task_spec import canonical_hash
    from .task_engine.image_planning import _assemble
    from .task_engine._image_run import _image_preview
    from .task_engine.semantic_plan_validation import (
        validate_semantic_plan,
        _validate_clause_coverage,
    )
    from .task_engine.orchestration.workflow_artifacts import (
        WorkflowRunArtifacts,
        write_run_json,
    )

    plan = load_task_plan(parent)
    if (
        plan.get("spec_digest") != expected_spec_digest
        or plan["semantic_graph_sha256"] != expected_graph_digest
    ):
        raise ValueError("Stale task plan revision")
    root = Path(plan["root"])
    target = Path(output).resolve()
    if target.is_relative_to(root):
        raise ValueError("Select a revision directory outside the parent plan")
    report = _load_json(root / "semantic_graph_validation.json", "plan validation")
    source = report["source"]
    graph, annotations, bindings = _assemble(dict(proposal), source["candidate"])
    source.update(
        annotations=annotations,
        bindings=bindings,
        requirements=proposal["requirements"],
    )
    _validate_clause_coverage(plan["task_document"], source["requirements"])
    report.update(
        graph_digest=canonical_hash(graph),
        coverage=validate_semantic_plan(graph, **source),
    )
    pairing = _load_json(root / "plan_coverage.json", "pairing")
    pairing["graph_digest"] = report["graph_digest"]
    revised = {
        "semantic_task_graph": graph,
        "semantic_graph_validation": report,
        "binding_requirements": bindings,
        "instruction_requirements": proposal["requirements"],
        "plan_coverage": pairing,
    }
    manifest = _load_json(root / "task_plan_manifest.json", "manifest")
    with WorkflowRunArtifacts(target) as publication:
        for name in manifest["artifacts"]:
            origin = _artifact_path(root, name)
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
        for key, value in revised.items():
            write_run_json(target / (key + ".json"), value)
        preview = {
            **revised,
            **{
                k: _load_json(target / (k + ".json"), k)
                for k in ("task_spec", "task_candidates", "image_grounding")
            },
        }
        summary = _load_json(target / "run_manifest.json", "summary")
        summary.update(
            pending_bindings=len(bindings),
            semantic_status=report["coverage"]["semantic_status"],
            visual_status=report["coverage"]["visual_status"],
        )
        write_run_json(target / "run_manifest.json", summary)
        (target / "semantic_task_graph.html").write_text(
            _image_preview(preview, _source_image(target), summary)
        )
        manifest.update(
            parent_plan_sha256=_file_hash(root / "task_plan_manifest.json"),
            semantic_graph_sha256=pairing["graph_digest"],
            semantic_status=summary["semantic_status"],
            visual_status=summary["visual_status"],
            artifacts={
                name: _file_hash(target / name) for name in manifest["artifacts"]
            },
        )
        write_run_json(target / "task_plan_manifest.json", manifest, overwrite=False)
        load_task_plan(target)
        publication.commit()
    return target


def load_generated_scene(path: str | Path) -> dict[str, Any]:
    """Load one provider-neutral generated scene and verify its contained files.

    Args:
        path: Directory containing the generated-scene manifest and assets.

    Returns:
        Verified scene manifest with its resolved root directory.
    """
    root = Path(path).expanduser().resolve(strict=True)
    manifest = _load_json(root / "generated_scene.json", "generated scene manifest")
    if manifest.get("schema_version") != "embodichain.generated-scene/v1":
        raise ValueError("Unsupported generated scene schema")
    if manifest.get("coordinates") != {
        "editable_frame": "Y-up",
        "runtime_frame": "Z-up",
        "units": "m",
    }:
        raise ValueError(
            "Generated scene must declare Y-up source and Z-up runtime meters"
        )
    _verify_inventory(root, manifest.get("artifacts"))
    runtime = manifest.get("runtime_asset")
    if not isinstance(runtime, Mapping):
        raise TypeError("Generated scene has no runtime asset")
    runtime_path = _artifact_path(root, runtime.get("path"))
    if runtime_path.suffix.lower() != ".usdz" or hashlib.sha256(
        runtime_path.read_bytes()
    ).hexdigest() != runtime.get("sha256"):
        raise ValueError("Generated scene runtime USDZ differs from its declaration")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Generated scene must contain objects")
    identifiers = []
    prim_paths = []
    for item in objects:
        if not isinstance(item, Mapping) or item.get("kind") != "rigid":
            raise ValueError("Generated scene v1 accepts rigid objects only")
        uid = item.get("uid")
        prim = item.get("source_prim_path")
        if (
            not isinstance(uid, str)
            or not uid
            or not isinstance(prim, str)
            or not prim.startswith("/")
        ):
            raise ValueError("Generated scene object identity is invalid")
        if any(
            not isinstance(item.get(field), str) or not item[field].strip()
            for field in ("name", "description", "category")
        ):
            raise ValueError(f"Generated scene object metadata is invalid for {uid}")
        pose = item.get("pose")
        if not isinstance(pose, Mapping) or set(pose) != {
            "position",
            "rotation_euler_xyz_degrees",
            "scale",
        }:
            raise ValueError(f"Generated scene pose is invalid for {uid}")
        for field in ("position", "rotation_euler_xyz_degrees", "scale"):
            values = pose[field]
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in values
                )
            ):
                raise ValueError(f"Generated scene {field} is invalid for {uid}")
        if not isinstance(item.get("physics"), Mapping):
            raise TypeError(f"Generated scene physics is invalid for {uid}")
        mesh = _artifact_path(root, item.get("mesh_path"))
        if hashlib.sha256(mesh.read_bytes()).hexdigest() != item.get("mesh_sha256"):
            raise ValueError(f"Generated scene mesh differs for {uid}")
        identifiers.append(uid)
        prim_paths.append(prim)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Generated scene object UIDs must be unique")
    if len(prim_paths) != len(set(prim_paths)):
        raise ValueError("Generated scene source prim paths must be unique")
    graph = manifest.get("graph")
    if not isinstance(graph, Mapping):
        raise TypeError("Generated scene has no graph")
    nodes = graph.get("nodes")
    if (
        not isinstance(nodes, list)
        or len(nodes) != len(identifiers)
        or {node.get("object_id") for node in nodes if isinstance(node, Mapping)}
        != set(identifiers)
    ):
        raise ValueError("Generated scene graph must cover every object exactly")
    return {**manifest, "root": str(root)}


def submit_scene_generation(
    provider: SceneGenerationProvider,
    request: SceneGenerationRequest,
    *,
    idempotency_key: str,
) -> SceneJobRef:
    """Submit a scene request through the selected provider.

    Args:
        provider: Selected scene-generation lifecycle owner.
        request: Source and requirements bound by content identity.
        idempotency_key: Provider-scoped submission identity.

    Returns:
        Provider-owned durable job reference.
    """
    return provider.submit(request, idempotency_key=idempotency_key)


def inspect_scene_generation(
    provider: SceneGenerationProvider, job: SceneJobRef
) -> SceneJobStatus:
    """Read the selected provider's current job status.

    Args:
        provider: Job lifecycle owner.
        job: Previously submitted job reference.

    Returns:
        Current provider-reported job status and artifact reference.
    """
    return provider.inspect(job)


def resume_scene_generation(
    provider: SceneGenerationProvider, job: SceneJobRef
) -> SceneJobStatus:
    """Reconcile an existing job through its selected provider.

    Args:
        provider: Job lifecycle owner.
        job: Durable job reference to reconcile.

    Returns:
        Reconciled provider status.
    """
    return provider.resume(job)


def cancel_scene_generation(
    provider: SceneGenerationProvider, job: SceneJobRef
) -> CancellationReport:
    """Request cancellation and return the provider's actual acknowledgement.

    Args:
        provider: Job lifecycle owner.
        job: Durable job reference to cancel.

    Returns:
        Acknowledged stop state and possible partial effects.
    """
    return provider.cancel(job)


def materialize_scene_generation(
    provider: SceneGenerationProvider, job: SceneJobRef, output: str | Path
) -> Path:
    """Materialize a completed provider job at the requested output location.

    Args:
        provider: Job and materialization owner.
        job: Successful durable job reference.
        output: Destination selected by the caller.

    Returns:
        Existing resolved directory returned by the provider.
    """
    path = provider.materialize(job, Path(output).expanduser().absolute())
    return Path(path).resolve(strict=True)


def _source_image(root: Path) -> Path:
    matches = sorted(root.glob("source_image.*"))
    if len(matches) != 1:
        raise ValueError("Task plan must contain exactly one source image")
    return matches[0]


def _file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _publish_qualification(
    output: Path,
    *,
    status: str,
    inputs: dict[str, str],
    issues: list[dict[str, Any]],
    bindings: dict[str, Any] | None = None,
    placements: dict[str, Any] | None = None,
    role_report: dict[str, Any] | None = None,
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    from .task_engine.orchestration.workflow_artifacts import write_run_json

    write_run_json(
        output / "scene_qualification.json",
        {
            "schema_version": "embodichain.scene-qualification/v1",
            "status": status,
            **inputs,
            "issues": issues,
            "bindings": bindings,
            "placements": placements or {},
            "roles": role_report or {},
        },
        overwrite=False,
    )
    return output


@service_entry
def qualify_task_scene(
    task_plan_dir: str | Path,
    generated_scene_dir: str | Path,
    runtime_config: str | Path,
    execution_policy: str | Path,
    output_dir: str | Path,
    *,
    services: GenSimServices | None = None,
) -> Path:
    """Bind and statically qualify one rigid E1 plan.

    Args:
        task_plan_dir: Verified Task Engine publication.
        generated_scene_dir: Provider-neutral generated scene publication.
        runtime_config: Trusted robot and physical runtime configuration.
        execution_policy: Trusted execution policy configuration.
        output_dir: Fresh qualification-report directory.
        services: Optional caller-owned services for logical role binding.

    Returns:
        Directory containing the qualification verdict and binding diagnostics.
    """
    from ._run_binding import (
        _generated_scene_data,
        _ground_ids,
        _load_runtime,
        _placement,
    )
    from ._run_config import _RunCfg

    plan = load_task_plan(task_plan_dir)
    scene = load_generated_scene(generated_scene_dir)
    plan_root, scene_root = Path(plan["root"]), Path(scene["root"])
    runtime_path = Path(runtime_config).expanduser().resolve(strict=True)
    policy_path = Path(execution_policy).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().absolute()
    inputs = {
        "task_plan_sha256": _file_hash(plan_root / "task_plan_manifest.json"),
        "scene_sha256": _file_hash(scene_root / "generated_scene.json"),
        "runtime_sha256": _file_hash(runtime_path),
        "policy_sha256": _file_hash(policy_path),
        "attempt_id": scene.get("attempt_id"),
        "candidate_id": scene.get("scene_id"),
        "parent_candidate_id": scene.get("parent_candidate_id"),
        "requirements_sha256": scene.get("requirements_sha256"),
        "feedback_sha256": scene.get("feedback_sha256"),
    }
    requirements_hash = _file_hash(plan_root / "scene_requirements.json")
    if scene.get("requirements_sha256") != requirements_hash:
        return _publish_qualification(
            output,
            status="rejected",
            inputs=inputs,
            issues=[
                {
                    "code": "scene_requirements_mismatch",
                    "owner": "scene_generation",
                    "retry_scope": "regenerate_scene",
                    "message": "Generated scene is bound to different task requirements.",
                }
            ],
        )
    if scene.get("source", {}).get("image_sha256") != plan["image_sha256"]:
        return _publish_qualification(
            output,
            status="rejected",
            inputs=inputs,
            issues=[
                {
                    "code": "source_image_mismatch",
                    "owner": "scene_generation",
                    "retry_scope": "regenerate_scene",
                    "message": "Generated scene and Task Plan use different source images.",
                }
            ],
        )
    if "task_document" in plan:
        goals = plan["task_document"]["objectives"]["goal"]
        placements = [p for p in goals if p["predicate"] == "semantic_goal"]
        steps = [
            {**p.get("arguments", {}), "relation": p.get("relation")}
            for p in placements
        ]
        if len(plan["task_document"]["objectives"]["milestones"]) != 1:
            steps = []
    else:
        candidates = _load_json(plan_root / "task_candidates.json", "task candidates")
        candidate = next(
            (
                c
                for c in candidates["candidates"]
                if c["candidate_id"] == plan["candidate_id"]
            ),
            {},
        )
        steps = candidate.get("draft", {}).get("steps")
    if (
        not isinstance(steps, list)
        or len(steps) != 1
        or steps[0].get("task_type") != "E1"
        or steps[0].get("relation") not in {"on", "inside"}
    ):
        return _publish_qualification(
            output,
            status="rejected",
            inputs=inputs,
            issues=[
                {
                    "code": "unsupported_task_scope",
                    "owner": "task_plan",
                    "retry_scope": "none",
                    "message": "Generated-scene workflow v1 requires one E1 on/inside step.",
                }
            ],
        )
    cfg = _RunCfg(
        image=_source_image(plan_root),
        instruction=plan["instruction"],
        output=output,
        runtime=runtime_path,
        execution_policy=policy_path,
        task_id=plan["task_id"],
    )
    try:
        runtime, policy, profile = _load_runtime(cfg)
    except (OSError, TypeError, ValueError) as error:
        message = str(error)
        policy_error = "execution_policy" in message
        return _publish_qualification(
            output,
            status="rejected",
            inputs=inputs,
            issues=[
                {
                    "code": "runtime_or_policy_incompatible",
                    "owner": "execution_policy" if policy_error else "runtime",
                    "retry_scope": (
                        "change_policy" if policy_error else "change_runtime"
                    ),
                    "message": message,
                }
            ],
        )
    _, entries = _generated_scene_data(scene_root, scene)
    validation = _load_json(
        plan_root / "semantic_graph_validation.json", "semantic graph validation"
    )
    graph = _load_json(plan_root / "semantic_task_graph.json", "semantic task graph")
    grounding = _load_json(plan_root / "image_grounding.json", "image grounding")
    try:
        bindings = _ground_ids(
            grounding,
            validation["source"]["annotations"],
            graph,
            entries,
            profile,
            services=services,
        )
    except ValueError as error:
        return _publish_qualification(
            output,
            status="ambiguous",
            inputs=inputs,
            issues=[
                {
                    "code": "scene_binding_failed",
                    "owner": "scene_generation",
                    "retry_scope": "regenerate_scene",
                    "message": str(error),
                }
            ],
        )
    role_report = {}
    if "task_document" in plan:
        members = _load_json(plan_root / "plan_coverage.json", "coverage")[
            "role_entities"
        ]
        native = {item["uid"]: item for item in scene["objects"]}
        issues = []
        for role in plan["task_document"]["configuration"]["roles"]:
            role_id = role["id"]
            selected = [bindings["objects"].get(e) for e in members[role_id]]
            role_report[role_id] = {"members": selected, "verified": [], "pending": []}
            for uid in selected:
                asset = native.get(uid, {})
                for capability in role["affordances"]:
                    if "affordances" in asset and capability in asset["affordances"]:
                        role_report[role_id]["verified"].append(capability)
                    else:
                        role_report[role_id]["pending"].append(capability)
                        issues.append(
                            {
                                "code": "missing_asset_capability",
                                "role_id": role_id,
                                "expected": capability,
                                "measured": asset.get("affordances"),
                                "uid": uid,
                            }
                        )
                for key, value in role["properties"].items():
                    if asset.get("properties", {}).get(key) != value:
                        issues.append(
                            {
                                "code": "asset_property_mismatch",
                                "role_id": role_id,
                                "expected": {key: value},
                                "measured": asset.get("properties", {}).get(key),
                                "uid": uid,
                            }
                        )
        if issues:
            return _publish_qualification(
                output,
                status="rejected",
                inputs=inputs,
                issues=issues,
                bindings=bindings,
                role_report=role_report,
            )
    placements = {}
    occupied: dict[str, list[tuple[float, float, float]]] = {}
    annotations = validation["source"]["annotations"]
    try:
        for node in graph["nodes"]:
            call = node["call"]
            if call["kind"] != "place" or not ({"on", "inside"} & call.keys()):
                continue
            subject = call["object"]
            reference = annotations[node["id"]]["reference"]
            pose, displacement = _placement(
                entries[bindings["objects"][subject]],
                entries[bindings["objects"][reference]],
                scene_root,
                occupied.setdefault(reference, []),
            )
            placements[node["id"]] = {
                "object_target_pose": pose,
                "displacement": displacement,
            }
    except ValueError as error:
        return _publish_qualification(
            output,
            status="rejected",
            inputs=inputs,
            issues=[
                {
                    "code": "placement_infeasible",
                    "owner": "scene_generation",
                    "retry_scope": "regenerate_scene",
                    "message": str(error),
                }
            ],
        )
    del runtime, policy
    return _publish_qualification(
        output,
        status="accepted",
        inputs=inputs,
        issues=[],
        bindings=bindings,
        placements=placements,
        role_report=role_report,
    )


def prepare_task_deployment(
    task_plan_dir: str | Path,
    generated_scene_dir: str | Path,
    qualification_dir: str | Path,
    runtime_config: str | Path,
    execution_policy: str | Path,
    output_dir: str | Path,
    *,
    expected_revisions: Mapping[str, str] | None = None,
) -> Path:
    """Publish an execution-ready deployment from one accepted qualification.

    Args:
        task_plan_dir: Verified Task Engine publication.
        generated_scene_dir: Qualified generated scene publication.
        qualification_dir: Matching accepted qualification report.
        runtime_config: Trusted robot and physical runtime configuration.
        execution_policy: Trusted execution policy configuration.
        output_dir: Fresh deployment publication directory.
        expected_revisions: Optional caller-selected input digests.

    Returns:
        Path to the runnable deployment configuration.

    Raises:
        ValueError: Qualification or expected input identities do not match.
    """
    from ._run_binding import _bind_deployment, _load_runtime
    from ._run_config import _RunCfg
    from .task_engine.orchestration.workflow_artifacts import write_run_json

    plan = load_task_plan(task_plan_dir)
    scene = load_generated_scene(generated_scene_dir)
    plan_root, scene_root = Path(plan["root"]), Path(scene["root"])
    qualification_root = Path(qualification_dir).expanduser().resolve(strict=True)
    qualification = _load_json(
        qualification_root / "scene_qualification.json", "scene qualification"
    )
    runtime_path = Path(runtime_config).expanduser().resolve(strict=True)
    policy_path = Path(execution_policy).expanduser().resolve(strict=True)
    expected = {
        "task_plan_sha256": _file_hash(plan_root / "task_plan_manifest.json"),
        "scene_sha256": _file_hash(scene_root / "generated_scene.json"),
        "runtime_sha256": _file_hash(runtime_path),
        "policy_sha256": _file_hash(policy_path),
    }
    revisions = {
        **expected,
        "spec_digest": plan.get("spec_digest"),
        "graph_digest": plan.get("semantic_graph_sha256"),
    }
    if expected_revisions is not None and any(
        key not in revisions or revisions[key] != value
        for key, value in expected_revisions.items()
    ):
        raise ValueError("Stale task deployment input revision")
    if (
        qualification.get("schema_version") != "embodichain.scene-qualification/v1"
        or qualification.get("status") != "accepted"
        or any(qualification.get(key) != value for key, value in expected.items())
        or not isinstance(qualification.get("bindings"), dict)
    ):
        raise ValueError(
            "Deployment preparation requires its matching accepted qualification"
        )
    output = Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    cfg = _RunCfg(
        image=_source_image(plan_root),
        instruction=plan["instruction"],
        output=output,
        runtime=runtime_path,
        execution_policy=policy_path,
        task_id=plan["task_id"],
    )
    runtime, policy, profile = _load_runtime(cfg)
    deployment = _bind_deployment(
        cfg,
        scene_root,
        plan_root,
        runtime,
        policy,
        profile,
        services=None,
        generated_scene=scene,
        resolved_bindings=qualification["bindings"],
        resolved_placements=qualification.get("placements", {}),
    )
    write_run_json(
        output / "deployment_manifest.json",
        {
            "schema_version": "embodichain.prepared-task-deployment/v1",
            **expected,
            "qualification_sha256": _file_hash(
                qualification_root / "scene_qualification.json"
            ),
            "deployment": str(deployment.relative_to(output)),
            "deployment_sha256": _file_hash(deployment),
            "binding_report_sha256": _file_hash(output / "binding_report.json"),
            "preflight_report_sha256": _file_hash(output / "preflight_report.json"),
            "execution_ready": True,
            "input_revisions": revisions,
            "evidence_requirements": [
                {"predicate_id": p["id"], "scope": scope}
                for scope, predicates in (
                    (
                        "final",
                        plan.get("task_document", {})
                        .get("objectives", {})
                        .get("goal", []),
                    ),
                    (
                        "recorded",
                        [
                            p
                            for m in plan.get("task_document", {})
                            .get("objectives", {})
                            .get("milestones", [])
                            for p in m["achieve"]
                        ],
                    ),
                )
                for p in predicates
            ],
            "runtime_evidence_status": "not_run",
        },
        overwrite=False,
    )
    return deployment
