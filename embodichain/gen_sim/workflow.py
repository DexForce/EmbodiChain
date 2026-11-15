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
    """Load a published Task Engine plan after verifying every artifact."""
    root = Path(path).expanduser().resolve(strict=True)
    manifest = _load_json(root / "task_plan_manifest.json", "task plan manifest")
    if manifest.get("schema_version") != "embodichain.task-plan-artifact/v1":
        raise ValueError("Unsupported task plan artifact schema")
    _verify_inventory(root, manifest.get("artifacts"))
    requirements = _load_json(root / "scene_requirements.json", "scene requirements")
    if (
        requirements.get("schema_version") != "embodichain.scene-requirements/v1"
        or requirements.get("task_id") != manifest.get("task_id")
        or requirements.get("candidate_id") != manifest.get("candidate_id")
        or requirements.get("image_sha256") != manifest.get("image_sha256")
    ):
        raise ValueError("Scene requirements do not match the task plan")
    return {**manifest, "root": str(root), "scene_requirements": requirements}


def load_generated_scene(path: str | Path) -> dict[str, Any]:
    """Load one provider-neutral generated scene and verify its contained files."""
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
    return provider.submit(request, idempotency_key=idempotency_key)


def inspect_scene_generation(
    provider: SceneGenerationProvider, job: SceneJobRef
) -> SceneJobStatus:
    return provider.inspect(job)


def resume_scene_generation(
    provider: SceneGenerationProvider, job: SceneJobRef
) -> SceneJobStatus:
    return provider.resume(job)


def cancel_scene_generation(
    provider: SceneGenerationProvider, job: SceneJobRef
) -> CancellationReport:
    return provider.cancel(job)


def materialize_scene_generation(
    provider: SceneGenerationProvider, job: SceneJobRef, output: str | Path
) -> Path:
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
    """Bind and statically qualify one rigid E1 plan without publishing a deployment."""
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
    candidates = _load_json(plan_root / "task_candidates.json", "task candidates")
    candidate = next(
        (
            value
            for value in candidates.get("candidates", [])
            if value.get("candidate_id") == plan["candidate_id"]
        ),
        None,
    )
    steps = None if candidate is None else candidate.get("draft", {}).get("steps")
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
    )


def prepare_task_deployment(
    task_plan_dir: str | Path,
    generated_scene_dir: str | Path,
    qualification_dir: str | Path,
    runtime_config: str | Path,
    execution_policy: str | Path,
    output_dir: str | Path,
) -> Path:
    """Publish an execution-ready deployment from one accepted qualification."""
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
            "runtime_evidence_status": "not_run",
        },
        overwrite=False,
    )
    return deployment
