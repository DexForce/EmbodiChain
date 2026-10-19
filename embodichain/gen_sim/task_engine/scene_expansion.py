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

"""TaskSpec role projection and immutable E1 scene revisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy, copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.utils import configclass
from embodichain.gen_sim.task_spec import (
    TaskSpec,
    scene_layout_requirement,
    validate_task_spec,
)
from embodichain.lab.sim.scene.expansion import (
    SceneLayoutInput,
    SceneRandomizationPolicyCfg,
    PreparedSceneRandomization,
    SceneVariant,
    prepare_scene_randomization,
    generate_scene_variants,
    save_scene_variants,
    load_scene_variants,
)

if TYPE_CHECKING:
    from .orchestration.source_scene import PreparedScene
    from .scene_backend import SceneRevision

__all__ = [
    "SceneExpansionExecutionCfg",
    "compile_scene_layout_requirement",
    "scene_layout_input_from_prepared_scene",
    "materialize_scene_variant",
    "evaluate_scene_task_spec",
]


@configclass
class SceneExpansionExecutionCfg:
    """Non-semantic layout execution settings; permissions come from TaskSpec.

    Args:
        device: Explicit CPU/CUDA sampling device.
        batch_size: Maximum proposals evaluated together.
        candidate_count: Requested valid scene variants.
        max_attempts: Whole-layout geometric proposal budget.
        layout_storage: JSON or packed layout persistence.
        layout_dir: Optional saved v4 package to replay without sampling.
    """

    device: str = "cpu"
    batch_size: int = 4096
    candidate_count: int = 10
    max_attempts: int = 10000
    layout_storage: str = "packed"
    layout_dir: str | Path | None = None

    def __post_init__(self) -> None:
        for name in ("batch_size", "candidate_count", "max_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_attempts < self.candidate_count:
            raise ValueError("max_attempts must cover the requested candidate_count.")
        if self.layout_storage not in ("json", "packed"):
            raise ValueError("layout_storage must be json or packed.")


def _bound_roles(value: Mapping) -> dict[str, tuple[str, ...]]:
    raw = value.get("reference_bindings", value)
    if not isinstance(raw, Mapping):
        raise ValueError("Role bindings must map logical roles to native UID lists.")
    result = {}
    for role, ids in raw.items():
        ids = (ids,) if isinstance(ids, str) else tuple(ids)
        if (
            not isinstance(role, str)
            or not ids
            or any(not isinstance(uid, str) or not uid for uid in ids)
        ):
            raise ValueError("Role bindings require nonempty role IDs and native UIDs.")
        result[role] = ids
    return result


def compile_scene_layout_requirement(
    task_spec: TaskSpec,
    scene_input: SceneLayoutInput,
    role_bindings: Mapping,
    execution: SceneExpansionExecutionCfg | None = None,
) -> PreparedSceneRandomization:
    """Resolve normative role permissions and ranges to shared geometric inputs.

    Args:
        task_spec: Frozen task semantics with expansion/scene_layout requirement.
        scene_input: Normalized source geometry in the runtime arena frame.
        role_bindings: Explicit role-to-native-UID bindings.
        execution: Device, batch and budget settings, without role overrides.

    Returns:
        Prepared shared generator with fixed roles and member/root constraints.
    """
    validate_task_spec(task_spec)
    declaration = scene_layout_requirement(task_spec)
    if declaration is None:
        raise ValueError("TaskSpec does not declare scene_layout expansion.")
    bindings = _bound_roles(role_bindings)
    referenced = set(declaration.movable_roles) | set(declaration.fixed_roles)
    if declaration.support_role is not None:
        referenced.add(declaration.support_role)
    if referenced - set(bindings):
        raise ValueError(
            f"Required scene-layout roles are unbound: {sorted(referenced - set(bindings))}"
        )
    if any(
        uid not in scene_input.poses for role in referenced for uid in bindings[role]
    ):
        raise ValueError(
            "Scene layout role bindings must select existing physical rigid entities."
        )
    if declaration.support_role is not None and bindings[declaration.support_role] != (
        scene_input.support_uid,
    ):
        raise ValueError("unsupported: scene layout requires one fixed main support.")
    fixed = {uid for role in declaration.fixed_roles for uid in bindings[role]}
    roots, members, root_bounds, physical_rules = [], {}, {}, {}
    for role in declaration.movable_roles:
        for uid in bindings[role]:
            if uid in fixed or uid == scene_input.support_uid:
                raise ValueError("TaskSpec movable and fixed physical roles conflict.")
            bounds = declaration.xy_offset_bounds_m.get(role)
            yaw = declaration.yaw_delta_range_deg.get(role)
            if uid in physical_rules:
                if physical_rules[uid] != (bounds, yaw):
                    raise ValueError(
                        "TaskSpec roles bind conflicting variation ranges to one physical entity."
                    )
                continue
            physical_rules[uid] = (bounds, yaw)
            if scene_input.parent_by_uid[uid] == scene_input.support_uid:
                if yaw is not None and tuple(yaw) != (0.0, 0.0):
                    raise ValueError(
                        "unsupported: group-root yaw is outside the current scene layout capability."
                    )
                roots.append(uid)
                if bounds is not None:
                    root_bounds[uid] = [list(row) for row in bounds]
            else:
                members[uid] = {
                    "xy_offset_bounds_m": (
                        None if bounds is None else [list(row) for row in bounds]
                    ),
                    "yaw_delta_range_deg": (
                        list(yaw) if yaw is not None else [-180.0, 180.0]
                    ),
                }
    # A fixed support may still host movable children; fixed means arena pose,
    # not a prohibition on independent child layout changes.
    source = replace(
        scene_input,
        movable_entity_ids=tuple(sorted(set(scene_input.movable_entity_ids) - fixed)),
    )
    settings = execution or SceneExpansionExecutionCfg()
    settings.validate()
    policy = SceneRandomizationPolicyCfg(
        device=settings.device,
        batch_size=settings.batch_size,
        randomize_groups=bool(roots),
        group_targets=tuple(sorted(set(roots))),
        group_xy_offset_bounds_m=root_bounds,
        intra_group={"targets": members},
    )
    return prepare_scene_randomization(source, policy)


def scene_layout_input_from_prepared_scene(scene: PreparedScene) -> SceneLayoutInput:
    """Project a Scene Engine export into Task Engine's normalized runtime frame.

    Args:
        scene: Prepared Scene Engine source with UID and world-transform evidence.

    Returns:
        Scaled geometry, explicit support topology and normalized arena poses.
    """
    from embodichain.gen_sim.scene_engine.randomization.adapter import (
        scene_layout_input_from_export,
    )

    if scene.articulations or scene.body_scale_policy != "preserve":
        raise ValueError(
            "unsupported: first scene-layout integration preserves rigid assets and their scale."
        )
    source = scene_layout_input_from_export(scene.source_config_path.parent)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler(
        "Z", scene.z_rotation_degrees, degrees=True
    ).as_matrix()
    world[:3, 3] = world[:3, :3] @ np.array([*scene.source_scene_xy_translation, 0.0])
    return SceneLayoutInput(
        source.scene_id,
        scene.uid_map[source.support_uid],
        {
            scene.uid_map[uid]: world @ np.asarray(pose)
            for uid, pose in source.poses.items()
        },
        {scene.uid_map[uid]: points for uid, points in source.vertices.items()},
        {scene.uid_map[uid]: faces for uid, faces in source.faces.items()},
        {
            scene.uid_map[uid]: scene.uid_map[parent]
            for uid, parent in source.parent_by_uid.items()
        },
    )


def materialize_scene_variant(
    scene: PreparedScene,
    variant: SceneVariant,
    output_root: str | Path,
) -> SceneRevision:
    """Write an independent revision in the source frame, preserving all assets.

    Args:
        scene: Normalized base source used for role binding and generation.
        variant: Saved runtime-arena poses; no sampling occurs here.
        output_root: Fresh variant revision directory.

    Returns:
        Immutable SceneRevision selecting the changed runtime configuration.
    """
    from .scene_backend import SceneRevision
    from .orchestration.scene_source import (
        scene_revision_id,
        fingerprint_scene_source,
        SceneSourceRef,
    )

    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    source_dir = scene.source_config_path.parent
    shutil.copytree(
        source_dir,
        destination / "scene_export",
        ignore=shutil.ignore_patterns("scene_randomization_geometry", "scene_layouts"),
    )
    config_path = destination / "scene_export" / scene.source_config_path.name
    config = json.loads(config_path.read_text())
    entries = {
        item["uid"]: item
        for section in ("background", "rigid_object")
        for item in config.get(section, [])
    }
    source_uid = {runtime: original for original, runtime in scene.uid_map.items()}
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler(
        "Z", scene.z_rotation_degrees, degrees=True
    ).as_matrix()
    world[:3, 3] = world[:3, :3] @ np.array([*scene.source_scene_xy_translation, 0.0])
    inverse = np.linalg.inv(world)
    for change in variant.pose_changes:
        uid = source_uid[change.entity_uid]
        matrix = inverse @ np.asarray(change.arena_pose)
        entry = entries[uid]
        entry["init_local_pose"] = matrix.tolist()
        entry["init_pos"] = matrix[:3, 3].tolist()
        entry["init_rot"] = (
            Rotation.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=True).tolist()
        )
    config_path.write_text(
        json.dumps(config, sort_keys=True, indent=2, allow_nan=False)
    )
    fingerprint = fingerprint_scene_source(SceneSourceRef(config_path))
    return SceneRevision(
        config_path,
        destination,
        scene_revision_id(config_path),
        variant.seed,
        {"kind": "scene_layout", "variant_id": variant.variant_id},
        fingerprint,
    )


def evaluate_scene_task_spec(
    task_spec: TaskSpec,
    graph: Mapping,
    runtime_result: Mapping | None,
    *,
    num_envs: int,
    initial_observations: Mapping | Sequence[Mapping] | None = None,
) -> tuple[dict, ...]:
    """Evaluate E1 task semantics from verified runtime-validator evidence.

    Args:
        task_spec: Frozen normative specification checked during bundle preparation.
        graph: Fingerprint-bound semantic task graph.
        runtime_result: Existing demo segment outcomes, including validator masks.
        num_envs: Number of independent rows in this one bundle execution.
        initial_observations: Optional independently measured initial predicates.

    Returns:
        Per-row TaskSpec reports. Missing segment or initial evidence stays
        unavailable and cannot be accepted from normal program completion.
    """
    from embodichain.gen_sim.task_spec import evaluate_task_spec
    from ._bundle_runner import _semantic_success_by_env
    from .contracts import canonical_hash

    spec = TaskSpec.from_dict(task_spec)
    validate_task_spec(spec)
    if any(node.get("task_type") != "E1" for node in graph["nodes"]):
        raise ValueError(
            "unsupported: first TaskSpec scene-layout execution qualifies E1 only."
        )
    semantics = _semantic_success_by_env(
        dict(graph),
        None if runtime_result is None else dict(runtime_result),
        num_envs=num_envs,
    )
    groups = list(graph["task_groups"])
    terminal_group = str(groups[-1]["id"]) if groups else None
    available = [set() for _ in range(num_envs)]
    for segment in (runtime_result or {}).get("segments", []):
        if (
            isinstance(segment, Mapping)
            and isinstance(segment.get("successes"), list)
            and len(segment["successes"]) == num_envs
            and isinstance(segment.get("active"), list)
            and len(segment["active"]) == num_envs
            and all(
                type(value) is bool
                for value in (*segment["successes"], *segment["active"])
            )
        ):
            for row, active in enumerate(segment["active"]):
                if active:
                    available[row].add(segment.get("name"))
    reports = []
    for row in range(num_envs):
        goal = {}
        last = groups[-1] if groups else None
        if last is not None and set(last["node_ids"]) <= available[row]:
            for predicate in spec.goal:
                goal[
                    predicate.observation or predicate.predicate_id or predicate.name
                ] = semantics[row][terminal_group]
        snapshots = []
        for group in groups:
            snapshot = {}
            if set(group["node_ids"]) <= available[row]:
                milestone = next(
                    (m for m in spec.milestones if m.id == group["id"]), None
                )
                if milestone is not None:
                    for index, predicate in enumerate(milestone.achieve):
                        snapshot[
                            predicate.observation
                            or predicate.predicate_id
                            or f"{milestone.id}.{index}"
                        ] = semantics[row][str(group["id"])]
            snapshots.append(snapshot)
        initial = (
            initial_observations[row]
            if isinstance(initial_observations, Sequence)
            else initial_observations or {}
        )
        report = evaluate_task_spec(
            spec,
            {"init": dict(initial), "goal": goal, "invariants": {}},
            trajectory=snapshots,
        )
        value = report.to_dict()
        value["semantic_hash"] = spec.semantic_hash
        value["evidence_source"] = "runtime_segment_validators"
        value["acceptance_status"] = (
            "passed"
            if report.certificate_ready
            else "unverified" if value["status"] == "unavailable" else "failed"
        )
        value["execution_result_sha256"] = canonical_hash(dict(runtime_result or {}))
        reports.append(value)
    return tuple(reports)


def _observe_initial_conditions(
    spec: TaskSpec, sim: object, bindings: Mapping, parents: Mapping
) -> tuple[dict, ...]:
    """Measure supported initial predicates without advancing simulation time.

    Distances use actor origins in metres (XY or XYZ axes). ``relation/on``
    requires the declared support edge and mesh coverage within 4 mm above or
    1 mm below an upward planar face. It is geometric support evidence, not a
    stability or contact-force certificate. Other predicates stay unavailable.
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    def array(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    poses, entities, meshes = {}, {}, {}
    for uids in bindings.values():
        for uid in uids:
            if uid not in entities:
                entity = sim.get_rigid_object(uid)
                if entity is not None:
                    entities[uid] = entity
                    poses[uid] = array(entity.get_local_pose(to_matrix=True))

    def points(uid: str, row: int) -> tuple[np.ndarray, np.ndarray]:
        if uid not in meshes:
            entity = entities[uid]
            meshes[uid] = (
                array(entity.get_vertices(env_ids=[0], scale=True)[0]),
                array(entity.get_triangles(env_ids=[0])[0]).astype(np.int64),
            )
        vertices, faces = meshes[uid]
        pose = poses[uid][row]
        return vertices @ pose[:3, :3].T + pose[:3, 3], faces

    rows = []
    for row in range(sim.num_envs):
        observed = {}
        for predicate in spec.init:
            args = predicate.arguments
            subject = bindings.get(args.get("subject"), ())
            reference = bindings.get(args.get("reference"), ())
            value = None
            if predicate.name == "exists" and subject:
                value = all(uid in poses for uid in subject)
            elif (
                len(subject) == len(reference) == 1
                and subject[0] in poses
                and reference[0] in poses
            ):
                uid, parent = subject[0], reference[0]
                if predicate.name == "distance" and args.get("axes", "xyz") in (
                    "xy",
                    "xyz",
                ):
                    dimensions = 2 if args.get("axes") == "xy" else 3
                    value = float(
                        np.linalg.norm(
                            poses[uid][row, :dimensions, 3]
                            - poses[parent][row, :dimensions, 3]
                        )
                    )
                elif predicate.name == "relation" and args.get("relation") == "on":
                    child_points, _ = points(uid, row)
                    parent_points, faces = points(parent, row)
                    triangles = parent_points[faces]
                    normal = np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    )
                    z = triangles[:, :, 2]
                    bottom = child_points[:, 2].min()
                    selected = (
                        (normal[:, 2] > 0.95 * np.linalg.norm(normal, axis=1))
                        & (z.max(1) - z.min(1) <= 0.001)
                        & (z.min(1) >= bottom - 0.004)
                        & (z.max(1) <= bottom + 0.001)
                    )
                    support = unary_union(
                        [Polygon(triangle[:, :2]) for triangle in triangles[selected]]
                    )
                    footprint = box(
                        *child_points[:, :2].min(0), *child_points[:, :2].max(0)
                    )
                    value = parents.get(uid) == parent and bool(
                        support.buffer(1e-7).covers(footprint)
                    )
            if value is not None:
                observed[
                    predicate.observation or predicate.predicate_id or predicate.name
                ] = value
        rows.append(observed)
    return tuple(rows)


def _run_task_spec_expansion(
    workflow: object,
    request: Mapping,
    task_spec: TaskSpec,
    layout_execution: SceneExpansionExecutionCfg,
    *,
    workflow_cfg: object,
    planning_cfg: object,
    execution_cfg: object,
    candidate_set: Mapping | None,
    options: dict,
) -> object:
    """Supply scene cases to the existing workflow, never a new action executor."""
    from .workflow import TaskEngineRunResult
    from .task_spec import (
        task_spec_from_candidate,
        task_spec_role_bindings,
        _preserves_task_spec,
    )
    from .contracts import validate_task_candidate_set, canonical_hash
    from .orchestration.artifacts import ArtifactTransaction
    from .orchestration.scene_source import SceneSourceRef

    spec = TaskSpec.from_dict(task_spec.to_dict())
    validate_task_spec(spec)
    if spec.task_id is not None and spec.task_id != request["task_id"]:
        raise ValueError("TaskSpec.task_id must match the workflow request.")
    if planning_cfg.fit_grasp_assets:
        raise ValueError(
            "Scene expansion preserves asset scales; fit_grasp_assets must be disabled."
        )
    generated = candidate_set
    if generated is None:
        generated = workflow.task_agent.generate(
            request["task_id"],
            request["task_instruction"],
            model=options.get("model"),
            candidate_count=planning_cfg.candidate_count,
        )
    generated = validate_task_candidate_set(generated)
    compatible = []
    for candidate in generated["candidates"]:
        if all(
            step["task_type"] == "E1" for step in candidate["draft"]["steps"]
        ) and _preserves_task_spec(spec, task_spec_from_candidate(candidate)):
            compatible.append(candidate)
    if not compatible:
        raise ValueError(
            "unsupported: no E1 candidate preserves the authoritative TaskSpec."
        )
    # Freeze one semantic candidate before sampling; geometry retries cannot
    # select another task meaning or broaden the fixed role assignments.
    selected = max(
        compatible, key=lambda item: (item["vote_count"], item["candidate_id"])
    )
    candidates = {
        **deepcopy(generated),
        "candidates": [deepcopy(selected)],
        "valid_response_count": selected["vote_count"],
        "requested_candidate_count": selected["vote_count"],
        "errors": [],
    }
    candidates = validate_task_candidate_set(candidates)
    output = Path(request["output_dir"]).expanduser().resolve()
    last_result = None
    accepted_result = None
    accepted = 0
    records = []
    with ArtifactTransaction(
        output, overwrite=bool(options.get("overwrite", False))
    ) as transaction:
        staging = transaction.staging_dir
        assert staging is not None
        (staging / "task_spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
        analysis = workflow.scene_backend.analyze(request, staging / "base_analysis")
        revision = workflow.scene_backend.materialize(
            analysis, request, staging / "base_scene", seed=options["base_seed"]
        )
        adapter = copy(workflow.coordinator.scene_adapter)
        adapter.audit_dir = staging / "binding_audit"
        base = adapter.adapt(
            candidates,
            SceneSourceRef(revision.source, robot_profile=adapter.robot_profile),
        )
        if base.role_bindings is None or base.binding_report["status"] != "bound":
            raise ValueError(
                "TaskSpec scene expansion requires resolved physical role bindings."
            )
        source = scene_layout_input_from_prepared_scene(base.prepared_scene)
        canonical_bindings = task_spec_role_bindings(selected, base.role_bindings)
        prepared = compile_scene_layout_requirement(
            spec, source, canonical_bindings, layout_execution
        )
        if layout_execution.layout_dir is None:
            layouts = generate_scene_variants(
                prepared,
                seed=options["base_seed"],
                candidate_count=layout_execution.candidate_count,
                max_attempts=layout_execution.max_attempts,
            )
            save_scene_variants(
                layouts,
                staging / "scene_layouts",
                storage=layout_execution.layout_storage,
            )
            variants = layouts.variants
            geometry_status, geometry_attempts = layouts.status, layouts.attempts
            layout_source = "online"
        else:
            saved = load_scene_variants(
                layout_execution.layout_dir, scene_input=prepared.source
            )

            def semantic_policy(policy: Mapping) -> dict:
                return {
                    key: value
                    for key, value in policy.items()
                    if key not in {"backend", "device", "batch_size"}
                }

            if semantic_policy(saved.prepared.policy) != semantic_policy(
                prepared.policy
            ):
                raise ValueError(
                    "Saved layout policy contradicts the authoritative TaskSpec role constraints."
                )
            prepared = saved.prepared
            variants = saved.variants
            geometry_status = saved.manifest["status"]
            geometry_attempts = saved.manifest["attempts"]
            saved_root = Path(layout_execution.layout_dir).expanduser().resolve()
            if not (saved_root / "run_result.json").is_file():
                saved_root = saved_root.parent
            shutil.copytree(saved_root, staging / "scene_layouts")
            layout_source = "saved"
        policy_fingerprint = canonical_hash(
            {
                "task_spec": spec.semantic_hash,
                "source": prepared.source.fingerprint,
                "policy": prepared.policy,
                "seed": options["base_seed"],
                "layout_execution": {
                    **layout_execution.to_dict(),
                    "layout_dir": (
                        None
                        if layout_execution.layout_dir is None
                        else str(layout_execution.layout_dir)
                    ),
                },
                "execution": execution_cfg.to_dict(),
            }
        )
        frozen_bindings = {
            role: sorted(uids)
            for role, uids in base.role_bindings["reference_bindings"].items()
        }
        selected_count = min(layout_execution.candidate_count, len(variants))
        for index in range(selected_count):
            variant = variants[index]
            variant_root = staging / "variants" / f"variant_{index:06d}"
            packet = {
                "schema_version": 1,
                "variant_id": variant.variant_id,
                "proposal_ordinal": variant.ordinal,
                "seed": variant.seed,
                "source_sha256": prepared.source.fingerprint,
                "geometry_sha256": prepared.geometry_sha256,
                "geometry_status": "passed",
                "task_spec_semantic_hash": spec.semantic_hash,
                "source": layout_source,
                "execution_fingerprint": policy_fingerprint,
                "role_bindings": frozen_bindings,
                "task_role_bindings": canonical_bindings,
                "parent_by_uid": dict(prepared.source.parent_by_uid),
                "layout": {
                    "parent_scene_id": variant.parent_scene_id,
                    "seed": variant.seed,
                    "ordinal": variant.ordinal,
                    "metadata": list(variant.metadata),
                    "poses": {
                        change.entity_uid: change.arena_pose
                        for change in variant.pose_changes
                    },
                },
            }
            record = {
                "variant_id": variant.variant_id,
                "proposal_ordinal": variant.ordinal,
                "status": "rejected",
                "scene_revision_id": None,
            }
            try:
                current = materialize_scene_variant(
                    base.prepared_scene, variant, variant_root / "scene_revision"
                )
                packet["scene_revision_id"] = current.revision_id
                record["scene_revision_id"] = current.revision_id
                child_request = {
                    **dict(request),
                    "image_path": None,
                    "gym_project": str(current.source),
                    "scene_edit_prompt": None,
                    "output_dir": str(variant_root / "task_run"),
                }
                child = copy(workflow)
                child.scene_adapter = copy(adapter)
                child.coordinator = copy(workflow.coordinator)
                child.coordinator.scene_adapter = child.scene_adapter
                child._owns_coordinator = True
                # Existing input, preparation, retry, execution and recording
                # stages remain owned by TaskEngineWorkflow.run.
                last_result = child.run(
                    child_request,
                    workflow_cfg=workflow_cfg,
                    planning_cfg=planning_cfg,
                    execution_cfg=execution_cfg,
                    task_spec=spec,
                    candidate_set=candidates,
                    _frozen_scene_variant=packet,
                    model=options.get("model"),
                    base_seed=options["base_seed"]
                    + index * workflow_cfg.max_action_attempts,
                    dataset_saving=options["dataset_saving"],
                    failure_policy=options["failure_policy"],
                    open_window=options["open_window"],
                    execute=options["execute"],
                    run_id=f"variant_{index:06d}",
                )
                record["status"] = last_result.status
                record["manifest_path"] = str(last_result.manifest_path)
                record["state_path"] = str(last_result.state_path)
                record["failure_class"] = last_result.failure_class
                if last_result.succeeded:
                    accepted += 1
                    accepted_result = last_result
            except (ValueError, RuntimeError, OSError) as exc:
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            records.append(record)
        prepared_count = sum(
            item["status"] in ("prepared", "succeeded") for item in records
        )
        status = (
            "succeeded"
            if accepted
            else "prepared" if not options["execute"] and prepared_count else "failed"
        )
        failure = (
            None if status in ("succeeded", "prepared") else "scene_expansion_exhausted"
        )
        manifest = {
            "schema_version": "task_engine_scene_expansion/v1",
            "status": status,
            "task_spec": {
                "path": "task_spec.json",
                "semantic_hash": spec.semantic_hash,
            },
            "geometry_status": geometry_status,
            "geometry_attempts": geometry_attempts,
            "source": layout_source,
            "execution_fingerprint": policy_fingerprint,
            "requested_count": layout_execution.candidate_count,
            "generated_count": selected_count,
            "accepted_count": accepted,
            "dataset_saving": options["dataset_saving"],
            "failure_class": failure,
            "variants": records,
        }
        (staging / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        audits = {
            "schema_version": "task_engine_scene_expansion_audit/v1",
            "status": status,
            "task_spec_semantic_hash": spec.semantic_hash,
            "variants": [],
        }
        for record in records:
            if "manifest_path" in record:
                audit_path = Path(record["state_path"])
                audits["variants"].append(
                    {
                        "variant_id": record["variant_id"],
                        "path": str(audit_path),
                        "state": json.loads(audit_path.read_text()),
                    }
                )
        (staging / "state_audit.json").write_text(json.dumps(audits, indent=2))
        representative = accepted_result or last_result
        final_relative = (
            None
            if representative is None or representative.final_bundle is None
            else representative.final_bundle.relative_to(staging)
        )
        published = transaction.commit()
    return TaskEngineRunResult(
        status,
        published,
        published / "run_manifest.json",
        published / "state_audit.json",
        None if final_relative is None else published / final_relative,
        failure,
    )
