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

"""Orchestrate source-scene preparation, planning, compilation, and publication."""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_engine.config import (
    generation_defaults,
    resolve_agent_runtime_policy,
)
from embodichain.gen_sim.action_engine.protocol import (
    ACTION_ENGINE_CONFIG_SCHEMA,
    EXECUTION_PROGRAM_FILENAME,
    SCENE_REQUIREMENTS_FILENAME,
    SCENE_REQUIREMENTS_SCHEMA,
    TASK_SPEC_FILENAME,
)

from .artifacts import artifact_paths, write_generation_artifacts
from .assets import normalize_scene_assets
from .config_builder import (
    VLM_CAMERA_UIDS,
    build_agent_config,
    build_fast_gym_config,
    canonical_robot_profile,
)
from .models import GeneratedConfigPaths
from .source_scene import prepare_scene

__all__ = ["generate_action_engine_config"]

_GENERATION_DEFAULTS = generation_defaults()
_TASK_DEFAULTS = _GENERATION_DEFAULTS["task"]
_SCENE_DEFAULTS = _GENERATION_DEFAULTS["scene"]
_DEFAULT_BODY_SCALE = tuple(float(value) for value in _SCENE_DEFAULTS["body_scale"])


def generate_action_engine_config(
    gym_project: str | Path,
    output_dir: str | Path,
    *,
    task_name: str,
    task_description: str | None = None,
    task_spec: Mapping[str, Any] | str | Path | None = None,
    robot_profile: str = str(_TASK_DEFAULTS["default_robot_profile"]),
    gripper_model: str = str(_TASK_DEFAULTS["default_gripper_model"]),
    ik_solver: str = str(_TASK_DEFAULTS["default_ik_solver"]),
    llm_model: str | None = None,
    source_scene_z_rotation_degrees: float | None = None,
    source_scene_xy_translation: Sequence[float] | None = None,
    body_scale_policy: str = str(_SCENE_DEFAULTS["body_scale_policy"]),
    body_scale: Sequence[float] = _DEFAULT_BODY_SCALE,
    overwrite: bool = False,
    max_episodes: int = int(_TASK_DEFAULTS["max_episodes"]),
    max_episode_steps: int = int(_TASK_DEFAULTS["max_episode_steps"]),
    randomize_scene: bool = False,
    randomize_table_material: bool = False,
    planning_mode: str = "offline",
    vlm_model: str | None = None,
    planner_policy: Mapping[str, Any] | None = None,
) -> GeneratedConfigPaths:
    """Generate the complete Action Engine input bundle.

    Natural-language input is interpreted and grounded by the structured LLM
    path. Callers may instead provide an already grounded v2 TaskSpec; that
    path never invokes a text model.
    """
    task_name = str(task_name).strip()
    task_description = "" if task_description is None else str(task_description).strip()
    if not task_name:
        raise ValueError("task_name must be a non-empty string.")
    if task_spec is not None and task_description:
        raise ValueError("task_spec cannot be combined with task_description.")
    if task_spec is None and not task_description:
        raise ValueError("task_description is required when task_spec is not supplied.")
    if planning_mode not in {"offline", "ab"}:
        raise ValueError("planning_mode must be 'offline' or 'ab'.")
    from embodichain.gen_sim.action_engine.gripper_profiles import get_gripper_profile
    from embodichain.gen_sim.action_engine.solver_profiles import (
        resolve_ik_solver_mode,
    )

    gripper_model = get_gripper_profile(gripper_model).model.value
    ik_solver = resolve_ik_solver_mode(
        ik_solver,
        canonical_robot_profile(robot_profile),
    )
    _raise_if_outputs_exist(
        output_dir,
        overwrite=overwrite,
        planning_mode=planning_mode,
    )

    scene = prepare_scene(
        gym_project,
        z_rotation_degrees=source_scene_z_rotation_degrees,
        source_scene_xy_translation=source_scene_xy_translation,
        body_scale_policy=body_scale_policy,
        body_scale=body_scale,
    )

    # Delayed imports keep scene/config tooling lightweight and avoid importing
    # an LLM client when callers only inspect exported projects.
    from embodichain.gen_sim.action_engine.capabilities import (
        build_atomic_capability_registry,
    )
    from embodichain.gen_sim.action_engine.domain import (
        seed_graph_hash,
        validate_seed_graph,
        validate_scene_requirements,
        validate_task_spec,
    )
    from embodichain.gen_sim.action_engine.tasks import (
        interpret_and_ground_task_spec,
        instantiate_seed_graph,
    )

    known_objects = [str(item["runtime_uid"]) for item in scene.planner_objects]
    if task_spec is not None:
        supplied_task_spec, source_path = _read_task_spec(task_spec)
        task_spec = _validated_mapping(
            supplied_task_spec,
            validator=validate_task_spec,
            label="TaskSpec",
        )
        _require_matching_task_spec(task_spec, task_name)
        task_description = str(task_spec["instruction"])
        supplied_requirements = _read_sibling_scene_requirements(
            source_path,
            task_name,
        )
        if supplied_requirements is not None:
            supplied_requirements = _validated_mapping(
                supplied_requirements,
                validator=validate_scene_requirements,
                label="SceneRequirements",
            )
        role_bindings = _task_spec_role_bindings(
            task_spec,
            known_objects,
            scene_requirements=supplied_requirements,
            scene_objects=scene.planner_objects,
            robot_profile=robot_profile,
        )
        task_spec = _with_role_bindings(task_spec, role_bindings)
        if supplied_requirements is None:
            scene_requirements = _scene_requirements_from_bindings(
                task_name,
                scene.planner_objects,
                role_bindings,
            )
        else:
            scene_requirements = supplied_requirements
            _validate_requirement_roles(scene_requirements, role_bindings)
        compiled = instantiate_seed_graph(task_spec, role_bindings)
    else:
        planned = interpret_and_ground_task_spec(
            task_name=task_name,
            task_description=task_description,
            scene_objects=[deepcopy(obj) for obj in scene.planner_objects],
            robot_profile=robot_profile,
            model=llm_model,
        )
        task_spec = _validated_mapping(
            planned.task_spec,
            validator=validate_task_spec,
            label="TaskSpec",
        )
        # Persist the validated Scene-Engine hand-off alongside the shared
        # semantic TaskSpec.  The binding is not an oracle for online planning,
        # but it is required for ``--regenerate`` and runtime-only loading.
        task_spec = _with_role_bindings(task_spec, planned.role_bindings)
        scene_requirements = _validated_mapping(
            planned.scene_requirements,
            validator=validate_scene_requirements,
            label="SceneRequirements",
        )
        compiled = instantiate_seed_graph(
            task_spec,
            planned.role_bindings,
        )
    if planning_mode == "ab":
        scene_requirements = _add_ab_camera_requirements(scene_requirements)
    capabilities = build_atomic_capability_registry()
    execution_program = _validated_mapping(
        compiled,
        validator=lambda value: validate_seed_graph(
            value,
            known_objects=known_objects,
            known_actions=capabilities.names(),
        ),
        label="SeedGraph",
    )
    if execution_program.get("task_id") != task_name:
        raise ValueError("SeedGraph task_id does not match requested task_name.")
    program_hash = str(seed_graph_hash(execution_program))
    if not program_hash:
        raise ValueError("SeedGraph hash must be non-empty.")

    # Validate planning before materializing normalized meshes in output_dir so
    # an ambiguous instruction cannot leave a half-generated bundle behind.
    scene = normalize_scene_assets(scene, output_dir)

    # Rendering consumes the exact validated in-memory program that runtime
    # consumes. The PNG is review-only and never appears in agent input fields.
    from embodichain.gen_sim.action_engine.graph_visualization import (
        render_seed_task_graph_png,
    )

    seed_task_graph_png = render_seed_task_graph_png(execution_program)
    if not isinstance(seed_task_graph_png, bytes):
        raise TypeError("render_seed_task_graph_png must return bytes.")

    paths = artifact_paths(output_dir, planning_mode=planning_mode)
    graph_relative_path = paths.seed_task_graph.relative_to(
        paths.agent_config.parent
    ).as_posix()
    vlm_camera_uids = list(VLM_CAMERA_UIDS)
    agent_config = build_agent_config(
        task_name=task_name,
        robot_profile=robot_profile,
        gripper_model=gripper_model,
        ik_solver=ik_solver,
        execution_program_hash=program_hash,
        source_config_path=scene.source_config_path,
        uid_map=scene.uid_map,
        static_obstacle_uids=[str(config["uid"]) for config in scene.background],
        dynamic_obstacle_uids=[str(config["uid"]) for config in scene.rigid_objects],
        table_top_z=scene.table_top_z,
        articulation_settings={
            str(config["uid"]): deepcopy(
                config.get("attributes", {}).get("joint_settings", {})
            )
            for config in scene.planner_objects
            if config.get("role") == "articulation"
            and isinstance(config.get("attributes"), Mapping)
            and config.get("attributes", {}).get("joint_settings")
        },
        planning_mode=planning_mode,
        seed_task_graph_path=graph_relative_path,
        vlm_model=vlm_model,
        vlm_camera_uids=vlm_camera_uids,
        planner_policy=planner_policy,
    )
    gym_config = build_fast_gym_config(
        scene,
        task_name=task_name,
        task_description=task_description,
        robot_profile=robot_profile,
        gripper_model=gripper_model,
        ik_solver=ik_solver,
        execution_program_hash=program_hash,
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
        randomize_scene=randomize_scene,
        randomize_table_material=randomize_table_material,
        planning_mode=planning_mode,
        seed_task_graph_path=graph_relative_path,
    )
    if planning_mode == "ab":
        output_root = Path(output_dir).expanduser().resolve()
        gym_config["env"]["events"]["record_camera"]["params"]["save_path"] = (
            output_root / ".ab_video_staging"
        ).as_posix()
        gym_config["env"]["dataset"]["lerobot"]["params"]["save_path"] = (
            output_root / ".ab_datasets"
        ).as_posix()
    _validate_agent_config(agent_config)
    return write_generation_artifacts(
        output_dir,
        gym_config=gym_config,
        agent_config=agent_config,
        task_spec=task_spec,
        scene_requirements=scene_requirements,
        seed_task_graph=execution_program,
        seed_task_graph_png=seed_task_graph_png,
        overwrite=overwrite,
        planning_mode=planning_mode,
    )


def _read_task_spec(
    source: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path | None]:
    """Read one existing v2 TaskSpec without invoking a text planner."""
    if isinstance(source, Mapping):
        return deepcopy(dict(source)), None
    path = Path(source).expanduser().resolve()
    return _read_json_mapping(path, label="TaskSpec"), path


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read {label} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} at {path} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} JSON must contain an object.")
    return deepcopy(dict(value))


def _read_sibling_scene_requirements(
    task_spec_path: Path | None,
    task_name: str,
) -> dict[str, Any] | None:
    """Load the canonical sidecar when a task-first batch supplied one."""
    if task_spec_path is None:
        return None
    candidate = task_spec_path.parent / SCENE_REQUIREMENTS_FILENAME
    if not candidate.is_file():
        return None
    requirements = _read_json_mapping(candidate, label="SceneRequirements")
    if requirements.get("task_id") != task_name:
        raise ValueError(
            "Sibling SceneRequirements task_id does not match the requested "
            "task_name."
        )
    return requirements


def _require_matching_task_spec(task_spec: Mapping[str, Any], task_name: str) -> None:
    if task_spec.get("task_id") != task_name:
        raise ValueError(
            f"TaskSpec task_id {task_spec.get('task_id')!r} does not match "
            f"requested task_name {task_name!r}."
        )


def _task_spec_role_bindings(
    task_spec: Mapping[str, Any],
    known_objects: Sequence[str],
    *,
    scene_requirements: Mapping[str, Any] | None = None,
    scene_objects: Sequence[Mapping[str, Any]] | None = None,
    robot_profile: str = "dual_ur10",
) -> dict[str, str]:
    """Resolve v2 roles from explicit hand-off data or a strict sidecar match.

    Task-first artifacts may contain abstract role IDs rather than scene UIDs.
    When their sibling SceneRequirements is available, match
    every still-unbound role against the source scene's static category,
    attributes, state, and affordance metadata.  This is a deterministic
    Scene-Engine hand-off, not a text-model fallback: missing or ambiguous
    evidence remains an error.
    """
    metadata = task_spec.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("TaskSpec.metadata must be a mapping.")
    metadata_bindings = metadata.get("role_bindings", {})
    if not isinstance(metadata_bindings, Mapping):
        raise ValueError("TaskSpec.metadata.role_bindings must be a mapping.")
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    if metadata_bindings:
        candidates.append(("TaskSpec.metadata", metadata_bindings))

    # Older grounded v2 TaskSpecs kept this private hand-off in ``oracle``
    # rather than metadata. Accept that representation while publishing the
    # normalized binding in metadata for runtime regeneration.
    oracle = task_spec.get("oracle", {})
    if isinstance(oracle, Mapping) and oracle.get("role_bindings"):
        oracle_bindings = oracle["role_bindings"]
        if not isinstance(oracle_bindings, Mapping):
            raise ValueError("TaskSpec.oracle.role_bindings must be a mapping.")
        candidates.append(("TaskSpec.oracle", oracle_bindings))
    if isinstance(oracle, Mapping):
        reference = oracle.get("reference_seed_graph")
        if isinstance(reference, Mapping):
            graph_metadata = reference.get("metadata", {})
            if isinstance(graph_metadata, Mapping) and graph_metadata.get(
                "role_bindings"
            ):
                graph_bindings = graph_metadata["role_bindings"]
                if not isinstance(graph_bindings, Mapping):
                    raise ValueError(
                        "SeedGraph.metadata.role_bindings must be a mapping."
                    )
                candidates.append(("SeedGraph.metadata", graph_bindings))

    if scene_requirements is not None:
        requirement_metadata = scene_requirements.get("metadata", {})
        if isinstance(requirement_metadata, Mapping) and requirement_metadata.get(
            "role_bindings"
        ):
            requirement_bindings = requirement_metadata["role_bindings"]
            if not isinstance(requirement_bindings, Mapping):
                raise ValueError(
                    "SceneRequirements.metadata.role_bindings must be a mapping."
                )
            candidates.append(("SceneRequirements.metadata", requirement_bindings))

    supplied: dict[str, Any] = {}
    supplied_sources: dict[str, str] = {}
    for source, candidate in candidates:
        for raw_role, uid in candidate.items():
            if not isinstance(raw_role, str) or not raw_role.strip():
                raise ValueError(f"{source}.role_bindings must use non-empty role IDs.")
            role = raw_role.strip()
            if role in supplied and supplied[role] != uid:
                raise ValueError(
                    "Conflicting role_bindings were supplied for "
                    f"{role!r} by {supplied_sources[role]} and {source}."
                )
            supplied[role] = uid
            supplied_sources[role] = source

    known = {str(uid) for uid in known_objects}
    required = _task_spec_role_references(task_spec.get("task_instances", []))
    required.discard("table")
    if not required:
        raise ValueError("TaskSpec must reference at least one non-table object role.")

    bindings: dict[str, str] = {}
    missing: list[str] = []
    for role in sorted(required):
        raw_uid = supplied.get(role, role if role in known else None)
        if raw_uid is None:
            missing.append(role)
            continue
        if not isinstance(raw_uid, str) or not raw_uid.strip():
            raise ValueError(
                "TaskSpec.metadata.role_bindings must map role IDs to non-empty "
                "runtime UIDs."
            )
        uid = raw_uid.strip()
        if uid not in known:
            raise ValueError(f"TaskSpec role {role!r} binds unknown scene UID {uid!r}.")
        bindings[role] = uid
    if missing and scene_requirements is not None and scene_objects is not None:
        bindings.update(
            _infer_role_bindings_from_scene_requirements(
                missing,
                known_objects=known,
                scene_objects=scene_objects,
                scene_requirements=scene_requirements,
                existing_bindings=bindings,
                robot_profile=robot_profile,
            )
        )
        missing = [role for role in missing if role not in bindings]
    if missing:
        raise ValueError(
            "TaskSpec requires explicit role_bindings or an unambiguous sibling "
            f"SceneRequirements match for roles {missing}; a task-first spec must "
            "be grounded by a Scene Engine before it can be compiled for this gym "
            "project."
        )
    if len(bindings.values()) != len(set(bindings.values())):
        raise ValueError("TaskSpec role bindings must resolve to unique scene UIDs.")
    if scene_requirements is not None and scene_objects is not None:
        _validate_bound_role_requirements(
            bindings,
            scene_requirements=scene_requirements,
            scene_objects=scene_objects,
            robot_profile=robot_profile,
        )
    return bindings


def _infer_role_bindings_from_scene_requirements(
    roles: Sequence[str],
    *,
    known_objects: set[str],
    scene_objects: Sequence[Mapping[str, Any]],
    scene_requirements: Mapping[str, Any],
    existing_bindings: Mapping[str, str],
    robot_profile: str,
) -> dict[str, str]:
    """Bind abstract task roles only when static evidence is unique."""
    from embodichain.gen_sim.action_engine.tasks.assembly import SceneInventory

    requirements = _requirements_by_role(scene_requirements)
    inventory = SceneInventory(scene_objects, robot_profile=robot_profile)
    entities = [entity for entity in inventory.entities if entity.uid in known_objects]
    used_uids = set(existing_bindings.values())
    inferred: dict[str, str] = {}
    for role in sorted(roles):
        requirement = requirements.get(role)
        if requirement is None:
            raise ValueError(
                "Sibling SceneRequirements is missing TaskSpec role " f"{role!r}."
            )
        count = requirement.get("count", 1)
        if count != 1:
            raise ValueError(
                f"TaskSpec role {role!r} has count={count}; direct SeedGraph "
                "binding requires exactly one concrete scene UID."
            )
        matches = [
            entity
            for entity in entities
            if entity.uid not in used_uids
            and _entity_matches_requirement(
                entity,
                requirement,
                require_complete_static_evidence=True,
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "TaskSpec role "
                f"{role!r} requires one unambiguous scene match, found "
                f"{[entity.uid for entity in matches]}."
            )
        uid = matches[0].uid
        inferred[role] = uid
        used_uids.add(uid)
    return inferred


def _validate_bound_role_requirements(
    bindings: Mapping[str, str],
    *,
    scene_requirements: Mapping[str, Any],
    scene_objects: Sequence[Mapping[str, Any]],
    robot_profile: str,
) -> None:
    """Ensure an explicit binding does not contradict its static sidecar."""
    from embodichain.gen_sim.action_engine.tasks.assembly import SceneInventory

    requirements = _requirements_by_role(scene_requirements)
    entities = SceneInventory(scene_objects, robot_profile=robot_profile).by_uid
    for role, uid in bindings.items():
        requirement = requirements.get(role)
        if requirement is None:
            raise ValueError(
                "Sibling SceneRequirements is missing TaskSpec role " f"{role!r}."
            )
        entity = entities.get(uid)
        if entity is None:
            raise ValueError(
                f"TaskSpec role {role!r} binds unavailable scene UID {uid!r}."
            )
        if not _entity_matches_requirement(
            entity,
            requirement,
            require_complete_static_evidence=False,
        ):
            raise ValueError(
                f"TaskSpec role {role!r} binding {uid!r} conflicts with its "
                "SceneRequirements category, attributes, state, or affordances."
            )


def _requirements_by_role(
    scene_requirements: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    objects = scene_requirements.get("objects", [])
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("SceneRequirements.objects must be a list.")
    result: dict[str, Mapping[str, Any]] = {}
    for requirement in objects:
        if not isinstance(requirement, Mapping):
            raise ValueError("SceneRequirements.objects must contain mappings.")
        role = requirement.get("role_id")
        if not isinstance(role, str) or not role:
            raise ValueError("SceneRequirements role_id must be a non-empty string.")
        result[role] = requirement
    return result


def _entity_matches_requirement(
    entity: Any,
    requirement: Mapping[str, Any],
    *,
    require_complete_static_evidence: bool,
) -> bool:
    """Match explicit metadata; UID inference requires complete evidence."""
    category = requirement.get("category")
    expected_category = category.strip().casefold() if isinstance(category, str) else ""
    actual_category = str(entity.category).strip().casefold()
    if expected_category:
        if not actual_category:
            if require_complete_static_evidence:
                return False
        elif expected_category != actual_category:
            return False
    required_affordances = requirement.get("affordances", [])
    if not isinstance(required_affordances, Sequence) or isinstance(
        required_affordances, (str, bytes)
    ):
        return False
    expected_affordances = {
        str(value).strip().casefold() for value in required_affordances
    }
    if (
        expected_affordances
        and (require_complete_static_evidence or entity.affordances)
        and not expected_affordances.issubset(entity.affordances)
    ):
        return False
    expected_attributes = requirement.get("attributes", {})
    if not isinstance(expected_attributes, Mapping):
        return False
    for name, expected in expected_attributes.items():
        if not _static_attribute_matches(
            entity,
            str(name),
            expected,
            require_complete_static_evidence=require_complete_static_evidence,
        ):
            return False
    expected_state = requirement.get("initial_state", {})
    if not isinstance(expected_state, Mapping):
        return False
    missing = object()
    for name, expected in expected_state.items():
        actual = entity.initial_state.get(str(name), missing)
        if actual is missing:
            if require_complete_static_evidence:
                return False
        elif actual != expected:
            return False
    return True


def _static_attribute_matches(
    entity: Any,
    name: str,
    expected: Any,
    *,
    require_complete_static_evidence: bool,
) -> bool:
    """Compare one requirement against explicit exported metadata only."""
    marker = object()
    actual = entity.color if name == "color" else entity.attributes.get(name, marker)
    if actual is marker or actual is None or actual == "":
        return not require_complete_static_evidence
    if name == "color" and isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().casefold() == expected.strip().casefold()
    return actual == expected


def _task_spec_role_references(value: Any, key: str = "") -> set[str]:
    if isinstance(value, Mapping):
        return {
            role
            for child_key, child in value.items()
            for role in _task_spec_role_references(child, str(child_key))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {
            role for child in value for role in _task_spec_role_references(child, key)
        }
    if isinstance(value, str) and (key.endswith("_role") or key.endswith("_roles")):
        return {value}
    return set()


def _with_role_bindings(
    task_spec: Mapping[str, Any],
    role_bindings: Mapping[str, str],
) -> dict[str, Any]:
    result = deepcopy(dict(task_spec))
    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("TaskSpec.metadata must be a mapping.")
    metadata["role_bindings"] = dict(sorted(role_bindings.items()))
    return result


def _validate_requirement_roles(
    requirements: Mapping[str, Any],
    role_bindings: Mapping[str, str],
) -> None:
    requirement_roles = {
        str(item["role_id"])
        for item in requirements["objects"]
        if isinstance(item, Mapping)
    }
    missing = sorted(set(role_bindings) - requirement_roles)
    if missing:
        raise ValueError(
            "SceneRequirements is missing TaskSpec role bindings for " f"{missing}."
        )


def _scene_requirements_from_bindings(
    task_id: str,
    planner_objects: Sequence[Mapping[str, Any]],
    role_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Derive a minimal concrete SceneRequirements view for grounded roles."""
    source = _scene_requirements_from_scene(task_id, planner_objects)
    by_uid = {str(item["role_id"]): item for item in source["objects"]}
    objects = []
    for role, uid in sorted(role_bindings.items()):
        requirement = by_uid.get(uid)
        if requirement is None:
            raise ValueError(
                f"TaskSpec role {role!r} binds UID {uid!r}, which has no "
                "source-scene requirement."
            )
        resolved = deepcopy(requirement)
        resolved["role_id"] = role
        objects.append(resolved)
    return {
        "schema_version": SCENE_REQUIREMENTS_SCHEMA,
        "task_id": task_id,
        "objects": objects,
        "cameras": [],
        "spatial_constraints": [{"type": "preserve_source_scene"}],
        "distractor_count": 0,
        "metadata": {"source": "task_spec_role_bindings"},
    }


def _validated_mapping(
    value: Any,
    *,
    validator: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{label} producer returned {type(value).__name__}, not a mapping."
        )
    candidate = deepcopy(dict(value))
    validated = validator(candidate)
    if validated is None:
        # Validators may either return a normalized mapping or validate in place.
        validated = candidate
    if not isinstance(validated, Mapping):
        raise TypeError(f"{label} validator must return a mapping or None.")
    return deepcopy(dict(validated))


def _validate_agent_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != ACTION_ENGINE_CONFIG_SCHEMA:
        raise ValueError("Agent config has an unexpected schema_version.")
    if config.get("task_spec") != TASK_SPEC_FILENAME:
        raise ValueError("Agent config must point to the canonical TaskSpec.")
    if config.get("scene_requirements") != SCENE_REQUIREMENTS_FILENAME:
        raise ValueError("Agent config must point to canonical SceneRequirements.")
    from embodichain.gen_sim.action_engine.gripper_profiles import get_gripper_profile
    from embodichain.gen_sim.action_engine.solver_profiles import (
        resolve_ik_solver_mode,
    )

    get_gripper_profile(config.get("gripper_model"))
    solver = config.get("ik_solver")
    if resolve_ik_solver_mode(solver, str(config.get("robot_profile"))) != solver:
        raise ValueError("Agent config must store a concrete IK solver mode.")
    graph_path = config.get("seed_task_graph")
    if (
        not isinstance(graph_path, str)
        or Path(graph_path).name != EXECUTION_PROGRAM_FILENAME
    ):
        raise ValueError("Agent config must point to the canonical SeedGraph.")
    planning_mode = config.get("planning_mode", "offline")
    if planning_mode not in {"offline", "ab"}:
        raise ValueError("Agent config planning_mode must be 'offline' or 'ab'.")
    if planning_mode == "ab":
        online = config.get("online_planning")
        if not isinstance(online, Mapping):
            raise ValueError("A/B agent config requires online_planning settings.")
        camera_uids = online.get("camera_uids")
        if camera_uids != list(VLM_CAMERA_UIDS):
            raise ValueError(
                "A/B agent config must list the canonical four VLM cameras."
            )
        model = online.get("vlm_model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("online_planning.vlm_model must be a string or null.")
        if config.get("offline_seed_task_graph") != graph_path:
            raise ValueError(
                "A/B agent config offline_seed_task_graph must match seed_task_graph."
            )
        if config.get("vlm_camera_uids") != camera_uids:
            raise ValueError(
                "A/B agent config vlm_camera_uids must match online_planning."
            )
        if config.get("vlm_model") != model:
            raise ValueError("A/B agent config vlm_model must match online_planning.")
    resolve_agent_runtime_policy(config)


def _raise_if_outputs_exist(
    output_dir: str | Path,
    *,
    overwrite: bool,
    planning_mode: str = "offline",
) -> None:
    if overwrite:
        return
    paths = artifact_paths(output_dir, planning_mode=planning_mode)
    existing = [
        path
        for path in (
            paths.gym_config,
            paths.agent_config,
            paths.task_spec,
            paths.scene_requirements,
            paths.seed_task_graph,
            paths.seed_task_graph_png,
        )
        if path.exists()
    ]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Generated artifacts already exist in {paths.gym_config.parent}: "
            f"{names}. Pass --overwrite to replace them."
        )


def _scene_requirements_from_scene(
    task_id: str,
    planner_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    objects = []
    for item in planner_objects:
        uid = str(item.get("runtime_uid", item.get("uid", ""))).strip()
        if not uid:
            raise ValueError("Planner scene object is missing a runtime UID.")
        role = str(item.get("role", "object")).strip().lower()
        raw_category = item.get("category", item.get("object_category", ""))
        category = str(raw_category).strip().lower() or role or "object"
        raw_attributes = item.get("attributes", {})
        attributes = (
            deepcopy(dict(raw_attributes))
            if isinstance(raw_attributes, Mapping)
            else {}
        )
        color = item.get("color")
        if color not in (None, ""):
            attributes.setdefault("color", color)
        objects.append(
            {
                "role_id": uid,
                "category": category,
                "count": 1,
                "affordances": [],
                "initial_state": {},
                "attributes": attributes,
            }
        )
    return {
        "schema_version": SCENE_REQUIREMENTS_SCHEMA,
        "task_id": task_id,
        "objects": objects,
        "cameras": [],
        "spatial_constraints": [{"type": "preserve_source_scene"}],
        "distractor_count": 0,
        "metadata": {"source": "existing_gym_project"},
    }


def _add_ab_camera_requirements(
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare fixed multi-view inputs in the shared A/B hand-off."""
    from embodichain.gen_sim.action_engine.domain import validate_scene_requirements

    result = deepcopy(dict(requirements))
    cameras = result.get("cameras", [])
    if not isinstance(cameras, list):
        raise ValueError("SceneRequirements.cameras must be a list.")
    existing_uids = {
        str(item.get("uid"))
        for item in cameras
        if isinstance(item, Mapping) and item.get("uid")
    }
    for uid in VLM_CAMERA_UIDS:
        if uid in existing_uids:
            continue
        cameras.append(
            {
                "uid": uid,
                "role": "vlm_view",
                "modalities": ["rgb", "depth"],
                "coverage": "all_interaction_objects",
                "resolution": [640, 480],
            }
        )
    result["cameras"] = cameras
    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        result["metadata"] = metadata
    metadata["planning_mode"] = "ab"
    metadata["vlm_camera_uids"] = list(VLM_CAMERA_UIDS)
    return validate_scene_requirements(result)
