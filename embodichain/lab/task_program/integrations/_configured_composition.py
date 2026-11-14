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

"""Compose configured Task Program deployments from typed YAML components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from embodichain.lab.evaluation import PredicateRegistry
    from embodichain.lab.task_program.language.schema import TaskProgramCfg
    from .catalog import TaskProgramIntegrationCatalog

from embodichain.lab.task_program.language.schema import TaskProgramIntegrationCfg
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config

from .composition import (
    ConfiguredTaskProgramDeployment as _ConfiguredTaskProgramDeployment,
)
from .configured import (
    _ConfiguredTaskProgramAdapterFactory,
    _ConfiguredTaskProgramIntegration,
    _decode_configured_task_program_integration,
    _decode_motion_policy,
    _decode_recovery_policy,
    _identifier,
    _identifier_mapping,
    _normalize_skill_options_key,
    _normalize_recovery_policy_keys,
    _mapping,
)

__all__: list[str] = []

_RUNTIME_SERVICE_FIELDS = frozenset(
    {
        "grasp_pose_generators",
        "hand_geometries",
        "handover_pose_providers",
        "registered_semantic_lowerers",
        "control_part_evidence",
        "settle_presets",
    }
)
_TASK_RUNTIME_SERVICE_FIELDS = _RUNTIME_SERVICE_FIELDS | frozenset(
    {"grasp_pose_generator_overrides", "nominal_motion_sources"}
)
_GRASP_GENERATOR_OVERRIDE_FIELDS = frozenset(
    {
        "sample_count",
        "approach_direction_samples",
        "point_spacing",
        "filter_ground_collision",
        "force_refresh",
    }
)
_PLANNER_TYPES = frozenset({"curobo", "neural", "toppra", "trapezoidal"})
_EXECUTION_POLICY_OVERRIDE_FIELDS = {
    "motion": frozenset(
        {
            "sample_count",
            "velocity_limit",
            "acceleration_limit",
            "dynamic_collision_mode",
            "ik_seed_mode",
        }
    ),
    "recovery": frozenset({"skill_timeout", "goal_rotation_threshold"}),
}


def _component_path(
    value: object,
    *,
    base_dir: Path,
    path: str,
    suffixes: frozenset[str],
) -> Path:
    """Resolve one packaged or owner-relative Task Program component path."""
    selected = _identifier(value, path=path)
    component_path = resolve_config_path(selected, base_dir=base_dir).resolve()
    if component_path.suffix.lower() not in suffixes:
        rendered = sorted(suffixes)
        raise ValueError(f"{path} must use one of {rendered}: {component_path}.")
    if not component_path.is_file():
        raise FileNotFoundError(f"{path} is not a file: {component_path}.")
    return component_path


def _owned_mapping(value: object, *, path: str) -> dict[str, object]:
    """Return an independently owned exact string-keyed mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    if not all(type(key) is str for key in value):
        raise TypeError(f"{path} keys must be exact strings.")
    return deepcopy(dict(value))


def _owned_sequence(value: object, *, path: str) -> list[object]:
    """Return an independently owned non-string sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence.")
    return deepcopy(list(value))


def _load_yaml_component(path: Path, *, field_name: str) -> dict[str, object]:
    """Load one YAML component mapping."""
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{field_name} must be a YAML file: {path}.")
    return _owned_mapping(load_config(path), path=field_name)


def _decode_planner_config(value: object, *, path: str) -> dict[str, object]:
    """Decode an executable-free planner selection from an execution policy."""
    config = _mapping(
        value,
        path=path,
        required=frozenset({"type"}),
        optional=frozenset({"config"}),
    )
    planner_type = _identifier(config["type"], path=f"{path}.type")
    if planner_type not in _PLANNER_TYPES:
        raise ValueError(
            f"{path}.type must be one of {sorted(_PLANNER_TYPES)}, "
            f"got {planner_type!r}."
        )
    options = config.get("config", {})
    if not isinstance(options, Mapping):
        raise TypeError(f"{path}.config must be a mapping.")
    planner_config = {
        "type": planner_type,
        "config": deepcopy(dict(options)),
    }
    from .simulation.environment import _planner_cfg_from_config

    _planner_cfg_from_config(planner_config, robot_uid="__configured_policy__")
    return planner_config


def _merge_runtime_services(
    skill_profile_services: object,
    task_services: object,
) -> dict[str, object]:
    """Compose disjoint typed runtime-service fields without deep merging."""
    skill_profile = _mapping(
        skill_profile_services,
        path="embodiment component.skill_profile.runtime_services",
        required=frozenset(),
        optional=_RUNTIME_SERVICE_FIELDS,
    )
    task = _mapping(
        task_services,
        path="integration.runtime_services",
        required=frozenset(),
        optional=_TASK_RUNTIME_SERVICE_FIELDS,
    )
    result: dict[str, object] = {}

    skill_profile_generators = _owned_mapping(
        skill_profile.get("grasp_pose_generators", {}),
        path=(
            "embodiment component.skill_profile.runtime_services."
            "grasp_pose_generators"
        ),
    )
    task_generators = _owned_mapping(
        task.get("grasp_pose_generators", {}),
        path="integration.runtime_services.grasp_pose_generators",
    )
    duplicate_generators = sorted(set(skill_profile_generators) & set(task_generators))
    if duplicate_generators:
        raise ValueError(
            "Skill profile and task runtime services define duplicate grasp generator "
            f"targets: {duplicate_generators}."
        )
    generators = {
        **skill_profile_generators,
        **task_generators,
    }
    generator_overrides = _owned_mapping(
        task.get("grasp_pose_generator_overrides", {}),
        path="integration.runtime_services.grasp_pose_generator_overrides",
    )
    for target_id, override_value in generator_overrides.items():
        if target_id not in generators:
            raise ValueError(
                f"Grasp generator override target {target_id!r} is not declared "
                "by the selected skill profile."
            )
        overrides = _mapping(
            override_value,
            path=(
                "integration.runtime_services.grasp_pose_generator_overrides."
                f"{target_id}"
            ),
            required=frozenset(),
            optional=_GRASP_GENERATOR_OVERRIDE_FIELDS,
        )
        generator = _owned_mapping(
            generators[target_id],
            path=f"runtime_services.grasp_pose_generators.{target_id}",
        )
        generator.update(deepcopy(dict(overrides)))
        generators[target_id] = generator
    if generators:
        result["grasp_pose_generators"] = generators

    profile_hands = _owned_mapping(
        skill_profile.get("hand_geometries", {}),
        path="embodiment component.skill_profile.runtime_services.hand_geometries",
    )
    task_hands = _owned_mapping(
        task.get("hand_geometries", {}),
        path="integration.runtime_services.hand_geometries",
    )
    duplicates = sorted(set(profile_hands) & set(task_hands))
    if duplicates:
        raise ValueError(
            f"Hand geometry targets must have exactly one owner; duplicates: {duplicates}."
        )
    if profile_hands or task_hands:
        result["hand_geometries"] = {**profile_hands, **task_hands}

    for field_name in (
        "handover_pose_providers",
        "registered_semantic_lowerers",
    ):
        skill_profile_values = _owned_sequence(
            skill_profile.get(field_name, ()),
            path=(
                "embodiment component.skill_profile.runtime_services." f"{field_name}"
            ),
        )
        task_values = _owned_sequence(
            task.get(field_name, ()),
            path=f"integration.runtime_services.{field_name}",
        )
        if skill_profile_values or task_values:
            result[field_name] = [*skill_profile_values, *task_values]

    skill_profile_evidence = skill_profile.get("control_part_evidence")
    task_evidence = task.get("control_part_evidence")
    if skill_profile_evidence is not None and task_evidence is not None:
        raise ValueError(
            "control_part_evidence must be owned by exactly one skill profile or "
            "task component."
        )
    evidence = task_evidence if task_evidence is not None else skill_profile_evidence
    if evidence is not None:
        result["control_part_evidence"] = deepcopy(evidence)
    if "nominal_motion_sources" in task:
        result["nominal_motion_sources"] = _owned_sequence(
            task["nominal_motion_sources"],
            path="integration.runtime_services.nominal_motion_sources",
        )
    if "settle_presets" in skill_profile or "settle_presets" in task:
        profile_presets = _owned_mapping(
            skill_profile.get("settle_presets", {}),
            path="skill_profile.runtime_services.settle_presets",
        )
        task_presets = _owned_mapping(
            task.get("settle_presets", {}),
            path="integration.runtime_services.settle_presets",
        )
        duplicate = set(profile_presets) & set(task_presets)
        if duplicate:
            raise ValueError(f"Duplicate settle-preset ownership: {sorted(duplicate)}.")
        result["settle_presets"] = {**profile_presets, **task_presets}
    return result


def _resolve_execution_policy(
    value: object,
    *,
    base_dir: Path,
) -> tuple[Path | None, dict[str, object]]:
    """Load a policy path or component selector with bounded scalar overrides."""
    selection_path = "task_program.execution_policy"
    overrides: Mapping[str, object] = {}
    component_field = selection_path
    inline = isinstance(value, Mapping) and "inline" in value
    if inline:
        component = None
    elif isinstance(value, str):
        component = value
    else:
        selection = _mapping(
            value,
            path=selection_path,
            required=frozenset({"component"}),
            optional=frozenset({"overrides"}),
        )
        component = selection["component"]
        component_field = f"{selection_path}.component"
        overrides = _mapping(
            selection.get("overrides", {}),
            path=f"{selection_path}.overrides",
            required=frozenset(),
            optional=frozenset(_EXECUTION_POLICY_OVERRIDE_FIELDS),
        )
    policy_path = (
        None
        if inline
        else _component_path(
            component,
            base_dir=base_dir,
            path=component_field,
            suffixes=frozenset({".yaml", ".yml"}),
        )
    )
    raw_policy = (
        _inline_component(value, path=selection_path)
        if inline
        else _load_yaml_component(policy_path, field_name="execution policy")
    )
    policy = dict(
        _mapping(
            raw_policy,
            path="execution policy",
            required=frozenset(
                {
                    "policy_id",
                    "preset_id",
                    "requires",
                    "motion",
                    "tracking",
                    "recovery",
                    "workflow_recovery",
                    "runner",
                    "effect_assurance",
                }
            ),
            optional=frozenset({"required_planner", "planner"}),
        )
    )
    _identifier(policy["policy_id"], path="execution policy.policy_id")
    policy["recovery"] = _normalize_recovery_policy_keys(
        policy["recovery"], path="execution policy.recovery"
    )
    for group, values in overrides.items():
        if group == "recovery":
            values = _normalize_recovery_policy_keys(
                values, path=f"{selection_path}.overrides.recovery"
            )
        patch = _mapping(
            values,
            path=f"{selection_path}.overrides.{group}",
            required=frozenset(),
            optional=_EXECUTION_POLICY_OVERRIDE_FIELDS[group],
        )
        merged = _owned_mapping(policy[group], path=f"execution policy.{group}")
        merged.update(deepcopy(dict(patch)))
        policy[group] = merged
    # Reuse the production value decoders after patching. Policy identity,
    # contracts, planner, tracking and effect assurance remain component-owned.
    _decode_motion_policy(policy["motion"], path="execution policy.motion")
    _decode_recovery_policy(policy["recovery"], path="execution policy.recovery")
    if "planner" in policy:
        policy["planner"] = _decode_planner_config(
            policy["planner"],
            path="execution policy.planner",
        )
        required_planner = policy.get("required_planner")
        if (
            required_planner is not None
            and required_planner != policy["planner"]["type"]
        ):
            raise ValueError(
                "execution policy.required_planner must match "
                "execution policy.planner.type when both are declared."
            )
    return policy_path, policy


def _apply_integration_overrides(
    source: object, overrides: object
) -> dict[str, object]:
    """Apply closed deployment tuning to an owned integration declaration."""
    result = _owned_mapping(source, path="task integration")
    patch = _mapping(
        overrides,
        path="integration.overrides",
        required=frozenset(),
        optional=frozenset({"integration_id", "requires", "profile"}),
    )
    if "integration_id" in patch:
        result["integration_id"] = _identifier(
            patch["integration_id"], path="integration.overrides.integration_id"
        )
    if "requires" in patch:
        requirements = _mapping(
            patch["requires"],
            path="integration.overrides.requires",
            required=frozenset(),
            optional=frozenset({"embodiment_contract"}),
        )
        result["requires"].update(deepcopy(dict(requirements)))
    if "profile" in patch:
        result["profile"] = _owned_mapping(
            _normalize_skill_options_key(
                result["profile"], path="task integration.profile"
            ),
            path="task integration.profile",
        )
        profile = _mapping(
            patch["profile"],
            path="integration.overrides.profile",
            required=frozenset(),
            optional=frozenset({"defaults", "skill_options"}),
        )
        defaults = _owned_mapping(
            profile.get("defaults", {}), path="integration.overrides.profile.defaults"
        )
        for name, value in defaults.items():
            existing = result["profile"]["defaults"]
            if name not in existing:
                raise ValueError(
                    f"Integration override names undeclared resource default {name!r}."
                )
            resources = _mapping(
                value,
                path=f"integration.overrides.profile.defaults.{name}",
                required=frozenset(),
                optional=frozenset(existing[name]),
            )
            for slot, resource in resources.items():
                existing[name][slot] = _identifier(
                    resource,
                    path=f"integration.overrides.profile.defaults.{name}.{slot}",
                )
        options = _owned_mapping(
            profile.get("skill_options", {}),
            path="integration.overrides.profile.skill_options",
        )
        allowed = frozenset(
            {
                "pre_grasp_distance",
                "lift_height",
                "hand_interp_steps",
                "grasp_settle_steps",
                "release_settle_steps",
                "approach_along_grasp_axis",
                "cartesian_lift",
                "cartesian_approach",
                "fixed_object_to_eef",
                "preserve_current_object_orientation",
            }
        )
        for name, value in options.items():
            if name not in result["profile"]["skill_options"]:
                raise ValueError(
                    f"Integration override names undeclared skill {name!r}."
                )
            fields = _mapping(
                value,
                path=f"integration.overrides.profile.skill_options.{name}",
                required=frozenset(),
                optional=allowed,
            )
            result["profile"]["skill_options"][name].update(deepcopy(dict(fields)))
    return result


def _resolve_task_program_components(
    value: object,
    *,
    base_dir: Path,
) -> tuple[Path | Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    """Resolve program, task integration, and execution policy selections."""
    declaration = _mapping(
        value,
        path="task_program",
        required=frozenset({"program", "integration", "execution_policy"}),
        optional=frozenset({"execution", "action_owners"}),
    )
    execution = declaration.get("execution", "expert")
    if execution not in ("expert", "policy"):
        raise ValueError("task_program.execution must be expert or policy.")
    if "action_owners" in declaration and execution != "policy":
        raise ValueError("task_program.action_owners requires policy execution.")
    program_path = (
        _inline_component(declaration["program"], path="task_program.program")
        if isinstance(declaration["program"], Mapping)
        else _component_path(
            declaration["program"],
            base_dir=base_dir,
            path="task_program.program",
            suffixes=frozenset({".json", ".yaml", ".yml"}),
        )
    )
    integration_selector = declaration["integration"]
    overrides = None
    if (
        isinstance(integration_selector, Mapping)
        and "component" in integration_selector
    ):
        selector = _mapping(
            integration_selector,
            path="task_program.integration",
            required=frozenset({"component"}),
            optional=frozenset({"overrides"}),
        )
        integration_selector = selector["component"]
        overrides = _owned_mapping(
            selector.get("overrides", {}), path="integration.overrides"
        )
    if isinstance(integration_selector, Mapping):
        integration_path = None
        integration_source = _inline_component(
            integration_selector, path="task_program.integration"
        )
    else:
        integration_path = _component_path(
            integration_selector,
            base_dir=base_dir,
            path="task_program.integration",
            suffixes=frozenset({".yaml", ".yml"}),
        )
        integration_source = _load_yaml_component(
            integration_path, field_name="task integration"
        )
    if overrides is not None:
        integration_source = _apply_integration_overrides(integration_source, overrides)
    integration_base = base_dir if integration_path is None else integration_path.parent
    integration = _mapping(
        integration_source,
        path="task integration",
        required=frozenset(
            {
                "integration_id",
                "program_id",
                "requires",
                "scene_binding",
                "profile",
            }
        ),
        optional=frozenset({"runtime_services", "asset_requirements"}),
    )
    integration = deepcopy(dict(integration))
    if "runtime_services" in integration:
        runtime_services = _owned_mapping(
            integration["runtime_services"], path="integration.runtime_services"
        )
        if "demo_skills" in runtime_services:
            from embodichain.data._references import _resolve_data_reference

            runtime_services["demo_skills"] = _owned_mapping(
                runtime_services["demo_skills"], path="runtime_services.demo_skills"
            )
            for alias, entry in runtime_services["demo_skills"].items():
                if not isinstance(entry, dict) or "asset_manifest" not in entry:
                    raise ValueError(f"Demo Skill {alias!r} requires asset_manifest")
                entry["asset_manifest"] = str(
                    _resolve_data_reference(
                        entry["asset_manifest"], base_dir=integration_base
                    )
                )
        if "registered_semantic_lowerers" in runtime_services:
            for lowerer in runtime_services["registered_semantic_lowerers"]:
                if not isinstance(lowerer, dict):
                    continue
                field = {
                    "demo_skill": "asset_manifest",
                }.get(lowerer.get("kind"))
                if field is not None and field in lowerer:
                    from embodichain.data._references import _resolve_data_reference

                    lowerer[field] = str(
                        _resolve_data_reference(
                            lowerer[field], base_dir=integration_base
                        )
                    )
        if "nominal_motion_sources" in runtime_services:
            sources = _owned_sequence(
                runtime_services["nominal_motion_sources"],
                path="integration.runtime_services.nominal_motion_sources",
            )
            resolved_sources = []
            for index, item in enumerate(sources):
                value = _owned_mapping(
                    item,
                    path=f"integration.runtime_services.nominal_motion_sources[{index}]",
                )
                manifest_key = (
                    "asset_manifest"
                    if value.get("provider_id") == "demo_skill"
                    else "prototype_manifest"
                )
                if manifest_key not in value:
                    raise ValueError(f"nominal source requires {manifest_key}")
                from embodichain.data._references import _resolve_data_reference

                value[manifest_key] = str(
                    _component_path(
                        str(
                            _resolve_data_reference(
                                value[manifest_key],
                                base_dir=integration_base,
                            )
                        ),
                        base_dir=integration_base,
                        path=f"integration.runtime_services.nominal_motion_sources[{index}].{manifest_key}",
                        suffixes=frozenset({".json"}),
                    )
                )
                resolved_sources.append(value)
            runtime_services["nominal_motion_sources"] = resolved_sources
        integration["runtime_services"] = runtime_services
    if "asset_requirements" in integration:
        from embodichain.lab.task_program.semantics.asset_requirements import (
            AssetRequirements,
        )

        AssetRequirements.from_dict(integration["asset_requirements"])
    _, policy = _resolve_execution_policy(
        declaration["execution_policy"], base_dir=base_dir
    )
    return program_path, integration, policy


def _compose_integration_payload(
    *,
    task: Mapping[str, object],
    policy: Mapping[str, object],
    skill_profile: Mapping[str, object],
    scene: Mapping[str, object],
) -> dict[str, object]:
    """Compose the existing strict integration payload from typed owners."""
    requirements = _mapping(
        task["requires"],
        path="task integration.requires",
        required=frozenset({"scene_contract", "embodiment_contract"}),
    )
    required_scene = _identifier(
        requirements["scene_contract"],
        path="task integration.requires.scene_contract",
    )
    required_embodiment = _identifier(
        requirements["embodiment_contract"],
        path="task integration.requires.embodiment_contract",
    )
    scene_contract = _identifier(
        scene["contract_id"],
        path="scene binding.contract_id",
    )
    embodiment_contract = _identifier(
        skill_profile["contract_id"],
        path="embodiment component.skill_profile.contract_id",
    )
    policy_requirements = _mapping(
        policy["requires"],
        path="execution policy.requires",
        required=frozenset({"embodiment_contract"}),
    )
    policy_embodiment_contract = _identifier(
        policy_requirements["embodiment_contract"],
        path="execution policy.requires.embodiment_contract",
    )
    if scene_contract != required_scene:
        raise ValueError(
            f"Scene contract {scene_contract!r} does not satisfy required "
            f"contract {required_scene!r}."
        )
    if embodiment_contract != required_embodiment:
        raise ValueError(
            f"Embodiment contract {embodiment_contract!r} does not satisfy "
            f"required contract {required_embodiment!r}."
        )
    if policy_embodiment_contract != required_embodiment:
        raise ValueError(
            "Execution policy requires embodiment contract "
            f"{policy_embodiment_contract!r}, not {required_embodiment!r}."
        )

    profile = _mapping(
        _normalize_skill_options_key(task["profile"], path="task integration.profile"),
        path="task integration.profile",
        required=frozenset({"defaults", "skill_options", "effect_monitors"}),
        optional=frozenset({"skill_presets", "grounding_providers"}),
    )
    preset = {
        "preset_id": deepcopy(policy["preset_id"]),
        "skill_options": deepcopy(profile["skill_options"]),
        "motion": deepcopy(policy["motion"]),
        "tracking": deepcopy(policy["tracking"]),
        "recovery": deepcopy(policy["recovery"]),
        "workflow_recovery": deepcopy(policy["workflow_recovery"]),
        "runner": deepcopy(policy["runner"]),
        "effect_assurance": deepcopy(policy["effect_assurance"]),
        "effect_monitors": deepcopy(profile["effect_monitors"]),
    }
    if "planner" in policy:
        preset["required_planner"] = deepcopy(policy["planner"]["type"])
    elif "required_planner" in policy:
        preset["required_planner"] = deepcopy(policy["required_planner"])

    scene_payload = {
        key: deepcopy(value) for key, value in scene.items() if key != "contract_id"
    }
    robot_profile = {
        "profile_id": deepcopy(skill_profile["profile_id"]),
        "resources": deepcopy(skill_profile["resources"]),
        "command_presets": deepcopy(skill_profile["command_presets"]),
        "defaults": deepcopy(profile["defaults"]),
        "presets": [preset],
        "default_preset": None,
        "skill_presets": deepcopy(profile.get("skill_presets", {})),
        "grounding_providers": deepcopy(profile.get("grounding_providers", {})),
    }
    runtime_services = _merge_runtime_services(
        skill_profile.get("runtime_services", {}),
        task.get("runtime_services", {}),
    )
    payload: dict[str, object] = {
        "scene": scene_payload,
        "robot_profile": robot_profile,
    }
    if runtime_services:
        payload["runtime_services"] = runtime_services
    return payload


def _load_configured_task_program_deployment(
    *,
    task_program: object,
    skill_profile: object,
    base_dir: str | Path,
    evaluation: Mapping[str, object] | None = None,
    evaluation_registry: PredicateRegistry | None = None,
) -> _ConfiguredTaskProgramDeployment:
    """Compose Task Program metadata after Gym resolves physical components."""
    selected_base_dir = Path(base_dir).expanduser()
    program_path, task, policy = _resolve_task_program_components(
        task_program,
        base_dir=selected_base_dir,
    )
    from .demo_skill import _prepare_named_demo_skills, _bind_named_demo_sources

    named_demo_sources = _prepare_named_demo_skills(task)
    selected_skill_profile = _mapping(
        skill_profile,
        path="embodiment component.skill_profile",
        required=frozenset(
            {"contract_id", "profile_id", "resources", "command_presets"}
        ),
        optional=frozenset({"runtime_services", "evaluation_links"}),
    )
    scene_binding = _mapping(
        task["scene_binding"],
        path="task integration.scene_binding",
        required=frozenset({"contract_id", "registry_id"}),
        optional=frozenset(
            {
                "rigid_objects",
                "rigidized_articulations",
                "articulations",
                "links",
                "collision_world_mode",
            }
        ),
    )
    payload = _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=selected_skill_profile,
        scene=scene_binding,
    )
    planner_config = deepcopy(policy.get("planner"))
    integration = _decode_configured_task_program_integration(
        payload,
        planner_config=planner_config,
        evaluation=evaluation,
        evaluation_registry=evaluation_registry,
        evaluation_robot_links=_identifier_mapping(
            selected_skill_profile.get("evaluation_links", {}),
            path="embodiment component.skill_profile.evaluation_links",
        ),
    )
    integration_id = _identifier(
        task["integration_id"],
        path="task integration.integration_id",
    )
    program_id = _identifier(
        task["program_id"],
        path="task integration.program_id",
    )
    selection = TaskProgramIntegrationCfg(
        robot_profile=integration.registration.robot_profile_binding.profile_id,
        scene_registry=integration.registration.scene_binding.registry_id,
        runtime_preset=_identifier(
            policy["preset_id"],
            path="execution policy.preset_id",
        ),
    )
    factory = cast(_ConfiguredTaskProgramAdapterFactory, integration.adapter_factory)
    if (
        factory.nominal_motion_sources
        or named_demo_sources
        or isinstance(program_path, Mapping)
    ):
        from .nominal import _ConfiguredTaskProgramNominalSources

        catalog = integration.registration.catalog
        program = _load_deployment_program(
            program_path, selection=selection, catalog=catalog
        )
        if program.program_id != program_id:
            raise ValueError(
                "configured nominal sources require the integration's program_id"
            )
        compiled = catalog.preflight(program)
        if named_demo_sources:
            from dataclasses import replace

            factory = replace(
                factory,
                nominal_motion_sources=_bind_named_demo_sources(
                    compiled, named_demo_sources, factory.nominal_motion_sources
                ),
            )
            from dataclasses import asdict
            import hashlib
            import json

            source_identity = [
                {
                    key: value
                    for key, value in asdict(item).items()
                    if key not in {"asset_manifest", "prototype_manifest"}
                }
                for item in factory.nominal_motion_sources
            ]
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "integration": integration.integration_fingerprint,
                        "nominal_motion_sources": source_identity,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            factory = replace(factory, integration_fingerprint=fingerprint)
            integration = replace(
                integration,
                adapter_factory=factory,
                integration_fingerprint=fingerprint,
            )
        if factory.nominal_motion_sources:
            _ConfiguredTaskProgramNominalSources(
                compiled, factory.nominal_motion_sources
            )
    return _ConfiguredTaskProgramDeployment(
        integration_id=integration_id,
        program_id=program_id,
        program_path=program_path if isinstance(program_path, Path) else None,
        program_declaration=(
            deepcopy(dict(program_path)) if isinstance(program_path, Mapping) else None
        ),
        selection=selection,
        integration=integration,
        scene_binding=deepcopy(dict(scene_binding)),
        planner_config=planner_config,
    )


def _inline_component(value: object, *, path: str) -> dict[str, object]:
    """Own an explicit inline declaration without reading a component file."""
    selection = _mapping(value, path=path, required=frozenset({"inline"}))
    return _owned_mapping(selection["inline"], path=f"{path}.inline")


def _load_deployment_program(
    source: Path | Mapping[str, object],
    *,
    selection: TaskProgramIntegrationCfg,
    catalog: TaskProgramIntegrationCatalog,
) -> TaskProgramCfg:
    """Use the same strict decoder for inline and file-backed programs."""
    from embodichain.lab.task_program import decode_task_program, load_task_program
    from embodichain.lab.task_program.language.loader import (
        _bind_task_program_integration,
    )

    if isinstance(source, Mapping):
        return decode_task_program(
            _bind_task_program_integration(dict(source), selection),
            validation_context=catalog,
        )
    return load_task_program(source, integration=selection, validation_context=catalog)
