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
from dataclasses import dataclass
from pathlib import Path

from embodichain.lab.task_program.language.schema import TaskProgramIntegrationCfg
from embodichain.utils.utility import load_config

from .configured import (
    _ConfiguredTaskProgramIntegration,
    _decode_configured_task_program_integration,
    _identifier,
    _mapping,
)

__all__: list[str] = []

_RUNTIME_SERVICE_FIELDS = frozenset(
    {
        "grasp_pose_generators",
        "handover_pose_providers",
        "registered_semantic_lowerers",
        "control_part_evidence",
    }
)
_TASK_RUNTIME_SERVICE_FIELDS = _RUNTIME_SERVICE_FIELDS | frozenset(
    {"grasp_pose_generator_overrides"}
)
_GRASP_GENERATOR_OVERRIDE_FIELDS = frozenset(
    {
        "sample_count",
        "approach_direction_samples",
        "opening_margin",
        "point_sample_density",
        "filter_ground_collision",
        "force_refresh",
    }
)


@dataclass(frozen=True, slots=True)
class _ConfiguredTaskProgramDeployment:
    """One fully resolved configured Task Program deployment."""

    integration_id: str
    program_id: str
    program_path: Path
    selection: TaskProgramIntegrationCfg
    integration: _ConfiguredTaskProgramIntegration


def _component_path(
    value: object,
    *,
    base_dir: Path,
    path: str,
    suffixes: frozenset[str],
) -> Path:
    """Resolve one exact component path relative to the Gym config."""
    selected = _identifier(value, path=path)
    component_path = Path(selected).expanduser()
    if not component_path.is_absolute():
        component_path = base_dir / component_path
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
    return result


def _resolve_task_program_components(
    value: object,
    *,
    base_dir: Path,
) -> tuple[Path, Mapping[str, object], Mapping[str, object]]:
    """Resolve program, task integration, and execution policy files."""
    declaration = _mapping(
        value,
        path="task_program",
        required=frozenset({"program", "integration", "execution_policy"}),
    )
    program_path = _component_path(
        declaration["program"],
        base_dir=base_dir,
        path="task_program.program",
        suffixes=frozenset({".json", ".yaml", ".yml"}),
    )
    integration_path = _component_path(
        declaration["integration"],
        base_dir=base_dir,
        path="task_program.integration",
        suffixes=frozenset({".yaml", ".yml"}),
    )
    policy_path = _component_path(
        declaration["execution_policy"],
        base_dir=base_dir,
        path="task_program.execution_policy",
        suffixes=frozenset({".yaml", ".yml"}),
    )
    integration = _mapping(
        _load_yaml_component(integration_path, field_name="task integration"),
        path="task integration",
        required=frozenset({"integration_id", "program_id", "requires", "profile"}),
        optional=frozenset({"runtime_services"}),
    )
    policy = _mapping(
        _load_yaml_component(policy_path, field_name="execution policy"),
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
        optional=frozenset({"required_planner"}),
    )
    _identifier(policy["policy_id"], path="execution policy.policy_id")
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
        path="scene component.task_program.contract_id",
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
        task["profile"],
        path="task integration.profile",
        required=frozenset({"defaults", "action_options", "effect_monitors"}),
        optional=frozenset({"skill_presets", "grounding_providers"}),
    )
    preset = {
        "preset_id": deepcopy(policy["preset_id"]),
        "action_options": deepcopy(profile["action_options"]),
        "motion": deepcopy(policy["motion"]),
        "tracking": deepcopy(policy["tracking"]),
        "recovery": deepcopy(policy["recovery"]),
        "workflow_recovery": deepcopy(policy["workflow_recovery"]),
        "runner": deepcopy(policy["runner"]),
        "effect_assurance": deepcopy(policy["effect_assurance"]),
        "effect_monitors": deepcopy(profile["effect_monitors"]),
    }
    if "required_planner" in policy:
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
    scene: object,
    base_dir: str | Path,
) -> _ConfiguredTaskProgramDeployment:
    """Compose Task Program metadata after Gym resolves physical components."""
    selected_base_dir = Path(base_dir).expanduser()
    program_path, task, policy = _resolve_task_program_components(
        task_program,
        base_dir=selected_base_dir,
    )
    selected_skill_profile = _mapping(
        skill_profile,
        path="embodiment component.skill_profile",
        required=frozenset(
            {"contract_id", "profile_id", "resources", "command_presets"}
        ),
        optional=frozenset({"runtime_services"}),
    )
    scene_task_program = _mapping(
        scene,
        path="scene component.task_program",
        required=frozenset({"contract_id", "registry_id"}),
        optional=frozenset(
            {
                "rigid_objects",
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
        scene=scene_task_program,
    )
    integration = _decode_configured_task_program_integration(payload)
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
    return _ConfiguredTaskProgramDeployment(
        integration_id=integration_id,
        program_id=program_id,
        program_path=program_path,
        selection=selection,
        integration=integration,
    )
