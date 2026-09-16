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

"""Resolve reusable environment and embodiment Gym components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from embodichain.utils.utility import load_config

__all__: list[str] = []

_EMBODIMENT_OVERRIDE_FIELDS = frozenset({"uid", "init_pos", "init_rot", "init_qpos"})
_ENVIRONMENT_COMPONENT_FIELDS = frozenset(
    {
        "max_episodes",
        "demo_max_attempts",
        "max_episode_steps",
        "num_envs",
        "arena_space",
        "physics",
        "physics_config",
        "render_cfg",
        "visualization",
        "simulation",
        "env",
    }
)
_ENVIRONMENT_DEPLOYMENT_OVERRIDE_FIELDS = frozenset(
    {
        "max_episodes",
        "num_envs",
        "arena_space",
        "visualization",
    }
)
_SCENE_SIMULATION_FIELDS = frozenset(
    {
        "light",
        "background",
        "rigid_object",
        "rigid_object_group",
        "articulation",
    }
)


@dataclass(frozen=True, slots=True)
class _ResolvedGymComponents:
    """One Gym config after its reusable components are expanded."""

    config: dict[str, object]
    embodiment_selected: bool
    scene_selected: bool
    embodiment_skill_profile: dict[str, object] | None
    scene_config: dict[str, object] | None


def _mapping(
    value: object,
    *,
    path: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    """Return one strict string-keyed mapping with an exact field set."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    if not all(type(key) is str for key in value):
        raise TypeError(f"{path} keys must be exact strings.")
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}.")
    unexpected = sorted(set(value).difference(required | optional))
    if unexpected:
        raise ValueError(f"{path} contains unsupported fields: {unexpected}.")
    return value


def _identifier(value: object, *, path: str) -> str:
    """Return one non-empty exact string without outer whitespace."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{path} must be a non-empty string without outer whitespace.")
    return value


def _component_path(
    value: object,
    *,
    base_dir: Path,
    path: str,
) -> Path:
    """Resolve one YAML component path relative to its owner."""
    selected = _identifier(value, path=path)
    component_path = Path(selected).expanduser()
    if not component_path.is_absolute():
        component_path = base_dir / component_path
    component_path = component_path.resolve()
    if component_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{path} must be a YAML file: {component_path}.")
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
    return _owned_mapping(load_config(path), path=field_name)


def _resolve_environment_component(
    config: Mapping[str, object],
    *,
    base_dir: Path,
) -> dict[str, object]:
    """Expand one backend-specific reusable physical environment component."""
    resolved = _owned_mapping(config, path="Gym deployment")
    declaration = _mapping(
        resolved.pop("environment"),
        path="environment",
        required=frozenset({"component"}),
    )
    component_path = _component_path(
        declaration["component"],
        base_dir=base_dir,
        path="environment.component",
    )
    component = _mapping(
        _load_yaml_component(component_path, field_name="environment component"),
        path="environment component",
        required=frozenset({"environment_id", "physics", "simulation", "env"}),
        optional=_ENVIRONMENT_COMPONENT_FIELDS - {"physics", "simulation", "env"},
    )
    _identifier(
        component["environment_id"],
        path="environment component.environment_id",
    )
    simulation = _mapping(
        component["simulation"],
        path="environment component.simulation",
        required=frozenset(),
        optional=_SCENE_SIMULATION_FIELDS,
    )
    environment_values = {
        key: deepcopy(value)
        for key, value in component.items()
        if key not in {"environment_id", "simulation"}
    }
    environment_values.update(deepcopy(dict(simulation)))

    deployment_physics_fields = sorted(
        {"physics", "physics_config"}.intersection(resolved)
    )
    if deployment_physics_fields:
        raise ValueError(
            "environment.component owns physics and physics_config; remove "
            f"{deployment_physics_fields} from the Gym deployment."
        )

    duplicate_fields = sorted(
        set(environment_values).intersection(resolved)
        - _ENVIRONMENT_DEPLOYMENT_OVERRIDE_FIELDS
    )
    if duplicate_fields:
        raise ValueError(
            "environment.component fields must not also appear in the Gym "
            "deployment: "
            f"{duplicate_fields}."
        )
    if "scene" in resolved:
        raise ValueError(
            "environment.component owns the physical scene; remove scene.component."
        )
    for field_name, value in environment_values.items():
        if field_name not in resolved:
            resolved[field_name] = value
    return resolved


def _resolve_embodiment_component(
    value: object,
    *,
    base_dir: Path,
) -> tuple[dict[str, object], list[object], dict[str, object] | None]:
    """Resolve robot, sensors, and an optional Task Program skill profile."""
    declaration = _mapping(
        value,
        path="embodiment",
        required=frozenset({"component"}),
        optional=frozenset({"overrides"}),
    )
    component_path = _component_path(
        declaration["component"],
        base_dir=base_dir,
        path="embodiment.component",
    )
    component = _mapping(
        _load_yaml_component(component_path, field_name="embodiment component"),
        path="embodiment component",
        required=frozenset({"embodiment_id", "simulation", "sensor"}),
        optional=frozenset({"skill_profile"}),
    )
    _identifier(component["embodiment_id"], path="embodiment component.embodiment_id")
    simulation = _owned_mapping(
        component["simulation"],
        path="embodiment component.simulation",
    )
    sensors = _owned_sequence(
        component["sensor"],
        path="embodiment component.sensor",
    )
    overrides = _mapping(
        declaration.get("overrides", {}),
        path="embodiment.overrides",
        required=frozenset(),
        optional=_EMBODIMENT_OVERRIDE_FIELDS,
    )
    simulation.update(deepcopy(dict(overrides)))
    skill_profile = None
    if "skill_profile" in component:
        skill_profile = _owned_mapping(
            component["skill_profile"],
            path="embodiment component.skill_profile",
        )
    return simulation, sensors, skill_profile


def _resolve_scene_component(
    value: object,
    *,
    base_dir: Path,
) -> dict[str, object]:
    """Resolve physical scene simulation fields."""
    declaration = _mapping(
        value,
        path="scene",
        required=frozenset({"component"}),
    )
    component_path = _component_path(
        declaration["component"],
        base_dir=base_dir,
        path="scene.component",
    )
    component = _mapping(
        _load_yaml_component(component_path, field_name="scene component"),
        path="scene component",
        required=frozenset({"scene_id", "simulation"}),
    )
    _identifier(component["scene_id"], path="scene component.scene_id")
    simulation = _mapping(
        component["simulation"],
        path="scene component.simulation",
        required=frozenset(),
        optional=_SCENE_SIMULATION_FIELDS,
    )
    return deepcopy(dict(simulation))


def _physical_scene_uids(
    simulation: Mapping[str, object],
    *,
    field_names: tuple[str, ...],
) -> frozenset[str]:
    """Collect exact native entity IDs from selected physical scene fields."""
    identifiers: set[str] = set()
    for field_name in field_names:
        values = simulation.get(field_name, ())
        for index, value in enumerate(
            _owned_sequence(values, path=f"physical environment.{field_name}")
        ):
            entity = _owned_mapping(
                value,
                path=f"physical environment.{field_name}[{index}]",
            )
            identifiers.add(
                _identifier(
                    entity.get("uid"),
                    path=f"physical environment.{field_name}[{index}].uid",
                )
            )
    return frozenset(identifiers)


def _validate_scene_binding_targets(
    binding: Mapping[str, object],
    *,
    simulation: Mapping[str, object],
) -> None:
    """Fail before runtime when semantic roots select absent native entities."""
    physical_ids = {
        "rigid_objects": _physical_scene_uids(
            simulation,
            field_names=("background", "rigid_object"),
        ),
        "articulations": _physical_scene_uids(
            simulation,
            field_names=("articulation",),
        ),
    }
    for binding_field, available in physical_ids.items():
        values = binding.get(binding_field, ())
        for index, value in enumerate(
            _owned_sequence(values, path=f"scene binding.{binding_field}")
        ):
            entity = _owned_mapping(
                value,
                path=f"scene binding.{binding_field}[{index}]",
            )
            entity_id = _identifier(
                entity.get("entity_id"),
                path=f"scene binding.{binding_field}[{index}].entity_id",
            )
            simulation_uid = _identifier(
                entity.get("simulation_uid", entity_id),
                path=f"scene binding.{binding_field}[{index}].simulation_uid",
            )
            if simulation_uid not in available:
                raise ValueError(
                    f"scene binding {binding_field}[{index}] selects simulation UID "
                    f"{simulation_uid!r}, which is not declared by the physical "
                    "environment."
                )


def _resolve_gym_components(
    config: Mapping[str, object],
    *,
    base_dir: str | Path,
) -> _ResolvedGymComponents:
    """Expand optional environment, embodiment, and scene selectors."""
    resolved = _owned_mapping(config, path="Gym config")
    selected_base_dir = Path(base_dir).expanduser()

    if "task" in resolved:
        raise ValueError(
            "task.component has been removed; use environment.component and "
            "declare task_program in the runnable task deployment."
        )
    environment_selected = "environment" in resolved
    if environment_selected:
        resolved = _resolve_environment_component(
            resolved,
            base_dir=selected_base_dir,
        )

    embodiment_selected = "embodiment" in resolved
    embodiment_skill_profile = None
    if embodiment_selected:
        duplicate_fields = sorted(
            field for field in ("robot", "sensor") if field in resolved
        )
        if duplicate_fields:
            raise ValueError(
                "embodiment.component owns robot and sensor configuration; "
                f"remove top-level fields {duplicate_fields}."
            )
        robot, sensors, embodiment_skill_profile = _resolve_embodiment_component(
            resolved.pop("embodiment"),
            base_dir=selected_base_dir,
        )
        resolved["robot"] = robot
        resolved["sensor"] = sensors

    scene_component_selected = "scene" in resolved
    scene_config = None
    if scene_component_selected:
        scene_config = _resolve_scene_component(
            resolved.pop("scene"),
            base_dir=selected_base_dir,
        )
        duplicate_fields = sorted(set(scene_config).intersection(resolved))
        if duplicate_fields:
            raise ValueError(
                "scene.component fields must not also appear in the Gym config: "
                f"{duplicate_fields}."
            )
        resolved.update(scene_config)
    elif environment_selected or any(
        field_name in resolved for field_name in _SCENE_SIMULATION_FIELDS
    ):
        scene_config = {
            field_name: deepcopy(resolved[field_name])
            for field_name in _SCENE_SIMULATION_FIELDS
            if field_name in resolved
        }

    return _ResolvedGymComponents(
        config=resolved,
        embodiment_selected=embodiment_selected,
        scene_selected=environment_selected
        or scene_component_selected
        or scene_config is not None,
        embodiment_skill_profile=embodiment_skill_profile,
        scene_config=scene_config,
    )
