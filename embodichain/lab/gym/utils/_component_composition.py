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

"""Resolve reusable physical components in Gym deployment configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from embodichain.utils.utility import load_config

__all__: list[str] = []

_EMBODIMENT_OVERRIDE_FIELDS = frozenset({"uid", "init_pos", "init_rot", "init_qpos"})
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
    """One Gym config after its reusable physical components are expanded."""

    config: dict[str, object]
    embodiment_selected: bool
    scene_selected: bool
    embodiment_skill_profile: dict[str, object] | None
    scene_task_program: dict[str, object] | None


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


def _component_path(value: object, *, base_dir: Path, path: str) -> Path:
    """Resolve one YAML component path relative to its Gym config."""
    selected = _identifier(value, path=path)
    component_path = Path(selected).expanduser()
    if not component_path.is_absolute():
        component_path = base_dir / component_path
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
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Resolve scene simulation fields and optional Task Program metadata."""
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
        optional=frozenset({"task_program"}),
    )
    _identifier(component["scene_id"], path="scene component.scene_id")
    simulation = _mapping(
        component["simulation"],
        path="scene component.simulation",
        required=frozenset(),
        optional=_SCENE_SIMULATION_FIELDS,
    )
    task_program = None
    if "task_program" in component:
        task_program = _owned_mapping(
            component["task_program"],
            path="scene component.task_program",
        )
    return deepcopy(dict(simulation)), task_program


def _resolve_gym_components(
    config: Mapping[str, object],
    *,
    base_dir: str | Path,
) -> _ResolvedGymComponents:
    """Expand optional embodiment and scene selectors in any Gym config."""
    resolved = _owned_mapping(config, path="Gym config")
    selected_base_dir = Path(base_dir).expanduser()

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

    scene_selected = "scene" in resolved
    scene_task_program = None
    if scene_selected:
        scene_config, scene_task_program = _resolve_scene_component(
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

    return _ResolvedGymComponents(
        config=resolved,
        embodiment_selected=embodiment_selected,
        scene_selected=scene_selected,
        embodiment_skill_profile=embodiment_skill_profile,
        scene_task_program=scene_task_program,
    )
