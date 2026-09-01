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

"""List tasks available through installed task packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.resources
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import PurePosixPath
from typing import Any

try:
    from importlib.resources.abc import Traversable
except ModuleNotFoundError:  # Python 3.10 compatibility.
    from importlib.abc import Traversable


_TASK_PROGRAM = "Expert Demo: Task Program"
_HANDWRITTEN_DEMO = "Expert Demo: Handwritten Trajectory"
_RL = "RL"
_CAPABILITY_ORDER = (_TASK_PROGRAM, _HANDWRITTEN_DEMO, _RL)


@dataclass
class _EnvironmentListEntry:
    """One environment and its task-catalog metadata."""

    env_id: str
    task_path: tuple[str, ...]
    capabilities: set[str]
    embodiments: set[str] = field(default_factory=set)
    config_names: set[str] = field(default_factory=set)


def _task_package_module_names() -> tuple[str, ...]:
    """Return module names declared by installed task-package entry points."""
    try:
        entry_points = importlib.metadata.entry_points(group="embodichain.tasks")
    except TypeError:
        entry_points = importlib.metadata.entry_points().get("embodichain.tasks", [])

    module_names: list[str] = []
    for entry_point in entry_points:
        module_name = getattr(entry_point, "module", None)
        if module_name is None:
            module_name = entry_point.value.partition(":")[0]
        if module_name not in module_names:
            module_names.append(module_name)
    return tuple(module_names)


def _task_path_from_module(
    module_name: str,
    task_package_modules: Sequence[str],
) -> tuple[str, ...] | None:
    """Resolve a task module to its path below an installed task package."""
    matching_packages = [
        package
        for package in task_package_modules
        if module_name == package or module_name.startswith(package + ".")
    ]
    if not matching_packages:
        return None
    package = max(matching_packages, key=len)
    relative_module = module_name.removeprefix(package).removeprefix(".")
    if not relative_module:
        return None
    return tuple(relative_module.split("."))


def _task_config_roots(
    task_package_modules: Sequence[str],
) -> tuple[Traversable, ...]:
    """Locate packaged ``configs/tasks`` trees for installed task packages."""
    roots: list[Traversable] = []
    visited_packages: set[str] = set()
    for task_package in task_package_modules:
        config_packages = (
            f"{task_package}.configs",
            f"{task_package.partition('.')[0]}.configs",
        )
        for config_package in config_packages:
            if config_package in visited_packages:
                continue
            visited_packages.add(config_package)
            try:
                root = importlib.resources.files(config_package).joinpath("tasks")
            except (ModuleNotFoundError, TypeError):
                continue
            if root.is_dir():
                roots.append(root)
                break
    return tuple(roots)


def _load_mapping(resource: Traversable) -> Mapping[str, Any]:
    """Load one JSON or YAML task configuration as a mapping."""
    text = resource.read_text(encoding="utf-8")
    if resource.name.endswith(".json"):
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise TypeError(f"Task config must contain a mapping: {resource}")
    return value


def _iter_task_directories(
    root: Traversable,
    relative_path: tuple[str, ...] = (),
):
    """Yield task directories and their paths below a config root."""
    children = sorted(root.iterdir(), key=lambda child: child.name.casefold())
    child_by_name = {child.name: child for child in children}
    config_resources = [
        child
        for child in children
        if child.is_file() and child.name.endswith((".json", ".yaml", ".yml"))
    ]
    agents = child_by_name.get("agents")
    if config_resources or (agents is not None and agents.is_dir()):
        yield relative_path, config_resources, agents

    for child in children:
        if not child.is_dir() or child.name in {"agents", "task_program"}:
            continue
        yield from _iter_task_directories(child, (*relative_path, child.name))


def _config_embodiment_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Infer display names for the embodiment selected by a task config."""
    embodiment = config.get("embodiment")
    if isinstance(embodiment, Mapping):
        component = embodiment.get("component")
        if type(component) is str and component and component == component.strip():
            component_name = PurePosixPath(component).stem
            if component_name:
                return (component_name,)

    robot = config.get("robot")
    if isinstance(robot, Mapping):
        for field_name in ("robot_type", "uid"):
            value = robot.get(field_name)
            if type(value) is str and value and value == value.strip():
                return (value,)
    return ()


def _merge_environment_entry(
    entries: dict[str, _EnvironmentListEntry],
    *,
    env_id: str,
    task_path: tuple[str, ...],
    capabilities: Sequence[str] = (),
    embodiments: Sequence[str] = (),
    config_names: Sequence[str] = (),
) -> _EnvironmentListEntry:
    """Merge one catalog source into a case-insensitive environment record."""
    key = env_id.casefold()
    entry = entries.get(key)
    if entry is None:
        entry = _EnvironmentListEntry(env_id, task_path, set())
        entries[key] = entry
    entry.capabilities.update(capabilities)
    entry.embodiments.update(embodiments)
    entry.config_names.update(config_names)
    return entry


def _config_environment_entries(
    config_roots: Sequence[Traversable],
) -> dict[str, _EnvironmentListEntry]:
    """Collect environment metadata encoded by task-local configs."""
    entries: dict[str, _EnvironmentListEntry] = {}
    for root in config_roots:
        for task_path, config_resources, agents in _iter_task_directories(root):
            env_ids: list[str] = []
            for resource in config_resources:
                try:
                    config = _load_mapping(resource)
                except TypeError:
                    continue
                env_id = config.get("id")
                if type(env_id) is not str or not env_id or env_id != env_id.strip():
                    continue
                capabilities = []
                if "task_program" in config:
                    capabilities.append(_TASK_PROGRAM)
                _merge_environment_entry(
                    entries,
                    env_id=env_id,
                    task_path=task_path,
                    capabilities=capabilities,
                    embodiments=_config_embodiment_names(config),
                    config_names=(resource.name,),
                )
                env_ids.append(env_id)

            if agents is None or not agents.is_dir():
                continue
            agent_resources = sorted(
                (
                    child
                    for child in agents.iterdir()
                    if child.is_file()
                    and child.name.endswith((".json", ".yaml", ".yml"))
                ),
                key=lambda child: child.name.casefold(),
            )
            if agent_resources:
                for env_id in env_ids:
                    _merge_environment_entry(
                        entries,
                        env_id=env_id,
                        task_path=task_path,
                        capabilities=(_RL,),
                    )
            for resource in agent_resources:
                config = _load_mapping(resource)
                trainer = config.get("trainer")
                if not isinstance(trainer, Mapping):
                    continue
                learning_env = trainer.get("learning_env")
                if isinstance(learning_env, Mapping):
                    learning_env = learning_env.get("name")
                if isinstance(learning_env, str) and learning_env:
                    _merge_environment_entry(
                        entries,
                        env_id=learning_env,
                        task_path=task_path,
                        capabilities=(_RL,),
                    )
    return entries


def _implements_handwritten_demo(env_cls: type[Any]) -> bool:
    """Return whether an environment overrides either demo authoring hook."""
    from embodichain.lab.gym.envs import EmbodiedEnv

    return any(
        getattr(env_cls, method_name, None)
        is not getattr(EmbodiedEnv, method_name, None)
        for method_name in ("create_demo_segments", "create_demo_action_list")
    )


def _collect_environment_entries() -> list[_EnvironmentListEntry]:
    """Build the task listing from packaged configs and runtime registries."""
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.learning.rl.env import get_registered_learning_env_names

    task_packages = _task_package_module_names()
    entries = _config_environment_entries(_task_config_roots(task_packages))

    for env_id, spec in REGISTERED_ENVS.items():
        task_path = _task_path_from_module(spec.cls.__module__, task_packages)
        existing = entries.get(env_id.casefold())
        if task_path is None and existing is None:
            continue
        capabilities: list[str] = []
        if (
            spec.task_program_registration is not None
            or spec.task_program_adapter_factory is not None
        ):
            capabilities.append(_TASK_PROGRAM)
        elif _implements_handwritten_demo(spec.cls):
            capabilities.append(_HANDWRITTEN_DEMO)
        if spec.supports_rl:
            capabilities.append(_RL)
        _merge_environment_entry(
            entries,
            env_id=env_id,
            task_path=task_path if existing is None else existing.task_path,
            capabilities=capabilities,
        )

    for env_id in get_registered_learning_env_names():
        existing = entries.get(env_id.casefold())
        _merge_environment_entry(
            entries,
            env_id=env_id,
            task_path=(
                ("uncategorized", env_id) if existing is None else existing.task_path
            ),
            capabilities=(_RL,),
        )

    return sorted(
        entries.values(),
        key=lambda entry: (
            tuple(part.casefold() for part in entry.task_path),
            entry.env_id.casefold(),
        ),
    )


def _print_environment_entries(entries: Sequence[_EnvironmentListEntry]) -> None:
    """Print environment entries as a table with a task-directory tree."""
    from prettytable import PrettyTable

    task_groups = [
        (task_path, list(task_entries))
        for task_path, task_entries in groupby(
            entries,
            key=lambda entry: entry.task_path,
        )
    ]
    table = PrettyTable()
    table.title = f"Tasks ({len(task_groups)}) / Environments ({len(entries)})"
    table.field_names = [
        "Task",
        "Environment ID",
        "Embodiment",
        "Capability",
        "Config",
    ]
    table.align = "l"
    active_categories: tuple[str, ...] = ()
    for group_index, (task_path, task_entries) in enumerate(task_groups):
        categories = task_path[:-1]
        shared_depth = 0
        for active, category in zip(active_categories, categories):
            if active != category:
                break
            shared_depth += 1
        for depth in range(shared_depth, len(categories)):
            table.add_row([f"{'  ' * depth}{categories[depth]}/", "", "", "", ""])

        task_name = task_path[-1]
        for entry_index, entry in enumerate(task_entries):
            labels = [
                capability
                for capability in _CAPABILITY_ORDER
                if capability in entry.capabilities
            ]
            if not labels:
                labels.append("Environment Only")
            table.add_row(
                [
                    (
                        f"{'  ' * len(categories)}{task_name}"
                        if entry_index == 0
                        else ""
                    ),
                    entry.env_id,
                    ", ".join(sorted(entry.embodiments, key=str.casefold)) or "-",
                    ", ".join(labels),
                    ", ".join(sorted(entry.config_names, key=str.casefold)) or "-",
                ]
            )
        if group_index < len(task_groups) - 1:
            table.add_divider()
        active_categories = categories
    print(table)


def main(argv: Sequence[str] | None = None) -> None:
    """List discovered EmbodiChain tasks.

    Args:
        argv: Arguments excluding the command name.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain list-task",
        description="List tasks by category, deployment, and capability.",
        epilog=(
            "Environment Only means the task currently exposes neither an "
            "Expert Demo entry point nor a supported RL configuration."
        ),
    )
    parser.parse_args(argv)

    from embodichain.lab.gym.utils.registration import discover_task_packages

    discover_task_packages()
    entries = _collect_environment_entries()
    if not entries:
        print("No registered tasks found.")
        return
    _print_environment_entries(entries)


if __name__ == "__main__":
    main()


__all__: list[str] = []
