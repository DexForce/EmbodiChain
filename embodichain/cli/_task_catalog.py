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

"""Internal task catalog shared by CLI listing, details, and HTML export."""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import importlib.resources
import json
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from importlib.resources.abc import Traversable
except ModuleNotFoundError:
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
        if task_package == "embodichain_tasks":
            # Editable imports can point at a different checkout after task
            # discovery. Mirror resolve_config_path's co-located precedence.
            colocated = (
                Path(__file__).resolve().parents[2] / "embodichain_tasks/configs/tasks"
            )
            if colocated.is_dir():
                roots.append(colocated)
                continue
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
    if resource.name.lower().endswith(".json"):
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
        if any(resource.name == "catalog.yaml" for resource in config_resources):
            # Authored metadata owns all named deployments under this task.
            return

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


@dataclass
class _Deployment:
    """One concrete runnable configuration or registry-only environment."""

    name: str
    env_id: str
    config_ref: str | None
    resource: Traversable | None
    physics: str | None
    embodiments: tuple[str, ...]
    capabilities: set[str]
    agent_refs: tuple[str, ...] = ()
    validation_resource: Traversable | None = None
    validation: Mapping[str, Any] | None = None

    @property
    def command(self) -> str | None:
        if self.config_ref is None:
            return None
        path = _local_path(self.resource)
        reference = str(path) if path is not None else self.config_ref
        return f"embodichain run-env --gym_config {shlex.quote(reference)}"


@dataclass
class _Task:
    """Logical identity and deployments, independently of runtime registration."""

    package: str
    key: str
    task_path: tuple[str, ...]
    title: str
    summary: str
    tags: tuple[str, ...]
    default_deployment: str | None
    deployments: list[_Deployment]
    readme: Traversable | None = None

    @property
    def qualified_key(self) -> str:
        return f"{self.package}:{self.key}"


def _local_path(resource: Traversable | None) -> Path | None:
    if isinstance(resource, Path) and resource.is_file():
        return resource.resolve()
    return None


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _fields(value: Any, allowed: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}")
    return value


def _load_metadata(resource: Traversable) -> Mapping[str, Any]:
    """Reject duplicate authored fields as well as malformed YAML."""
    import yaml

    class CatalogLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Catalog mapping keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate catalog field: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    CatalogLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        return yaml.load(resource.read_text(encoding="utf-8"), Loader=CatalogLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Malformed catalog {resource}: {error}") from error


def _relative_resource(directory: Traversable, value: Any) -> Traversable:
    reference = _text(value, "config")
    path = PurePosixPath(reference)
    if path.is_absolute() or ".." in path.parts or "\\" in reference:
        raise ValueError(f"config must be a task-relative resource: {reference}")
    resource = directory.joinpath(*path.parts)
    if not resource.is_file():
        raise ValueError(f"Catalog references missing config: {reference}")
    if isinstance(directory, Path) and isinstance(resource, Path):
        if not resource.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"config must stay task-relative: {reference}")
    return resource


def _backend(config: Mapping[str, Any], resource: Traversable) -> str | None:
    environment = config.get("environment")
    if isinstance(environment, Mapping) and isinstance(
        environment.get("component"), str
    ):
        # Components are normally siblings or ancestors. Path resources support
        # this without importing simulator-dependent configuration decoders.
        if isinstance(resource, Path):
            component = resource.parent / environment["component"]
            if not component.is_file():
                raise ValueError(f"Missing environment component: {component}")
            return _load_mapping(component).get("physics")
    return config.get("physics")


def _reference(package: str, task_path: tuple[str, ...], name: str) -> str:
    return "/".join((package, "configs", "tasks", *task_path, name))


def _agent_matches(
    reference: Any, deployment: _Deployment, agent_resource: Traversable
) -> bool:
    """Mirror trainer reference precedence without importing the RL runtime."""
    if not isinstance(reference, str):
        return False
    reference_path = Path(reference).expanduser()
    deployment_path = _local_path(deployment.resource)
    if reference_path.is_absolute():
        return (
            deployment_path is not None and reference_path.resolve() == deployment_path
        )
    if isinstance(agent_resource, Path):
        candidate = agent_resource.parent.resolve() / reference_path
        if candidate.exists():
            return (
                deployment_path is not None and candidate.resolve() == deployment_path
            )
    if reference_path.exists():
        return (
            deployment_path is not None and reference_path.resolve() == deployment_path
        )
    # Missing repository-style references resolve through packaged resources.
    return reference == deployment.config_ref


def _load_catalog(roots: Mapping[str, Traversable]) -> list[_Task]:
    """Read static task metadata; never import environment or robot modules."""
    tasks: list[_Task] = []
    identities: set[str] = set()
    for package, root in roots.items():
        for task_path, resources, agents in _iter_task_directories(root):
            directory = root.joinpath(*task_path)
            metadata_file = directory.joinpath("catalog.yaml")
            metadata: Mapping[str, Any] = {}
            if metadata_file.is_file():
                metadata = _fields(
                    _load_metadata(metadata_file),
                    {
                        "task_key",
                        "title",
                        "summary",
                        "tags",
                        "default_deployment",
                        "deployments",
                    },
                    "catalog",
                )
                for required in (
                    "task_key",
                    "title",
                    "summary",
                    "default_deployment",
                    "deployments",
                ):
                    if required not in metadata:
                        raise ValueError(f"Missing catalog field: {required}")
            key = _text(
                metadata.get("task_key", task_path[-1] if task_path else package),
                "task_key",
            )
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
                raise ValueError(f"Invalid task_key: {key}")
            title = _text(metadata.get("title", key.replace("_", " ").title()), "title")
            summary = (
                _text(metadata["summary"], "summary")
                if metadata
                else "No authored summary."
            )
            tags = metadata.get("tags", [])
            if not isinstance(tags, list):
                raise TypeError("tags must be a list")
            tags = tuple(_text(tag, "tags entry") for tag in tags)
            selected: list[tuple[str, Traversable]] = []
            if metadata:
                declarations = metadata["deployments"]
                if not isinstance(declarations, Mapping) or not declarations:
                    raise ValueError("deployments must be a nonempty mapping")
                for name, declaration in declarations.items():
                    name = _text(name, "deployment name")
                    declaration = _fields(
                        declaration, {"config", "validation"}, "deployment"
                    )
                    selected.append(
                        (name, _relative_resource(directory, declaration.get("config")))
                    )
                default = _text(metadata["default_deployment"], "default_deployment")
                if default not in declarations:
                    raise ValueError(
                        "default_deployment must name a declared deployment"
                    )
            else:
                default = None
                selected = [(resource.name, resource) for resource in resources]
            deployments: list[_Deployment] = []
            for name, resource in selected:
                if not resource.name.lower().endswith((".json", ".yaml", ".yml")):
                    raise ValueError(
                        f"Unsupported deployment config extension: {resource.name}"
                    )
                try:
                    config = _load_mapping(resource)
                except TypeError:
                    if metadata:
                        raise
                    continue
                env_id = config.get("id")
                if (
                    not isinstance(env_id, str)
                    or not env_id.strip()
                    or env_id != env_id.strip()
                ):
                    if metadata:
                        raise ValueError(
                            f"Catalog config is not runnable (missing id): {resource}"
                        )
                    continue
                relative_name = (
                    metadata["deployments"][name]["config"]
                    if metadata
                    else resource.name
                )
                deployments.append(
                    _Deployment(
                        name,
                        env_id,
                        _reference(package, task_path, relative_name),
                        resource,
                        _backend(config, resource),
                        _config_embodiment_names(config),
                        {_TASK_PROGRAM} if "task_program" in config else set(),
                    )
                )
                if metadata and "validation" in metadata["deployments"][name]:
                    validation_resource = _relative_resource(
                        directory, metadata["deployments"][name]["validation"]
                    )
                    if not validation_resource.name.endswith(".json"):
                        raise ValueError("validation must reference a JSON report")
                    deployments[-1].validation_resource = validation_resource
                    deployments[-1].validation = _load_mapping(validation_resource)
            if agents is not None and agents.is_dir():
                for agent in sorted(agents.iterdir(), key=lambda item: item.name):
                    if not agent.is_file() or not agent.name.endswith(
                        (".json", ".yaml", ".yml")
                    ):
                        continue
                    trainer = _load_mapping(agent).get("trainer")
                    if not isinstance(trainer, Mapping):
                        continue
                    agent_ref = _reference(package, task_path, f"agents/{agent.name}")
                    for deployment in deployments:
                        if _agent_matches(trainer.get("gym_config"), deployment, agent):
                            deployment.capabilities.add(_RL)
                            deployment.agent_refs += (agent_ref,)
                    learning = trainer.get("learning_env")
                    if isinstance(learning, Mapping):
                        learning = learning.get("name")
                    if isinstance(learning, str) and learning:
                        existing = next(
                            (
                                d
                                for d in deployments
                                if d.env_id == learning and d.config_ref is None
                            ),
                            None,
                        )
                        if existing is None:
                            existing = _Deployment(
                                learning, learning, None, None, None, (), {_RL}
                            )
                            deployments.append(existing)
                        existing.agent_refs += (agent_ref,)
            if not deployments:
                continue
            identity = f"{package}:{key}".casefold()
            if identity in identities:
                raise ValueError(f"Duplicate task identity: {package}:{key}")
            identities.add(identity)
            readme = directory.joinpath("README.md")
            tasks.append(
                _Task(
                    package,
                    key,
                    task_path or (key,),
                    title,
                    summary,
                    tags,
                    default,
                    deployments,
                    readme if readme.is_file() else None,
                )
            )
    return sorted(tasks, key=lambda task: (task.task_path, task.package))


def _catalog_entries(tasks: Sequence[_Task]) -> dict[str, _EnvironmentListEntry]:
    entries: dict[str, _EnvironmentListEntry] = {}
    for task in tasks:
        for deployment in task.deployments:
            _merge_environment_entry(
                entries,
                env_id=deployment.env_id,
                task_path=task.task_path,
                capabilities=deployment.capabilities,
                embodiments=deployment.embodiments,
                config_names=(
                    () if deployment.resource is None else (deployment.resource.name,)
                ),
            )
    return entries


def _config_environment_entries(
    config_roots: Sequence[Traversable],
) -> dict[str, _EnvironmentListEntry]:
    # Retained for callers of the original private list-task helper.
    return _catalog_entries(
        _load_catalog(
            {
                "embodichain_tasks" if index == 0 else f"package_{index}": root
                for index, root in enumerate(config_roots)
            }
        )
    )


def _implements_handwritten_demo(env_cls: type[Any]) -> bool:
    """Return whether an environment overrides either demo authoring hook."""
    from embodichain.lab.gym.envs import EmbodiedEnv

    return any(
        getattr(env_cls, method_name, None)
        is not getattr(EmbodiedEnv, method_name, None)
        for method_name in ("create_demo_segments", "create_demo_action_list")
    )


def _augment_runtime(tasks: list[_Task], task_packages: Sequence[str]) -> None:
    """Augment capabilities and preserve entries only available at runtime."""
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.learning.rl.env import get_registered_learning_env_names

    for env_id, spec in REGISTERED_ENVS.items():
        matches = [
            d
            for task in tasks
            for d in task.deployments
            if d.env_id.casefold() == env_id.casefold()
        ]
        task_path = _task_path_from_module(spec.cls.__module__, task_packages)
        if not matches and task_path is None:
            continue
        capabilities = set()
        if (
            spec.task_program_registration is not None
            or spec.task_program_adapter_factory is not None
        ):
            capabilities.add(_TASK_PROGRAM)
        elif _implements_handwritten_demo(spec.cls):
            capabilities.add(_HANDWRITTEN_DEMO)
        if spec.supports_rl:
            capabilities.add(_RL)
        if not matches:
            package = max(
                (p for p in task_packages if spec.cls.__module__.startswith(p + ".")),
                key=len,
            )
            task = next(
                (t for t in tasks if t.package == package and t.task_path == task_path),
                None,
            )
            if task is None:
                task = _Task(
                    package,
                    task_path[-1],
                    task_path,
                    task_path[-1],
                    "No authored summary.",
                    (),
                    None,
                    [],
                )
                tasks.append(task)
            deployment = _Deployment(env_id, env_id, None, None, None, (), set())
            task.deployments.append(deployment)
            matches = [deployment]
        for deployment in matches:
            deployment.capabilities.update(capabilities)
    for env_id in get_registered_learning_env_names():
        matches = [
            d
            for task in tasks
            for d in task.deployments
            if d.env_id.casefold() == env_id.casefold()
        ]
        if matches:
            for deployment in matches:
                deployment.capabilities.add(_RL)
        else:
            tasks.append(
                _Task(
                    "runtime",
                    env_id,
                    ("uncategorized", env_id),
                    env_id,
                    "No authored summary.",
                    (),
                    None,
                    [_Deployment(env_id, env_id, None, None, None, (), {_RL})],
                )
            )


def _discover_catalog() -> list[_Task]:
    packages = _task_package_module_names()
    roots = {
        package: root for package in packages for root in _task_config_roots((package,))
    }
    tasks = _load_catalog(roots)
    _augment_runtime(tasks, packages)
    return tasks


def _select_task(tasks: Sequence[_Task], key: str) -> _Task:
    matches = [
        task
        for task in tasks
        if (task.qualified_key if ":" in key else task.key).casefold() == key.casefold()
    ]
    if not matches:
        raise ValueError(f"Unknown task: {key}")
    if len(matches) != 1:
        raise ValueError(
            f"Task key is ambiguous; use one of: {', '.join(task.qualified_key for task in matches)}"
        )
    return matches[0]


def _add_source_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-root",
        action="append",
        metavar="PACKAGE=PATH",
        help="Read a configs/tasks directory without importing simulator registries; repeat for multiple packages.",
    )


def _catalog_from_args(args: argparse.Namespace) -> list[_Task]:
    if args.config_root:
        roots: dict[str, Path] = {}
        for argument in args.config_root:
            package, separator, value = argument.partition("=")
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", package):
                raise ValueError("--config-root requires PACKAGE=PATH")
            if package in roots:
                raise ValueError(f"Duplicate package root: {package}")
            root = Path(value).expanduser().resolve()
            if not root.is_dir():
                raise ValueError(f"Config root is not a directory: {root}")
            roots[package] = root
        return _load_catalog(roots)
    from embodichain.lab.gym.utils.registration import discover_task_packages

    discover_task_packages()
    return _discover_catalog()


def _capability_label(deployment: _Deployment) -> str:
    return (
        ", ".join(
            label for label in _CAPABILITY_ORDER if label in deployment.capabilities
        )
        or "Environment Only"
    )


def _validation_summary(deployment: _Deployment) -> str:
    """Display supplied observations without deriving physical truth from execution."""
    report = deployment.validation
    if report is None:
        return "Qualification: unavailable"
    parts = [f"Report status: {report.get('status', 'unavailable')}"]
    code = report.get("code")
    if isinstance(code, Mapping):
        parts.append(
            f"revision={code.get('revision', 'unavailable')}, dirty={str(code.get('dirty', 'unavailable')).lower()}"
        )
    results = report.get("results")
    if isinstance(results, list) and results:
        for result in results:
            if not isinstance(result, Mapping):
                continue
            physical = result.get("physical_outcome")
            success = physical.get("success") if isinstance(physical, Mapping) else None
            physical_label = (
                str(success).lower() if type(success) is bool else "unavailable"
            )
            execution = result.get("execution_outcome")
            status = (
                execution.get("status", "unavailable")
                if isinstance(execution, Mapping)
                else "unavailable"
            )
            parts.append(
                f"{result.get('action_source', 'unspecified')}: execution={status}, physical={physical_label}"
            )
    else:
        parts.append("physical=unavailable")
    return "; ".join(parts)


def _render_html(tasks: Sequence[_Task], *, static_only: bool = False) -> str:
    """Render escaped static cards with verified local resource links."""
    escape = html.escape
    cards = []
    if static_only:
        cards.append(
            "<p>Static config catalog. Runtime registrations were not inspected; additional expert or RL capabilities may be available.</p>"
        )
    for task in tasks:
        rows = []
        for deployment in task.deployments:
            path = _local_path(deployment.resource)
            source = (
                f'<a href="{escape(path.as_uri(), quote=True)}">{escape(deployment.resource.name)}</a>'
                if path is not None
                else "unavailable"
            )
            validation_path = _local_path(deployment.validation_resource)
            validation_link = (
                f'<a href="{escape(validation_path.as_uri(), quote=True)}">Validation report</a><br>'
                if validation_path
                else ""
            )
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{escape(value)}</td>"
                    for value in (
                        deployment.name,
                        deployment.env_id,
                        deployment.physics or "unavailable",
                        ", ".join(deployment.embodiments) or "unavailable",
                        _capability_label(deployment),
                    )
                )
                + f"<td>{source}</td><td><code>{escape(deployment.command or 'No runnable config')}</code></td>"
                + f"<td>{escape(', '.join(deployment.agent_refs) or 'None')}</td><td>{validation_link}{escape(_validation_summary(deployment))}</td></tr>"
            )
        readme = _local_path(task.readme)
        readme_link = (
            f'<p><a href="{escape(readme.as_uri(), quote=True)}">Task README</a></p>'
            if readme
            else ""
        )
        cards.append(
            f"<article><h2>{escape(task.title)}</h2><p>{escape(task.qualified_key)} · {escape('/'.join(task.task_path[:-1]))}</p>"
            f"<p>{escape(task.summary)}</p><p>Tags: {escape(', '.join(task.tags) or 'None')} · Default: {escape(task.default_deployment or 'unspecified')}</p>"
            "<p>Preview: unavailable. Validation reports describe observed outcomes for the recorded run only.</p>"
            + readme_link
            + "<div class='scroll'><table><thead><tr><th>Deployment</th><th>Environment</th><th>Physics</th><th>Embodiment</th><th>Supported uses</th><th>Source</th><th>Run command</th><th>Agents</th><th>Qualification</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div></article>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>EmbodiChain Task Catalog</title><style>body{font:16px system-ui;margin:2rem;background:#f4f6f8;color:#17212b}article{background:white;border:1px solid #ccd5df;border-radius:12px;padding:1.5rem;margin:1.5rem 0}table{border-collapse:collapse}td,th{padding:.6rem;border:1px solid #ccd5df;text-align:left}code{white-space:pre-wrap}.scroll{overflow:auto}</style><h1>EmbodiChain Task Catalog</h1>"
        + "".join(cards)
        + "</html>"
    )


__all__: list[str] = []
