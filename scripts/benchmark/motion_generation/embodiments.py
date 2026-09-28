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

"""Static embodiment resolution for configuration-driven benchmark suites."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import yaml

from .config import EmbodimentSpecCfg, RobotSpecCfg, stable_hash
from .contracts import CapabilityDecision

__all__ = [
    "EmbodimentResolution",
    "EmbodimentProvider",
    "YamlEmbodimentProvider",
    "check_embodiment_capabilities",
    "resolve_embodiment",
]


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    """Validate one YAML mapping with string keys."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping with string keys.")
    return value


@dataclass(frozen=True, slots=True)
class EmbodimentResolution:
    """Resolved embodiment metadata used by cases and benchmark artifacts."""

    embodiment_id: str
    component_path: Path | None
    config_hash: str
    capabilities: frozenset[str]
    endpoint_bindings: Mapping[str, Mapping[str, str]]
    runtime_services: Mapping[str, object]
    source: str

    def __post_init__(self) -> None:
        if not self.embodiment_id or self.embodiment_id != self.embodiment_id.strip():
            raise ValueError("embodiment_id must be a non-empty string.")
        if not self.config_hash:
            raise ValueError("config_hash must be non-empty.")
        if not self.source:
            raise ValueError("source must be non-empty.")
        endpoint_bindings = {
            str(resource): {
                str(endpoint): str(control_part)
                for endpoint, control_part in endpoints.items()
            }
            for resource, endpoints in self.endpoint_bindings.items()
        }
        if any(not endpoints for endpoints in endpoint_bindings.values()):
            raise ValueError("endpoint_bindings must not contain empty resources.")
        object.__setattr__(
            self,
            "capabilities",
            frozenset(str(value) for value in self.capabilities),
        )
        object.__setattr__(
            self, "endpoint_bindings", MappingProxyType(endpoint_bindings)
        )
        object.__setattr__(
            self,
            "runtime_services",
            MappingProxyType(dict(self.runtime_services)),
        )

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-safe resolved embodiment metadata."""
        return {
            "embodiment_id": self.embodiment_id,
            "component": (
                None if self.component_path is None else str(self.component_path)
            ),
            "config_hash": self.config_hash,
            "capabilities": sorted(self.capabilities),
            "endpoint_bindings": {
                resource: dict(endpoints)
                for resource, endpoints in self.endpoint_bindings.items()
            },
            "runtime_services": dict(self.runtime_services),
            "source": self.source,
        }


@runtime_checkable
class EmbodimentProvider(Protocol):
    """Resolve one configured embodiment without creating a simulator robot."""

    def resolve(
        self,
        spec: EmbodimentSpecCfg,
        *,
        base_dir: Path | None = None,
    ) -> EmbodimentResolution:
        """Resolve component metadata and endpoint capabilities."""
        ...


class YamlEmbodimentProvider:
    """Resolve official embodiment YAML metadata for benchmark preflight."""

    def resolve(
        self,
        spec: EmbodimentSpecCfg,
        *,
        base_dir: Path | None = None,
    ) -> EmbodimentResolution:
        """Load, validate, and hash one embodiment component."""
        if not isinstance(spec, EmbodimentSpecCfg):
            raise TypeError("spec must be an EmbodimentSpecCfg.")
        path = _resolve_component_path(spec.component, base_dir=base_dir)
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        component = _mapping(raw, name="embodiment component")
        embodiment_id = component.get("embodiment_id")
        if not isinstance(embodiment_id, str) or not embodiment_id.strip():
            raise ValueError("embodiment component.embodiment_id must be non-empty.")
        _mapping(component.get("simulation"), name="embodiment component.simulation")
        sensor = component.get("sensor", [])
        if not isinstance(sensor, list):
            raise TypeError("embodiment component.sensor must be a list.")
        skill_profile = component.get("skill_profile", {})
        skill_profile = _mapping(
            skill_profile, name="embodiment component.skill_profile"
        )
        endpoints: dict[str, dict[str, str]] = {}
        capabilities: set[str] = set()
        resources = skill_profile.get("resources", [])
        if not isinstance(resources, list):
            raise TypeError("embodiment skill_profile.resources must be a list.")
        for resource in resources:
            resource_mapping = _mapping(resource, name="embodiment resource")
            resource_id = resource_mapping.get("resource_id")
            if not isinstance(resource_id, str) or not resource_id.strip():
                raise ValueError("embodiment resources need resource_id.")
            endpoint_map: dict[str, str] = {}
            raw_endpoints = resource_mapping.get("endpoints", [])
            if not isinstance(raw_endpoints, list):
                raise TypeError("embodiment resource endpoints must be a list.")
            for endpoint in raw_endpoints:
                endpoint_mapping = _mapping(endpoint, name="embodiment endpoint")
                endpoint_id = endpoint_mapping.get("endpoint_id")
                control_part = endpoint_mapping.get("control_part")
                if not isinstance(endpoint_id, str) or not endpoint_id.strip():
                    raise ValueError("embodiment endpoints need endpoint_id.")
                if not isinstance(control_part, str) or not control_part.strip():
                    raise ValueError("embodiment endpoints need control_part.")
                endpoint_map[endpoint_id] = control_part
                raw_capabilities = endpoint_mapping.get("capabilities", [])
                if not isinstance(raw_capabilities, list):
                    raise TypeError("endpoint capabilities must be a list.")
                capabilities.update(str(value) for value in raw_capabilities)
            endpoints[resource_id] = endpoint_map
        resolved = {
            "component": component,
            "overrides": spec.overrides,
            "initial_qpos": spec.initial_qpos,
            "endpoint_bindings": spec.endpoint_bindings,
            "runtime_services": spec.runtime_services,
        }
        runtime_services = skill_profile.get("runtime_services", {})
        runtime_services = _mapping(
            runtime_services, name="embodiment runtime_services"
        )
        return EmbodimentResolution(
            embodiment_id=embodiment_id,
            component_path=path,
            config_hash=stable_hash(resolved),
            capabilities=frozenset(capabilities),
            endpoint_bindings=endpoints,
            runtime_services=runtime_services,
            source="yaml_component",
        )


def _resolve_component_path(component: str, *, base_dir: Path | None) -> Path:
    """Resolve a component path against the suite, cwd, and repository roots."""
    if not isinstance(component, str) or not component.strip():
        raise ValueError("embodiment.component must be a non-empty path.")
    requested = Path(component).expanduser()
    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        if base_dir is not None:
            candidates.append(base_dir / requested)
        candidates.append(Path.cwd() / requested)
        candidates.append(Path(__file__).resolve().parents[3] / requested)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    rendered = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"Embodiment component {component!r} was not found; checked: {rendered}"
    )


def resolve_embodiment(
    spec: EmbodimentSpecCfg | None,
    robot: RobotSpecCfg,
    *,
    base_dir: Path | None = None,
) -> EmbodimentResolution:
    """Resolve official component metadata or legacy robot-provider metadata."""
    if not isinstance(robot, RobotSpecCfg):
        raise TypeError("robot must be a RobotSpecCfg.")
    if spec is None:
        return EmbodimentResolution(
            embodiment_id=robot.id,
            component_path=None,
            config_hash=stable_hash(robot.config),
            capabilities=frozenset(),
            endpoint_bindings={},
            runtime_services={},
            source=f"robot_provider:{robot.provider}",
        )
    return YamlEmbodimentProvider().resolve(spec, base_dir=base_dir)


def check_embodiment_capabilities(
    resolution: EmbodimentResolution,
    required_capabilities: frozenset[str] | set[str] | tuple[str, ...],
) -> CapabilityDecision:
    """Check a skill's declared capabilities against one embodiment profile."""
    if not isinstance(resolution, EmbodimentResolution):
        raise TypeError("resolution must be an EmbodimentResolution.")
    required = frozenset(required_capabilities)
    if any(not isinstance(value, str) or not value.strip() for value in required):
        raise ValueError("required_capabilities must contain non-empty strings.")
    missing = sorted(required.difference(resolution.capabilities))
    if missing:
        return CapabilityDecision(
            status="unsupported",
            reason=(
                f"Embodiment {resolution.embodiment_id!r} is missing capabilities: "
                + ", ".join(missing)
            ),
        )
    return CapabilityDecision(status="supported")
