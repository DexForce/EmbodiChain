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

"""Core role contract shared by the GenSim TaskSpec protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .canonicalization import json_snapshot

__all__ = ["RoleSpec"]

_LEGACY_EXPORTS = {
    "ACTION_WITNESS_SCHEMA",
    "EXPANSION_MANIFEST_SCHEMA",
    "SCENE_INSTANCE_SCHEMA",
    "TASK_TEMPLATE_SCHEMA",
    "VALIDATION_CERTIFICATE_SCHEMA",
    "ActionWitness",
    "ExpansionManifest",
    "RequirementSpec",
    "SceneInstance",
    "TaskTemplate",
    "ValidationCertificate",
}


def __getattr__(name: str) -> Any:
    """Keep explicit legacy contract imports working for one transition release."""
    if name in _LEGACY_EXPORTS or name == "TaskSpec":
        from . import compat

        return getattr(compat, "TaskTemplate" if name == "TaskSpec" else name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_ROLE_KINDS: Final = frozenset(
    {
        "object",
        "target",
        "surface",
        "container",
        "robot",
        "articulation",
        "button",
        "tool",
        "agent",
    }
)
_QUANTIFIERS: Final = frozenset({"one", "all", "count"})


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _optional_text(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _text(value, context)


def _bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{context} must be a boolean.")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}.")
    return value


def _strings(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence of strings.")
    result = tuple(
        _text(item, f"{context}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{context} must not contain duplicates.")
    return result


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    snapshot = json_snapshot(dict(value))
    if not isinstance(snapshot, dict):  # pragma: no cover - guarded by snapshot
        raise TypeError(f"{context} must be a mapping.")
    return snapshot


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """A logical task role that asset and scene engines must ground."""

    name: str
    kind: str = "object"
    required: bool = True
    quantifier: str = "one"
    count: int = 0
    source_structure: str | None = None
    affordances: tuple[str, ...] = ()
    initial_state: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _text(self.name, "RoleSpec.name")
        kind = _text(self.kind, "RoleSpec.kind")
        if kind not in _ROLE_KINDS:
            raise ValueError(f"RoleSpec.kind must be one of {sorted(_ROLE_KINDS)}.")
        required = _bool(self.required, "RoleSpec.required")
        quantifier = _text(self.quantifier, "RoleSpec.quantifier")
        if quantifier not in _QUANTIFIERS:
            raise ValueError(
                f"RoleSpec.quantifier must be one of {sorted(_QUANTIFIERS)}."
            )
        count = _integer(self.count, "RoleSpec.count")
        if quantifier == "count" and count < 1:
            raise ValueError("RoleSpec.count must be >= 1 for quantifier='count'.")
        if quantifier != "count" and count != 0:
            raise ValueError("RoleSpec.count must be 0 unless quantifier='count'.")
        source_structure = _optional_text(
            self.source_structure, "RoleSpec.source_structure"
        )
        affordances = _strings(self.affordances, "RoleSpec.affordances")
        initial_state = _mapping(self.initial_state, "RoleSpec.initial_state")
        attributes = _mapping(self.attributes, "RoleSpec.attributes")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "quantifier", quantifier)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "source_structure", source_structure)
        object.__setattr__(self, "affordances", affordances)
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "attributes", attributes)

    @property
    def role(self) -> str:
        """Compatibility alias used by scene binding adapters."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical role mapping."""
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "quantifier": self.quantifier,
            "count": self.count,
            "source_structure": self.source_structure,
            "affordances": list(self.affordances),
            "initial_state": json_snapshot(self.initial_state),
            "attributes": json_snapshot(self.attributes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | RoleSpec) -> RoleSpec:
        """Decode one role mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("RoleSpec must be a mapping.")
        raw = dict(value)
        if "name" not in raw and "role" in raw:
            raw["name"] = raw.pop("role")
        allowed = {
            "name",
            "kind",
            "required",
            "quantifier",
            "count",
            "source_structure",
            "affordances",
            "initial_state",
            "attributes",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"RoleSpec has unknown fields {sorted(unknown)}.")
        return cls(
            name=raw.get("name"),
            kind=raw.get("kind", "object"),
            required=raw.get("required", True),
            quantifier=raw.get("quantifier", "one"),
            count=raw.get("count", 0),
            source_structure=raw.get("source_structure"),
            affordances=raw.get("affordances", ()),
            initial_state=raw.get("initial_state", {}),
            attributes=raw.get("attributes", {}),
        )
