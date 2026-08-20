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

"""Immutable task intent and canonical semantic task-graph contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType

from embodichain.gen_sim.task_engine.ontology import TASK_TYPES
from embodichain.lab.gym.envs.expert_program.cfg import SemanticCallCfg
from embodichain.lab.gym.envs.expert_program.decoder import (
    ExpertProgramValidationContext,
    decode_semantic_call,
    encode_semantic_call,
)

__all__ = [
    "FORBIDDEN_SEMANTIC_GRAPH_FIELDS",
    "PLANNER_ROUTES",
    "REASONING_TYPES",
    "SEMANTIC_TASK_GRAPH_FILENAME",
    "SEMANTIC_TASK_GRAPH_SCHEMA",
    "TASK_LEVELS",
    "TASK_SPEC_FILENAME",
    "TASK_SPEC_SCHEMA",
    "FailurePolicy",
    "PlannerProvenance",
    "SemanticTaskGraph",
    "SemanticTaskNode",
    "SuccessSpec",
    "TaskGroupSpec",
    "TaskInstanceSpec",
    "TaskSpec",
    "decode_semantic_task_graph",
    "decode_task_spec",
    "semantic_task_graph_hash",
    "task_spec_hash",
]

TASK_SPEC_SCHEMA = "task_spec/v1"
SEMANTIC_TASK_GRAPH_SCHEMA = "semantic_task_graph/v1"
TASK_SPEC_FILENAME = "task_spec.json"
SEMANTIC_TASK_GRAPH_FILENAME = "semantic_task_graph.json"

TASK_LEVELS = frozenset({"L1", "L2", "L3", "L4"})
REASONING_TYPES = frozenset(
    {
        "none",
        "memory",
        "visual_semantics",
        "pattern",
        "logic",
        "common_sense",
        "constraint",
    }
)
PLANNER_ROUTES = frozenset({"offline", "online", "selected", "fused"})
_INSTANCE_ROLES = frozenset({"primary", "recovery"})
_NODE_ROLES = frozenset({"primary", "recovery", "cleanup"})

# These names describe already-grounded execution state or duplicate lower-layer
# policy. They are forbidden at every nesting level, including metadata and
# registered-call arguments.
FORBIDDEN_SEMANTIC_GRAPH_FIELDS = frozenset(
    {
        "absolute_position",
        "action_invocation",
        "action_options",
        "actor",
        "arm",
        "atomic_action",
        "command_frame",
        "control",
        "control_part",
        "controller",
        "eef_pose",
        "effect",
        "goal_type",
        "grasp_pose",
        "held_object",
        "joint_positions",
        "motion_policy",
        "planner_backend",
        "postcondition",
        "precondition",
        "qpos",
        "receive_arm",
        "required_arm",
        "resource_claim",
        "resource_claims",
        "solver",
        "trajectory",
        "transfer_arm",
        "waypoint",
        "waypoints",
    }
)

_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 4096


def _identifier(value: object, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{path} must be a non-empty, trimmed string.")
    return value


def _enum(value: object, allowed: frozenset[str], path: str) -> str:
    result = _identifier(value, path)
    if result not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}.")
    return result


def _owned_json(value: object, path: str) -> object:
    budget = [_MAX_JSON_NODES]
    frozen = _freeze_json(value, path=path, active=set(), budget=budget, depth=0)
    return _thaw_json(frozen)


def _freeze_json(
    value: object,
    *,
    path: str,
    active: set[int],
    budget: list[int],
    depth: int,
) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{path} exceeds the maximum JSON depth.")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError(f"{path} exceeds the maximum JSON node count.")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a cyclic mapping.")
        active.add(identity)
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str or not key or key != key.strip():
                raise ValueError(f"{path} keys must be non-empty, trimmed strings.")
            result[key] = _freeze_json(
                nested,
                path=f"{path}.{key}",
                active=active,
                budget=budget,
                depth=depth + 1,
            )
        active.remove(identity)
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a cyclic sequence.")
        active.add(identity)
        result = tuple(
            _freeze_json(
                nested,
                path=f"{path}[{index}]",
                active=active,
                budget=budget,
                depth=depth + 1,
            )
            for index, nested in enumerate(value)
        )
        active.remove(identity)
        return result
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}.")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


def _frozen_mapping(value: object, path: str) -> Mapping[str, object]:
    frozen = _freeze_json(
        value,
        path=path,
        active=set(),
        budget=[_MAX_JSON_NODES],
        depth=0,
    )
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{path} must be a JSON object.")
    return frozen


def _mapping(value: object, path: str) -> dict[str, object]:
    result = _owned_json(value, path)
    if type(result) is not dict:
        raise TypeError(f"{path} must be a JSON object.")
    return result


def _sequence(value: object, path: str) -> list[object]:
    result = _owned_json(value, path)
    if type(result) is not list:
        raise TypeError(f"{path} must be a JSON array.")
    return result


def _keys(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    path: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}.")
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {unknown}.")


def _identifiers(value: object, path: str) -> tuple[str, ...]:
    result = tuple(
        _identifier(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{path} must contain unique identifiers.")
    return result


def _reject_forbidden_fields(value: object, path: str = "SemanticTaskGraph") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_SEMANTIC_GRAPH_FIELDS:
                raise ValueError(
                    f"{path}.{key} is forbidden in a semantic planning artifact."
                )
            _reject_forbidden_fields(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_forbidden_fields(nested, f"{path}[{index}]")


def _validate_dag(dependencies: Mapping[str, tuple[str, ...]], path: str) -> None:
    known = set(dependencies)
    for node_id, parents in dependencies.items():
        unknown = sorted(set(parents) - known)
        if unknown:
            raise ValueError(f"{path} {node_id!r} has unknown dependencies: {unknown}.")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError(f"{path} contains a dependency cycle at {node_id!r}.")
        if node_id in visited:
            return
        visiting.add(node_id)
        for parent_id in dependencies[node_id]:
            visit(parent_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in dependencies:
        visit(node_id)


def _digest(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SuccessSpec:
    """Extensible task-level success predicate with immutable JSON arguments."""

    kind: str
    arguments: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _identifier(self.kind, "SuccessSpec.kind")
        object.__setattr__(
            self,
            "arguments",
            _frozen_mapping(self.arguments, "SuccessSpec.arguments"),
        )

    @classmethod
    def from_dict(cls, value: object, path: str = "SuccessSpec") -> SuccessSpec:
        """Decode one strict success predicate from JSON-safe values."""
        result = _mapping(value, path)
        kind = _identifier(result.pop("kind", None), f"{path}.kind")
        return cls(kind=kind, arguments=result)

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe predicate mapping."""
        return {"kind": self.kind, **_mapping(self.arguments, "SuccessSpec.arguments")}


@dataclass(frozen=True, slots=True)
class TaskInstanceSpec:
    """One E1-E9 task instance in a scene-independent task DAG."""

    id: str
    task_type: str
    params: Mapping[str, object]
    depends_on: tuple[str, ...] = ()
    role: str = "primary"

    def __post_init__(self) -> None:
        _identifier(self.id, "TaskInstanceSpec.id")
        _enum(self.task_type, TASK_TYPES, "TaskInstanceSpec.task_type")
        _enum(self.role, _INSTANCE_ROLES, "TaskInstanceSpec.role")
        object.__setattr__(self, "params", _frozen_mapping(self.params, "params"))
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("TaskInstanceSpec.depends_on must be unique.")
        for index, dependency in enumerate(self.depends_on):
            _identifier(dependency, f"TaskInstanceSpec.depends_on[{index}]")

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe task-instance mapping."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "params": _mapping(self.params, "TaskInstanceSpec.params"),
            "depends_on": list(self.depends_on),
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Immutable task intent, independent of scene and robot execution state."""

    task_id: str
    level: str
    instruction: str
    reasoning_type: str
    task_instances: tuple[TaskInstanceSpec, ...]
    success: SuccessSpec
    oracle: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    schema_version: str = TASK_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TASK_SPEC_SCHEMA:
            raise ValueError(f"TaskSpec.schema_version must be {TASK_SPEC_SCHEMA!r}.")
        _identifier(self.task_id, "TaskSpec.task_id")
        level = _enum(self.level, TASK_LEVELS, "TaskSpec.level")
        _identifier(self.instruction, "TaskSpec.instruction")
        reasoning = _enum(
            self.reasoning_type,
            REASONING_TYPES,
            "TaskSpec.reasoning_type",
        )
        if (level == "L4") != (reasoning != "none"):
            raise ValueError(
                "TaskSpec.reasoning_type must be non-'none' exactly for L4."
            )
        if not self.task_instances:
            raise ValueError("TaskSpec.task_instances must not be empty.")
        if not all(type(item) is TaskInstanceSpec for item in self.task_instances):
            raise TypeError(
                "TaskSpec.task_instances must contain TaskInstanceSpec values."
            )
        if type(self.success) is not SuccessSpec:
            raise TypeError("TaskSpec.success must be exactly SuccessSpec.")
        ids = [instance.id for instance in self.task_instances]
        if len(ids) != len(set(ids)):
            raise ValueError("TaskSpec task instance IDs must be unique.")
        _validate_dag(
            {instance.id: instance.depends_on for instance in self.task_instances},
            "TaskSpec.task_instances",
        )
        primary = [item for item in self.task_instances if item.role == "primary"]
        task_types = {item.task_type for item in primary}
        if level == "L1" and len(primary) != 1:
            raise ValueError("L1 requires exactly one primary task instance.")
        if level == "L2" and (len(primary) < 2 or len(task_types) != 1):
            raise ValueError(
                "L2 requires at least two primary instances of one E type."
            )
        if level == "L3" and (len(primary) < 2 or len(task_types) < 2):
            raise ValueError(
                "L3 requires at least two primary instances of different E types."
            )
        object.__setattr__(self, "oracle", _frozen_mapping(self.oracle, "oracle"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, value: object) -> TaskSpec:
        """Decode and validate one strict TaskSpec JSON value."""
        _reject_forbidden_fields(value, "TaskSpec")
        result = _mapping(value, "TaskSpec")
        _keys(
            result,
            required=frozenset(
                {
                    "schema_version",
                    "task_id",
                    "level",
                    "instruction",
                    "task_instances",
                    "success",
                }
            ),
            optional=frozenset({"reasoning_type", "oracle", "metadata"}),
            path="TaskSpec",
        )
        if result["schema_version"] != TASK_SPEC_SCHEMA:
            raise ValueError(f"TaskSpec.schema_version must be {TASK_SPEC_SCHEMA!r}.")
        instances: list[TaskInstanceSpec] = []
        for index, raw_instance in enumerate(
            _sequence(result["task_instances"], "TaskSpec.task_instances")
        ):
            path = f"TaskSpec.task_instances[{index}]"
            instance = _mapping(raw_instance, path)
            _keys(
                instance,
                required=frozenset({"id", "task_type"}),
                optional=frozenset({"params", "depends_on", "role"}),
                path=path,
            )
            instances.append(
                TaskInstanceSpec(
                    id=_identifier(instance["id"], f"{path}.id"),
                    task_type=_enum(
                        instance["task_type"], TASK_TYPES, f"{path}.task_type"
                    ),
                    params=_mapping(instance.get("params", {}), f"{path}.params"),
                    depends_on=_identifiers(
                        instance.get("depends_on", []), f"{path}.depends_on"
                    ),
                    role=_enum(
                        instance.get("role", "primary"),
                        _INSTANCE_ROLES,
                        f"{path}.role",
                    ),
                )
            )
        return cls(
            task_id=_identifier(result["task_id"], "TaskSpec.task_id"),
            level=_enum(result["level"], TASK_LEVELS, "TaskSpec.level"),
            instruction=_identifier(result["instruction"], "TaskSpec.instruction"),
            reasoning_type=_enum(
                result.get("reasoning_type", "none"),
                REASONING_TYPES,
                "TaskSpec.reasoning_type",
            ),
            task_instances=tuple(instances),
            success=SuccessSpec.from_dict(result["success"], "TaskSpec.success"),
            oracle=_mapping(result.get("oracle", {}), "TaskSpec.oracle"),
            metadata=_mapping(result.get("metadata", {}), "TaskSpec.metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe TaskSpec mapping."""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "level": self.level,
            "instruction": self.instruction,
            "reasoning_type": self.reasoning_type,
            "task_instances": [item.to_dict() for item in self.task_instances],
            "success": self.success.to_dict(),
            "oracle": _mapping(self.oracle, "TaskSpec.oracle"),
            "metadata": _mapping(self.metadata, "TaskSpec.metadata"),
        }

    def to_public_dict(self) -> dict[str, object]:
        """Return the oracle-free TaskSpec view available to online planners."""
        result = self.to_dict()
        result.pop("oracle")
        if self.level == "L4":
            result.pop("task_instances")
        return result


@dataclass(frozen=True, slots=True)
class FailurePolicy:
    """Bounded task-level route substitution policy."""

    max_attempts: int = 1
    alternate_task_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 0 <= self.max_attempts <= 100:
            raise ValueError("FailurePolicy.max_attempts must be in [0, 100].")
        if len(self.alternate_task_group_ids) != len(
            set(self.alternate_task_group_ids)
        ):
            raise ValueError("FailurePolicy alternate group IDs must be unique.")
        for index, group_id in enumerate(self.alternate_task_group_ids):
            _identifier(group_id, f"alternate_task_group_ids[{index}]")

    @classmethod
    def from_dict(cls, value: object, path: str) -> FailurePolicy:
        """Decode one bounded failure policy."""
        result = _mapping(value, path)
        _keys(
            result,
            required=frozenset(),
            optional=frozenset({"max_attempts", "alternate_task_group_ids"}),
            path=path,
        )
        return cls(
            max_attempts=result.get("max_attempts", 1),  # type: ignore[arg-type]
            alternate_task_group_ids=_identifiers(
                result.get("alternate_task_group_ids", []),
                f"{path}.alternate_task_group_ids",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe failure policy."""
        return {
            "max_attempts": self.max_attempts,
            "alternate_task_group_ids": list(self.alternate_task_group_ids),
        }


@dataclass(frozen=True, slots=True)
class PlannerProvenance:
    """Audit-only identity and confidence for the planner that built a graph."""

    planner_id: str
    revision: str
    confidence: float | None = None
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _identifier(self.planner_id, "PlannerProvenance.planner_id")
        _identifier(self.revision, "PlannerProvenance.revision")
        if self.confidence is not None and (
            type(self.confidence) not in (int, float)
            or isinstance(self.confidence, bool)
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("PlannerProvenance.confidence must be in [0, 1].")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, value: object, path: str) -> PlannerProvenance:
        """Decode one planner provenance record."""
        result = _mapping(value, path)
        _keys(
            result,
            required=frozenset({"planner_id", "revision"}),
            optional=frozenset({"confidence", "metadata"}),
            path=path,
        )
        return cls(
            planner_id=_identifier(result["planner_id"], f"{path}.planner_id"),
            revision=_identifier(result["revision"], f"{path}.revision"),
            confidence=result.get("confidence"),  # type: ignore[arg-type]
            metadata=_mapping(result.get("metadata", {}), f"{path}.metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe provenance record."""
        result: dict[str, object] = {
            "planner_id": self.planner_id,
            "revision": self.revision,
            "metadata": _mapping(self.metadata, "PlannerProvenance.metadata"),
        }
        if self.confidence is not None:
            result["confidence"] = float(self.confidence)
        return result


@dataclass(frozen=True, slots=True)
class SemanticTaskNode:
    """One canonical semantic call and its task-level dependencies."""

    id: str
    _call_payload: Mapping[str, object]
    depends_on: tuple[str, ...]
    task_instance_id: str
    task_type: str
    role: str = "primary"
    _target_ids: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        _identifier(self.id, "SemanticTaskNode.id")
        _identifier(self.task_instance_id, "SemanticTaskNode.task_instance_id")
        _enum(self.task_type, TASK_TYPES, "SemanticTaskNode.task_type")
        _enum(self.role, _NODE_ROLES, "SemanticTaskNode.role")
        object.__setattr__(
            self,
            "_call_payload",
            _frozen_mapping(self._call_payload, "SemanticTaskNode.call"),
        )
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("SemanticTaskNode.depends_on must be unique.")
        for index, dependency in enumerate(self.depends_on):
            _identifier(dependency, f"SemanticTaskNode.depends_on[{index}]")
        if type(self._target_ids) is not frozenset:
            raise TypeError("SemanticTaskNode target IDs must be a frozenset.")

    @property
    def call(self) -> SemanticCallCfg:
        """Return a fresh canonical semantic-call config."""
        return decode_semantic_call(
            _mapping(self._call_payload, "SemanticTaskNode.call"),
            target_ids=self._target_ids,
        )

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe task-node mapping."""
        return {
            "id": self.id,
            "call": _mapping(self._call_payload, "SemanticTaskNode.call"),
            "depends_on": list(self.depends_on),
            "task_instance_id": self.task_instance_id,
            "task_type": self.task_type,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class TaskGroupSpec:
    """Complete semantic subtask boundary used for selection and recovery."""

    id: str
    task_type: str
    node_ids: tuple[str, ...]
    depends_on: tuple[str, ...]
    success: SuccessSpec
    role: str = "primary"
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)

    def __post_init__(self) -> None:
        _identifier(self.id, "TaskGroupSpec.id")
        _enum(self.task_type, TASK_TYPES, "TaskGroupSpec.task_type")
        _enum(self.role, _INSTANCE_ROLES, "TaskGroupSpec.role")
        if not self.node_ids or len(self.node_ids) != len(set(self.node_ids)):
            raise ValueError("TaskGroupSpec.node_ids must be non-empty and unique.")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("TaskGroupSpec.depends_on must be unique.")
        for index, node_id in enumerate(self.node_ids):
            _identifier(node_id, f"TaskGroupSpec.node_ids[{index}]")
        for index, dependency in enumerate(self.depends_on):
            _identifier(dependency, f"TaskGroupSpec.depends_on[{index}]")
        if type(self.success) is not SuccessSpec:
            raise TypeError("TaskGroupSpec.success must be exactly SuccessSpec.")
        if type(self.failure_policy) is not FailurePolicy:
            raise TypeError(
                "TaskGroupSpec.failure_policy must be exactly FailurePolicy."
            )

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-safe TaskGroup mapping."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "node_ids": list(self.node_ids),
            "depends_on": list(self.depends_on),
            "success": self.success.to_dict(),
            "role": self.role,
            "failure_policy": self.failure_policy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SemanticTaskGraph:
    """Immutable semantic DAG with no grounded or atomic execution data."""

    task_id: str
    instruction: str
    planner_route: str
    integration_fingerprint: str
    nodes: tuple[SemanticTaskNode, ...]
    task_groups: tuple[TaskGroupSpec, ...]
    success: SuccessSpec
    planner_provenance: PlannerProvenance | None = None
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    schema_version: str = SEMANTIC_TASK_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_TASK_GRAPH_SCHEMA:
            raise ValueError(
                "SemanticTaskGraph.schema_version must be "
                f"{SEMANTIC_TASK_GRAPH_SCHEMA!r}."
            )
        _identifier(self.task_id, "SemanticTaskGraph.task_id")
        _identifier(self.instruction, "SemanticTaskGraph.instruction")
        _enum(self.planner_route, PLANNER_ROUTES, "SemanticTaskGraph.planner_route")
        if (
            type(self.integration_fingerprint) is not str
            or len(self.integration_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.integration_fingerprint
            )
        ):
            raise ValueError(
                "SemanticTaskGraph.integration_fingerprint must be a lowercase "
                "SHA-256 digest."
            )
        if not self.nodes or not self.task_groups:
            raise ValueError("SemanticTaskGraph requires nodes and task_groups.")
        if not all(type(node) is SemanticTaskNode for node in self.nodes):
            raise TypeError(
                "SemanticTaskGraph.nodes must contain SemanticTaskNode values."
            )
        if not all(type(group) is TaskGroupSpec for group in self.task_groups):
            raise TypeError(
                "SemanticTaskGraph.task_groups must contain TaskGroupSpec values."
            )
        if type(self.success) is not SuccessSpec:
            raise TypeError("SemanticTaskGraph.success must be exactly SuccessSpec.")
        if (
            self.planner_provenance is not None
            and type(self.planner_provenance) is not PlannerProvenance
        ):
            raise TypeError(
                "SemanticTaskGraph.planner_provenance must be PlannerProvenance or None."
            )
        node_ids = [node.id for node in self.nodes]
        group_ids = [group.id for group in self.task_groups]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("SemanticTaskGraph node IDs must be unique.")
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("SemanticTaskGraph TaskGroup IDs must be unique.")
        _validate_dag(
            {node.id: node.depends_on for node in self.nodes},
            "SemanticTaskGraph.nodes",
        )
        _validate_dag(
            {group.id: group.depends_on for group in self.task_groups},
            "SemanticTaskGraph.task_groups",
        )
        self._validate_task_groups()
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        target_ids: frozenset[str] = frozenset(),
        validation_context: ExpertProgramValidationContext | None = None,
    ) -> SemanticTaskGraph:
        """Decode a graph through the canonical semantic-call codec.

        Args:
            value: Untrusted JSON-compatible graph mapping.
            target_ids: Declared target IDs available to canonical target refs.
            validation_context: Optional canonical provider-free integration validator.

        Returns:
            Immutable and fully validated semantic task graph.

        Raises:
            TypeError: If the input is not strict JSON-compatible data.
            ValueError: If graph topology or task metadata is invalid.
            ExpertProgramDecodeError: If a call violates the canonical call schema.
            ExpertProgramValidationError: If canonical provider-free validation fails.
        """
        _reject_forbidden_fields(value)
        result = _mapping(value, "SemanticTaskGraph")
        _keys(
            result,
            required=frozenset(
                {
                    "schema_version",
                    "task_id",
                    "instruction",
                    "planner_route",
                    "integration_fingerprint",
                    "nodes",
                    "task_groups",
                    "success",
                }
            ),
            optional=frozenset({"planner_provenance", "metadata"}),
            path="SemanticTaskGraph",
        )
        if result["schema_version"] != SEMANTIC_TASK_GRAPH_SCHEMA:
            raise ValueError(
                "SemanticTaskGraph.schema_version must be "
                f"{SEMANTIC_TASK_GRAPH_SCHEMA!r}."
            )
        if type(target_ids) is not frozenset:
            raise TypeError("target_ids must be a frozenset of exact identifiers.")
        for index, target_id in enumerate(target_ids):
            _identifier(target_id, f"target_ids[{index}]")

        nodes: list[SemanticTaskNode] = []
        for index, raw_node in enumerate(
            _sequence(result["nodes"], "SemanticTaskGraph.nodes")
        ):
            path = f"SemanticTaskGraph.nodes[{index}]"
            node = _mapping(raw_node, path)
            _keys(
                node,
                required=frozenset(
                    {
                        "id",
                        "call",
                        "depends_on",
                        "task_instance_id",
                        "task_type",
                        "role",
                    }
                ),
                optional=frozenset(),
                path=path,
            )
            call = decode_semantic_call(
                node["call"],
                target_ids=target_ids,
                validation_context=validation_context,
                path=("nodes", index, "call"),
            )
            nodes.append(
                SemanticTaskNode(
                    id=_identifier(node["id"], f"{path}.id"),
                    _call_payload=encode_semantic_call(call),
                    depends_on=_identifiers(node["depends_on"], f"{path}.depends_on"),
                    task_instance_id=_identifier(
                        node["task_instance_id"], f"{path}.task_instance_id"
                    ),
                    task_type=_enum(node["task_type"], TASK_TYPES, f"{path}.task_type"),
                    role=_enum(node["role"], _NODE_ROLES, f"{path}.role"),
                    _target_ids=target_ids,
                )
            )

        task_groups: list[TaskGroupSpec] = []
        for index, raw_group in enumerate(
            _sequence(result["task_groups"], "SemanticTaskGraph.task_groups")
        ):
            path = f"SemanticTaskGraph.task_groups[{index}]"
            group = _mapping(raw_group, path)
            _keys(
                group,
                required=frozenset(
                    {"id", "task_type", "node_ids", "depends_on", "success"}
                ),
                optional=frozenset({"role", "failure_policy"}),
                path=path,
            )
            task_groups.append(
                TaskGroupSpec(
                    id=_identifier(group["id"], f"{path}.id"),
                    task_type=_enum(
                        group["task_type"], TASK_TYPES, f"{path}.task_type"
                    ),
                    node_ids=_identifiers(group["node_ids"], f"{path}.node_ids"),
                    depends_on=_identifiers(group["depends_on"], f"{path}.depends_on"),
                    success=SuccessSpec.from_dict(group["success"], f"{path}.success"),
                    role=_enum(
                        group.get("role", "primary"),
                        _INSTANCE_ROLES,
                        f"{path}.role",
                    ),
                    failure_policy=FailurePolicy.from_dict(
                        group.get("failure_policy", {}), f"{path}.failure_policy"
                    ),
                )
            )

        provenance_value = result.get("planner_provenance")
        provenance = (
            None
            if provenance_value is None
            else PlannerProvenance.from_dict(
                provenance_value,
                "SemanticTaskGraph.planner_provenance",
            )
        )
        return cls(
            task_id=_identifier(result["task_id"], "SemanticTaskGraph.task_id"),
            instruction=_identifier(
                result["instruction"], "SemanticTaskGraph.instruction"
            ),
            planner_route=_enum(
                result["planner_route"],
                PLANNER_ROUTES,
                "SemanticTaskGraph.planner_route",
            ),
            integration_fingerprint=_identifier(
                result["integration_fingerprint"],
                "SemanticTaskGraph.integration_fingerprint",
            ),
            nodes=tuple(nodes),
            task_groups=tuple(task_groups),
            success=SuccessSpec.from_dict(
                result["success"], "SemanticTaskGraph.success"
            ),
            planner_provenance=provenance,
            metadata=_mapping(result.get("metadata", {}), "SemanticTaskGraph.metadata"),
        )

    def _validate_task_groups(self) -> None:
        node_by_id = {node.id: node for node in self.nodes}
        group_by_id = {group.id: group for group in self.task_groups}
        memberships: dict[str, str] = {}
        for group in self.task_groups:
            for node_id in group.node_ids:
                node = node_by_id.get(node_id)
                if node is None:
                    raise ValueError(
                        f"TaskGroup {group.id!r} references unknown node {node_id!r}."
                    )
                if node_id in memberships:
                    raise ValueError(
                        f"Semantic task node {node_id!r} belongs to multiple TaskGroups."
                    )
                if (
                    node.task_instance_id != group.id
                    or node.task_type != group.task_type
                ):
                    raise ValueError(
                        f"Semantic task node {node_id!r} does not match TaskGroup "
                        f"{group.id!r}."
                    )
                memberships[node_id] = group.id
            for alternate_id in group.failure_policy.alternate_task_group_ids:
                alternate = group_by_id.get(alternate_id)
                if alternate is None:
                    raise ValueError(
                        f"TaskGroup {group.id!r} references unknown alternate "
                        f"{alternate_id!r}."
                    )
                if alternate_id == group.id:
                    raise ValueError("A TaskGroup cannot be its own alternate route.")
                if alternate.task_type != group.task_type:
                    raise ValueError(
                        "Alternate TaskGroups must implement the same task_type."
                    )
        missing = sorted(set(node_by_id) - set(memberships))
        if missing:
            raise ValueError(
                f"Semantic task nodes are missing TaskGroup membership: {missing}."
            )

        group_dependencies = {
            group.id: set(group.depends_on) for group in self.task_groups
        }

        def reaches(child: str, parent: str) -> bool:
            pending = list(group_dependencies[child])
            visited: set[str] = set()
            while pending:
                current = pending.pop()
                if current == parent:
                    return True
                if current not in visited:
                    visited.add(current)
                    pending.extend(group_dependencies[current])
            return False

        for node in self.nodes:
            child_group = memberships[node.id]
            for dependency in node.depends_on:
                parent_group = memberships[dependency]
                if parent_group != child_group and not reaches(
                    child_group, parent_group
                ):
                    raise ValueError(
                        f"Node {node.id!r} crosses from TaskGroup {parent_group!r} "
                        f"without a matching dependency on {child_group!r}."
                    )

    def to_dict(self) -> dict[str, object]:
        """Return an owned canonical JSON-safe graph mapping."""
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "planner_route": self.planner_route,
            "integration_fingerprint": self.integration_fingerprint,
            "nodes": [node.to_dict() for node in self.nodes],
            "task_groups": [group.to_dict() for group in self.task_groups],
            "success": self.success.to_dict(),
            "metadata": _mapping(self.metadata, "SemanticTaskGraph.metadata"),
        }
        if self.planner_provenance is not None:
            result["planner_provenance"] = self.planner_provenance.to_dict()
        return result


def decode_task_spec(value: object) -> TaskSpec:
    """Decode one immutable TaskSpec from untrusted JSON-compatible data."""
    return TaskSpec.from_dict(value)


def task_spec_hash(value: object) -> str:
    """Return the deterministic SHA-256 hash of one validated TaskSpec."""
    task_spec = value if type(value) is TaskSpec else decode_task_spec(value)
    assert type(task_spec) is TaskSpec
    return _digest(task_spec.to_dict())


def decode_semantic_task_graph(
    value: object,
    *,
    target_ids: frozenset[str] = frozenset(),
    validation_context: ExpertProgramValidationContext | None = None,
) -> SemanticTaskGraph:
    """Decode one immutable SemanticTaskGraph through the canonical call codec."""
    return SemanticTaskGraph.from_dict(
        value,
        target_ids=target_ids,
        validation_context=validation_context,
    )


def semantic_task_graph_hash(value: object) -> str:
    """Return the deterministic SHA-256 hash of one validated semantic graph."""
    graph = (
        value if type(value) is SemanticTaskGraph else decode_semantic_task_graph(value)
    )
    assert type(graph) is SemanticTaskGraph
    return _digest(graph.to_dict())
