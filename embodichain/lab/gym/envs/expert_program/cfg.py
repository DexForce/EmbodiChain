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

"""Typed configuration values for declarative Expert Programs."""

from __future__ import annotations

import math
import re
from dataclasses import MISSING, field
from typing import TypeAlias

from embodichain.utils import configclass

EXPERT_PROGRAM_SCHEMA_VERSION = 1
"""Stable sequential Expert Program schema version."""

EXPERT_PROGRAM_SCHEMA_VERSION_V2 = 2
"""Schema version adding deterministic ``Parallel`` and ``Barrier`` nodes."""

SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS = (
    EXPERT_PROGRAM_SCHEMA_VERSION,
    EXPERT_PROGRAM_SCHEMA_VERSION_V2,
)
"""Exact schema revisions accepted by the strict decoder."""

MAX_REPEAT_COUNT = 1_000
"""Maximum repeat count accepted by one Expert Program repeat node."""

MAX_EXPANDED_CALLS = 10_000
"""Maximum statically expanded semantic calls in one Expert Program."""

MAX_PROGRAM_DEPTH = 64
"""Maximum nesting depth of a supported Expert Program AST."""

MAX_PROGRAM_NODES = 10_000
"""Maximum number of stored nodes in a supported Expert Program AST."""

MAX_DECLARATIVE_DEPTH = 32
"""Maximum nesting depth of a registered-call declarative payload."""

MAX_DECLARATIVE_NODES = 10_000
"""Maximum number of values in a registered-call declarative payload."""

_REGISTERED_CALL_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_ENV_TRAVERSAL_PATTERN = re.compile(
    r"(?:\$?(?:env|environment)(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
    r"\$\{(?:env|environment)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\})"
)
_FORBIDDEN_DECLARATIVE_KEYS = frozenset(
    {
        "__import__",
        "attribute_path",
        "callable",
        "environment_path",
        "env_path",
        "eval",
        "exec",
        "expression",
        "getattr",
        "import",
        "module",
        "python",
    }
)

DeclarativeCfgValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["DeclarativeCfgValue", ...]
    | dict[str, "DeclarativeCfgValue"]
)
"""Executable-free value accepted by a registered semantic call config."""


def _validate_identifier(value: object, *, field_name: str) -> str:
    """Return one exact non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _validate_kind(value: object, *, expected: str, field_name: str) -> None:
    """Require one exact discriminator value."""
    if type(value) is not str or value != expected:
        raise ValueError(f"{field_name} must be exactly {expected!r}.")


def _validate_number(value: object, *, field_name: str) -> float:
    """Return one finite number while rejecting bool values."""
    if type(value) not in (int, float):
        raise TypeError(f"{field_name} must be an int or float.")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _validate_resources(value: object, *, field_name: str) -> dict[str, str]:
    """Own one strict slot-to-resource mapping."""
    if type(value) is not dict:
        raise TypeError(f"{field_name} must be an exact dict.")
    resources: dict[str, str] = {}
    for slot_id, resource_id in value.items():
        resources[
            _validate_identifier(slot_id, field_name=f"{field_name} slot IDs")
        ] = _validate_identifier(
            resource_id,
            field_name=f"{field_name} resource IDs",
        )
    return resources


def _validate_declarative_string(value: str, *, path: str) -> str:
    """Reject strings that request executable or environment traversal behavior."""
    stripped = value.strip()
    lowered = stripped.lower()
    forbidden_prefixes = (
        "__import__(",
        "eval(",
        "exec(",
        "import ",
        "from ",
    )
    if lowered.startswith(forbidden_prefixes):
        raise ValueError(f"{path} contains an executable import/eval expression.")
    if _ENV_TRAVERSAL_PATTERN.fullmatch(stripped) is not None:
        raise ValueError(f"{path} contains dotted environment attribute traversal.")
    return value


def _snapshot_declarative_value(
    value: object,
    *,
    path: str,
    _active: set[int] | None = None,
    _budget: list[int] | None = None,
    _depth: int = 0,
) -> DeclarativeCfgValue:
    """Validate and own one bounded executable-free declarative value."""
    active = set() if _active is None else _active
    budget = [MAX_DECLARATIVE_NODES] if _budget is None else _budget
    if _depth > MAX_DECLARATIVE_DEPTH:
        raise ValueError(
            f"{path} exceeds declarative depth limit {MAX_DECLARATIVE_DEPTH}."
        )
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError(
            f"{path} exceeds declarative node limit {MAX_DECLARATIVE_NODES}."
        )
    if value is None or type(value) in (bool, int):
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if type(value) is str:
        return _validate_declarative_string(value, path=path)
    if type(value) in (list, tuple):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a cyclic sequence.")
        active.add(identity)
        try:
            return tuple(
                _snapshot_declarative_value(
                    item,
                    path=f"{path}[{index}]",
                    _active=active,
                    _budget=budget,
                    _depth=_depth + 1,
                )
                for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)
    if type(value) is dict:
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a cyclic mapping.")
        active.add(identity)
        try:
            result: dict[str, DeclarativeCfgValue] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} keys must be exact strings.")
                if key.lower() in _FORBIDDEN_DECLARATIVE_KEYS:
                    raise ValueError(
                        f"{path}.{key} requests forbidden executable behavior."
                    )
                result[key] = _snapshot_declarative_value(
                    item,
                    path=f"{path}.{key}",
                    _active=active,
                    _budget=budget,
                    _depth=_depth + 1,
                )
            return result
        finally:
            active.remove(identity)
    raise TypeError(
        f"{path} contains non-declarative {type(value).__name__}; callables, "
        "classes, modules, tensors, and live objects are not allowed."
    )


@configclass
class ExpertProgramIntegrationCfg:
    """Static integration references selected by one Expert Program."""

    robot_profile: str = MISSING
    scene_registry: str = MISSING
    runtime_preset: str = MISSING

    def __post_init__(self) -> None:
        """Validate stable integration identifiers."""
        _validate_identifier(self.robot_profile, field_name="robot_profile")
        _validate_identifier(self.scene_registry, field_name="scene_registry")
        _validate_identifier(self.runtime_preset, field_name="runtime_preset")


@configclass
class PoseCfg:
    """One declarative Cartesian pose using a WXYZ quaternion."""

    position: tuple[float, float, float] = MISSING
    quaternion_wxyz: tuple[float, float, float, float] = MISSING

    def __post_init__(self) -> None:
        """Validate pose shape, finiteness, and quaternion magnitude."""
        if type(self.position) not in (list, tuple) or len(self.position) != 3:
            raise ValueError("position must contain exactly three numbers.")
        if (
            type(self.quaternion_wxyz) not in (list, tuple)
            or len(self.quaternion_wxyz) != 4
        ):
            raise ValueError("quaternion_wxyz must contain exactly four numbers.")
        position = tuple(
            _validate_number(value, field_name=f"position[{index}]")
            for index, value in enumerate(self.position)
        )
        quaternion = tuple(
            _validate_number(value, field_name=f"quaternion_wxyz[{index}]")
            for index, value in enumerate(self.quaternion_wxyz)
        )
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1.0e-12:
            raise ValueError("quaternion_wxyz must have non-zero magnitude.")
        self.position = position  # type: ignore[assignment]
        self.quaternion_wxyz = quaternion  # type: ignore[assignment]


@configclass
class TargetRefCfg:
    """Reference to one top-level typed target provider."""

    target: str = MISSING
    kind: str = "target_ref"

    def __post_init__(self) -> None:
        """Validate the target identifier and discriminator."""
        _validate_identifier(self.target, field_name="target")
        _validate_kind(self.kind, expected="target_ref", field_name="kind")


@configclass
class CyclicPoseTargetCfg:
    """Finite pose values selected cyclically by the enclosing repeat index."""

    values: tuple[PoseCfg, ...] = MISSING
    kind: str = "cyclic_pose"

    def __post_init__(self) -> None:
        """Validate a non-empty owned pose sequence."""
        _validate_kind(self.kind, expected="cyclic_pose", field_name="kind")
        if type(self.values) not in (list, tuple) or not self.values:
            raise ValueError("values must contain at least one PoseCfg.")
        values = tuple(self.values)
        if not all(type(value) is PoseCfg for value in values):
            raise TypeError("values must contain exact PoseCfg values.")
        self.values = values  # type: ignore[assignment]


TargetCfg: TypeAlias = CyclicPoseTargetCfg


@configclass
class PickCfg:
    """Declarative request to acquire one registered object."""

    object: str = MISSING
    grasp: str | None = None
    resources: dict[str, str] = field(default_factory=dict)
    kind: str = "pick"

    def __post_init__(self) -> None:
        """Validate object, optional affordance, resources, and kind."""
        _validate_identifier(self.object, field_name="object")
        if self.grasp is not None:
            _validate_identifier(self.grasp, field_name="grasp")
        self.resources = _validate_resources(self.resources, field_name="resources")
        _validate_kind(self.kind, expected="pick", field_name="kind")


@configclass
class PlaceCfg:
    """Declarative request to place one held object at one destination."""

    object: str = MISSING
    at: TargetRefCfg | None = None
    on: str | None = None
    inside: str | None = None
    resources: dict[str, str] = field(default_factory=dict)
    kind: str = "place"

    def __post_init__(self) -> None:
        """Require exactly one typed destination."""
        _validate_identifier(self.object, field_name="object")
        selected = sum(value is not None for value in (self.at, self.on, self.inside))
        if selected != 1:
            raise ValueError("Place requires exactly one of at, on, or inside.")
        if self.at is not None and type(self.at) is not TargetRefCfg:
            raise TypeError("at must be exactly TargetRefCfg or None.")
        if self.on is not None:
            _validate_identifier(self.on, field_name="on")
        if self.inside is not None:
            _validate_identifier(self.inside, field_name="inside")
        self.resources = _validate_resources(self.resources, field_name="resources")
        _validate_kind(self.kind, expected="place", field_name="kind")


@configclass
class HandOverCfg:
    """Declarative request to transfer one held object between resources."""

    object: str = MISSING
    final_target: TargetRefCfg | None = None
    resources: dict[str, str] = field(default_factory=dict)
    kind: str = "hand_over"

    def __post_init__(self) -> None:
        """Validate object, destination resource, and optional target."""
        _validate_identifier(self.object, field_name="object")
        if (
            self.final_target is not None
            and type(self.final_target) is not TargetRefCfg
        ):
            raise TypeError("final_target must be exactly TargetRefCfg or None.")
        self.resources = _validate_resources(self.resources, field_name="resources")
        _validate_kind(self.kind, expected="hand_over", field_name="kind")


@configclass
class OperateArticulationCfg:
    """Declarative request to operate one articulated joint through a handle."""

    articulation: str = MISSING
    handle: str | None = None
    target: str | None = None
    target_position: float | None = None
    target_displacement: float | None = None
    resources: dict[str, str] = field(default_factory=dict)
    kind: str = "operate_articulation"

    def __post_init__(self) -> None:
        """Require one named target or one complete explicit target pair."""
        _validate_identifier(self.articulation, field_name="articulation")
        if self.handle is not None:
            _validate_identifier(self.handle, field_name="handle")
        named = self.target is not None
        explicit_position = self.target_position is not None
        explicit_displacement = self.target_displacement is not None
        if named:
            _validate_identifier(self.target, field_name="target")
            if explicit_position or explicit_displacement:
                raise ValueError(
                    "target is mutually exclusive with target_position and "
                    "target_displacement."
                )
        elif not (explicit_position and explicit_displacement):
            raise ValueError(
                "OperateArticulation requires either target or the explicit "
                "target_position and target_displacement pair."
            )
        else:
            self.target_position = _validate_number(
                self.target_position,
                field_name="target_position",
            )
            self.target_displacement = _validate_number(
                self.target_displacement,
                field_name="target_displacement",
            )
        self.resources = _validate_resources(self.resources, field_name="resources")
        _validate_kind(
            self.kind,
            expected="operate_articulation",
            field_name="kind",
        )


@configclass
class RegisteredSemanticCallCfg:
    """Safe declarative payload for one catalog-registered semantic call."""

    call_id: str = MISSING
    schema_version: int = EXPERT_PROGRAM_SCHEMA_VERSION
    arguments: dict[str, DeclarativeCfgValue] = field(default_factory=dict)
    resources: dict[str, str] = field(default_factory=dict)
    kind: str = "registered"

    def __post_init__(self) -> None:
        """Validate versioned ID and recursively executable-free arguments."""
        _validate_identifier(self.call_id, field_name="call_id")
        if _REGISTERED_CALL_ID_PATTERN.fullmatch(self.call_id) is None:
            raise ValueError(
                "call_id must contain two or more lowercase identifier segments "
                "separated by single dots."
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Registered call schema_version must be exactly 1.")
        if type(self.arguments) is not dict:
            raise TypeError("arguments must be an exact dict.")
        arguments = _snapshot_declarative_value(
            self.arguments,
            path="arguments",
        )
        assert type(arguments) is dict
        self.arguments = arguments
        self.resources = _validate_resources(self.resources, field_name="resources")
        _validate_kind(self.kind, expected="registered", field_name="kind")


SemanticCallCfg: TypeAlias = (
    PickCfg
    | PlaceCfg
    | HandOverCfg
    | OperateArticulationCfg
    | RegisteredSemanticCallCfg
)


@configclass
class WaitStablePostCfg:
    """Wait for one registered entity to satisfy a named stability preset."""

    entity: str = MISSING
    preset: str = "rigid_object"
    kind: str = "wait_stable"

    def __post_init__(self) -> None:
        """Validate entity, preset, and discriminator."""
        _validate_identifier(self.entity, field_name="entity")
        _validate_identifier(self.preset, field_name="preset")
        _validate_kind(self.kind, expected="wait_stable", field_name="kind")


PostPolicyCfg: TypeAlias = WaitStablePostCfg


@configclass
class ObjectNearTargetValidatorCfg:
    """Validate an object's position against one resolved target."""

    object: str = MISSING
    target: str = MISSING
    position_tolerance: float = 0.03
    kind: str = "object_near_target"

    def __post_init__(self) -> None:
        """Validate reference IDs and a positive finite tolerance."""
        _validate_identifier(self.object, field_name="object")
        _validate_identifier(self.target, field_name="target")
        tolerance = _validate_number(
            self.position_tolerance,
            field_name="position_tolerance",
        )
        if tolerance <= 0.0:
            raise ValueError("position_tolerance must be positive.")
        self.position_tolerance = tolerance
        _validate_kind(
            self.kind,
            expected="object_near_target",
            field_name="kind",
        )


ValidatorCfg: TypeAlias = ObjectNearTargetValidatorCfg


@configclass
class InvokeCfg:
    """Invoke exactly one semantic call at the current program boundary."""

    call: SemanticCallCfg = MISSING
    kind: str = "invoke"

    def __post_init__(self) -> None:
        """Validate the semantic-call union and discriminator."""
        if type(self.call) not in _SEMANTIC_CALL_TYPES:
            raise TypeError("call must be an exact SemanticCallCfg value.")
        _validate_kind(self.kind, expected="invoke", field_name="kind")


@configclass
class BarrierCfg:
    """Explicit synchronization boundary owned by one parallel node."""

    name: str = "join"
    timeout_steps: int = 1_000
    failure_policy: str = "fail_fast"
    kind: str = "barrier"

    def __post_init__(self) -> None:
        """Validate deterministic timeout and cancellation semantics."""
        _validate_kind(self.kind, expected="barrier", field_name="kind")
        _validate_identifier(self.name, field_name="name")
        if type(self.timeout_steps) is not int or self.timeout_steps <= 0:
            raise ValueError("timeout_steps must be a positive integer.")
        if self.failure_policy != "fail_fast":
            raise ValueError("failure_policy must be exactly 'fail_fast'.")


@configclass
class SequenceCfg:
    """Execute one non-empty ordered tuple of program nodes."""

    items: tuple[ProgramNodeCfg, ...] = MISSING
    kind: str = "sequence"

    def __post_init__(self) -> None:
        """Validate ordered child nodes and discriminator."""
        _validate_kind(self.kind, expected="sequence", field_name="kind")
        if type(self.items) not in (list, tuple) or not self.items:
            raise ValueError("items must contain at least one program node.")
        items = tuple(self.items)
        if not all(type(item) in _PROGRAM_NODE_TYPES for item in items):
            raise TypeError("items must contain exact ProgramNodeCfg values.")
        self.items = items  # type: ignore[assignment]


@configclass
class RepeatCfg:
    """Repeat one child node a finite validated number of times."""

    count: int = MISSING
    body: ProgramNodeCfg = MISSING
    kind: str = "repeat"

    def __post_init__(self) -> None:
        """Validate a bounded positive repeat and its child node."""
        if type(self.count) is not int or not 1 <= self.count <= MAX_REPEAT_COUNT:
            raise ValueError(f"count must be an integer in [1, {MAX_REPEAT_COUNT}].")
        if type(self.body) not in _PROGRAM_NODE_TYPES:
            raise TypeError("body must be an exact ProgramNodeCfg value.")
        _validate_kind(self.kind, expected="repeat", field_name="kind")


@configclass
class SegmentCfg:
    """Logical program transaction with post-policies and validators."""

    name: str = MISSING
    steps: ProgramNodeCfg = MISSING
    post: tuple[PostPolicyCfg, ...] = field(default_factory=tuple)
    validators: tuple[ValidatorCfg, ...] = field(default_factory=tuple)
    kind: str = "segment"

    def __post_init__(self) -> None:
        """Validate the segment boundary and its declarative hooks."""
        _validate_identifier(self.name, field_name="name")
        if type(self.steps) not in _PROGRAM_NODE_TYPES:
            raise TypeError("steps must be an exact ProgramNodeCfg value.")
        if type(self.post) not in (list, tuple):
            raise TypeError("post must be a list or tuple.")
        if type(self.validators) not in (list, tuple):
            raise TypeError("validators must be a list or tuple.")
        post = tuple(self.post)
        validators = tuple(self.validators)
        if not all(type(value) in _POST_POLICY_TYPES for value in post):
            raise TypeError("post must contain exact PostPolicyCfg values.")
        if not all(type(value) in _VALIDATOR_TYPES for value in validators):
            raise TypeError("validators must contain exact ValidatorCfg values.")
        self.post = post  # type: ignore[assignment]
        self.validators = validators  # type: ignore[assignment]
        _validate_kind(self.kind, expected="segment", field_name="kind")


@configclass
class ParallelCfg:
    """Execute two or more branches concurrently and join at one barrier."""

    branches: tuple[ProgramNodeCfg, ...] = MISSING
    barrier: BarrierCfg = MISSING
    kind: str = "parallel"

    def __post_init__(self) -> None:
        """Validate branch ownership and an explicit synchronization node."""
        _validate_kind(self.kind, expected="parallel", field_name="kind")
        if type(self.branches) not in (list, tuple) or len(self.branches) < 2:
            raise ValueError("branches must contain at least two program nodes.")
        branches = tuple(self.branches)
        if not all(type(branch) in _PROGRAM_NODE_TYPES for branch in branches):
            raise TypeError("branches must contain exact ProgramNodeCfg values.")
        if any(type(branch) in (ParallelCfg, BarrierCfg) for branch in branches):
            raise ValueError(
                "Nested Parallel and standalone Barrier branches are forbidden."
            )
        if type(self.barrier) is not BarrierCfg:
            raise TypeError("barrier must be exactly BarrierCfg.")
        self.branches = branches  # type: ignore[assignment]


ProgramNodeCfg: TypeAlias = (
    SequenceCfg | RepeatCfg | SegmentCfg | InvokeCfg | ParallelCfg | BarrierCfg
)

_SEMANTIC_CALL_TYPES = (
    PickCfg,
    PlaceCfg,
    HandOverCfg,
    OperateArticulationCfg,
    RegisteredSemanticCallCfg,
)
_POST_POLICY_TYPES = (WaitStablePostCfg,)
_VALIDATOR_TYPES = (ObjectNearTargetValidatorCfg,)
_PROGRAM_NODE_TYPES = (
    SequenceCfg,
    RepeatCfg,
    SegmentCfg,
    InvokeCfg,
    ParallelCfg,
    BarrierCfg,
)


def _validate_target_reference(target: str, targets: dict[str, TargetCfg]) -> None:
    """Require one target reference to exist in the top-level registry."""
    if target not in targets:
        raise ValueError(f"Unknown target reference {target!r}.")


def _validate_program(
    node: ProgramNodeCfg,
    *,
    targets: dict[str, TargetCfg],
    depth: int,
    budget: list[int],
    schema_version: int,
    inside_parallel: bool = False,
) -> int:
    """Validate references and return the statically expanded call count."""
    if depth > MAX_PROGRAM_DEPTH:
        raise ValueError(f"Program exceeds depth limit {MAX_PROGRAM_DEPTH}.")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError(f"Program exceeds node limit {MAX_PROGRAM_NODES}.")
    if type(node) is InvokeCfg:
        call = node.call
        if type(call) is PlaceCfg and call.at is not None:
            _validate_target_reference(call.at.target, targets)
        if type(call) is HandOverCfg and call.final_target is not None:
            _validate_target_reference(call.final_target.target, targets)
        return 1
    if type(node) is BarrierCfg:
        if schema_version < EXPERT_PROGRAM_SCHEMA_VERSION_V2:
            raise ValueError("Barrier requires Expert Program schema version 2.")
        if not inside_parallel:
            raise ValueError("Barrier nodes may only be owned by Parallel.")
        return 0
    if type(node) is SequenceCfg:
        expanded = sum(
            _validate_program(
                child,
                targets=targets,
                depth=depth + 1,
                budget=budget,
                schema_version=schema_version,
                inside_parallel=inside_parallel,
            )
            for child in node.items
        )
    elif type(node) is RepeatCfg:
        expanded = node.count * _validate_program(
            node.body,
            targets=targets,
            depth=depth + 1,
            budget=budget,
            schema_version=schema_version,
            inside_parallel=inside_parallel,
        )
    elif type(node) is SegmentCfg:
        if inside_parallel:
            raise ValueError(
                "Parallel branches may contain only Invoke, Sequence, and Repeat "
                "nodes; wrap the Parallel node in one Segment instead."
            )
        for validator in node.validators:
            _validate_target_reference(validator.target, targets)
        expanded = _validate_program(
            node.steps,
            targets=targets,
            depth=depth + 1,
            budget=budget,
            schema_version=schema_version,
            inside_parallel=inside_parallel,
        )
    elif type(node) is ParallelCfg:
        if schema_version < EXPERT_PROGRAM_SCHEMA_VERSION_V2:
            raise ValueError("Parallel requires Expert Program schema version 2.")
        if inside_parallel:
            raise ValueError("Nested Parallel nodes are forbidden in schema version 2.")
        branch_counts = tuple(
            _validate_program(
                branch,
                targets=targets,
                depth=depth + 1,
                budget=budget,
                schema_version=schema_version,
                inside_parallel=True,
            )
            for branch in node.branches
        )
        if any(count <= 0 for count in branch_counts):
            raise ValueError("Every Parallel branch must contain a semantic call.")
        _validate_program(
            node.barrier,
            targets=targets,
            depth=depth + 1,
            budget=budget,
            schema_version=schema_version,
            inside_parallel=True,
        )
        expanded = sum(branch_counts)
    else:  # pragma: no cover - exact construction prevents this branch
        raise TypeError("program must contain exact ProgramNodeCfg values.")
    if expanded > MAX_EXPANDED_CALLS:
        raise ValueError(
            f"Program expands to more than {MAX_EXPANDED_CALLS} semantic calls."
        )
    return expanded


@configclass
class ExpertProgramCfg:
    """Strict, versioned, executable-free Expert Program configuration."""

    schema_version: int = MISSING
    program_id: str = MISSING
    integration: ExpertProgramIntegrationCfg = MISSING
    program: ProgramNodeCfg = MISSING
    targets: dict[str, TargetCfg] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the complete static configuration and target graph."""
        if (
            type(self.schema_version) is not int
            or self.schema_version not in SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS
        ):
            raise ValueError(
                "schema_version must be one of "
                f"{SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS}."
            )
        _validate_identifier(self.program_id, field_name="program_id")
        if type(self.integration) is not ExpertProgramIntegrationCfg:
            raise TypeError("integration must be ExpertProgramIntegrationCfg.")
        if type(self.targets) is not dict:
            raise TypeError("targets must be an exact dict.")
        targets: dict[str, TargetCfg] = {}
        for target_id, target in self.targets.items():
            normalized_id = _validate_identifier(
                target_id,
                field_name="target IDs",
            )
            if type(target) is not CyclicPoseTargetCfg:
                raise TypeError("targets must contain exact TargetCfg values.")
            targets[normalized_id] = target
        if type(self.program) not in _PROGRAM_NODE_TYPES:
            raise TypeError("program must be an exact ProgramNodeCfg value.")
        expanded = _validate_program(
            self.program,
            targets=targets,
            depth=0,
            budget=[MAX_PROGRAM_NODES],
            schema_version=self.schema_version,
        )
        if expanded <= 0:
            raise ValueError("program must contain at least one semantic call.")
        self.targets = targets


__all__ = [
    "BarrierCfg",
    "CyclicPoseTargetCfg",
    "DeclarativeCfgValue",
    "EXPERT_PROGRAM_SCHEMA_VERSION",
    "EXPERT_PROGRAM_SCHEMA_VERSION_V2",
    "ExpertProgramCfg",
    "ExpertProgramIntegrationCfg",
    "HandOverCfg",
    "InvokeCfg",
    "MAX_DECLARATIVE_DEPTH",
    "MAX_DECLARATIVE_NODES",
    "MAX_EXPANDED_CALLS",
    "MAX_PROGRAM_DEPTH",
    "MAX_PROGRAM_NODES",
    "MAX_REPEAT_COUNT",
    "ObjectNearTargetValidatorCfg",
    "OperateArticulationCfg",
    "ParallelCfg",
    "PickCfg",
    "PlaceCfg",
    "PoseCfg",
    "PostPolicyCfg",
    "ProgramNodeCfg",
    "RegisteredSemanticCallCfg",
    "RepeatCfg",
    "SegmentCfg",
    "SemanticCallCfg",
    "SequenceCfg",
    "SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS",
    "TargetCfg",
    "TargetRefCfg",
    "ValidatorCfg",
    "WaitStablePostCfg",
]
