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

"""Strict JSON/YAML-value decoder for Expert Program schema versions 1 and 2."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from .cfg import (
    BarrierCfg,
    EXPERT_PROGRAM_SCHEMA_VERSION,
    EXPERT_PROGRAM_SCHEMA_VERSION_V2,
    MAX_PROGRAM_DEPTH,
    MAX_REPEAT_COUNT,
    CyclicPoseTargetCfg,
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    HandOverCfg,
    InvokeCfg,
    ObjectNearTargetValidatorCfg,
    OperateArticulationCfg,
    ParallelCfg,
    PickCfg,
    PlaceCfg,
    PoseCfg,
    PostPolicyCfg,
    ProgramNodeCfg,
    RegisteredSemanticCallCfg,
    RepeatCfg,
    SegmentCfg,
    SemanticCallCfg,
    SequenceCfg,
    SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS,
    TargetCfg,
    TargetRefCfg,
    ValidatorCfg,
    WaitStablePostCfg,
)

ConfigPathPart: TypeAlias = str | int
ConfigPath: TypeAlias = tuple[ConfigPathPart, ...]
SceneReferenceRole: TypeAlias = Literal[
    "entity",
    "object",
    "articulation",
    "affordance",
    "object_or_affordance",
]

_MAX_INPUT_DEPTH = 128
_MAX_INPUT_NODES = 100_000
_ENV_TRAVERSAL_PATTERN = re.compile(
    r"(?:\$?(?:env|environment)(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
    r"\$\{(?:env|environment)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\})"
)
_FORBIDDEN_KEYS = frozenset(
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


def render_config_path(path: ConfigPath) -> str:
    """Render one configuration path using JSONPath-like notation.

    Args:
        path: Tuple of mapping keys and sequence indices.

    Returns:
        Stable human-readable path beginning at ``$``.
    """
    rendered = "$"
    for part in path:
        if type(part) is int:
            rendered += f"[{part}]"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) is not None:
            rendered += f".{part}"
        else:
            rendered += f"[{part!r}]"
    return rendered


class ExpertProgramConfigError(ValueError):
    """Base pathful diagnostic for Expert Program configuration failures."""

    def __init__(self, code: str, path: ConfigPath, message: str) -> None:
        """Create one stable pathful diagnostic.

        Args:
            code: Machine-readable failure code.
            path: Exact configuration location.
            message: Human-readable explanation.
        """
        self.code = code
        self.path = tuple(path)
        self.message = message
        super().__init__(f"{render_config_path(self.path)}: {message} [{code}]")


class ExpertProgramDecodeError(ExpertProgramConfigError):
    """Raised when untrusted data does not match a supported strict schema."""


class ExpertProgramValidationError(ExpertProgramConfigError):
    """Raised when an explicit static integration context rejects a reference."""


@runtime_checkable
class ExpertProgramValidationContext(Protocol):
    """Provider-free static validation boundary for external references.

    Implementations may resolve profile, scene, preset, catalog, affordance, and
    resource IDs, but must not observe simulation state, construct planners, or
    execute calls.
    """

    def validate_integration(
        self,
        integration: ExpertProgramIntegrationCfg,
        *,
        path: ConfigPath,
    ) -> None:
        """Validate integration references at ``path``."""

    def validate_semantic_call(
        self,
        call: SemanticCallCfg,
        *,
        path: ConfigPath,
    ) -> None:
        """Validate catalog identity, schema revision, and resource overrides."""

    def validate_scene_reference(
        self,
        reference: str,
        *,
        role: SceneReferenceRole,
        path: ConfigPath,
    ) -> None:
        """Validate one canonical scene reference with its semantic role."""

    def validate_post_policy(
        self,
        policy: PostPolicyCfg,
        *,
        path: ConfigPath,
    ) -> None:
        """Validate a post-policy kind and its named preset."""

    def validate_validator(
        self,
        validator: ValidatorCfg,
        *,
        path: ConfigPath,
    ) -> None:
        """Validate one registered segment-validator contract."""


def _error(code: str, path: ConfigPath, message: str) -> ExpertProgramDecodeError:
    """Build one decoder diagnostic."""
    return ExpertProgramDecodeError(code, path, message)


def _clone_untrusted_value(
    value: object,
    *,
    path: ConfigPath,
    active: set[int],
    budget: list[int],
    depth: int,
) -> object:
    """Own and validate one bounded JSON-compatible value tree."""
    if depth > _MAX_INPUT_DEPTH:
        raise _error(
            "input_too_deep",
            path,
            f"Input exceeds nesting depth limit {_MAX_INPUT_DEPTH}.",
        )
    budget[0] -= 1
    if budget[0] < 0:
        raise _error(
            "input_too_large",
            path,
            f"Input exceeds node limit {_MAX_INPUT_NODES}.",
        )
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _error("non_finite_number", path, "Floats must be finite.")
        return value
    if type(value) is str:
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered.startswith(("__import__(", "eval(", "exec(", "import ", "from ")):
            raise _error(
                "executable_expression",
                path,
                "Imports, eval, exec, and executable expressions are forbidden.",
            )
        if _ENV_TRAVERSAL_PATTERN.fullmatch(stripped) is not None:
            raise _error(
                "environment_traversal",
                path,
                "Dotted environment attribute traversal is forbidden.",
            )
        return value
    if type(value) is list:
        identity = id(value)
        if identity in active:
            raise _error("cyclic_input", path, "Input contains a cyclic list.")
        active.add(identity)
        try:
            return [
                _clone_untrusted_value(
                    item,
                    path=(*path, index),
                    active=active,
                    budget=budget,
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    if type(value) is dict:
        identity = id(value)
        if identity in active:
            raise _error("cyclic_input", path, "Input contains a cyclic mapping.")
        active.add(identity)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise _error(
                        "invalid_mapping_key",
                        path,
                        "Mapping keys must be exact strings.",
                    )
                if key.lower() in _FORBIDDEN_KEYS:
                    raise _error(
                        "forbidden_construct",
                        (*path, key),
                        f"Field {key!r} requests executable or traversal behavior.",
                    )
                result[key] = _clone_untrusted_value(
                    item,
                    path=(*path, key),
                    active=active,
                    budget=budget,
                    depth=depth + 1,
                )
            return result
        finally:
            active.remove(identity)
    raise _error(
        "non_declarative_value",
        path,
        f"{type(value).__name__} is not JSON-compatible declarative data; "
        "callables, classes, modules, tensors, and live objects are forbidden.",
    )


def _expect_mapping(value: object, *, path: ConfigPath) -> dict[str, object]:
    """Require one exact mapping."""
    if type(value) is not dict:
        raise _error("expected_mapping", path, "Expected an object mapping.")
    return value


def _expect_list(value: object, *, path: ConfigPath) -> list[object]:
    """Require one exact JSON list."""
    if type(value) is not list:
        raise _error("expected_list", path, "Expected a list.")
    return value


def _validate_fields(
    value: dict[str, object],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    path: ConfigPath,
) -> None:
    """Reject unknown fields and report the first missing required field."""
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        field_name = unknown[0]
        raise _error(
            "unknown_field",
            (*path, field_name),
            f"Unknown field {field_name!r}; allowed fields are {sorted(allowed)}.",
        )
    missing = sorted(required.difference(value))
    if missing:
        field_name = missing[0]
        raise _error(
            "missing_field",
            (*path, field_name),
            f"Missing required field {field_name!r}.",
        )


def _expect_identifier(value: object, *, path: ConfigPath) -> str:
    """Require one exact non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise _error(
            "invalid_identifier",
            path,
            "Expected a non-empty string without outer whitespace.",
        )
    return value


def _expect_discriminator(
    value: dict[str, object],
    *,
    path: ConfigPath,
    supported: tuple[str, ...],
) -> str:
    """Read one required exact string discriminator."""
    if "kind" not in value:
        raise _error(
            "missing_discriminator",
            (*path, "kind"),
            "Missing required discriminator 'kind'.",
        )
    kind = value["kind"]
    if type(kind) is not str or kind not in supported:
        raise _error(
            "unknown_discriminator",
            (*path, "kind"),
            f"Unsupported discriminator {kind!r}; expected one of {supported}.",
        )
    return kind


def _decode_resources(value: object, *, path: ConfigPath) -> dict[str, str]:
    """Decode one strict slot-to-resource mapping."""
    mapping = _expect_mapping(value, path=path)
    return {
        _expect_identifier(slot_id, path=(*path, slot_id)): _expect_identifier(
            resource_id,
            path=(*path, slot_id),
        )
        for slot_id, resource_id in mapping.items()
    }


def _construct(
    constructor: Callable[..., object],
    *,
    path: ConfigPath,
    **kwargs: object,
) -> object:
    """Construct one config value and wrap invariant failures pathfully."""
    try:
        return constructor(**kwargs)
    except ExpertProgramConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise _error("invalid_value", path, str(exc)) from exc


def _decode_pose(value: object, *, path: ConfigPath) -> PoseCfg:
    """Decode one finite pose value."""
    mapping = _expect_mapping(value, path=path)
    _validate_fields(
        mapping,
        allowed=frozenset({"position", "quaternion_wxyz"}),
        required=frozenset({"position", "quaternion_wxyz"}),
        path=path,
    )
    position_values = _expect_list(mapping["position"], path=(*path, "position"))
    quaternion_values = _expect_list(
        mapping["quaternion_wxyz"],
        path=(*path, "quaternion_wxyz"),
    )
    if len(position_values) != 3:
        raise _error(
            "invalid_pose_shape",
            (*path, "position"),
            "position must contain exactly three numbers.",
        )
    if len(quaternion_values) != 4:
        raise _error(
            "invalid_pose_shape",
            (*path, "quaternion_wxyz"),
            "quaternion_wxyz must contain exactly four numbers.",
        )
    for name, values in (
        ("position", position_values),
        ("quaternion_wxyz", quaternion_values),
    ):
        for index, number in enumerate(values):
            if type(number) not in (int, float):
                raise _error(
                    "invalid_number",
                    (*path, name, index),
                    "Pose components must be finite numbers, not bool values.",
                )
    return _construct(
        PoseCfg,
        path=path,
        position=tuple(position_values),
        quaternion_wxyz=tuple(quaternion_values),
    )  # type: ignore[return-value]


def _decode_target(value: object, *, path: ConfigPath) -> TargetCfg:
    """Decode one target provider shared by the supported schema versions."""
    mapping = _expect_mapping(value, path=path)
    kind = _expect_discriminator(
        mapping,
        path=path,
        supported=("cyclic_pose",),
    )
    assert kind == "cyclic_pose"
    _validate_fields(
        mapping,
        allowed=frozenset({"kind", "values"}),
        required=frozenset({"kind", "values"}),
        path=path,
    )
    values = tuple(
        _decode_pose(item, path=(*path, "values", index))
        for index, item in enumerate(
            _expect_list(mapping["values"], path=(*path, "values"))
        )
    )
    return _construct(
        CyclicPoseTargetCfg,
        path=path,
        kind=kind,
        values=values,
    )  # type: ignore[return-value]


def _decode_target_ref(
    value: object,
    *,
    path: ConfigPath,
    target_ids: frozenset[str],
) -> TargetRefCfg:
    """Decode and statically resolve one target reference."""
    mapping = _expect_mapping(value, path=path)
    kind = _expect_discriminator(mapping, path=path, supported=("target_ref",))
    _validate_fields(
        mapping,
        allowed=frozenset({"kind", "target"}),
        required=frozenset({"kind", "target"}),
        path=path,
    )
    target = _expect_identifier(mapping["target"], path=(*path, "target"))
    if target not in target_ids:
        raise _error(
            "unknown_target",
            (*path, "target"),
            f"Unknown target {target!r}; available targets are {sorted(target_ids)}.",
        )
    return _construct(
        TargetRefCfg,
        path=path,
        kind=kind,
        target=target,
    )  # type: ignore[return-value]


def _decode_call(
    value: object,
    *,
    path: ConfigPath,
    target_ids: frozenset[str],
) -> SemanticCallCfg:
    """Decode one discriminated semantic call."""
    mapping = _expect_mapping(value, path=path)
    kind = _expect_discriminator(
        mapping,
        path=path,
        supported=(
            "pick",
            "place",
            "hand_over",
            "operate_articulation",
            "registered",
        ),
    )
    resources = _decode_resources(
        mapping.get("resources", {}),
        path=(*path, "resources"),
    )
    if kind == "pick":
        _validate_fields(
            mapping,
            allowed=frozenset({"kind", "object", "grasp", "resources"}),
            required=frozenset({"kind", "object"}),
            path=path,
        )
        grasp = mapping.get("grasp")
        if grasp is not None:
            grasp = _expect_identifier(grasp, path=(*path, "grasp"))
        return _construct(
            PickCfg,
            path=path,
            kind=kind,
            object=_expect_identifier(mapping["object"], path=(*path, "object")),
            grasp=grasp,
            resources=resources,
        )  # type: ignore[return-value]
    if kind == "place":
        _validate_fields(
            mapping,
            allowed=frozenset({"kind", "object", "at", "on", "inside", "resources"}),
            required=frozenset({"kind", "object"}),
            path=path,
        )
        at = (
            None
            if mapping.get("at") is None
            else _decode_target_ref(
                mapping["at"],
                path=(*path, "at"),
                target_ids=target_ids,
            )
        )
        on = mapping.get("on")
        inside = mapping.get("inside")
        if on is not None:
            on = _expect_identifier(on, path=(*path, "on"))
        if inside is not None:
            inside = _expect_identifier(inside, path=(*path, "inside"))
        return _construct(
            PlaceCfg,
            path=path,
            kind=kind,
            object=_expect_identifier(mapping["object"], path=(*path, "object")),
            at=at,
            on=on,
            inside=inside,
            resources=resources,
        )  # type: ignore[return-value]
    if kind == "hand_over":
        _validate_fields(
            mapping,
            allowed=frozenset({"kind", "object", "final_target", "resources"}),
            required=frozenset({"kind", "object"}),
            path=path,
        )
        final_target = (
            None
            if mapping.get("final_target") is None
            else _decode_target_ref(
                mapping["final_target"],
                path=(*path, "final_target"),
                target_ids=target_ids,
            )
        )
        return _construct(
            HandOverCfg,
            path=path,
            kind=kind,
            object=_expect_identifier(mapping["object"], path=(*path, "object")),
            final_target=final_target,
            resources=resources,
        )  # type: ignore[return-value]
    if kind == "operate_articulation":
        _validate_fields(
            mapping,
            allowed=frozenset(
                {
                    "kind",
                    "articulation",
                    "handle",
                    "target",
                    "target_position",
                    "target_displacement",
                    "resources",
                }
            ),
            required=frozenset({"kind", "articulation"}),
            path=path,
        )
        handle = mapping.get("handle")
        target = mapping.get("target")
        if handle is not None:
            handle = _expect_identifier(handle, path=(*path, "handle"))
        if target is not None:
            target = _expect_identifier(target, path=(*path, "target"))
        target_position = mapping.get("target_position")
        target_displacement = mapping.get("target_displacement")
        for field_name, value in (
            ("target_position", target_position),
            ("target_displacement", target_displacement),
        ):
            if value is not None and type(value) not in (int, float):
                raise _error(
                    "invalid_number",
                    (*path, field_name),
                    f"{field_name} must be a finite number, not bool.",
                )
        named = target is not None
        explicit_position = target_position is not None
        explicit_displacement = target_displacement is not None
        if named and (explicit_position or explicit_displacement):
            raise _error(
                "conflicting_articulation_target",
                path,
                "target is mutually exclusive with target_position and "
                "target_displacement.",
            )
        if not named and not (explicit_position and explicit_displacement):
            raise _error(
                "incomplete_articulation_target",
                path,
                "Specify target or both target_position and target_displacement.",
            )
        return _construct(
            OperateArticulationCfg,
            path=path,
            kind=kind,
            articulation=_expect_identifier(
                mapping["articulation"],
                path=(*path, "articulation"),
            ),
            handle=handle,
            target=target,
            target_position=target_position,
            target_displacement=target_displacement,
            resources=resources,
        )  # type: ignore[return-value]

    _validate_fields(
        mapping,
        allowed=frozenset(
            {"kind", "call_id", "schema_version", "arguments", "resources"}
        ),
        required=frozenset({"kind", "call_id", "schema_version"}),
        path=path,
    )
    arguments = _expect_mapping(
        mapping.get("arguments", {}),
        path=(*path, "arguments"),
    )
    schema_version = mapping["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise _error(
            "invalid_schema_version",
            (*path, "schema_version"),
            "Registered call schema_version must be exactly 1.",
        )
    return _construct(
        RegisteredSemanticCallCfg,
        path=path,
        kind=kind,
        call_id=_expect_identifier(mapping["call_id"], path=(*path, "call_id")),
        schema_version=schema_version,
        arguments=arguments,
        resources=resources,
    )  # type: ignore[return-value]


def _decode_post_policy(value: object, *, path: ConfigPath) -> PostPolicyCfg:
    """Decode one segment post-policy shared by the supported schemas."""
    mapping = _expect_mapping(value, path=path)
    kind = _expect_discriminator(mapping, path=path, supported=("wait_stable",))
    _validate_fields(
        mapping,
        allowed=frozenset({"kind", "entity", "preset"}),
        required=frozenset({"kind", "entity"}),
        path=path,
    )
    return _construct(
        WaitStablePostCfg,
        path=path,
        kind=kind,
        entity=_expect_identifier(mapping["entity"], path=(*path, "entity")),
        preset=_expect_identifier(
            mapping.get("preset", "rigid_object"),
            path=(*path, "preset"),
        ),
    )  # type: ignore[return-value]


def _decode_validator(
    value: object,
    *,
    path: ConfigPath,
    target_ids: frozenset[str],
) -> ValidatorCfg:
    """Decode one segment validator shared by the supported schemas."""
    mapping = _expect_mapping(value, path=path)
    kind = _expect_discriminator(
        mapping,
        path=path,
        supported=("object_near_target",),
    )
    _validate_fields(
        mapping,
        allowed=frozenset({"kind", "object", "target", "position_tolerance"}),
        required=frozenset({"kind", "object", "target"}),
        path=path,
    )
    target = _expect_identifier(mapping["target"], path=(*path, "target"))
    if target not in target_ids:
        raise _error(
            "unknown_target",
            (*path, "target"),
            f"Unknown target {target!r}; available targets are {sorted(target_ids)}.",
        )
    tolerance = mapping.get("position_tolerance", 0.03)
    if type(tolerance) not in (int, float):
        raise _error(
            "invalid_number",
            (*path, "position_tolerance"),
            "position_tolerance must be a finite number, not bool.",
        )
    return _construct(
        ObjectNearTargetValidatorCfg,
        path=path,
        kind=kind,
        object=_expect_identifier(mapping["object"], path=(*path, "object")),
        target=target,
        position_tolerance=tolerance,
    )  # type: ignore[return-value]


def _decode_program_node(
    value: object,
    *,
    path: ConfigPath,
    target_ids: frozenset[str],
    depth: int,
    schema_version: int,
) -> ProgramNodeCfg:
    """Recursively decode one bounded versioned program node."""
    if depth > MAX_PROGRAM_DEPTH:
        raise _error(
            "program_too_deep",
            path,
            "Program AST exceeds the configured nesting depth.",
        )
    mapping = _expect_mapping(value, path=path)
    supported_kinds = ["sequence", "repeat", "segment", "invoke"]
    if schema_version >= EXPERT_PROGRAM_SCHEMA_VERSION_V2:
        supported_kinds.extend(("parallel", "barrier"))
    kind = _expect_discriminator(
        mapping,
        path=path,
        supported=tuple(supported_kinds),
    )
    if kind == "sequence":
        _validate_fields(
            mapping,
            allowed=frozenset({"kind", "items"}),
            required=frozenset({"kind", "items"}),
            path=path,
        )
        items = tuple(
            _decode_program_node(
                item,
                path=(*path, "items", index),
                target_ids=target_ids,
                depth=depth + 1,
                schema_version=schema_version,
            )
            for index, item in enumerate(
                _expect_list(mapping["items"], path=(*path, "items"))
            )
        )
        return _construct(
            SequenceCfg,
            path=path,
            kind=kind,
            items=items,
        )  # type: ignore[return-value]
    if kind == "repeat":
        _validate_fields(
            mapping,
            allowed=frozenset({"kind", "count", "body"}),
            required=frozenset({"kind", "count", "body"}),
            path=path,
        )
        count = mapping["count"]
        if type(count) is not int or not 1 <= count <= MAX_REPEAT_COUNT:
            raise _error(
                "invalid_repeat_count",
                (*path, "count"),
                f"Repeat count must be an integer in [1, {MAX_REPEAT_COUNT}].",
            )
        return _construct(
            RepeatCfg,
            path=path,
            kind=kind,
            count=count,
            body=_decode_program_node(
                mapping["body"],
                path=(*path, "body"),
                target_ids=target_ids,
                depth=depth + 1,
                schema_version=schema_version,
            ),
        )  # type: ignore[return-value]
    if kind == "segment":
        _validate_fields(
            mapping,
            allowed=frozenset({"kind", "name", "steps", "post", "validators"}),
            required=frozenset({"kind", "name", "steps"}),
            path=path,
        )
        post = tuple(
            _decode_post_policy(item, path=(*path, "post", index))
            for index, item in enumerate(
                _expect_list(mapping.get("post", []), path=(*path, "post"))
            )
        )
        validators = tuple(
            _decode_validator(
                item,
                path=(*path, "validators", index),
                target_ids=target_ids,
            )
            for index, item in enumerate(
                _expect_list(
                    mapping.get("validators", []),
                    path=(*path, "validators"),
                )
            )
        )
        return _construct(
            SegmentCfg,
            path=path,
            kind=kind,
            name=_expect_identifier(mapping["name"], path=(*path, "name")),
            steps=_decode_program_node(
                mapping["steps"],
                path=(*path, "steps"),
                target_ids=target_ids,
                depth=depth + 1,
                schema_version=schema_version,
            ),
            post=post,
            validators=validators,
        )  # type: ignore[return-value]

    if kind == "parallel":
        _validate_fields(
            mapping,
            allowed=frozenset({"kind", "branches", "barrier"}),
            required=frozenset({"kind", "branches", "barrier"}),
            path=path,
        )
        branches_values = _expect_list(
            mapping["branches"],
            path=(*path, "branches"),
        )
        if len(branches_values) < 2:
            raise _error(
                "parallel_branch_count",
                (*path, "branches"),
                "Parallel requires at least two branches.",
            )
        barrier = _decode_program_node(
            mapping["barrier"],
            path=(*path, "barrier"),
            target_ids=target_ids,
            depth=depth + 1,
            schema_version=schema_version,
        )
        if type(barrier) is not BarrierCfg:
            raise _error(
                "parallel_barrier_required",
                (*path, "barrier"),
                "Parallel.barrier must be an explicit barrier node.",
            )
        return _construct(
            ParallelCfg,
            path=path,
            kind=kind,
            branches=tuple(
                _decode_program_node(
                    branch,
                    path=(*path, "branches", index),
                    target_ids=target_ids,
                    depth=depth + 1,
                    schema_version=schema_version,
                )
                for index, branch in enumerate(branches_values)
            ),
            barrier=barrier,
        )  # type: ignore[return-value]
    if kind == "barrier":
        _validate_fields(
            mapping,
            allowed=frozenset({"kind", "name", "timeout_steps", "failure_policy"}),
            required=frozenset({"kind", "name"}),
            path=path,
        )
        timeout_steps = mapping.get("timeout_steps", 1_000)
        if type(timeout_steps) is not int or timeout_steps <= 0:
            raise _error(
                "invalid_barrier_timeout",
                (*path, "timeout_steps"),
                "Barrier timeout_steps must be a positive integer.",
            )
        failure_policy = mapping.get("failure_policy", "fail_fast")
        if failure_policy != "fail_fast":
            raise _error(
                "unsupported_failure_policy",
                (*path, "failure_policy"),
                "Barrier failure_policy must be exactly 'fail_fast'.",
            )
        return _construct(
            BarrierCfg,
            path=path,
            kind=kind,
            name=_expect_identifier(mapping["name"], path=(*path, "name")),
            timeout_steps=timeout_steps,
            failure_policy=failure_policy,
        )  # type: ignore[return-value]

    _validate_fields(
        mapping,
        allowed=frozenset({"kind", "call"}),
        required=frozenset({"kind", "call"}),
        path=path,
    )
    return _construct(
        InvokeCfg,
        path=path,
        kind=kind,
        call=_decode_call(
            mapping["call"],
            path=(*path, "call"),
            target_ids=target_ids,
        ),
    )  # type: ignore[return-value]


def _walk_program(
    node: ProgramNodeCfg,
    *,
    path: ConfigPath,
) -> list[tuple[ProgramNodeCfg, ConfigPath]]:
    """Return deterministic node/path pairs for static context validation."""
    values = [(node, path)]
    if type(node) is SequenceCfg:
        for index, child in enumerate(node.items):
            values.extend(_walk_program(child, path=(*path, "items", index)))
    elif type(node) is RepeatCfg:
        values.extend(_walk_program(node.body, path=(*path, "body")))
    elif type(node) is SegmentCfg:
        values.extend(_walk_program(node.steps, path=(*path, "steps")))
    elif type(node) is ParallelCfg:
        for index, branch in enumerate(node.branches):
            values.extend(_walk_program(branch, path=(*path, "branches", index)))
        values.extend(_walk_program(node.barrier, path=(*path, "barrier")))
    return values


def _call_context(
    callback: Callable[..., None],
    *args: object,
    path: ConfigPath,
    **kwargs: object,
) -> None:
    """Call one static validation hook and preserve pathful failures."""
    try:
        callback(*args, path=path, **kwargs)
    except ExpertProgramConfigError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ExpertProgramValidationError(
            "reference_validation_failed",
            path,
            str(exc),
        ) from exc


def _validate_decoded_semantic_call(
    call: SemanticCallCfg,
    context: ExpertProgramValidationContext,
    *,
    path: ConfigPath,
) -> None:
    """Validate one decoded call against provider-free catalog and scene ports."""
    _call_context(context.validate_semantic_call, call, path=path)
    if type(call) in (PickCfg, PlaceCfg, HandOverCfg):
        _call_context(
            context.validate_scene_reference,
            call.object,
            role="object",
            path=(*path, "object"),
        )
    if type(call) is PickCfg and call.grasp is not None:
        _call_context(
            context.validate_scene_reference,
            call.grasp,
            role="affordance",
            path=(*path, "grasp"),
        )
    if type(call) is PlaceCfg:
        for field_name in ("on", "inside"):
            reference = getattr(call, field_name)
            if reference is not None:
                _call_context(
                    context.validate_scene_reference,
                    reference,
                    role="object_or_affordance",
                    path=(*path, field_name),
                )
    if type(call) is OperateArticulationCfg:
        _call_context(
            context.validate_scene_reference,
            call.articulation,
            role="articulation",
            path=(*path, "articulation"),
        )
        if call.handle is not None:
            _call_context(
                context.validate_scene_reference,
                call.handle,
                role="affordance",
                path=(*path, "handle"),
            )


def decode_semantic_call(
    data: object,
    *,
    target_ids: frozenset[str] = frozenset(),
    validation_context: ExpertProgramValidationContext | None = None,
    path: ConfigPath = ("call",),
) -> SemanticCallCfg:
    """Decode one untrusted canonical semantic-call payload.

    Args:
        data: Exact JSON-compatible call mapping produced by a trusted parser.
        target_ids: Target IDs available to ``at`` or ``final_target`` refs.
        validation_context: Optional provider-free catalog and scene validator.
        path: Diagnostic root used when the call is embedded in another schema.

    Returns:
        Fully owned and strictly validated canonical semantic-call config.

    Raises:
        ExpertProgramDecodeError: If the payload violates the call schema.
        ExpertProgramValidationError: If provider-free validation rejects it.
    """
    if type(target_ids) is not frozenset or not all(
        type(target_id) is str and target_id and target_id == target_id.strip()
        for target_id in target_ids
    ):
        raise TypeError("target_ids must be a frozenset of exact identifiers.")
    if type(path) is not tuple:
        raise TypeError("path must be a ConfigPath tuple.")
    owned = _clone_untrusted_value(
        data,
        path=path,
        active=set(),
        budget=[_MAX_INPUT_NODES],
        depth=0,
    )
    call = _decode_call(owned, path=path, target_ids=target_ids)
    if validation_context is not None:
        if not isinstance(validation_context, ExpertProgramValidationContext):
            raise TypeError(
                "validation_context must implement " "ExpertProgramValidationContext."
            )
        _validate_decoded_semantic_call(call, validation_context, path=path)
    return call


def encode_semantic_call(call: SemanticCallCfg) -> dict[str, object]:
    """Encode one canonical semantic-call config as owned JSON-safe values.

    Args:
        call: Exact supported semantic-call config.

    Returns:
        Deterministic mapping accepted by :func:`decode_semantic_call`.
    """
    if type(call) not in (
        PickCfg,
        PlaceCfg,
        HandOverCfg,
        OperateArticulationCfg,
        RegisteredSemanticCallCfg,
    ):
        raise TypeError("call must be an exact SemanticCallCfg value.")
    result: dict[str, object] = {"kind": call.kind}
    if type(call) is PickCfg:
        result["object"] = call.object
        if call.grasp is not None:
            result["grasp"] = call.grasp
    elif type(call) is PlaceCfg:
        result["object"] = call.object
        if call.at is not None:
            result["at"] = {"kind": call.at.kind, "target": call.at.target}
        if call.on is not None:
            result["on"] = call.on
        if call.inside is not None:
            result["inside"] = call.inside
    elif type(call) is HandOverCfg:
        result["object"] = call.object
        if call.final_target is not None:
            result["final_target"] = {
                "kind": call.final_target.kind,
                "target": call.final_target.target,
            }
    elif type(call) is OperateArticulationCfg:
        result["articulation"] = call.articulation
        if call.handle is not None:
            result["handle"] = call.handle
        if call.target is not None:
            result["target"] = call.target
        else:
            result["target_position"] = call.target_position
            result["target_displacement"] = call.target_displacement
    else:
        assert type(call) is RegisteredSemanticCallCfg
        result["call_id"] = call.call_id
        result["schema_version"] = call.schema_version
        result["arguments"] = _encode_declarative_value(call.arguments)
    if call.resources:
        result["resources"] = dict(call.resources)
    return result


def _encode_declarative_value(value: object) -> object:
    """Convert an owned config snapshot back to exact JSON-compatible values."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _encode_declarative_value(nested) for key, nested in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_encode_declarative_value(nested) for nested in value]
    return deepcopy(value)


def validate_expert_program(
    config: ExpertProgramCfg,
    context: ExpertProgramValidationContext,
) -> None:
    """Resolve external references without observing or executing an environment.

    Args:
        config: Fully decoded and internally validated Expert Program.
        context: Provider-free static integration/catalog/scene validator.

    Raises:
        TypeError: If either argument has the wrong contract.
        ExpertProgramValidationError: If an external reference is unavailable.
    """
    if type(config) is not ExpertProgramCfg:
        raise TypeError("config must be exactly ExpertProgramCfg.")
    if not isinstance(context, ExpertProgramValidationContext):
        raise TypeError(
            "context must implement ExpertProgramValidationContext exactly."
        )
    _call_context(
        context.validate_integration,
        config.integration,
        path=("integration",),
    )
    for node, path in _walk_program(config.program, path=("program",)):
        if type(node) is InvokeCfg:
            call = node.call
            call_path = (*path, "call")
            _validate_decoded_semantic_call(call, context, path=call_path)
        elif type(node) is SegmentCfg:
            for index, post in enumerate(node.post):
                post_path = (*path, "post", index)
                _call_context(
                    context.validate_post_policy,
                    post,
                    path=post_path,
                )
                _call_context(
                    context.validate_scene_reference,
                    post.entity,
                    role="entity",
                    path=(*post_path, "entity"),
                )
            for index, validator in enumerate(node.validators):
                validator_path = (*path, "validators", index)
                _call_context(
                    context.validate_validator,
                    validator,
                    path=validator_path,
                )
                _call_context(
                    context.validate_scene_reference,
                    validator.object,
                    role="object",
                    path=(*validator_path, "object"),
                )


def decode_expert_program(
    data: object,
    *,
    validation_context: ExpertProgramValidationContext | None = None,
) -> ExpertProgramCfg:
    """Decode untrusted JSON/YAML-shaped values into strict versioned config.

    Schema versions 1 and 2 are supported. Version 2 adds deterministic
    parallel blocks with explicit barriers while preserving the Version 1
    sequential nodes and semantic calls.

    Args:
        data: Exact JSON-compatible mapping produced by a trusted parser.
        validation_context: Optional provider-free static reference validator.

    Returns:
        Fully owned and internally validated Expert Program configuration.

    Raises:
        ExpertProgramDecodeError: If data is unsafe or violates the schema.
        ExpertProgramValidationError: If an explicit context rejects a reference.
    """
    owned = _clone_untrusted_value(
        data,
        path=(),
        active=set(),
        budget=[_MAX_INPUT_NODES],
        depth=0,
    )
    mapping = _expect_mapping(owned, path=())
    _validate_fields(
        mapping,
        allowed=frozenset(
            {"schema_version", "program_id", "integration", "targets", "program"}
        ),
        required=frozenset(
            {"schema_version", "program_id", "integration", "targets", "program"}
        ),
        path=(),
    )
    schema_version = mapping["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version not in SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS
    ):
        raise _error(
            "unsupported_schema_version",
            ("schema_version",),
            "Supported schema versions are "
            f"{list(SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS)}.",
        )

    integration_mapping = _expect_mapping(
        mapping["integration"],
        path=("integration",),
    )
    _validate_fields(
        integration_mapping,
        allowed=frozenset({"robot_profile", "scene_registry", "runtime_preset"}),
        required=frozenset({"robot_profile", "scene_registry", "runtime_preset"}),
        path=("integration",),
    )
    integration = _construct(
        ExpertProgramIntegrationCfg,
        path=("integration",),
        robot_profile=_expect_identifier(
            integration_mapping["robot_profile"],
            path=("integration", "robot_profile"),
        ),
        scene_registry=_expect_identifier(
            integration_mapping["scene_registry"],
            path=("integration", "scene_registry"),
        ),
        runtime_preset=_expect_identifier(
            integration_mapping["runtime_preset"],
            path=("integration", "runtime_preset"),
        ),
    )

    target_mapping = _expect_mapping(mapping["targets"], path=("targets",))
    targets: dict[str, TargetCfg] = {}
    for target_id, target_value in target_mapping.items():
        normalized_id = _expect_identifier(target_id, path=("targets", target_id))
        targets[normalized_id] = _decode_target(
            target_value,
            path=("targets", normalized_id),
        )
    target_ids = frozenset(targets)
    program = _decode_program_node(
        mapping["program"],
        path=("program",),
        target_ids=target_ids,
        depth=0,
        schema_version=schema_version,
    )
    config = _construct(
        ExpertProgramCfg,
        path=(),
        schema_version=schema_version,
        program_id=_expect_identifier(mapping["program_id"], path=("program_id",)),
        integration=integration,
        targets=targets,
        program=program,
    )
    assert type(config) is ExpertProgramCfg
    if validation_context is not None:
        validate_expert_program(config, validation_context)
    return config


__all__ = [
    "ConfigPath",
    "ConfigPathPart",
    "ExpertProgramConfigError",
    "ExpertProgramDecodeError",
    "ExpertProgramValidationContext",
    "ExpertProgramValidationError",
    "SceneReferenceRole",
    "decode_expert_program",
    "decode_semantic_call",
    "encode_semantic_call",
    "render_config_path",
    "validate_expert_program",
]
