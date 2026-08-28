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

"""Typed standard-runtime extension declarations for Expert Programs.

The values in this module deliberately describe extension wiring without
creating a simulator or resolving one live robot endpoint.  A task
registration owns the corresponding adapter, transport, and safety-factory
instances, while its provider-free catalog owns the exact declarations below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, Protocol, runtime_checkable, TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions.bindings import (
    JointPositionTarget,
    RuntimeEndpointTarget,
)
from embodichain.lab.sim.atomic_actions.runtime_commands import RuntimeCommandPayload
from embodichain.lab.semantic_skills.effects import (
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
)
from embodichain.lab.expert_program._parallel_executor import (
    ParallelCommandSafetyValidator,
)
from embodichain.lab.semantic_skills.profiles import (
    ControlPartEndpoint,
    ControlPartEndpointAdapter,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotSkillProfile,
)

from .bridge import (
    JointPositionGymTransportEncoder,
    RuntimeTransportActionEncoder,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine, SceneProvider
    from embodichain.lab.semantic_skills import (
        EffectEvidenceProvider,
        SceneRegistry,
    )
    from embodichain.lab.expert_program._semantic_compiler import (
        RegisteredSemanticLowerer,
    )

VersionedKey = tuple[str, str]
"""Exact ``(provider_or_projector_id, revision)`` registry key."""

_BUILTIN_TRACKING_FEEDBACK_SOURCE_KEYS = frozenset({("planning_context.robot", "1")})
_BUILTIN_TRACKING_PROJECTOR_KEYS = frozenset({("joint_position_payload", "1")})
_BUILTIN_EFFECT_EVIDENCE_SOURCE_KEYS = frozenset(
    {
        (
            CONTROL_PART_EVIDENCE_PROVIDER_ID,
            CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
        )
    }
)


def _identifier(value: object, *, field_name: str) -> str:
    """Validate one exact, non-empty registration identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _qualified_name(value: type[object] | object) -> str:
    """Return one deterministic diagnostic name."""
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _class_attribute(value: object, name: str, *, field_name: str) -> object:
    """Read registration metadata from the provider type, never instance state."""
    owner = type(value)
    if not hasattr(owner, name):
        raise TypeError(f"{field_name} must be declared on {owner.__name__}.")
    return getattr(owner, name)


def _versioned_keys(value: object, *, field_name: str) -> frozenset[VersionedKey]:
    """Validate one exact immutable set of versioned registry keys."""
    if type(value) is not frozenset:
        raise TypeError(f"{field_name} must be an exact frozenset.")
    normalized: set[VersionedKey] = set()
    for key in value:
        if type(key) is not tuple or len(key) != 2:
            raise TypeError(f"{field_name} must contain exact 2-tuples.")
        identifier, revision = key
        normalized.add(
            (
                _identifier(identifier, field_name=f"{field_name} IDs"),
                _identifier(revision, field_name=f"{field_name} revisions"),
            )
        )
    return frozenset(normalized)


def _identifier_set(value: object, *, field_name: str) -> frozenset[str]:
    """Validate one exact immutable set of identifiers."""
    if type(value) is not frozenset:
        raise TypeError(f"{field_name} must be an exact frozenset.")
    return frozenset(_identifier(item, field_name=field_name) for item in value)


def _type_tuple(
    value: object,
    *,
    base_type: type[object],
    field_name: str,
) -> tuple[type[object], ...]:
    """Validate one non-empty exact tuple of unique exact value types."""
    if type(value) is not tuple or not value:
        raise TypeError(f"{field_name} must be a non-empty exact tuple.")
    normalized: list[type[object]] = []
    for item in value:
        if not isinstance(item, type) or not issubclass(item, base_type):
            raise TypeError(
                f"{field_name} values must be {base_type.__name__} subclasses."
            )
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate exact types.")
    return tuple(normalized)


def validate_immutable_extension_declaration(
    value: object,
    *,
    field_name: str,
) -> None:
    """Accept only a deeply immutable frozen dataclass or stateless instance.

    Frozen dataclass fields may contain only immutable scalar values, types,
    enums with immutable values, exact tuples, exact frozensets, and recursively
    frozen dataclasses.
    Mutable leaves such as mappings, lists, sets, bytearrays, and tensors are
    rejected because registration-owned live extensions are shared with an
    assembled runtime.  A non-dataclass extension must not have instance or
    slot state at all.
    """
    if isinstance(value, type):
        raise TypeError(f"{field_name} must contain instances, not types.")

    def validate_state(
        declaration: object,
        *,
        path: str,
    ) -> tuple[bool, tuple[str, ...]]:
        """Validate declared state and return dataclass field names."""
        dataclass_declaration = is_dataclass(declaration)
        dataclass_field_names: set[str] = set()
        if dataclass_declaration:
            params = getattr(type(declaration), "__dataclass_params__", None)
            if params is None or not params.frozen:
                raise TypeError(
                    f"{path} stateful declarations must be frozen dataclasses."
                )
            dataclass_field_names.update(item.name for item in fields(declaration))

        state_names: set[str] = set()
        instance_state = getattr(declaration, "__dict__", None)
        if isinstance(instance_state, Mapping):
            state_names.update(instance_state)
        for owner in type(declaration).__mro__:
            declared_slots = getattr(owner, "__slots__", ())
            slots = (
                (declared_slots,) if isinstance(declared_slots, str) else declared_slots
            )
            for slot_name in slots:
                if slot_name in {"__dict__", "__weakref__"}:
                    continue
                storage_name = (
                    f"_{owner.__name__.lstrip('_')}{slot_name}"
                    if slot_name.startswith("__") and not slot_name.endswith("__")
                    else slot_name
                )
                if hasattr(declaration, storage_name):
                    state_names.add(storage_name)
        undeclared_state = (
            state_names.difference(dataclass_field_names)
            if dataclass_declaration
            else state_names
        )
        if undeclared_state:
            raise TypeError(
                f"{path} contains unfingerprinted state "
                f"{sorted(undeclared_state)}; Use a frozen dataclass with every "
                "configuration field declared, or a stateless instance."
            )
        return dataclass_declaration, tuple(sorted(dataclass_field_names))

    def validate_nested(
        nested: object,
        *,
        path: str,
        active: set[int],
    ) -> None:
        """Reject every mutable or opaque leaf in one declaration graph."""
        if nested is None or type(nested) in {bool, int, float, str}:
            return
        if isinstance(nested, type):
            return
        if isinstance(nested, Enum):
            validate_nested(
                nested.value,
                path=f"{path}.value",
                active=active,
            )
            return
        if isinstance(nested, torch.Tensor) or type(nested) in {
            list,
            dict,
            set,
            bytearray,
        }:
            raise TypeError(
                f"{path} must be deeply immutable; mutable value type "
                f"{_qualified_name(nested)!r} is forbidden."
            )
        if isinstance(nested, Mapping):
            raise TypeError(
                f"{path} must be deeply immutable; mapping values are forbidden."
            )

        nested_id = id(nested)
        if nested_id in active:
            raise TypeError(f"{path} must not contain a cyclic declaration graph.")
        if type(nested) in {tuple, frozenset}:
            active.add(nested_id)
            try:
                for index, item in enumerate(nested):
                    validate_nested(
                        item,
                        path=f"{path}[{index}]",
                        active=active,
                    )
            finally:
                active.remove(nested_id)
            return
        if is_dataclass(nested) and not isinstance(nested, type):
            active.add(nested_id)
            try:
                _, nested_field_names = validate_state(nested, path=path)
                for nested_field_name in nested_field_names:
                    validate_nested(
                        getattr(nested, nested_field_name),
                        path=f"{path}.{nested_field_name}",
                        active=active,
                    )
            finally:
                active.remove(nested_id)
            return
        raise TypeError(
            f"{path} contains unsupported value type "
            f"{_qualified_name(nested)!r}; extension declarations must be "
            "complete deeply immutable data."
        )

    dataclass_declaration, dataclass_field_names = validate_state(
        value,
        path=field_name,
    )
    if dataclass_declaration:
        for dataclass_field_name in dataclass_field_names:
            validate_nested(
                getattr(value, dataclass_field_name),
                path=f"{field_name}.{dataclass_field_name}",
                active={id(value)},
            )


@dataclass(frozen=True, slots=True)
class EndpointAdapterDeclaration:
    """Provider-free declaration of one exact endpoint adapter."""

    endpoint_type: type[ResourceEndpoint]
    adapter_type: type[ResourceEndpointAdapter]
    adapter_id: str
    runtime_transport_ids: frozenset[str]
    runtime_target_types: tuple[type[RuntimeEndpointTarget], ...]
    tracking_feedback_source_keys: frozenset[VersionedKey]
    tracking_projector_keys: frozenset[VersionedKey]
    effect_evidence_source_keys: frozenset[VersionedKey]

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_type, type) or not issubclass(
            self.endpoint_type, ResourceEndpoint
        ):
            raise TypeError("endpoint_type must be a ResourceEndpoint subclass.")
        if not isinstance(self.adapter_type, type) or not issubclass(
            self.adapter_type, ResourceEndpointAdapter
        ):
            raise TypeError("adapter_type must be a ResourceEndpointAdapter subclass.")
        _identifier(self.adapter_id, field_name="adapter_id")
        object.__setattr__(
            self,
            "runtime_transport_ids",
            _identifier_set(
                self.runtime_transport_ids,
                field_name="runtime_transport_ids",
            ),
        )
        if not self.runtime_transport_ids:
            raise ValueError("runtime_transport_ids must not be empty.")
        object.__setattr__(
            self,
            "runtime_target_types",
            _type_tuple(
                self.runtime_target_types,
                base_type=RuntimeEndpointTarget,
                field_name="runtime_target_types",
            ),
        )
        object.__setattr__(
            self,
            "tracking_feedback_source_keys",
            _versioned_keys(
                self.tracking_feedback_source_keys,
                field_name="tracking_feedback_source_keys",
            ),
        )
        object.__setattr__(
            self,
            "tracking_projector_keys",
            _versioned_keys(
                self.tracking_projector_keys,
                field_name="tracking_projector_keys",
            ),
        )
        object.__setattr__(
            self,
            "effect_evidence_source_keys",
            _versioned_keys(
                self.effect_evidence_source_keys,
                field_name="effect_evidence_source_keys",
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeTransportDeclaration:
    """Provider-free declaration of one ordered runtime transport encoder."""

    transport_type: type[RuntimeTransportActionEncoder]
    transport_id: str
    target_types: tuple[type[RuntimeEndpointTarget], ...]
    payload_types: tuple[type[RuntimeCommandPayload], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.transport_type, type):
            raise TypeError("transport_type must be a type.")
        _identifier(self.transport_id, field_name="transport_id")
        object.__setattr__(
            self,
            "target_types",
            _type_tuple(
                self.target_types,
                base_type=RuntimeEndpointTarget,
                field_name="target_types",
            ),
        )
        object.__setattr__(
            self,
            "payload_types",
            _type_tuple(
                self.payload_types,
                base_type=RuntimeCommandPayload,
                field_name="payload_types",
            ),
        )
        for field_name, declared_types in (
            ("target_types", self.target_types),
            ("payload_types", self.payload_types),
        ):
            for declared_type in declared_types:
                try:
                    type_transport_id = declared_type.__dict__["TRANSPORT_ID"]
                except KeyError as exc:
                    raise TypeError(
                        f"{field_name} value {declared_type.__name__} must declare "
                        "an exact ClassVar TRANSPORT_ID on that type; inherited or "
                        "instance-only transport IDs are forbidden."
                    ) from exc
                _identifier(
                    type_transport_id,
                    field_name=f"{declared_type.__name__}.TRANSPORT_ID",
                )
                if type_transport_id != self.transport_id:
                    raise ValueError(
                        f"{field_name} value {declared_type.__name__} declares "
                        f"transport {type_transport_id!r}, not "
                        f"{self.transport_id!r}."
                    )


@dataclass(frozen=True, slots=True)
class ParallelSafetyDeclaration:
    """Provider-free identity and transport coverage of one safety factory."""

    factory_type: type[object]
    validator_id: str
    revision: str
    supported_transport_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.factory_type, type):
            raise TypeError("factory_type must be a type.")
        _identifier(self.validator_id, field_name="validator_id")
        _identifier(self.revision, field_name="revision")
        object.__setattr__(
            self,
            "supported_transport_ids",
            _identifier_set(
                self.supported_transport_ids,
                field_name="supported_transport_ids",
            ),
        )
        if not self.supported_transport_ids:
            raise ValueError("supported_transport_ids must not be empty.")


@dataclass(frozen=True, slots=True)
class ControlPartEvidenceProviderDeclaration:
    """Provider-free identity of one control-part evidence factory.

    Args:
        factory_type: Exact immutable factory implementation type.
        provider_id: Stable effect-evidence provider identifier.
        revision: Exact provider contract revision.
    """

    factory_type: type[object]
    provider_id: str
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.factory_type, type):
            raise TypeError("factory_type must be a type.")
        _identifier(self.provider_id, field_name="provider_id")
        _identifier(self.revision, field_name="revision")
        expected = (
            CONTROL_PART_EVIDENCE_PROVIDER_ID,
            CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
        )
        if (self.provider_id, self.revision) != expected:
            raise ValueError(
                "A control-part evidence factory must provide the exact built-in "
                f"route {expected!r}."
            )


@dataclass(frozen=True, slots=True)
class RegisteredSemanticLowererDeclaration:
    """Provider-free identity of one registered semantic lowerer factory.

    Args:
        factory_type: Exact immutable factory implementation type.
        call_id: Registered semantic-call ID owned by the factory.
        revision: Exact factory contract revision.
    """

    factory_type: type[object]
    call_id: str
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.factory_type, type):
            raise TypeError("factory_type must be a type.")
        _identifier(self.call_id, field_name="call_id")
        _identifier(self.revision, field_name="revision")


@runtime_checkable
class ParallelCommandSafetyValidatorFactory(Protocol):
    """Registration-owned factory for one authoritative live safety gate."""

    validator_id: ClassVar[str]
    revision: ClassVar[str]
    supported_transport_ids: ClassVar[frozenset[str]]

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> ParallelCommandSafetyValidator:
        """Create one live gate bound to the exact assembled runtime."""


@runtime_checkable
class ControlPartEvidenceProviderFactory(Protocol):
    """Registration-owned factory for one live control-part evidence provider."""

    provider_id: ClassVar[str]
    revision: ClassVar[str]

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        scene_provider: SceneProvider,
    ) -> EffectEvidenceProvider:
        """Create one live provider bound to the exact assembled runtime.

        Args:
            simulation: Simulation that owns the selected robot and sensors.
            robot: Exact robot selected by the environment factory.
            scene_registry: Exact live semantic scene registry.
            engine: Atomic-action engine assembled for the robot.
            scene_provider: Shared synchronized live scene provider.

        Returns:
            Fresh effect-evidence provider for the assembled runtime.
        """


@runtime_checkable
class RegisteredSemanticLowererFactory(Protocol):
    """Registration-owned factory for one fresh live semantic lowerer."""

    call_id: ClassVar[str]
    revision: ClassVar[str]

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Create a lowerer bound to the exact assembled simulation runtime.

        Args:
            simulation: Simulation that owns the selected robot and entities.
            robot: Exact robot selected by the environment factory.
            scene_registry: Exact live semantic scene registry.
            engine: Atomic-action engine assembled for the robot.

        Returns:
            A fresh registered semantic lowerer matching the declared call.
        """


@dataclass(frozen=True, slots=True)
class StandardExtensionDeclarations:
    """Cross-checked provider-free declarations for the standard factory."""

    endpoint_adapters: Mapping[type[ResourceEndpoint], EndpointAdapterDeclaration]
    runtime_transports: tuple[RuntimeTransportDeclaration, ...]
    parallel_safety: ParallelSafetyDeclaration | None
    control_part_evidence: ControlPartEvidenceProviderDeclaration | None
    registered_semantic_lowerers: Mapping[str, RegisteredSemanticLowererDeclaration] = (
        field(default_factory=dict)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_adapters, Mapping):
            raise TypeError("endpoint_adapters must be a mapping.")
        normalized: dict[type[ResourceEndpoint], EndpointAdapterDeclaration] = {}
        for endpoint_type, declaration in self.endpoint_adapters.items():
            if type(declaration) is not EndpointAdapterDeclaration:
                raise TypeError(
                    "endpoint_adapters values must be EndpointAdapterDeclaration "
                    "values."
                )
            if endpoint_type is not declaration.endpoint_type:
                raise ValueError(
                    "endpoint_adapters keys must exactly match declaration "
                    "endpoint_type values."
                )
            normalized[endpoint_type] = declaration
        object.__setattr__(self, "endpoint_adapters", MappingProxyType(normalized))
        transports = tuple(self.runtime_transports)
        if not transports or not all(
            type(value) is RuntimeTransportDeclaration for value in transports
        ):
            raise TypeError(
                "runtime_transports must contain RuntimeTransportDeclaration values."
            )
        object.__setattr__(self, "runtime_transports", transports)
        if (
            self.parallel_safety is not None
            and type(self.parallel_safety) is not ParallelSafetyDeclaration
        ):
            raise TypeError(
                "parallel_safety must be ParallelSafetyDeclaration or None."
            )
        if (
            self.control_part_evidence is not None
            and type(self.control_part_evidence)
            is not ControlPartEvidenceProviderDeclaration
        ):
            raise TypeError(
                "control_part_evidence must be "
                "ControlPartEvidenceProviderDeclaration or None."
            )
        if not isinstance(self.registered_semantic_lowerers, Mapping):
            raise TypeError("registered_semantic_lowerers must be a mapping.")
        normalized_lowerers: dict[str, RegisteredSemanticLowererDeclaration] = {}
        for call_id, declaration in self.registered_semantic_lowerers.items():
            if type(declaration) is not RegisteredSemanticLowererDeclaration:
                raise TypeError(
                    "registered_semantic_lowerers values must be "
                    "RegisteredSemanticLowererDeclaration values."
                )
            if call_id != declaration.call_id:
                raise ValueError(
                    "registered_semantic_lowerers keys must exactly match "
                    "declaration call_id values."
                )
            normalized_lowerers[call_id] = declaration
        object.__setattr__(
            self,
            "registered_semantic_lowerers",
            MappingProxyType(normalized_lowerers),
        )
        adapter_ids = [value.adapter_id for value in normalized.values()]
        if len(set(adapter_ids)) != len(adapter_ids):
            raise ValueError("Endpoint adapter IDs must be unique.")
        transport_ids = [value.transport_id for value in transports]
        if len(set(transport_ids)) != len(transport_ids):
            raise ValueError("Runtime transport IDs must be unique.")
        transport_by_id = {value.transport_id: value for value in transports}
        required_transport_ids = frozenset(
            transport_id
            for declaration in normalized.values()
            for transport_id in declaration.runtime_transport_ids
        )
        if required_transport_ids != frozenset(transport_by_id):
            raise ValueError(
                "Provider-free runtime transports must exactly cover endpoint "
                f"adapter transport IDs; expected {sorted(required_transport_ids)}, "
                f"got {sorted(transport_by_id)}."
            )
        target_owners: dict[type[RuntimeEndpointTarget], str] = {}
        for transport in transports:
            for target_type in transport.target_types:
                if target_type in target_owners:
                    raise ValueError(
                        f"Runtime target type {_qualified_name(target_type)!r} has "
                        "multiple transport owners."
                    )
                target_owners[target_type] = transport.transport_id
        declared_target_types: set[type[RuntimeEndpointTarget]] = set()
        for adapter in normalized.values():
            counts = {transport_id: 0 for transport_id in adapter.runtime_transport_ids}
            for target_type in adapter.runtime_target_types:
                owner = target_owners.get(target_type)
                if owner is None or owner not in counts:
                    raise ValueError(
                        f"Endpoint adapter {adapter.adapter_id!r} target type "
                        f"{_qualified_name(target_type)!r} has no matching "
                        "declared transport."
                    )
                counts[owner] += 1
                declared_target_types.add(target_type)
            unused = sorted(key for key, count in counts.items() if count == 0)
            if unused:
                raise ValueError(
                    f"Endpoint adapter {adapter.adapter_id!r} declares unused "
                    f"transport IDs {unused}."
                )
        if declared_target_types != set(target_owners):
            raise ValueError(
                "Provider-free runtime target types must be covered exactly by "
                "endpoint adapter declarations."
            )
        _validate_builtin_routes(normalized)
        if self.parallel_safety is not None and (
            self.parallel_safety.supported_transport_ids != frozenset(transport_by_id)
        ):
            raise ValueError(
                "Parallel safety transport coverage must exactly match the "
                "provider-free runtime transports."
            )


def declare_endpoint_adapter(
    adapter: ResourceEndpointAdapter,
) -> EndpointAdapterDeclaration:
    """Read one endpoint adapter's exact static extension contract."""
    if not isinstance(adapter, ResourceEndpointAdapter):
        raise TypeError("endpoint adapters must be ResourceEndpointAdapter instances.")
    validate_immutable_extension_declaration(
        adapter,
        field_name="endpoint_adapters",
    )
    endpoint_type = _class_attribute(
        adapter,
        "endpoint_type",
        field_name="ResourceEndpointAdapter.endpoint_type",
    )
    if not isinstance(endpoint_type, type) or not issubclass(
        endpoint_type, ResourceEndpoint
    ):
        raise TypeError(
            "ResourceEndpointAdapter.endpoint_type must be a ResourceEndpoint "
            "subclass."
        )
    return EndpointAdapterDeclaration(
        endpoint_type=endpoint_type,
        adapter_type=type(adapter),
        adapter_id=_identifier(
            _class_attribute(
                adapter,
                "adapter_id",
                field_name="ResourceEndpointAdapter.adapter_id",
            ),
            field_name="ResourceEndpointAdapter.adapter_id",
        ),
        runtime_transport_ids=_class_attribute(
            adapter,
            "runtime_transport_ids",
            field_name="ResourceEndpointAdapter.runtime_transport_ids",
        ),
        runtime_target_types=_class_attribute(
            adapter,
            "runtime_target_types",
            field_name="ResourceEndpointAdapter.runtime_target_types",
        ),
        tracking_feedback_source_keys=_class_attribute(
            adapter,
            "tracking_feedback_source_keys",
            field_name="ResourceEndpointAdapter.tracking_feedback_source_keys",
        ),
        tracking_projector_keys=_class_attribute(
            adapter,
            "tracking_projector_keys",
            field_name="ResourceEndpointAdapter.tracking_projector_keys",
        ),
        effect_evidence_source_keys=_class_attribute(
            adapter,
            "effect_evidence_source_keys",
            field_name="ResourceEndpointAdapter.effect_evidence_source_keys",
        ),
    )


def declare_runtime_transport(
    transport: RuntimeTransportActionEncoder,
) -> RuntimeTransportDeclaration:
    """Read one runtime encoder's exact static target/payload contract."""
    if not isinstance(transport, RuntimeTransportActionEncoder):
        raise TypeError(
            "runtime_transports must implement RuntimeTransportActionEncoder."
        )
    validate_immutable_extension_declaration(
        transport,
        field_name="runtime_transports",
    )
    return RuntimeTransportDeclaration(
        transport_type=type(transport),
        transport_id=_identifier(
            _class_attribute(
                transport,
                "transport_id",
                field_name="RuntimeTransportActionEncoder.transport_id",
            ),
            field_name="RuntimeTransportActionEncoder.transport_id",
        ),
        target_types=_class_attribute(
            transport,
            "target_types",
            field_name="RuntimeTransportActionEncoder.target_types",
        ),
        payload_types=_class_attribute(
            transport,
            "payload_types",
            field_name="RuntimeTransportActionEncoder.payload_types",
        ),
    )


def declare_parallel_safety_factory(
    factory: ParallelCommandSafetyValidatorFactory,
) -> ParallelSafetyDeclaration:
    """Read one safety factory's exact static identity and coverage."""
    create = getattr(factory, "create", None)
    if not callable(create):
        raise TypeError("parallel_safety_factory must define create().")
    validate_immutable_extension_declaration(
        factory,
        field_name="parallel_safety_factory",
    )
    return ParallelSafetyDeclaration(
        factory_type=type(factory),
        validator_id=_identifier(
            _class_attribute(
                factory,
                "validator_id",
                field_name="ParallelCommandSafetyValidatorFactory.validator_id",
            ),
            field_name="ParallelCommandSafetyValidatorFactory.validator_id",
        ),
        revision=_identifier(
            _class_attribute(
                factory,
                "revision",
                field_name="ParallelCommandSafetyValidatorFactory.revision",
            ),
            field_name="ParallelCommandSafetyValidatorFactory.revision",
        ),
        supported_transport_ids=_class_attribute(
            factory,
            "supported_transport_ids",
            field_name=(
                "ParallelCommandSafetyValidatorFactory.supported_transport_ids"
            ),
        ),
    )


def declare_control_part_evidence_factory(
    factory: ControlPartEvidenceProviderFactory,
) -> ControlPartEvidenceProviderDeclaration:
    """Read one control-part evidence factory's exact static identity.

    Args:
        factory: Immutable registration-owned factory declaration.

    Returns:
        Provider-free exact factory identity.
    """
    create = getattr(factory, "create", None)
    if not callable(create):
        raise TypeError("control_part_evidence_factory must define create().")
    validate_immutable_extension_declaration(
        factory,
        field_name="control_part_evidence_factory",
    )
    return ControlPartEvidenceProviderDeclaration(
        factory_type=type(factory),
        provider_id=_identifier(
            _class_attribute(
                factory,
                "provider_id",
                field_name="ControlPartEvidenceProviderFactory.provider_id",
            ),
            field_name="ControlPartEvidenceProviderFactory.provider_id",
        ),
        revision=_identifier(
            _class_attribute(
                factory,
                "revision",
                field_name="ControlPartEvidenceProviderFactory.revision",
            ),
            field_name="ControlPartEvidenceProviderFactory.revision",
        ),
    )


def declare_registered_semantic_lowerer_factory(
    factory: RegisteredSemanticLowererFactory,
) -> RegisteredSemanticLowererDeclaration:
    """Read one semantic lowerer factory's exact static identity.

    Args:
        factory: Immutable registration-owned lowerer factory.

    Returns:
        Provider-free call, revision, and factory-type declaration.

    Raises:
        TypeError: If the factory has no callable creator or is mutable.
        ValueError: If its call ID or revision is not an exact identifier.
    """
    create = getattr(factory, "create", None)
    if not callable(create):
        raise TypeError("registered_semantic_lowerer_factories must define create().")
    validate_immutable_extension_declaration(
        factory,
        field_name="registered_semantic_lowerer_factories",
    )
    return RegisteredSemanticLowererDeclaration(
        factory_type=type(factory),
        call_id=_identifier(
            _class_attribute(
                factory,
                "call_id",
                field_name="RegisteredSemanticLowererFactory.call_id",
            ),
            field_name="RegisteredSemanticLowererFactory.call_id",
        ),
        revision=_identifier(
            _class_attribute(
                factory,
                "revision",
                field_name="RegisteredSemanticLowererFactory.revision",
            ),
            field_name="RegisteredSemanticLowererFactory.revision",
        ),
    )


def _profile_endpoint_types(
    profile: RobotSkillProfile,
) -> frozenset[type[ResourceEndpoint]]:
    """Return every exact endpoint declaration type used by one profile."""
    if type(profile) is not RobotSkillProfile:
        raise TypeError("profile must be exactly RobotSkillProfile.")
    return frozenset(
        type(endpoint)
        for resource in profile.resources.values()
        for endpoint in resource.endpoints.values()
    )


def _validate_builtin_routes(
    declarations: Mapping[type[ResourceEndpoint], EndpointAdapterDeclaration],
) -> None:
    """Keep C1 custom endpoints open-loop and preserve exact built-in routes."""
    for endpoint_type, declaration in declarations.items():
        if endpoint_type is ControlPartEndpoint:
            if (
                declaration.tracking_feedback_source_keys
                != _BUILTIN_TRACKING_FEEDBACK_SOURCE_KEYS
                or declaration.tracking_projector_keys
                != _BUILTIN_TRACKING_PROJECTOR_KEYS
                or declaration.effect_evidence_source_keys
                != _BUILTIN_EFFECT_EVIDENCE_SOURCE_KEYS
            ):
                raise ValueError(
                    "The built-in ControlPartEndpoint adapter must retain its "
                    "exact tracking and effect-evidence routes."
                )
            continue
        if (
            declaration.tracking_feedback_source_keys
            or declaration.tracking_projector_keys
            or declaration.effect_evidence_source_keys
        ):
            raise ValueError(
                f"Custom endpoint adapter {declaration.adapter_id!r} must declare "
                "empty tracking and effect-evidence routes; the C1 standard "
                "simulation factory does not install custom closed-loop providers."
            )


def build_standard_extension_declarations(
    *,
    profile: RobotSkillProfile,
    endpoint_adapters: tuple[ResourceEndpointAdapter, ...],
    runtime_transports: tuple[RuntimeTransportActionEncoder, ...],
    parallel_safety_factory: ParallelCommandSafetyValidatorFactory | None,
    control_part_evidence_factory: ControlPartEvidenceProviderFactory | None = None,
    registered_semantic_lowerer_factories: tuple[
        RegisteredSemanticLowererFactory, ...
    ] = (),
) -> StandardExtensionDeclarations:
    """Cross-check standard-runtime extensions against one exact profile.

    The built-in control-part adapter and joint-position transport cannot be
    overridden. They are installed first only when the profile uses a
    :class:`ControlPartEndpoint`; a pure-custom profile contains only its custom
    declarations. Custom adapters and transports must cover exactly the endpoint
    types and transport IDs used by the registered profile; unused declarations
    fail closed.
    """
    if type(endpoint_adapters) is not tuple:
        raise TypeError("endpoint_adapters must be an exact tuple.")
    if type(runtime_transports) is not tuple:
        raise TypeError("runtime_transports must be an exact tuple.")
    if type(registered_semantic_lowerer_factories) is not tuple:
        raise TypeError("registered_semantic_lowerer_factories must be an exact tuple.")

    builtin_adapter = declare_endpoint_adapter(ControlPartEndpointAdapter())
    custom_adapters = tuple(
        declare_endpoint_adapter(adapter) for adapter in endpoint_adapters
    )
    adapter_declarations = (builtin_adapter, *custom_adapters)
    endpoint_types = [value.endpoint_type for value in adapter_declarations]
    adapter_ids = [value.adapter_id for value in adapter_declarations]
    if len(set(endpoint_types)) != len(endpoint_types):
        raise ValueError(
            "Endpoint adapter declarations contain a duplicate exact endpoint "
            "type or attempt to override the built-in ControlPartEndpoint."
        )
    if len(set(adapter_ids)) != len(adapter_ids):
        raise ValueError(
            "Endpoint adapter declarations contain a duplicate adapter ID or "
            "attempt to override a built-in adapter."
        )
    installed_by_type = {
        declaration.endpoint_type: declaration for declaration in adapter_declarations
    }
    used_endpoint_types = _profile_endpoint_types(profile)
    missing_adapters = used_endpoint_types - set(installed_by_type)
    unused_adapters = set(installed_by_type) - used_endpoint_types
    unused_adapters.discard(ControlPartEndpoint)
    if missing_adapters or unused_adapters:
        raise ValueError(
            "Endpoint adapter coverage must exactly match profile endpoint types; "
            f"missing={sorted(_qualified_name(value) for value in missing_adapters)}, "
            f"unused={sorted(_qualified_name(value) for value in unused_adapters)}."
        )
    if ControlPartEndpoint not in used_endpoint_types:
        installed_by_type.pop(ControlPartEndpoint)

    _validate_builtin_routes(installed_by_type)

    builtin_transport = declare_runtime_transport(JointPositionGymTransportEncoder())
    custom_transports = tuple(
        declare_runtime_transport(transport) for transport in runtime_transports
    )
    transport_declarations = (builtin_transport, *custom_transports)
    transport_ids = [value.transport_id for value in transport_declarations]
    if len(set(transport_ids)) != len(transport_ids):
        raise ValueError(
            "Runtime transport declarations contain a duplicate transport ID or "
            "attempt to override the built-in joint-position transport."
        )
    transport_by_id = {
        declaration.transport_id: declaration for declaration in transport_declarations
    }
    required_transport_ids = frozenset(
        transport_id
        for declaration in installed_by_type.values()
        for transport_id in declaration.runtime_transport_ids
    )
    missing_transports = required_transport_ids - set(transport_by_id)
    unused_transports = set(transport_by_id) - required_transport_ids
    unused_transports.discard(JointPositionTarget.TRANSPORT_ID)
    if missing_transports or unused_transports:
        raise ValueError(
            "Runtime transport coverage must exactly match endpoint adapters; "
            f"missing={sorted(missing_transports)}, "
            f"unused={sorted(unused_transports)}."
        )
    if JointPositionTarget.TRANSPORT_ID not in required_transport_ids:
        transport_declarations = custom_transports
        transport_by_id.pop(JointPositionTarget.TRANSPORT_ID)

    target_owners: dict[type[RuntimeEndpointTarget], str] = {}
    for transport in transport_declarations:
        for target_type in transport.target_types:
            previous = target_owners.get(target_type)
            if previous is not None:
                raise ValueError(
                    f"Runtime target type {_qualified_name(target_type)!r} is "
                    f"declared by both transports {previous!r} and "
                    f"{transport.transport_id!r}."
                )
            target_owners[target_type] = transport.transport_id

    adapter_target_types: set[type[RuntimeEndpointTarget]] = set()
    for adapter in installed_by_type.values():
        for transport_id in adapter.runtime_transport_ids:
            if transport_id not in transport_by_id:
                raise ValueError(
                    f"Endpoint adapter {adapter.adapter_id!r} requires missing "
                    f"transport {transport_id!r}."
                )
        per_transport_counts = {
            transport_id: 0 for transport_id in adapter.runtime_transport_ids
        }
        for target_type in adapter.runtime_target_types:
            owner = target_owners.get(target_type)
            if owner is None or owner not in adapter.runtime_transport_ids:
                raise ValueError(
                    f"Endpoint adapter {adapter.adapter_id!r} target type "
                    f"{_qualified_name(target_type)!r} is not covered by one of "
                    f"its transports {sorted(adapter.runtime_transport_ids)}."
                )
            per_transport_counts[owner] += 1
            adapter_target_types.add(target_type)
        unused_adapter_transport_ids = sorted(
            transport_id
            for transport_id, count in per_transport_counts.items()
            if count == 0
        )
        if unused_adapter_transport_ids:
            raise ValueError(
                f"Endpoint adapter {adapter.adapter_id!r} declares unused "
                f"transport IDs {unused_adapter_transport_ids}."
            )
    extra_transport_target_types = set(target_owners) - adapter_target_types
    if extra_transport_target_types:
        raise ValueError(
            "Runtime transports declare target types unused by endpoint adapters: "
            f"{sorted(_qualified_name(value) for value in extra_transport_target_types)}."
        )

    parallel_safety = (
        None
        if parallel_safety_factory is None
        else declare_parallel_safety_factory(parallel_safety_factory)
    )
    if parallel_safety is not None:
        installed_transport_ids = frozenset(transport_by_id)
        if parallel_safety.supported_transport_ids != installed_transport_ids:
            raise ValueError(
                "parallel_safety_factory must support exactly the registered "
                f"runtime transports; expected {sorted(installed_transport_ids)}, "
                f"got {sorted(parallel_safety.supported_transport_ids)}."
            )

    control_part_evidence = (
        None
        if control_part_evidence_factory is None
        else declare_control_part_evidence_factory(control_part_evidence_factory)
    )
    if (
        control_part_evidence is not None
        and ControlPartEndpoint not in installed_by_type
    ):
        raise ValueError(
            "control_part_evidence_factory requires a registered "
            "ControlPartEndpoint."
        )
    lowerer_declarations = tuple(
        declare_registered_semantic_lowerer_factory(factory)
        for factory in registered_semantic_lowerer_factories
    )
    lowerer_ids = [declaration.call_id for declaration in lowerer_declarations]
    if len(set(lowerer_ids)) != len(lowerer_ids):
        raise ValueError("Registered semantic lowerer call IDs must be unique.")

    return StandardExtensionDeclarations(
        endpoint_adapters=installed_by_type,
        runtime_transports=transport_declarations,
        parallel_safety=parallel_safety,
        control_part_evidence=control_part_evidence,
        registered_semantic_lowerers={
            declaration.call_id: declaration for declaration in lowerer_declarations
        },
    )


__all__ = [
    "ControlPartEvidenceProviderDeclaration",
    "ControlPartEvidenceProviderFactory",
    "EndpointAdapterDeclaration",
    "ParallelCommandSafetyValidatorFactory",
    "ParallelSafetyDeclaration",
    "RegisteredSemanticLowererDeclaration",
    "RegisteredSemanticLowererFactory",
    "RuntimeTransportDeclaration",
    "StandardExtensionDeclarations",
    "VersionedKey",
    "build_standard_extension_declarations",
    "declare_control_part_evidence_factory",
    "declare_endpoint_adapter",
    "declare_parallel_safety_factory",
    "declare_registered_semantic_lowerer_factory",
    "declare_runtime_transport",
    "validate_immutable_extension_declaration",
]
