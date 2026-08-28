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

"""Tests for exact standard-runtime Expert Program extension declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch

from embodichain.lab.gym.envs.expert_program import (
    SimulationExpertProgramRegistration,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
)
from embodichain.lab.gym.envs.expert_program.bridge import (
    JointPositionGymTransportEncoder,
)
from embodichain.lab.gym.envs.expert_program.extensions import (
    RuntimeTransportDeclaration,
    build_standard_extension_declarations,
    validate_immutable_extension_declaration,
)
from embodichain.lab.sim.atomic_actions import PlanningContext
from embodichain.lab.sim.atomic_actions.bindings import RuntimeEndpointTarget
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    RuntimeCommandPayload,
)
from embodichain.lab.semantic_skills import (
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    ControlPartEndpoint,
    ControlPartEndpointAdapter,
    EndpointResolution,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotResource,
    RobotSkillProfile,
)
from embodichain.lab.sim.types import EnvAction


@dataclass(frozen=True, slots=True, kw_only=True)
class _MobileEndpoint(ResourceEndpoint):
    """Custom endpoint declaration used by the catalog-only tests."""

    controller: str = "base"


@dataclass(frozen=True, slots=True, kw_only=True)
class _ToolEndpoint(ResourceEndpoint):
    """Second exact endpoint type used to prove transport ordering."""

    controller: str = "tool"


@dataclass(frozen=True, slots=True)
class _MobileTarget(RuntimeEndpointTarget):
    """Immutable custom runtime destination."""

    TRANSPORT_ID: ClassVar[str] = "test.mobile"
    controller: str

    @property
    def transport_id(self) -> str:
        return self.TRANSPORT_ID

    @property
    def target_id(self) -> str:
        return self.controller


@dataclass(frozen=True, slots=True)
class _ToolTarget(RuntimeEndpointTarget):
    """Immutable destination owned by the second transport."""

    TRANSPORT_ID: ClassVar[str] = "test.tool"
    controller: str

    @property
    def transport_id(self) -> str:
        return self.TRANSPORT_ID

    @property
    def target_id(self) -> str:
        return self.controller


@dataclass(frozen=True, slots=True, eq=False)
class _MobilePayload(RuntimeCommandPayload):
    """Minimal typed payload declaration for the mobile transport."""

    TRANSPORT_ID: ClassVar[str] = _MobileTarget.TRANSPORT_ID
    values: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def transport_id(self) -> str:
        return self.TRANSPORT_ID

    def snapshot(self) -> _MobilePayload:
        return _MobilePayload(self.values.clone())


@dataclass(frozen=True, slots=True, eq=False)
class _ToolPayload(RuntimeCommandPayload):
    """Minimal typed payload declaration for the tool transport."""

    TRANSPORT_ID: ClassVar[str] = _ToolTarget.TRANSPORT_ID
    values: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def transport_id(self) -> str:
        return self.TRANSPORT_ID

    def snapshot(self) -> _ToolPayload:
        return _ToolPayload(self.values.clone())


class _MobileAdapter(ResourceEndpointAdapter):
    """Stateless custom adapter with only standard-factory provider routes."""

    adapter_id: ClassVar[str] = "test.mobile"
    endpoint_type: ClassVar[type[ResourceEndpoint]] = _MobileEndpoint
    runtime_transport_ids: ClassVar[frozenset[str]] = frozenset(
        {_MobileTarget.TRANSPORT_ID}
    )
    runtime_target_types: ClassVar[tuple[type[RuntimeEndpointTarget], ...]] = (
        _MobileTarget,
    )
    tracking_feedback_source_keys: ClassVar[frozenset[tuple[str, str]]] = frozenset()
    tracking_projector_keys: ClassVar[frozenset[tuple[str, str]]] = frozenset()
    effect_evidence_source_keys: ClassVar[frozenset[tuple[str, str]]] = frozenset()

    def resolve(
        self, endpoint: ResourceEndpoint, *, engine: object
    ) -> EndpointResolution:
        del endpoint, engine
        return EndpointResolution(
            runtime_target=_MobileTarget("base"),
            claim_tokens=frozenset({"test.mobile:base"}),
        )


class _ToolAdapter(ResourceEndpointAdapter):
    """Second stateless adapter used by ordering tests."""

    adapter_id: ClassVar[str] = "test.tool"
    endpoint_type: ClassVar[type[ResourceEndpoint]] = _ToolEndpoint
    runtime_transport_ids: ClassVar[frozenset[str]] = frozenset(
        {_ToolTarget.TRANSPORT_ID}
    )
    runtime_target_types: ClassVar[tuple[type[RuntimeEndpointTarget], ...]] = (
        _ToolTarget,
    )
    tracking_feedback_source_keys: ClassVar[frozenset[tuple[str, str]]] = frozenset()
    tracking_projector_keys: ClassVar[frozenset[tuple[str, str]]] = frozenset()
    effect_evidence_source_keys: ClassVar[frozenset[tuple[str, str]]] = frozenset()

    def resolve(
        self, endpoint: ResourceEndpoint, *, engine: object
    ) -> EndpointResolution:
        del endpoint, engine
        return EndpointResolution(
            runtime_target=_ToolTarget("tool"),
            claim_tokens=frozenset({"test.tool:tool"}),
        )


class _MobileTransport:
    """Stateless action composition transport for the mobile target."""

    transport_id: ClassVar[str] = _MobileTarget.TRANSPORT_ID
    target_types: ClassVar[tuple[type[RuntimeEndpointTarget], ...]] = (_MobileTarget,)
    payload_types: ClassVar[tuple[type[RuntimeCommandPayload], ...]] = (_MobilePayload,)

    def encode(
        self,
        command: EndpointCommand,
        *,
        base_action: EnvAction,
        active_mask: torch.Tensor,
    ) -> EnvAction:
        del command, active_mask
        return base_action

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        base_action: EnvAction,
        context: PlanningContext,
    ) -> EnvAction:
        del targets, context
        return base_action


class _ToolTransport(_MobileTransport):
    """Second stateless action composition transport."""

    transport_id: ClassVar[str] = _ToolTarget.TRANSPORT_ID
    target_types: ClassVar[tuple[type[RuntimeEndpointTarget], ...]] = (_ToolTarget,)
    payload_types: ClassVar[tuple[type[RuntimeCommandPayload], ...]] = (_ToolPayload,)


class _SafetyValidator:
    """Protocol-compatible no-op validator used only for factory typing."""

    def validate(self, *, branch_frames: object, merged_frame: object) -> None:
        del branch_frames, merged_frame


class _MobileSafetyFactory:
    """Stateless exact safety-factory declaration."""

    validator_id: ClassVar[str] = "test.mobile_safety"
    revision: ClassVar[str] = "1"
    supported_transport_ids: ClassVar[frozenset[str]] = frozenset(
        {_MobileTarget.TRANSPORT_ID}
    )

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: object,
        engine: object,
    ) -> _SafetyValidator:
        del simulation, robot, scene_registry, engine
        return _SafetyValidator()


class _ControlPartEvidenceFactory:
    """Stateless declaration for the exact built-in evidence route."""

    provider_id: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_REVISION

    def create(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("Declaration tests must not create live providers.")


def _custom_profile(*, include_tool: bool = False) -> RobotSkillProfile:
    """Return a pure provider-free profile with exact custom endpoint types."""
    endpoints: dict[str, ResourceEndpoint] = {
        "motion": _MobileEndpoint(capabilities=frozenset())
    }
    if include_tool:
        endpoints["tool"] = _ToolEndpoint(capabilities=frozenset())
    resource = RobotResource(resource_id="custom", endpoints=endpoints)
    return RobotSkillProfile(profile_id="custom", resources={"custom": resource})


def test_custom_endpoint_transport_and_safety_declarations_are_exact() -> None:
    """A complete custom extension set produces an immutable provider-free catalog."""
    declarations = build_standard_extension_declarations(
        profile=_custom_profile(),
        endpoint_adapters=(_MobileAdapter(),),
        runtime_transports=(_MobileTransport(),),
        parallel_safety_factory=_MobileSafetyFactory(),
    )

    assert declarations.endpoint_adapters[_MobileEndpoint].adapter_id == "test.mobile"
    assert tuple(value.transport_id for value in declarations.runtime_transports) == (
        "test.mobile",
    )
    assert declarations.parallel_safety is not None
    assert declarations.parallel_safety.supported_transport_ids == frozenset(
        {"test.mobile"}
    )


def test_parallel_safety_transport_coverage_must_match_registration() -> None:
    """A safety factory must cover the exact installed transport set."""

    class MismatchedSafetyFactory(_MobileSafetyFactory):
        supported_transport_ids: ClassVar[frozenset[str]] = frozenset({"test.other"})

    with pytest.raises(ValueError, match="must support exactly"):
        build_standard_extension_declarations(
            profile=_custom_profile(),
            endpoint_adapters=(_MobileAdapter(),),
            runtime_transports=(_MobileTransport(),),
            parallel_safety_factory=MismatchedSafetyFactory(),
        )


def test_control_part_evidence_factory_requires_exact_builtin_route() -> None:
    """The standard factory cannot attach control-part evidence to custom routes."""
    with pytest.raises(ValueError, match="requires a registered ControlPartEndpoint"):
        build_standard_extension_declarations(
            profile=_custom_profile(),
            endpoint_adapters=(_MobileAdapter(),),
            runtime_transports=(_MobileTransport(),),
            parallel_safety_factory=None,
            control_part_evidence_factory=_ControlPartEvidenceFactory(),
        )

    class WrongRouteFactory(_ControlPartEvidenceFactory):
        provider_id: ClassVar[str] = "test.wrong_evidence"

    control_part_profile = RobotSkillProfile(
        profile_id="joint",
        resources={
            "hand": RobotResource(
                resource_id="hand",
                endpoints={
                    "grasp": ControlPartEndpoint(
                        control_part="hand",
                        capabilities=frozenset(),
                    )
                },
            )
        },
    )
    with pytest.raises(ValueError, match="exact built-in route"):
        build_standard_extension_declarations(
            profile=control_part_profile,
            endpoint_adapters=(),
            runtime_transports=(),
            parallel_safety_factory=None,
            control_part_evidence_factory=WrongRouteFactory(),
        )


@pytest.mark.parametrize("declaration_kind", ("target", "payload"))
def test_runtime_transport_types_require_direct_transport_id(
    declaration_kind: str,
) -> None:
    """Every registered runtime value type owns its transport ID directly."""

    class MissingTarget(RuntimeEndpointTarget):
        @property
        def transport_id(self) -> str:
            return _MobileTarget.TRANSPORT_ID

        @property
        def target_id(self) -> str:
            return "missing"

    class MissingPayload(RuntimeCommandPayload):
        @property
        def batch_size(self) -> int:
            return 1

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        @property
        def transport_id(self) -> str:
            return _MobileTarget.TRANSPORT_ID

        def snapshot(self) -> MissingPayload:
            return MissingPayload()

    target_types = (
        (MissingTarget,) if declaration_kind == "target" else (_MobileTarget,)
    )
    payload_types = (
        (MissingPayload,) if declaration_kind == "payload" else (_MobilePayload,)
    )

    with pytest.raises(TypeError, match="must declare an exact ClassVar TRANSPORT_ID"):
        RuntimeTransportDeclaration(
            transport_type=_MobileTransport,
            transport_id=_MobileTarget.TRANSPORT_ID,
            target_types=target_types,
            payload_types=payload_types,
        )


@pytest.mark.parametrize("declaration_kind", ("target", "payload"))
def test_runtime_transport_types_cannot_inherit_transport_id(
    declaration_kind: str,
) -> None:
    """A subtype cannot silently inherit another runtime type's transport owner."""

    class InheritedTarget(_MobileTarget):
        pass

    class InheritedPayload(_MobilePayload):
        pass

    target_types = (
        (InheritedTarget,) if declaration_kind == "target" else (_MobileTarget,)
    )
    payload_types = (
        (InheritedPayload,) if declaration_kind == "payload" else (_MobilePayload,)
    )

    with pytest.raises(TypeError, match="inherited or instance-only"):
        RuntimeTransportDeclaration(
            transport_type=_MobileTransport,
            transport_id=_MobileTarget.TRANSPORT_ID,
            target_types=target_types,
            payload_types=payload_types,
        )


@pytest.mark.parametrize("declaration_kind", ("target", "payload"))
def test_runtime_transport_type_transport_id_must_match_encoder(
    declaration_kind: str,
) -> None:
    """Static runtime value ownership must match the encoder transport exactly."""

    class MismatchedTarget(_MobileTarget):
        TRANSPORT_ID: ClassVar[str] = "test.mismatched"

    class MismatchedPayload(_MobilePayload):
        TRANSPORT_ID: ClassVar[str] = "test.mismatched"

    target_types = (
        (MismatchedTarget,) if declaration_kind == "target" else (_MobileTarget,)
    )
    payload_types = (
        (MismatchedPayload,) if declaration_kind == "payload" else (_MobilePayload,)
    )

    with pytest.raises(ValueError, match="not 'test.mobile'"):
        RuntimeTransportDeclaration(
            transport_type=_MobileTransport,
            transport_id=_MobileTarget.TRANSPORT_ID,
            target_types=target_types,
            payload_types=payload_types,
        )


@pytest.mark.parametrize(
    ("adapters", "transports", "message"),
    (
        ((), (_MobileTransport(),), "missing"),
        ((_MobileAdapter(),), (), "missing"),
        (
            (_MobileAdapter(), _ToolAdapter()),
            (_MobileTransport(), _ToolTransport()),
            "unused",
        ),
    ),
)
def test_extension_coverage_rejects_missing_and_unused_declarations(
    adapters: tuple[ResourceEndpointAdapter, ...],
    transports: tuple[object, ...],
    message: str,
) -> None:
    """Every custom adapter and transport must be necessary and complete."""
    with pytest.raises(ValueError, match=message):
        build_standard_extension_declarations(
            profile=_custom_profile(),
            endpoint_adapters=adapters,
            runtime_transports=transports,  # type: ignore[arg-type]
            parallel_safety_factory=None,
        )


def test_builtin_adapter_and_transport_cannot_be_overridden() -> None:
    """Standard built-ins retain exact ownership of their endpoint and transport."""
    profile = RobotSkillProfile(
        profile_id="joint",
        resources={
            "arm": RobotResource(
                resource_id="arm",
                endpoints={
                    "motion": ControlPartEndpoint(
                        control_part="arm",
                        capabilities=frozenset(),
                    )
                },
            )
        },
    )

    with pytest.raises(ValueError, match="override the built-in ControlPartEndpoint"):
        build_standard_extension_declarations(
            profile=profile,
            endpoint_adapters=(ControlPartEndpointAdapter(),),
            runtime_transports=(),
            parallel_safety_factory=None,
        )
    with pytest.raises(ValueError, match="override the built-in joint-position"):
        build_standard_extension_declarations(
            profile=profile,
            endpoint_adapters=(),
            runtime_transports=(JointPositionGymTransportEncoder(),),
            parallel_safety_factory=None,
        )


def test_nonbuiltin_provider_route_is_rejected_by_standard_registration() -> None:
    """Provider declarations cannot name a live registry absent from the factory."""

    class UnsupportedProviderAdapter(_MobileAdapter):
        adapter_id: ClassVar[str] = "test.unsupported_provider"
        tracking_feedback_source_keys: ClassVar[frozenset[tuple[str, str]]] = frozenset(
            {("test.feedback", "1")}
        )

    with pytest.raises(ValueError, match="does not install"):
        build_standard_extension_declarations(
            profile=_custom_profile(),
            endpoint_adapters=(UnsupportedProviderAdapter(),),
            runtime_transports=(_MobileTransport(),),
            parallel_safety_factory=None,
        )


@pytest.mark.parametrize(
    "mutable_leaf",
    (
        [0.1],
        {"gain": 0.1},
        {0.1},
        bytearray(b"gain"),
        torch.tensor((0.1,)),
    ),
    ids=("list", "dict", "set", "bytearray", "tensor"),
)
def test_extension_declarations_reject_nested_mutable_state(
    mutable_leaf: object,
) -> None:
    """Frozen wrappers cannot retain mutable state used by a live extension."""

    @dataclass(frozen=True, slots=True)
    class NestedDeclaration:
        config: tuple[object, ...]

    with pytest.raises(TypeError, match="deeply immutable"):
        validate_immutable_extension_declaration(
            NestedDeclaration((mutable_leaf,)),
            field_name="runtime_transports",
        )


def test_runtime_transport_tuple_order_changes_registration_fingerprint() -> None:
    """Transport composition order is semantic registration data."""
    profile_binding = SimulationRobotSkillProfileBinding(
        profile_id="custom",
        resources=(
            RobotResource(
                resource_id="custom",
                endpoints={
                    "motion": _MobileEndpoint(capabilities=frozenset()),
                    "tool": _ToolEndpoint(capabilities=frozenset()),
                },
            ),
        ),
    )
    common = {
        "scene_binding": SimulationSceneBinding(registry_id="custom_scene"),
        "robot_profile_binding": profile_binding,
        "endpoint_adapters": (_MobileAdapter(), _ToolAdapter()),
    }
    forward = SimulationExpertProgramRegistration(
        **common,
        runtime_transports=(_MobileTransport(), _ToolTransport()),
    )
    reversed_registration = SimulationExpertProgramRegistration(
        **common,
        runtime_transports=(_ToolTransport(), _MobileTransport()),
    )

    assert forward.fingerprint != reversed_registration.fingerprint


__all__: list[str] = []
