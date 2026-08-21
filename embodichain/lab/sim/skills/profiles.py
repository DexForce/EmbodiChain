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

"""Declarative robot resources, skill binding, and policy presets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import ClassVar, Mapping, TYPE_CHECKING

from embodichain.lab.sim.atomic_actions.bindings import (
    ActionBinding,
    EndpointBinding,
    JointPositionTarget,
    RuntimeEndpointTarget,
)
from embodichain.lab.sim.atomic_actions.control import (
    ControlCommand,
    ControlPartCommandProfile,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import SkillDescriptor
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy, RecoveryPolicy
from embodichain.lab.sim.atomic_actions.requirements import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    DisjointResourceSlots,
    DisjointSlotEndpoints,
    FORWARD_KINEMATICS_CAPABILITY,
    INVERSE_KINEMATICS_CAPABILITY,
    SkillBindingContract,
    SkillResourceSlot,
)
from embodichain.lab.sim.atomic_actions.runner import ExecutionRunnerCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine


class ProfileValidationError(ValueError):
    """Raised when a robot skill profile disagrees with its engine or robot."""


class UnsupportedSkillError(ValueError):
    """Raised when no robot-resource assignment can satisfy a skill."""


class AmbiguousSkillBindingError(ValueError):
    """Raised when multiple assignments remain without a complete default."""


_SOLVER_BACKED_CAPABILITIES = frozenset(
    {
        BATCH_INVERSE_KINEMATICS_CAPABILITY,
        FORWARD_KINEMATICS_CAPABILITY,
        INVERSE_KINEMATICS_CAPABILITY,
    }
)


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Return one strict, whitespace-free identifier."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _normalize_identifier_set(
    values: frozenset[str],
    *,
    field_name: str,
) -> frozenset[str]:
    """Validate one immutable set of identifiers."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of strings, not a string.")
    try:
        normalized = frozenset(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings.") from exc
    for value in normalized:
        _validate_identifier(value, field_name=field_name)
    return normalized


def _snapshot_endpoint_commands(
    values: Mapping[str, ControlCommand],
    *,
    field_name: str,
) -> Mapping[str, ControlCommand]:
    """Validate, snapshot, and freeze commands exposed by one endpoint."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    snapshots: dict[str, ControlCommand] = {}
    for command_name, command in values.items():
        _validate_identifier(command_name, field_name=f"{field_name} keys")
        if not isinstance(command, ControlCommand):
            raise TypeError(f"{field_name} values must be ControlCommand instances.")
        snapshot = command.snapshot()
        if type(snapshot) is not type(command) or snapshot is command:
            raise TypeError(
                f"{field_name}[{command_name!r}].snapshot() must return an "
                "independently owned value of the same ControlCommand type."
            )
        snapshots[command_name] = snapshot
    return MappingProxyType(snapshots)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceEndpoint(ABC):
    """Extensible execution endpoint in a robot resource graph.

    Endpoint subclasses add controller-specific addressing data. Capabilities
    stay on this common base so skill matching does not depend on any one
    controller kind.
    """

    capabilities: frozenset[str] = frozenset()
    """Open, namespaced capabilities provided by this exact endpoint."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            _normalize_identifier_set(
                self.capabilities,
                field_name="ResourceEndpoint.capabilities",
            ),
        )

    def snapshot(self) -> ResourceEndpoint:
        """Return an independently owned endpoint declaration.

        Endpoint subclasses with payloads that cannot be deep-copied must
        override this method and return a new value of their exact type.
        """
        return deepcopy(self)


@dataclass(frozen=True, slots=True)
class ControlPartEndpoint(ResourceEndpoint):
    """One named execution endpoint backed by a robot control part.

    Capabilities are explicit and never inferred from the endpoint name, joint
    count, other endpoints, or composite resource members.
    """

    control_part: str
    """Key from the bound robot's ``control_parts`` mapping."""

    command_profile: str | None = None
    """Optional generic command-profile ID; defaults to ``control_part``."""

    def __post_init__(self) -> None:
        ResourceEndpoint.__post_init__(self)
        _validate_identifier(
            self.control_part,
            field_name="ControlPartEndpoint.control_part",
        )
        if self.command_profile is not None:
            _validate_identifier(
                self.command_profile,
                field_name="ControlPartEndpoint.command_profile",
            )


@dataclass(frozen=True, slots=True)
class EndpointResolution:
    """Adapter-produced runtime destination and claim metadata for one endpoint."""

    runtime_target: RuntimeEndpointTarget
    """Typed immutable destination consumed by an endpoint command transport."""

    command_profile_key: str | None = None
    """Profile key that owns semantic commands for this endpoint, when any."""

    requires_command_profile: bool = False
    """Whether a missing ``command_profile_key`` entry invalidates binding."""

    claim_tokens: frozenset[str] = frozenset()
    """Adapter-defined physical/controller claims beyond robot joint IDs."""

    joint_ids: tuple[int, ...] = ()
    """Ordered robot joint IDs controlled by the endpoint, when applicable."""

    exclusive: bool = True
    """Whether this execution endpoint must declare a physical claim."""

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_target, RuntimeEndpointTarget):
            raise TypeError(
                "EndpointResolution.runtime_target must be a " "RuntimeEndpointTarget."
            )
        target = self.runtime_target.snapshot()
        if (
            type(target) is not type(self.runtime_target)
            or target is self.runtime_target
        ):
            raise TypeError(
                "RuntimeEndpointTarget.snapshot() must return an independently "
                "owned value of the same target type."
            )
        _validate_identifier(
            target.transport_id,
            field_name="RuntimeEndpointTarget.transport_id",
        )
        _validate_identifier(
            target.target_id,
            field_name="RuntimeEndpointTarget.target_id",
        )
        object.__setattr__(self, "runtime_target", target)
        if self.command_profile_key is not None:
            _validate_identifier(
                self.command_profile_key,
                field_name="EndpointResolution.command_profile_key",
            )
        if not isinstance(self.requires_command_profile, bool):
            raise TypeError("requires_command_profile must be a bool.")
        if self.requires_command_profile and self.command_profile_key is None:
            raise ValueError(
                "requires_command_profile needs a non-None command_profile_key."
            )
        object.__setattr__(
            self,
            "claim_tokens",
            _normalize_identifier_set(
                self.claim_tokens,
                field_name="EndpointResolution.claim_tokens",
            ),
        )
        joint_ids = tuple(self.joint_ids)
        if not all(
            isinstance(joint_id, int)
            and not isinstance(joint_id, bool)
            and joint_id >= 0
            for joint_id in joint_ids
        ):
            raise ValueError(
                "EndpointResolution.joint_ids must be non-negative integers."
            )
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError("EndpointResolution.joint_ids must be unique.")
        if isinstance(target, JointPositionTarget) and joint_ids != target.joint_ids:
            raise ValueError(
                "EndpointResolution.joint_ids must exactly match its "
                "JointPositionTarget."
            )
        object.__setattr__(self, "joint_ids", joint_ids)
        if not isinstance(self.exclusive, bool):
            raise TypeError("EndpointResolution.exclusive must be a bool.")
        if self.exclusive and not joint_ids and not self.claim_tokens:
            raise ValueError(
                "An exclusive EndpointResolution must declare joint_ids or "
                "claim_tokens."
            )


class ResourceEndpointAdapter(ABC):
    """Resolve one endpoint kind without coupling profiles to its controller."""

    adapter_id: ClassVar[str]
    """Stable adapter identifier used in diagnostics and resolved metadata."""

    endpoint_type: ClassVar[type[ResourceEndpoint]]
    """Exact endpoint declaration type accepted by this adapter."""

    @abstractmethod
    def resolve(
        self,
        endpoint: ResourceEndpoint,
        *,
        engine: AtomicActionEngine,
    ) -> EndpointResolution:
        """Validate and resolve one endpoint against an action engine.

        Args:
            endpoint: Endpoint declaration of :attr:`endpoint_type`.
            engine: Engine whose robot, planner, and command profiles are bound.

        Returns:
            Physical claims and supported lowering metadata.
        """


class ControlPartEndpointAdapter(ResourceEndpointAdapter):
    """Resolve joint-backed :class:`ControlPartEndpoint` declarations."""

    adapter_id: ClassVar[str] = "control_part"
    endpoint_type: ClassVar[type[ResourceEndpoint]] = ControlPartEndpoint

    def resolve(
        self,
        endpoint: ResourceEndpoint,
        *,
        engine: AtomicActionEngine,
    ) -> EndpointResolution:
        """Resolve a robot control part and verify its standard capabilities."""
        if not isinstance(endpoint, ControlPartEndpoint):
            raise TypeError("ControlPartEndpointAdapter requires ControlPartEndpoint.")
        control_parts = getattr(engine.robot, "control_parts", None)
        if not isinstance(control_parts, Mapping):
            raise ProfileValidationError(
                "ControlPartEndpoint requires Robot.control_parts."
            )
        if endpoint.control_part not in control_parts:
            available = sorted(str(name) for name in control_parts)
            raise ProfileValidationError(
                f"ControlPartEndpoint references unknown control part "
                f"{endpoint.control_part!r}; Robot.control_parts contains "
                f"{available}."
            )
        joint_ids = tuple(engine.robot.get_joint_ids(name=endpoint.control_part))
        if not joint_ids:
            raise ProfileValidationError(
                f"Control part {endpoint.control_part!r} contains no joints."
            )
        if len(set(joint_ids)) != len(joint_ids):
            raise ProfileValidationError(
                f"Control part {endpoint.control_part!r} contains duplicate joint IDs."
            )
        declared = endpoint.capabilities & _SOLVER_BACKED_CAPABILITIES
        if declared:
            get_solver = getattr(engine.robot, "get_solver", None)
            if not callable(get_solver):
                raise ProfileValidationError(
                    f"Control part {endpoint.control_part!r} declares solver-backed "
                    f"capabilities {sorted(declared)}, but the robot exposes no "
                    "get_solver()."
                )
            try:
                solver = get_solver(name=endpoint.control_part)
            except Exception as exc:
                raise ProfileValidationError(
                    f"Could not validate solver-backed capabilities for control "
                    f"part {endpoint.control_part!r}: {exc}"
                ) from exc
            if solver is None:
                raise ProfileValidationError(
                    f"Control part {endpoint.control_part!r} declares solver-backed "
                    f"capabilities {sorted(declared)}, but has no configured solver."
                )
        return EndpointResolution(
            runtime_target=JointPositionTarget(
                control_part=endpoint.control_part,
                joint_ids=joint_ids,
            ),
            command_profile_key=(
                endpoint.control_part
                if endpoint.command_profile is None
                else endpoint.command_profile
            ),
            requires_command_profile=endpoint.command_profile is not None,
            claim_tokens=frozenset({f"robot.control_part:{endpoint.control_part}"}),
            joint_ids=joint_ids,
        )


@dataclass(frozen=True, slots=True)
class ResolvedResourceEndpoint:
    """Endpoint declaration resolved by one registered adapter."""

    endpoint: ResourceEndpoint
    adapter_id: str
    runtime_target: RuntimeEndpointTarget
    command_profile_key: str | None = None
    requires_command_profile: bool = False
    commands: Mapping[str, ControlCommand] = field(default_factory=dict)
    claim_tokens: frozenset[str] = frozenset()
    joint_ids: tuple[int, ...] = ()
    exclusive: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, ResourceEndpoint):
            raise TypeError("endpoint must be a ResourceEndpoint.")
        endpoint_snapshot = self.endpoint.snapshot()
        if (
            type(endpoint_snapshot) is not type(self.endpoint)
            or endpoint_snapshot is self.endpoint
        ):
            raise TypeError(
                "endpoint.snapshot() must return an independently owned value of "
                "the same endpoint type."
            )
        object.__setattr__(self, "endpoint", endpoint_snapshot)
        _validate_identifier(
            self.adapter_id,
            field_name="ResolvedResourceEndpoint.adapter_id",
        )
        resolution = EndpointResolution(
            runtime_target=self.runtime_target,
            command_profile_key=self.command_profile_key,
            requires_command_profile=self.requires_command_profile,
            claim_tokens=self.claim_tokens,
            joint_ids=self.joint_ids,
            exclusive=self.exclusive,
        )
        object.__setattr__(self, "runtime_target", resolution.runtime_target)
        object.__setattr__(
            self,
            "command_profile_key",
            resolution.command_profile_key,
        )
        object.__setattr__(
            self,
            "requires_command_profile",
            resolution.requires_command_profile,
        )
        object.__setattr__(
            self,
            "commands",
            _snapshot_endpoint_commands(
                self.commands,
                field_name="ResolvedResourceEndpoint.commands",
            ),
        )
        object.__setattr__(self, "claim_tokens", resolution.claim_tokens)
        object.__setattr__(self, "joint_ids", resolution.joint_ids)
        object.__setattr__(self, "exclusive", resolution.exclusive)

    @property
    def capabilities(self) -> frozenset[str]:
        """Return capabilities declared by the source endpoint."""
        return self.endpoint.capabilities

    def conflicts_with(self, other: ResolvedResourceEndpoint) -> bool:
        """Return whether two endpoints address overlapping physical channels."""
        if not isinstance(other, ResolvedResourceEndpoint):
            raise TypeError("other must be a ResolvedResourceEndpoint.")
        return bool(
            (
                self.runtime_target.transport_id,
                self.runtime_target.target_id,
            )
            == (
                other.runtime_target.transport_id,
                other.runtime_target.target_id,
            )
            or self.claim_tokens & other.claim_tokens
            or set(self.joint_ids) & set(other.joint_ids)
        )


def _normalize_endpoints(
    values: Mapping[str, ResourceEndpoint],
) -> Mapping[str, ResourceEndpoint]:
    """Validate and freeze resource endpoint declarations."""
    if not isinstance(values, Mapping):
        raise TypeError("RobotResource.endpoints must be a mapping.")
    normalized: dict[str, ResourceEndpoint] = {}
    for endpoint_id, endpoint in values.items():
        _validate_identifier(endpoint_id, field_name="resource endpoint identifiers")
        if not isinstance(endpoint, ResourceEndpoint):
            raise TypeError(
                "RobotResource.endpoints values must be ResourceEndpoint " "instances."
            )
        snapshot = endpoint.snapshot()
        if type(snapshot) is not type(endpoint) or snapshot is endpoint:
            raise TypeError(
                f"Endpoint {endpoint_id!r}.snapshot() must return an independently "
                f"owned {type(endpoint).__name__}."
            )
        normalized[endpoint_id] = snapshot
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class RobotResource:
    """Generic leaf or composite resource in one robot's resource DAG.

    A resource may expose any number of named endpoints. For example, one
    manipulation participant may expose ``motion`` and ``grasp`` endpoints,
    while a mobile base or whole-body controller may expose only ``motion``.
    ``members`` describes physical claim composition and does not inherit
    endpoint capabilities.
    """

    resource_id: str
    endpoints: Mapping[str, ResourceEndpoint] = field(default_factory=dict)
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.resource_id, field_name="RobotResource.resource_id")
        object.__setattr__(self, "endpoints", _normalize_endpoints(self.endpoints))
        if isinstance(self.members, (str, bytes)):
            raise TypeError(
                "RobotResource.members must be an iterable of strings, not a string."
            )
        try:
            members = tuple(self.members)
        except TypeError as exc:
            raise TypeError(
                "RobotResource.members must be an iterable of strings."
            ) from exc
        for member in members:
            _validate_identifier(member, field_name="RobotResource.members")
        if len(set(members)) != len(members):
            raise ValueError("RobotResource.members must be unique.")
        if self.resource_id in members:
            raise ValueError("A robot resource cannot contain itself.")
        if not members and not self.endpoints:
            raise ValueError(
                "A leaf RobotResource must expose at least one execution endpoint."
            )
        object.__setattr__(self, "members", members)

    def snapshot(self) -> RobotResource:
        """Return an independently owned resource declaration."""
        return RobotResource(
            resource_id=self.resource_id,
            endpoints=self.endpoints,
            members=self.members,
        )


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    """Generic mapping from skill-local slots to robot resource IDs."""

    resources: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.resources, Mapping):
            raise TypeError("ResourceBinding.resources must be a mapping.")
        normalized: dict[str, str] = {}
        for slot_id, resource_id in self.resources.items():
            _validate_identifier(slot_id, field_name="ResourceBinding slot IDs")
            _validate_identifier(resource_id, field_name="ResourceBinding resource IDs")
            normalized[slot_id] = resource_id
        object.__setattr__(self, "resources", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True, init=False)
class SkillPolicyPreset:
    """Versioned planning, recovery, and runner policy bundle.

    Args:
        preset_id: Stable preset identifier.
        schema_version: Preset schema version. Version 1 is currently supported.
        motion_policy: Reusable atomic motion policy.
        recovery_policy: Bounded action recovery policy.
        runner_cfg: Execution transport and scheduling policy.
        required_planner: Optional planner backend required by this preset.
    """

    preset_id: str
    schema_version: int
    required_planner: str | None
    """Optional planner backend required by this preset."""
    _motion_policy: MotionPolicy
    _recovery_policy: RecoveryPolicy
    _runner_cfg: ExecutionRunnerCfg

    def __init__(
        self,
        preset_id: str,
        schema_version: int = 1,
        motion_policy: MotionPolicy | None = None,
        recovery_policy: RecoveryPolicy | None = None,
        runner_cfg: ExecutionRunnerCfg | None = None,
        required_planner: str | None = None,
    ) -> None:
        """Own one policy bundle without exposing mutable nested configuration."""
        _validate_identifier(preset_id, field_name="SkillPolicyPreset.preset_id")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise TypeError("SkillPolicyPreset.schema_version must be an integer.")
        if schema_version != 1:
            raise ValueError(
                "Unsupported SkillPolicyPreset.schema_version "
                f"{schema_version}; supported versions are [1]."
            )
        if required_planner is not None:
            _validate_identifier(
                required_planner,
                field_name="SkillPolicyPreset.required_planner",
            )
        selected_motion = MotionPolicy() if motion_policy is None else motion_policy
        selected_recovery = (
            RecoveryPolicy() if recovery_policy is None else recovery_policy
        )
        selected_runner = ExecutionRunnerCfg() if runner_cfg is None else runner_cfg
        if not isinstance(selected_motion, MotionPolicy):
            raise TypeError("motion_policy must be a MotionPolicy.")
        if not isinstance(selected_recovery, RecoveryPolicy):
            raise TypeError("recovery_policy must be a RecoveryPolicy.")
        if not isinstance(selected_runner, ExecutionRunnerCfg):
            raise TypeError("runner_cfg must be an ExecutionRunnerCfg.")
        object.__setattr__(self, "preset_id", preset_id)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "required_planner", required_planner)
        object.__setattr__(self, "_motion_policy", deepcopy(selected_motion))
        object.__setattr__(self, "_recovery_policy", deepcopy(selected_recovery))
        object.__setattr__(self, "_runner_cfg", deepcopy(selected_runner))

    @property
    def motion_policy(self) -> MotionPolicy:
        """Return an independently owned motion policy."""
        return deepcopy(self._motion_policy)

    @property
    def recovery_policy(self) -> RecoveryPolicy:
        """Return an independently owned recovery policy."""
        return deepcopy(self._recovery_policy)

    @property
    def runner_cfg(self) -> ExecutionRunnerCfg:
        """Return an independently owned runner configuration."""
        return deepcopy(self._runner_cfg)

    def snapshot(self) -> SkillPolicyPreset:
        """Return an independently owned preset value."""
        return SkillPolicyPreset(
            preset_id=self.preset_id,
            schema_version=self.schema_version,
            motion_policy=self.motion_policy,
            recovery_policy=self.recovery_policy,
            runner_cfg=self.runner_cfg,
            required_planner=self.required_planner,
        )


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    """Physical leaf and joint claim used for deterministic conflict checks."""

    leaf_resource_ids: frozenset[str]
    joint_ids: tuple[int, ...]
    claim_tokens: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "leaf_resource_ids",
            _normalize_identifier_set(
                self.leaf_resource_ids,
                field_name="ResourceClaim.leaf_resource_ids",
            ),
        )
        joint_ids = tuple(self.joint_ids)
        if not all(
            isinstance(joint_id, int)
            and not isinstance(joint_id, bool)
            and joint_id >= 0
            for joint_id in joint_ids
        ):
            raise ValueError("ResourceClaim.joint_ids must be non-negative integers.")
        if tuple(sorted(set(joint_ids))) != joint_ids:
            raise ValueError("ResourceClaim.joint_ids must be sorted and unique.")
        object.__setattr__(self, "joint_ids", joint_ids)
        object.__setattr__(
            self,
            "claim_tokens",
            _normalize_identifier_set(
                self.claim_tokens,
                field_name="ResourceClaim.claim_tokens",
            ),
        )

    def conflicts_with(self, other: ResourceClaim) -> bool:
        """Return whether two claims overlap in a leaf or concrete joint."""
        if not isinstance(other, ResourceClaim):
            raise TypeError("other must be a ResourceClaim.")
        return bool(
            self.leaf_resource_ids & other.leaf_resource_ids
            or self.claim_tokens & other.claim_tokens
            or set(self.joint_ids) & set(other.joint_ids)
        )

    @classmethod
    def combine(cls, claims: tuple[ResourceClaim, ...]) -> ResourceClaim:
        """Return the union of zero or more resource claims."""
        leaves: set[str] = set()
        joints: set[int] = set()
        tokens: set[str] = set()
        for claim in claims:
            if not isinstance(claim, ResourceClaim):
                raise TypeError("claims values must be ResourceClaim instances.")
            leaves.update(claim.leaf_resource_ids)
            joints.update(claim.joint_ids)
            tokens.update(claim.claim_tokens)
        return cls(
            frozenset(leaves),
            tuple(sorted(joints)),
            frozenset(tokens),
        )


@dataclass(frozen=True, slots=True)
class ResolvedRobotResource:
    """Robot-validated resource with concrete endpoint joint IDs and claim."""

    resource_id: str
    endpoints: Mapping[str, ResolvedResourceEndpoint]
    members: tuple[str, ...]
    claim: ResourceClaim

    def __post_init__(self) -> None:
        _validate_identifier(
            self.resource_id,
            field_name="ResolvedRobotResource.resource_id",
        )
        if not isinstance(self.endpoints, Mapping):
            raise TypeError("endpoints must be a mapping.")
        normalized_endpoints: dict[str, ResolvedResourceEndpoint] = {}
        for endpoint_id, endpoint in self.endpoints.items():
            _validate_identifier(endpoint_id, field_name="resolved endpoint IDs")
            if not isinstance(endpoint, ResolvedResourceEndpoint):
                raise TypeError(
                    "ResolvedRobotResource.endpoints values must be "
                    "ResolvedResourceEndpoint instances."
                )
            normalized_endpoints[endpoint_id] = endpoint
        object.__setattr__(
            self,
            "endpoints",
            MappingProxyType(normalized_endpoints),
        )
        if isinstance(self.members, (str, bytes)):
            raise TypeError("members must be an iterable of resource IDs.")
        members = tuple(self.members)
        for member in members:
            _validate_identifier(member, field_name="resolved resource members")
        if len(set(members)) != len(members):
            raise ValueError("Resolved resource members must be unique.")
        object.__setattr__(self, "members", members)
        if not isinstance(self.claim, ResourceClaim):
            raise TypeError("claim must be a ResourceClaim.")
        endpoint_joints = {
            joint_id
            for endpoint in normalized_endpoints.values()
            for joint_id in endpoint.joint_ids
        }
        missing_claim_joints = sorted(endpoint_joints - set(self.claim.joint_ids))
        if missing_claim_joints:
            raise ValueError(
                "Resolved resource claim does not cover endpoint joints "
                f"{missing_claim_joints}."
            )
        endpoint_tokens = {
            token
            for endpoint in normalized_endpoints.values()
            for token in endpoint.claim_tokens
        }
        missing_claim_tokens = sorted(endpoint_tokens - self.claim.claim_tokens)
        if missing_claim_tokens:
            raise ValueError(
                "Resolved resource claim does not cover endpoint claim tokens "
                f"{missing_claim_tokens}."
            )
        if not members:
            if self.claim.leaf_resource_ids != frozenset({self.resource_id}):
                raise ValueError(
                    "A resolved leaf resource claim must contain exactly its own "
                    "resource ID."
                )
            if set(self.claim.joint_ids) != endpoint_joints:
                raise ValueError(
                    "A resolved leaf resource claim must contain exactly its "
                    "endpoint joints."
                )
            if self.claim.claim_tokens != frozenset(endpoint_tokens):
                raise ValueError(
                    "A resolved leaf resource claim must contain exactly its "
                    "endpoint claim tokens."
                )

    @property
    def endpoint_joint_ids(self) -> Mapping[str, tuple[int, ...]]:
        """Return ordered joint IDs for each resolved endpoint."""
        return MappingProxyType(
            {
                endpoint_id: endpoint.joint_ids
                for endpoint_id, endpoint in self.endpoints.items()
            }
        )


@dataclass(frozen=True, slots=True)
class ResolvedSkillBinding:
    """One generic resource assignment lowered for the current action core."""

    skill_id: str
    resources: Mapping[str, ResolvedRobotResource]
    action_binding: ActionBinding
    claim: ResourceClaim

    def __post_init__(self) -> None:
        _validate_identifier(self.skill_id, field_name="ResolvedSkillBinding.skill_id")
        if not isinstance(self.resources, Mapping):
            raise TypeError("resources must be a mapping.")
        normalized: dict[str, ResolvedRobotResource] = {}
        for slot_id, resource in self.resources.items():
            _validate_identifier(slot_id, field_name="resolved skill slot IDs")
            if not isinstance(resource, ResolvedRobotResource):
                raise TypeError(
                    "ResolvedSkillBinding.resources values must be "
                    "ResolvedRobotResource instances."
                )
            normalized[slot_id] = resource
        object.__setattr__(self, "resources", MappingProxyType(normalized))
        if not isinstance(self.action_binding, ActionBinding):
            raise TypeError("action_binding must be an ActionBinding.")
        if not isinstance(self.claim, ResourceClaim):
            raise TypeError("claim must be a ResourceClaim.")

    @property
    def resource_ids(self) -> Mapping[str, str]:
        """Return the selected logical resource ID for each skill-local slot."""
        return MappingProxyType(
            {
                slot_id: resource.resource_id
                for slot_id, resource in self.resources.items()
            }
        )


def _normalize_resources(
    values: Mapping[str, RobotResource],
) -> Mapping[str, RobotResource]:
    """Validate profile resource ownership and mapping keys."""
    if not isinstance(values, Mapping):
        raise TypeError("RobotSkillProfile.resources must be a mapping.")
    normalized: dict[str, RobotResource] = {}
    for resource_id, resource in values.items():
        _validate_identifier(resource_id, field_name="profile resource IDs")
        if not isinstance(resource, RobotResource):
            raise TypeError(
                "RobotSkillProfile.resources values must be RobotResource instances."
            )
        if resource_id != resource.resource_id:
            raise ValueError(
                f"Resource mapping key {resource_id!r} does not match "
                f"RobotResource.resource_id {resource.resource_id!r}."
            )
        normalized[resource_id] = resource.snapshot()
    return MappingProxyType(normalized)


def _normalize_command_profiles(
    values: Mapping[str, ControlPartCommandProfile],
) -> Mapping[str, ControlPartCommandProfile]:
    """Own generic endpoint command-profile snapshots by stable profile ID."""
    if not isinstance(values, Mapping):
        raise TypeError("command_profiles must be a mapping.")
    normalized: dict[str, ControlPartCommandProfile] = {}
    for profile_id, profile in values.items():
        _validate_identifier(profile_id, field_name="command profile IDs")
        if not isinstance(profile, ControlPartCommandProfile):
            raise TypeError(
                "command_profiles values must be ControlPartCommandProfile instances."
            )
        normalized[profile_id] = profile.snapshot()
    return MappingProxyType(normalized)


def _normalize_defaults(
    values: Mapping[str, ResourceBinding],
) -> Mapping[str, ResourceBinding]:
    """Validate and freeze per-skill complete default bindings."""
    if not isinstance(values, Mapping):
        raise TypeError("defaults must be a mapping.")
    normalized: dict[str, ResourceBinding] = {}
    for skill_id, binding in values.items():
        _validate_identifier(skill_id, field_name="default skill IDs")
        if not isinstance(binding, ResourceBinding):
            raise TypeError("defaults values must be ResourceBinding instances.")
        normalized[skill_id] = binding
    return MappingProxyType(normalized)


def _normalize_presets(
    values: Mapping[str, SkillPolicyPreset],
) -> Mapping[str, SkillPolicyPreset]:
    """Validate preset keys and own independent snapshots."""
    if not isinstance(values, Mapping):
        raise TypeError("presets must be a mapping.")
    normalized: dict[str, SkillPolicyPreset] = {}
    for preset_id, preset in values.items():
        _validate_identifier(preset_id, field_name="preset IDs")
        if not isinstance(preset, SkillPolicyPreset):
            raise TypeError("presets values must be SkillPolicyPreset instances.")
        if preset_id != preset.preset_id:
            raise ValueError(
                f"Preset mapping key {preset_id!r} does not match preset_id "
                f"{preset.preset_id!r}."
            )
        normalized[preset_id] = preset.snapshot()
    return MappingProxyType(normalized)


def _normalize_named_mapping(
    values: Mapping[str, str],
    *,
    field_name: str,
) -> Mapping[str, str]:
    """Validate and freeze one identifier-to-identifier mapping."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        _validate_identifier(key, field_name=f"{field_name} keys")
        _validate_identifier(value, field_name=f"{field_name} values")
        normalized[key] = value
    return MappingProxyType(normalized)


def _normalize_endpoint_adapters(
    values: Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None,
) -> Mapping[type[ResourceEndpoint], ResourceEndpointAdapter]:
    """Install the built-in adapter plus exact-type endpoint extensions."""
    normalized: dict[type[ResourceEndpoint], ResourceEndpointAdapter] = {
        ControlPartEndpoint: ControlPartEndpointAdapter()
    }
    if values is not None:
        if not isinstance(values, Mapping):
            raise TypeError("endpoint_adapters must be a mapping or None.")
        for endpoint_type, adapter in values.items():
            if not isinstance(endpoint_type, type) or not issubclass(
                endpoint_type, ResourceEndpoint
            ):
                raise TypeError(
                    "endpoint_adapters keys must be ResourceEndpoint subclasses."
                )
            if endpoint_type is ControlPartEndpoint:
                raise ValueError(
                    "The built-in ControlPartEndpoint adapter cannot be overridden; "
                    "declare a distinct ResourceEndpoint subtype for custom "
                    "controller semantics."
                )
            if not isinstance(adapter, ResourceEndpointAdapter):
                raise TypeError(
                    "endpoint_adapters values must be ResourceEndpointAdapter "
                    "instances."
                )
            declared_endpoint_type = getattr(adapter, "endpoint_type", None)
            if not isinstance(declared_endpoint_type, type) or not issubclass(
                declared_endpoint_type, ResourceEndpoint
            ):
                raise TypeError(
                    f"Endpoint adapter {type(adapter).__name__} must declare a "
                    "ResourceEndpoint subclass as endpoint_type."
                )
            if declared_endpoint_type is not endpoint_type:
                raise ValueError(
                    f"Endpoint adapter {type(adapter).__name__} declares "
                    f"endpoint_type {declared_endpoint_type.__name__}, but is "
                    f"registered for {endpoint_type.__name__}."
                )
            adapter_id = getattr(adapter, "adapter_id", None)
            _validate_identifier(
                adapter_id,
                field_name="ResourceEndpointAdapter.adapter_id",
            )
            normalized[endpoint_type] = adapter
    adapter_ids = [
        getattr(adapter, "adapter_id", None) for adapter in normalized.values()
    ]
    if len(set(adapter_ids)) != len(adapter_ids):
        raise ValueError("Installed ResourceEndpointAdapter IDs must be unique.")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class RobotSkillProfile:
    """Reusable declarative skill integration for one robot embodiment."""

    profile_id: str
    resources: Mapping[str, RobotResource]
    command_profiles: Mapping[str, ControlPartCommandProfile] = field(
        default_factory=dict
    )
    defaults: Mapping[str, ResourceBinding] = field(default_factory=dict)
    presets: Mapping[str, SkillPolicyPreset] = field(default_factory=dict)
    default_preset: str | None = None
    skill_presets: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.profile_id, field_name="RobotSkillProfile.profile_id")
        resources = _normalize_resources(self.resources)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(
            self,
            "command_profiles",
            _normalize_command_profiles(self.command_profiles),
        )
        object.__setattr__(self, "defaults", _normalize_defaults(self.defaults))
        presets = _normalize_presets(self.presets)
        object.__setattr__(self, "presets", presets)
        if self.default_preset is not None:
            _validate_identifier(
                self.default_preset,
                field_name="RobotSkillProfile.default_preset",
            )
            if self.default_preset not in presets:
                raise ValueError(
                    f"Unknown default preset {self.default_preset!r}; available "
                    f"presets are {sorted(presets)}."
                )
        skill_presets = _normalize_named_mapping(
            self.skill_presets,
            field_name="skill_presets",
        )
        unknown_presets = sorted(set(skill_presets.values()) - set(presets))
        if unknown_presets:
            raise ValueError(
                f"skill_presets references unknown presets {unknown_presets}."
            )
        object.__setattr__(self, "skill_presets", skill_presets)
        self._validate_resource_graph(resources)
        self.action_control_profiles()

    def action_control_profiles(self) -> Mapping[str, ControlPartCommandProfile]:
        """Lower endpoint command profiles for the current action core.

        Returns:
            Owned command profiles keyed by concrete robot control-part name.

        Raises:
            ValueError: If two endpoint declarations assign non-equivalent
                commands with the same semantic name to one control part.
        """
        commands_by_control_part: dict[str, dict[str, ControlCommand]] = {}
        for resource in self.resources.values():
            for endpoint in resource.endpoints.values():
                if type(endpoint) is not ControlPartEndpoint:
                    continue
                profile_id = (
                    endpoint.control_part
                    if endpoint.command_profile is None
                    else endpoint.command_profile
                )
                profile = self.command_profiles.get(profile_id)
                if profile is None:
                    continue
                merged = commands_by_control_part.setdefault(
                    endpoint.control_part,
                    {},
                )
                for command_name, command in profile.commands.items():
                    previous = merged.get(command_name)
                    if previous is not None and not previous.equivalent_to(command):
                        raise ValueError(
                            f"Control part {endpoint.control_part!r} receives "
                            f"non-equivalent {command_name!r} commands from profile "
                            f"{profile_id!r}."
                        )
                    merged[command_name] = command
        return MappingProxyType(
            {
                control_part: ControlPartCommandProfile(commands=commands)
                for control_part, commands in commands_by_control_part.items()
            }
        )

    @staticmethod
    def _validate_resource_graph(resources: Mapping[str, RobotResource]) -> None:
        """Reject unknown members and cycles in the resource DAG."""
        for resource in resources.values():
            unknown = sorted(set(resource.members) - set(resources))
            if unknown:
                raise ValueError(
                    f"Robot resource {resource.resource_id!r} references unknown "
                    f"members {unknown}."
                )

        visiting: list[str] = []
        visited: set[str] = set()

        def visit(resource_id: str) -> None:
            if resource_id in visited:
                return
            if resource_id in visiting:
                cycle_start = visiting.index(resource_id)
                cycle = visiting[cycle_start:] + [resource_id]
                raise ValueError(
                    "Robot resource graph contains a cycle: " + " -> ".join(cycle)
                )
            visiting.append(resource_id)
            for member in resources[resource_id].members:
                visit(member)
            visiting.pop()
            visited.add(resource_id)

        for resource_id in resources:
            visit(resource_id)

    def bind(
        self,
        engine: AtomicActionEngine,
        *,
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
    ) -> BoundRobotSkillProfile:
        """Validate this profile against one fully configured action engine.

        Args:
            engine: Installed atomic-action engine for the target robot.
            endpoint_adapters: Optional exact endpoint-type adapters. Explicit
                entries extend the non-overridable built-in control-part adapter.

        Returns:
            Robot-, engine-, and adapter-validated profile view.
        """
        return BoundRobotSkillProfile(
            self,
            engine,
            endpoint_adapters=endpoint_adapters,
        )


class BoundRobotSkillProfile:
    """Robot- and engine-validated view of a :class:`RobotSkillProfile`."""

    def __init__(
        self,
        profile: RobotSkillProfile,
        engine: AtomicActionEngine,
        *,
        endpoint_adapters: (
            Mapping[type[ResourceEndpoint], ResourceEndpointAdapter] | None
        ) = None,
    ) -> None:
        from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine

        if not isinstance(profile, RobotSkillProfile):
            raise TypeError("profile must be a RobotSkillProfile.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        self._profile = profile
        self._engine = engine
        self._endpoint_adapters = _normalize_endpoint_adapters(endpoint_adapters)
        self._validate_presets()
        self._resources = self._resolve_resources()
        self._validate_engine_control_profiles()
        self._validate_leaf_ownership()
        self._installed_skills = MappingProxyType(dict(engine.skills))
        self._validate_named_skill_configuration()
        self._validate_defaults()
        self._skills = MappingProxyType(
            {
                skill_id: descriptor
                for skill_id, descriptor in self._installed_skills.items()
                if self._assignments(descriptor.binding_contract, {})
            }
        )

    @property
    def profile_id(self) -> str:
        """Return the stable profile identifier."""
        return self._profile.profile_id

    @property
    def resources(self) -> Mapping[str, ResolvedRobotResource]:
        """Return resolved generic robot resources keyed by logical ID."""
        return self._resources

    @property
    def skills(self) -> Mapping[str, SkillDescriptor]:
        """Return installed semantic skills fully supported by this profile."""
        self._assert_catalog_current()
        return self._skills

    def preset(
        self,
        preset_id: str | None = None,
        *,
        skill_id: str | None = None,
    ) -> SkillPolicyPreset:
        """Resolve an explicit, per-skill, or profile-default policy preset."""
        selected = preset_id
        if skill_id is not None:
            descriptor = self._require_installed_skill(skill_id)
            if skill_id not in self._skills:
                raise UnsupportedSkillError(
                    self._unsupported_message(
                        skill_id,
                        descriptor.binding_contract,
                        {},
                    )
                )
            if selected is None:
                selected = self._profile.skill_presets.get(skill_id)
        if selected is None:
            selected = self._profile.default_preset
        if selected is None:
            raise KeyError(
                "No policy preset was selected and no default is configured."
            )
        try:
            preset = self._profile.presets[selected]
        except KeyError as exc:
            raise KeyError(
                f"Unknown policy preset {selected!r}; available presets are "
                f"{sorted(self._profile.presets)}."
            ) from exc
        return preset.snapshot()

    def candidates(
        self,
        skill_id: str,
        selections: Mapping[str, str] | None = None,
    ) -> tuple[ResourceBinding, ...]:
        """Return every valid complete resource assignment deterministically."""
        descriptor = self._require_installed_skill(skill_id)
        normalized = self._normalize_selections(descriptor, selections)
        return tuple(
            ResourceBinding(
                resources={
                    slot_id: resource.resource_id
                    for slot_id, resource in assignment.items()
                }
            )
            for assignment in self._assignments(
                descriptor.binding_contract,
                normalized,
            )
        )

    def resolve(
        self,
        skill_id: str,
        selections: Mapping[str, str] | None = None,
    ) -> ResolvedSkillBinding:
        """Resolve one skill with strict capability matching and disambiguation."""
        descriptor = self._require_installed_skill(skill_id)
        normalized = self._normalize_selections(descriptor, selections)
        contract = descriptor.binding_contract
        assignments = self._assignments(contract, normalized)
        if not assignments:
            raise UnsupportedSkillError(
                self._unsupported_message(skill_id, contract, normalized)
            )
        if len(assignments) == 1:
            assignment = assignments[0]
        else:
            default = self._profile.defaults.get(skill_id)
            assignment = None
            if default is not None:
                selected_ids = dict(default.resources)
                selected_ids.update(normalized)
                for candidate in assignments:
                    if all(
                        candidate[slot_id].resource_id == resource_id
                        for slot_id, resource_id in selected_ids.items()
                    ):
                        assignment = candidate
                        break
            if assignment is None:
                rendered = [
                    "{"
                    + ", ".join(
                        f"{slot}={resource.resource_id}"
                        for slot, resource in candidate.items()
                    )
                    + "}"
                    for candidate in assignments
                ]
                raise AmbiguousSkillBindingError(
                    f"Skill {skill_id!r} has {len(assignments)} valid resource "
                    f"bindings: {rendered}. Configure a complete per-skill "
                    "default or provide enough explicit slot selections."
                )
        return self._lower_binding(skill_id, contract, assignment)

    def _require_installed_skill(self, skill_id: str) -> SkillDescriptor:
        """Return one installed explicit descriptor or fail at the right boundary."""
        self._assert_catalog_current()
        _validate_identifier(skill_id, field_name="skill_id")
        descriptor = self._installed_skills.get(skill_id)
        if descriptor is None:
            raise KeyError(
                f"Skill {skill_id!r} is not an installed, agent-visible skill with "
                "an explicit binding contract."
            )
        return descriptor

    def _assert_catalog_current(self) -> None:
        """Prevent stale contracts after engine registration or replacement."""
        if dict(self._engine.skills) != dict(self._installed_skills):
            raise RuntimeError(
                "AtomicActionEngine semantic skills changed after the robot skill "
                "profile was bound; bind the profile again before discovery or "
                "resolution."
            )

    def _normalize_selections(
        self,
        descriptor: SkillDescriptor,
        selections: Mapping[str, str] | None,
    ) -> Mapping[str, str]:
        """Validate caller selections against one skill's local slots."""
        normalized = _normalize_named_mapping(
            {} if selections is None else selections,
            field_name="selections",
        )
        contract = descriptor.binding_contract
        assert contract is not None
        unknown_slots = sorted(set(normalized) - set(contract.slot_ids))
        if unknown_slots:
            raise ValueError(
                f"Skill {descriptor.skill_id!r} selections contain unknown slots "
                f"{unknown_slots}; expected a subset of {list(contract.slot_ids)}."
            )
        unknown_resources = sorted(set(normalized.values()) - set(self._resources))
        if unknown_resources:
            raise ValueError(
                f"Selections reference unknown resources {unknown_resources}; "
                f"available resources are {sorted(self._resources)}."
            )
        return normalized

    def _validate_presets(self) -> None:
        """Validate planner-pinned presets against the selected engine backend."""
        configured = self._engine.planning_services.planner_name
        for preset in self._profile.presets.values():
            required = preset.required_planner
            if required is not None and required != configured:
                raise ProfileValidationError(
                    f"Preset {preset.preset_id!r} requires planner {required!r}, "
                    f"but this engine uses {configured!r}."
                )

    def _validate_engine_control_profiles(self) -> None:
        """Require current-core endpoint commands to be installed on the engine."""
        engine_profiles = self._engine.control_profiles
        try:
            expected_control_profiles = self._profile.action_control_profiles()
        except (TypeError, ValueError) as exc:
            raise ProfileValidationError(
                f"Could not lower profile commands to action control parts: {exc}"
            ) from exc
        for control_part, expected in expected_control_profiles.items():
            installed = engine_profiles.get(control_part)
            if installed is None:
                raise ProfileValidationError(
                    f"Profile command set for control part {control_part!r} is not "
                    "installed on the AtomicActionEngine."
                )
            for command_name, command in expected.commands.items():
                installed_command = installed.commands.get(command_name)
                if installed_command is None:
                    raise ProfileValidationError(
                        f"Engine control profile {control_part!r} is missing profile "
                        f"command {command_name!r}."
                    )
                if not command.equivalent_to(installed_command):
                    raise ProfileValidationError(
                        f"Engine command {control_part!r}.{command_name} is not "
                        "semantically equivalent to the profile-owned command."
                    )
        for resource in self._resources.values():
            for endpoint in resource.endpoints.values():
                if not endpoint.commands or not isinstance(
                    endpoint.runtime_target,
                    JointPositionTarget,
                ):
                    continue
                control_part = endpoint.runtime_target.control_part
                installed = engine_profiles.get(control_part)
                if installed is None:
                    raise ProfileValidationError(
                        f"Endpoint command profile "
                        f"{endpoint.command_profile_key!r} for control part "
                        f"{control_part!r} is not installed on the "
                        "AtomicActionEngine."
                    )
                for command_name, command in endpoint.commands.items():
                    installed_command = installed.commands.get(command_name)
                    if installed_command is None:
                        raise ProfileValidationError(
                            f"Engine control profile {control_part!r} is missing "
                            f"profile command {command_name!r}."
                        )
                    if not command.equivalent_to(installed_command):
                        raise ProfileValidationError(
                            f"Engine command {control_part!r}.{command_name} is "
                            "not semantically equivalent to the profile-owned "
                            "command."
                        )

    def _resolve_resources(self) -> Mapping[str, ResolvedRobotResource]:
        """Resolve adapter endpoints, graph closure, commands, and claims."""
        resolved_endpoints: dict[str, dict[str, ResolvedResourceEndpoint]] = {}
        direct_joints: dict[str, set[int]] = {}
        direct_tokens: dict[str, set[str]] = {}
        for resource_id, resource in self._profile.resources.items():
            resource_endpoints: dict[str, ResolvedResourceEndpoint] = {}
            for endpoint_id, endpoint in resource.endpoints.items():
                adapter = self._endpoint_adapters.get(type(endpoint))
                if adapter is None:
                    raise ProfileValidationError(
                        f"Resource {resource_id!r} endpoint {endpoint_id!r} uses "
                        f"unsupported endpoint type {type(endpoint).__name__}; "
                        "register a ResourceEndpointAdapter for that exact type."
                    )
                try:
                    resolution = adapter.resolve(endpoint, engine=self._engine)
                except Exception as exc:
                    raise ProfileValidationError(
                        f"Endpoint adapter {adapter.adapter_id!r} failed for resource "
                        f"{resource_id!r} endpoint {endpoint_id!r}: {exc}"
                    ) from exc
                if not isinstance(resolution, EndpointResolution):
                    raise ProfileValidationError(
                        f"Endpoint adapter {adapter.adapter_id!r} returned "
                        f"{type(resolution).__name__}, expected EndpointResolution."
                    )
                invalid_joint_ids = sorted(
                    joint_id
                    for joint_id in resolution.joint_ids
                    if joint_id >= self._engine.robot.dof
                )
                if invalid_joint_ids:
                    raise ProfileValidationError(
                        f"Endpoint adapter {adapter.adapter_id!r} resolved resource "
                        f"{resource_id!r} endpoint {endpoint_id!r} to joint IDs "
                        f"{invalid_joint_ids} outside robot DOF "
                        f"{self._engine.robot.dof}."
                    )
                command_profile = (
                    None
                    if resolution.command_profile_key is None
                    else self._profile.command_profiles.get(
                        resolution.command_profile_key
                    )
                )
                if resolution.requires_command_profile and command_profile is None:
                    raise ProfileValidationError(
                        f"Endpoint adapter {adapter.adapter_id!r} resolved resource "
                        f"{resource_id!r} endpoint {endpoint_id!r} to required "
                        f"command profile {resolution.command_profile_key!r}, but "
                        "the RobotSkillProfile does not define it."
                    )
                resource_endpoints[endpoint_id] = ResolvedResourceEndpoint(
                    endpoint=endpoint,
                    adapter_id=adapter.adapter_id,
                    runtime_target=resolution.runtime_target,
                    command_profile_key=resolution.command_profile_key,
                    requires_command_profile=resolution.requires_command_profile,
                    commands=(
                        {} if command_profile is None else command_profile.commands
                    ),
                    claim_tokens=resolution.claim_tokens,
                    joint_ids=resolution.joint_ids,
                    exclusive=resolution.exclusive,
                )
            resolved_endpoints[resource_id] = resource_endpoints
            direct_joints[resource_id] = {
                joint_id
                for endpoint in resource_endpoints.values()
                for joint_id in endpoint.joint_ids
            }
            direct_tokens[resource_id] = {
                token
                for endpoint in resource_endpoints.values()
                for token in endpoint.claim_tokens
            }

        leaf_cache: dict[str, frozenset[str]] = {}
        joint_cache: dict[str, frozenset[int]] = {}
        token_cache: dict[str, frozenset[str]] = {}

        def resolve_claim(
            resource_id: str,
        ) -> tuple[frozenset[str], frozenset[int], frozenset[str]]:
            cached_leaves = leaf_cache.get(resource_id)
            if cached_leaves is not None:
                return (
                    cached_leaves,
                    joint_cache[resource_id],
                    token_cache[resource_id],
                )
            resource = self._profile.resources[resource_id]
            if not resource.members:
                leaves = frozenset({resource_id})
                joints = frozenset(direct_joints[resource_id])
                tokens = frozenset(direct_tokens[resource_id])
            else:
                leaves_set: set[str] = set()
                member_joints: set[int] = set()
                member_tokens: set[str] = set()
                for member in resource.members:
                    member_leaves, nested_joints, nested_tokens = resolve_claim(member)
                    leaves_set.update(member_leaves)
                    member_joints.update(nested_joints)
                    member_tokens.update(nested_tokens)
                uncovered = direct_joints[resource_id] - member_joints
                if uncovered:
                    raise ProfileValidationError(
                        f"Composite resource {resource_id!r} endpoints control joints "
                        f"{sorted(uncovered)} not claimed by its members."
                    )
                leaves = frozenset(leaves_set)
                joints = frozenset(member_joints | direct_joints[resource_id])
                tokens = frozenset(member_tokens | direct_tokens[resource_id])
            leaf_cache[resource_id] = leaves
            joint_cache[resource_id] = joints
            token_cache[resource_id] = tokens
            return leaves, joints, tokens

        resolved: dict[str, ResolvedRobotResource] = {}
        for resource_id, resource in self._profile.resources.items():
            leaves, joints, tokens = resolve_claim(resource_id)
            resolved[resource_id] = ResolvedRobotResource(
                resource_id=resource_id,
                endpoints=resolved_endpoints[resource_id],
                members=resource.members,
                claim=ResourceClaim(
                    leaves,
                    tuple(sorted(joints)),
                    tokens,
                ),
            )
        self._validate_command_shapes(resolved_endpoints)
        return MappingProxyType(resolved)

    def _validate_command_shapes(
        self,
        endpoints_by_resource: Mapping[str, Mapping[str, ResolvedResourceEndpoint]],
    ) -> None:
        """Validate profile joint commands against every referenced endpoint DOF."""
        checked: set[tuple[str, int]] = set()
        for endpoints in endpoints_by_resource.values():
            for endpoint in endpoints.values():
                if not endpoint.commands:
                    continue
                dof = len(endpoint.joint_ids)
                profile_label = endpoint.command_profile_key or endpoint.adapter_id
                key = (profile_label, dof)
                if key in checked:
                    continue
                checked.add(key)
                for command_name, command in endpoint.commands.items():
                    if not isinstance(command, JointPositionCommand):
                        continue
                    positions = command.positions
                    if positions.dim() != 1:
                        raise ProfileValidationError(
                            f"Profile command {profile_label!r}."
                            f"{command_name} must be one-dimensional and "
                            "broadcastable across environments; use invocation "
                            "overrides for per-environment commands."
                        )
                    if positions.shape[-1] != dof:
                        raise ProfileValidationError(
                            f"Command {profile_label!r}.{command_name} has "
                            f"{positions.shape[-1]} joints, expected {dof}."
                        )

    def _validate_leaf_ownership(self) -> None:
        """Require physical leaves to own disjoint claims and runtime targets."""
        leaves = [
            resource for resource in self._resources.values() if not resource.members
        ]
        for index, left in enumerate(leaves):
            for right in leaves[index + 1 :]:
                overlapping_joints = sorted(
                    set(left.claim.joint_ids) & set(right.claim.joint_ids)
                )
                overlapping_tokens = sorted(
                    left.claim.claim_tokens & right.claim.claim_tokens
                )
                if overlapping_joints or overlapping_tokens:
                    raise ProfileValidationError(
                        f"Leaf resources {left.resource_id!r} and "
                        f"{right.resource_id!r} overlap on robot joints "
                        f"{overlapping_joints} or adapter claims "
                        f"{overlapping_tokens}. "
                        "Model one physical leaf and reference it from composites."
                    )
                left_targets = {
                    (
                        endpoint.runtime_target.transport_id,
                        endpoint.runtime_target.target_id,
                    )
                    for endpoint in left.endpoints.values()
                }
                right_targets = {
                    (
                        endpoint.runtime_target.transport_id,
                        endpoint.runtime_target.target_id,
                    )
                    for endpoint in right.endpoints.values()
                }
                overlapping_targets = sorted(left_targets & right_targets)
                if overlapping_targets:
                    raise ProfileValidationError(
                        f"Leaf resources {left.resource_id!r} and "
                        f"{right.resource_id!r} share runtime targets "
                        f"{overlapping_targets}. Model one physical leaf and "
                        "reference it from composites."
                    )

    def _validate_named_skill_configuration(self) -> None:
        """Reject defaults and preset selections for absent semantic skills."""
        configured_skill_ids = set(self._profile.defaults) | set(
            self._profile.skill_presets
        )
        unknown = sorted(configured_skill_ids - set(self._installed_skills))
        if unknown:
            raise ProfileValidationError(
                f"Profile references skills not installed with explicit contracts: "
                f"{unknown}."
            )

    def _validate_defaults(self) -> None:
        """Require every configured default to be complete and currently valid."""
        for skill_id, default in self._profile.defaults.items():
            descriptor = self._installed_skills[skill_id]
            contract = descriptor.binding_contract
            assert contract is not None
            expected = set(contract.slot_ids)
            actual = set(default.resources)
            if actual != expected:
                raise ProfileValidationError(
                    f"Default binding for skill {skill_id!r} must cover exactly "
                    f"{sorted(expected)}; missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}."
                )
            unknown_resources = sorted(
                set(default.resources.values()) - set(self._resources)
            )
            if unknown_resources:
                raise ProfileValidationError(
                    f"Default binding for skill {skill_id!r} references unknown "
                    f"resources {unknown_resources}."
                )
            assignments = self._assignments(contract, default.resources)
            if len(assignments) != 1:
                raise ProfileValidationError(
                    f"Default binding for skill {skill_id!r} does not satisfy its "
                    "capabilities, commands, endpoints, and resource constraints."
                )

    def _assignments(
        self,
        contract: SkillBindingContract | None,
        selections: Mapping[str, str],
    ) -> tuple[dict[str, ResolvedRobotResource], ...]:
        """Enumerate valid complete assignments in declaration order."""
        if contract is None:
            return ()
        if not contract.slots:
            return ({},)
        slot_candidates: list[tuple[ResolvedRobotResource, ...]] = []
        for slot in contract.slots:
            selected = selections.get(slot.slot_id)
            candidates = tuple(
                resource
                for resource in self._resources.values()
                if (selected is None or resource.resource_id == selected)
                and not self._rejection_reasons(resource, slot)
            )
            if not candidates:
                return ()
            slot_candidates.append(candidates)
        assignments: list[dict[str, ResolvedRobotResource]] = []
        for combination in product(*slot_candidates):
            assignment = {
                slot.slot_id: resource
                for slot, resource in zip(contract.slots, combination, strict=True)
            }
            if self._constraints_match(contract, assignment):
                assignments.append(assignment)
        return tuple(assignments)

    @staticmethod
    def _constraints_match(
        contract: SkillBindingContract,
        assignment: Mapping[str, ResolvedRobotResource],
    ) -> bool:
        """Apply declared graph/claim constraints to one assignment."""
        for constraint in contract.constraints:
            if isinstance(constraint, DisjointResourceSlots):
                resources = [assignment[slot] for slot in constraint.slots]
                for index, left in enumerate(resources):
                    if any(
                        left.claim.conflicts_with(right.claim)
                        for right in resources[index + 1 :]
                    ):
                        return False
        return True

    def _unsupported_message(
        self,
        skill_id: str,
        contract: SkillBindingContract | None,
        selections: Mapping[str, str],
    ) -> str:
        """Render deterministic per-slot rejection reasons."""
        if contract is None:
            return f"Skill {skill_id!r} has no explicit binding contract."
        lines = [f"Skill {skill_id!r} has no compatible resource binding."]
        every_slot_has_candidate = True
        for slot in contract.slots:
            selected = selections.get(slot.slot_id)
            lines.append(f"slot {slot.slot_id!r}:")
            slot_has_candidate = False
            for resource in self._resources.values():
                if selected is not None and resource.resource_id != selected:
                    continue
                reasons = self._rejection_reasons(resource, slot)
                status = "compatible" if not reasons else "; ".join(reasons)
                slot_has_candidate |= not reasons
                lines.append(f"  {resource.resource_id}: {status}")
            every_slot_has_candidate &= slot_has_candidate
        if contract.constraints and every_slot_has_candidate:
            lines.append(
                "All individually compatible combinations violate constraints."
            )
        return "\n".join(lines)

    def _rejection_reasons(
        self,
        resource: ResolvedRobotResource,
        slot: SkillResourceSlot,
    ) -> tuple[str, ...]:
        """Explain why one resource fails one slot requirement."""
        reasons: list[str] = []
        matched_endpoints: dict[str, ResolvedResourceEndpoint] = {}
        for requirement in slot.endpoints:
            endpoint = resource.endpoints.get(requirement.endpoint_id)
            if endpoint is None:
                reasons.append(f"missing endpoint {requirement.endpoint_id!r}")
                continue
            missing_capabilities = sorted(
                requirement.capabilities - endpoint.capabilities
            )
            if missing_capabilities:
                reasons.append(
                    f"endpoint {requirement.endpoint_id!r} missing capabilities "
                    f"{missing_capabilities}"
                )
            for command_name, command_type in requirement.required_commands.items():
                command = endpoint.commands.get(command_name)
                if command is None:
                    reasons.append(
                        f"endpoint {requirement.endpoint_id!r} missing command "
                        f"{command_name!r}"
                    )
                elif not isinstance(command, command_type):
                    reasons.append(
                        f"command {command_name!r} is {type(command).__name__}, "
                        f"expected {command_type.__name__}"
                    )
            matched_endpoints[requirement.endpoint_id] = endpoint
        for constraint in slot.constraints:
            if not isinstance(constraint, DisjointSlotEndpoints):
                continue
            endpoint_ids = constraint.endpoint_ids
            for index, left_id in enumerate(endpoint_ids):
                left = matched_endpoints.get(left_id)
                if left is None:
                    continue
                for right_id in endpoint_ids[index + 1 :]:
                    right = matched_endpoints.get(right_id)
                    if right is None or not left.conflicts_with(right):
                        continue
                    overlapping_joints = sorted(
                        set(left.joint_ids) & set(right.joint_ids)
                    )
                    overlapping_tokens = sorted(left.claim_tokens & right.claim_tokens)
                    shared_target = (
                        (
                            left.runtime_target.transport_id,
                            left.runtime_target.target_id,
                        )
                        if (
                            left.runtime_target.transport_id,
                            left.runtime_target.target_id,
                        )
                        == (
                            right.runtime_target.transport_id,
                            right.runtime_target.target_id,
                        )
                        else None
                    )
                    reasons.append(
                        f"endpoints {left_id!r} and {right_id!r} overlap on joints "
                        f"{overlapping_joints} or adapter claims "
                        f"{overlapping_tokens} or share runtime target "
                        f"{shared_target}"
                    )
        return tuple(reasons)

    def _lower_binding(
        self,
        skill_id: str,
        contract: SkillBindingContract | None,
        assignment: Mapping[str, ResolvedRobotResource],
    ) -> ResolvedSkillBinding:
        """Lower every required endpoint to one engine-owned action binding."""
        assert contract is not None
        endpoints: list[EndpointBinding] = []
        for slot in contract.slots:
            resource = assignment[slot.slot_id]
            for requirement in slot.endpoints:
                endpoint = resource.endpoints[requirement.endpoint_id]
                endpoints.append(
                    EndpointBinding(
                        slot_id=slot.slot_id,
                        endpoint_id=requirement.endpoint_id,
                        resource_id=resource.resource_id,
                        adapter_id=endpoint.adapter_id,
                        target=endpoint.runtime_target,
                        capabilities=endpoint.capabilities,
                        commands=endpoint.commands,
                        claim_tokens=endpoint.claim_tokens,
                        joint_ids=endpoint.joint_ids,
                    )
                )
        return ResolvedSkillBinding(
            skill_id=skill_id,
            resources=assignment,
            action_binding=ActionBinding(
                owner_id=self._engine.binding_owner_id,
                endpoints=tuple(endpoints),
            ),
            claim=ResourceClaim.combine(
                tuple(resource.claim for resource in assignment.values())
            ),
        )


__all__ = [
    "AmbiguousSkillBindingError",
    "BoundRobotSkillProfile",
    "ControlPartEndpoint",
    "ControlPartEndpointAdapter",
    "EndpointResolution",
    "ProfileValidationError",
    "ResourceEndpoint",
    "ResourceEndpointAdapter",
    "ResolvedRobotResource",
    "ResolvedResourceEndpoint",
    "ResolvedSkillBinding",
    "ResourceBinding",
    "ResourceClaim",
    "RobotResource",
    "RobotSkillProfile",
    "SkillPolicyPreset",
    "UnsupportedSkillError",
]
