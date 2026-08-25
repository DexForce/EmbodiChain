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

"""Single-source AtomicAction capability descriptors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

__all__ = [
    "ACTION_CONTRACT_VERSION",
    "AtomicCapability",
    "AtomicCapabilityRegistry",
    "ResolvedActionContract",
    "ResourceClaim",
    "StateAtom",
    "StateEffect",
    "build_atomic_capability_registry",
    "capability_precondition",
]

_RETRY_MODES = frozenset({"direct", "recover_then_retry", "non_retryable"})
ACTION_CONTRACT_VERSION = "action_contract_v2"
_PREDICATES = frozenset(
    {
        "arm_free",
        "object_free",
        "object_held",
        "object_coordinated_held",
        "handover_complete",
        "arm_clear",
        "arm_home",
    }
)
_EFFECT_OPERATIONS = frozenset({"add", "delete"})
_RESOURCE_ACCESS = frozenset({"shared_read", "exclusive"})
_RESOURCE_LIFETIMES = frozenset({"action", "until_release"})
_COMPLETION_MODES = frozenset({"ordinary", "cleanup", "terminal_barrier"})
_FAILURE_POLICIES = frozenset({"task_required", "safety_required", "best_effort"})


@dataclass(frozen=True)
class StateAtom:
    """One symbolic state fact used by an Action Contract."""

    predicate: str
    object_uid: str | None = None
    arm: str | None = None

    def __post_init__(self) -> None:
        if self.predicate not in _PREDICATES:
            raise ValueError(f"Unknown Action Contract predicate {self.predicate!r}.")
        if self.object_uid is not None and not self.object_uid:
            raise ValueError("StateAtom.object_uid must not be empty.")
        if self.arm is not None and not self.arm:
            raise ValueError("StateAtom.arm must not be empty.")

    def as_mapping(self) -> dict[str, str]:
        """Return the stable JSON representation of this fact."""
        result = {"predicate": self.predicate}
        if self.object_uid is not None:
            result["object_uid"] = self.object_uid
        if self.arm is not None:
            result["arm"] = self.arm
        return result


@dataclass(frozen=True)
class StateEffect:
    """Add or delete one symbolic state fact."""

    op: str
    atom: StateAtom

    def __post_init__(self) -> None:
        if self.op not in _EFFECT_OPERATIONS:
            raise ValueError(f"Unknown Action Contract effect operation {self.op!r}.")

    def as_mapping(self) -> dict[str, Any]:
        """Return the stable JSON representation of this effect."""
        return {"op": self.op, "atom": self.atom.as_mapping()}


@dataclass(frozen=True)
class ResourceClaim:
    """One resource access claim made by an AtomicAction."""

    resource: str
    access: str = "exclusive"
    lifetime: str = "action"

    def __post_init__(self) -> None:
        if not self.resource:
            raise ValueError("ResourceClaim.resource must not be empty.")
        if self.access not in _RESOURCE_ACCESS:
            raise ValueError(f"Unknown resource access mode {self.access!r}.")
        if self.lifetime not in _RESOURCE_LIFETIMES:
            raise ValueError(f"Unknown resource lifetime {self.lifetime!r}.")

    def as_mapping(self) -> dict[str, str]:
        """Return the stable JSON representation of this claim."""
        return {
            "resource": self.resource,
            "access": self.access,
            "lifetime": self.lifetime,
        }


@dataclass(frozen=True)
class ResolvedActionContract:
    """Fully resolved, serializable contract for one action node."""

    requires: tuple[StateAtom, ...] = ()
    effects: tuple[StateEffect, ...] = ()
    claims: tuple[ResourceClaim, ...] = ()
    completion: str = "ordinary"
    failure_policy: str = "task_required"
    version: str = ACTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version != ACTION_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported Action Contract version {self.version!r}; "
                f"expected {ACTION_CONTRACT_VERSION!r}."
            )
        if self.completion not in _COMPLETION_MODES:
            raise ValueError(f"Unknown Action Contract completion {self.completion!r}.")
        if self.failure_policy not in _FAILURE_POLICIES:
            raise ValueError(
                f"Unknown Action Contract failure policy {self.failure_policy!r}."
            )

    def as_mapping(self) -> dict[str, Any]:
        """Return the stable JSON representation persisted in SeedGraph v3."""
        return {
            "version": self.version,
            "requires": [atom.as_mapping() for atom in self.requires],
            "effects": [effect.as_mapping() for effect in self.effects],
            "claims": [claim.as_mapping() for claim in self.claims],
            "completion": self.completion,
            "failure_policy": self.failure_policy,
        }


@dataclass(frozen=True)
class AtomicCapability:
    """Describe planning, grounding, execution, and recovery for one skill."""

    name: str
    action_type: type | None
    config_type: type | None
    binding_kinds: frozenset[str]
    controls: frozenset[str]
    resource_mode: str
    state_effect: str
    target_materializer: str
    motion_base: str | None = None
    config_materializer: str = "single_arm"
    verifier: str = "postcondition"
    failure_classifier: str = "default"
    retry_mode: str = "direct"
    runtime_available: bool = True
    unavailable_reason: str | None = None
    target_materializer_hook: Callable[..., Any] | None = None
    config_materializer_hook: Callable[..., Any] | None = None
    verifier_hook: Callable[..., Any] | None = None
    failure_classifier_hook: Callable[..., str] | None = None
    contract_resolver_hook: (
        Callable[[Mapping[str, Any]], ResolvedActionContract] | None
    ) = None
    allows_target_contact: bool = False
    """Whether motion planning may temporarily exclude the action target."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AtomicCapability.name must not be empty.")
        if self.motion_base is not None and not self.motion_base:
            raise ValueError(
                f"AtomicCapability {self.name!r} motion_base must not be empty."
            )
        if not self.binding_kinds or not self.controls:
            raise ValueError(
                f"AtomicCapability {self.name!r} requires bindings and controls."
            )
        if self.retry_mode not in _RETRY_MODES:
            raise ValueError(
                f"AtomicCapability {self.name!r} has invalid retry_mode {self.retry_mode!r}."
            )
        if not isinstance(self.allows_target_contact, bool):
            raise TypeError("allows_target_contact must be a boolean.")
        if self.runtime_available:
            if self.action_type is None or self.config_type is None:
                raise ValueError(
                    f"Executable AtomicCapability {self.name!r} requires action/config types."
                )
            if self.unavailable_reason is not None:
                raise ValueError(
                    f"Executable AtomicCapability {self.name!r} cannot have an unavailable reason."
                )
        elif not self.unavailable_reason:
            raise ValueError(
                f"Planning-only AtomicCapability {self.name!r} requires unavailable_reason."
            )
        for field_name in (
            "target_materializer_hook",
            "config_materializer_hook",
            "verifier_hook",
            "failure_classifier_hook",
            "contract_resolver_hook",
        ):
            value = getattr(self, field_name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"AtomicCapability {self.name!r} {field_name} must be callable."
                )

    def resolve_contract(self, node: Mapping[str, Any]) -> ResolvedActionContract:
        """Resolve the deterministic Action Contract for one bound node."""
        if self.contract_resolver_hook is not None:
            contract = self.contract_resolver_hook(node)
            if not isinstance(contract, ResolvedActionContract):
                raise TypeError(
                    f"AtomicCapability {self.name!r} contract resolver must return "
                    "ResolvedActionContract."
                )
            return contract
        return _resolve_default_contract(self, node)

    def as_catalog_entry(self) -> dict[str, Any]:
        """Return the stable, JSON-safe planning view of this capability."""
        return {
            "name": self.name,
            "binding_kinds": sorted(self.binding_kinds),
            "controls": sorted(self.controls),
            "resource_mode": self.resource_mode,
            "state_effect": self.state_effect,
            "target_materializer": self.target_materializer,
            "motion_base": self.motion_base or self.name,
            "config_materializer": self.config_materializer,
            "verifier": self.verifier,
            "failure_classifier": self.failure_classifier,
            "retry_mode": self.retry_mode,
            "runtime_available": self.runtime_available,
            "unavailable_reason": self.unavailable_reason,
            "allows_target_contact": self.allows_target_contact,
            "custom_target_materializer": _callable_name(self.target_materializer_hook),
            "custom_config_materializer": _callable_name(self.config_materializer_hook),
            "custom_verifier": _callable_name(self.verifier_hook),
            "custom_failure_classifier": _callable_name(self.failure_classifier_hook),
            "contract_version": ACTION_CONTRACT_VERSION,
            "contract_resolver": _callable_name(self.contract_resolver_hook)
            or f"{__name__}._resolve_default_contract",
        }


class AtomicCapabilityRegistry:
    """Strict registry shared by planners, validators, grounders, and runtime."""

    def __init__(self) -> None:
        self._capabilities: dict[str, AtomicCapability] = {}

    def register(self, capability: AtomicCapability) -> None:
        if capability.name in self._capabilities:
            raise ValueError(
                f"AtomicCapability {capability.name!r} is already registered."
            )
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> AtomicCapability:
        try:
            return self._capabilities[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown AtomicAction {name!r}; available actions are {list(self.names())}."
            ) from exc

    def require_executable(self, name: str) -> AtomicCapability:
        capability = self.get(name)
        if not capability.runtime_available:
            raise ValueError(
                f"AtomicAction {name!r} is planning-only and cannot be executed: "
                f"{capability.unavailable_reason}"
            )
        return capability

    def validate_binding(self, action: Mapping[str, Any]) -> None:
        name = str(action.get("atomic_action", action.get("atomic_action_class", "")))
        capability = self.get(name)
        binding = action.get("target_binding")
        if not isinstance(binding, Mapping):
            raise ValueError(
                f"AtomicAction {name!r} requires a target_binding mapping."
            )
        kind = str(binding.get("kind", ""))
        if kind not in capability.binding_kinds:
            raise ValueError(
                f"AtomicAction {name!r} does not accept binding kind {kind!r}; "
                f"expected one of {sorted(capability.binding_kinds)}."
            )
        control = str(action.get("control", "arm"))
        if control not in capability.controls:
            raise ValueError(
                f"AtomicAction {name!r} does not support control {control!r}; "
                f"expected one of {sorted(capability.controls)}."
            )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def executable_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.names() if self._capabilities[name].runtime_available
        )

    def catalog(self) -> dict[str, dict[str, Any]]:
        return {
            name: self._capabilities[name].as_catalog_entry() for name in self.names()
        }

    def catalog_hash(self) -> str:
        payload = json.dumps(
            self.catalog(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def build_atomic_capability_registry() -> AtomicCapabilityRegistry:
    """Build the default catalog, including explicit planning-only skills."""
    from embodichain.lab.sim.atomic_actions import (
        AxisAlign,
        CoordinatedPickment,
        CoordinatedPickmentOptions,
        CoordinatedPlacement,
        CoordinatedPlacementOptions,
        AxisAlignOptions,
        MoveEndEffector,
        MoveEndEffectorOptions,
        MoveHeldObject,
        MoveHeldObjectOptions,
        MoveJoints,
        MoveJointsOptions,
        PickUp,
        PickUpOptions,
        Place,
        PlaceOptions,
        Pour,
        PourOptions,
        Press,
        PressOptions,
        Slide,
        SlideOptions,
        Twist,
        TwistOptions,
    )
    from .held_hand_over import HeldObjectHandOver, HeldObjectHandOverOptions

    registry = AtomicCapabilityRegistry()
    definitions = (
        AtomicCapability(
            "AxisAlign",
            AxisAlign,
            AxisAlignOptions,
            frozenset({"object"}),
            frozenset({"arm"}),
            "single_arm_object",
            "preserve",
            "axis_align",
            motion_base="AxisAlign",
            verifier="postcondition",
            verifier_hook=_verify_axis_alignment,
            failure_classifier="grasp",
            contract_resolver_hook=_resolve_axis_align_contract,
            allows_target_contact=True,
        ),
        AtomicCapability(
            "PickUp",
            PickUp,
            PickUpOptions,
            frozenset({"object"}),
            frozenset({"arm"}),
            "single_arm_object",
            "hold",
            "object_grasp",
            verifier="held_object",
            failure_classifier="grasp",
            allows_target_contact=True,
        ),
        AtomicCapability(
            "MoveHeldObject",
            MoveHeldObject,
            MoveHeldObjectOptions,
            frozenset({"semantic_goal", "visual_constraint", "handover_staging"}),
            frozenset({"arm"}),
            "single_arm_object",
            "preserve_hold",
            "semantic_held_object",
        ),
        AtomicCapability(
            "MoveEndEffector",
            MoveEndEffector,
            MoveEndEffectorOptions,
            frozenset({"policy_pose", "visual_constraint"}),
            frozenset({"arm"}),
            "single_arm",
            "preserve",
            "eef_pose",
            verifier_hook=_verify_arm_clearance,
            contract_resolver_hook=_resolve_end_effector_contract,
        ),
        AtomicCapability(
            "MoveJoints",
            MoveJoints,
            MoveJointsOptions,
            frozenset({"joint_state"}),
            frozenset({"arm", "hand"}),
            "control_part",
            "preserve",
            "joint_state",
            contract_resolver_hook=_resolve_joints_contract,
        ),
        AtomicCapability(
            "Place",
            Place,
            PlaceOptions,
            frozenset({"current_held_pose"}),
            frozenset({"arm"}),
            "single_arm_object",
            "release",
            "current_held_pose",
        ),
        AtomicCapability(
            "Pour",
            Pour,
            PourOptions,
            frozenset({"pour_goal"}),
            frozenset({"arm"}),
            "single_arm_object",
            "preserve_hold",
            "pour",
            motion_base="MoveHeldObject",
            verifier="postcondition",
            retry_mode="non_retryable",
            contract_resolver_hook=_resolve_pour_contract,
        ),
        AtomicCapability(
            "PullArticulatedPart",
            Slide,
            SlideOptions,
            frozenset({"articulation_goal"}),
            frozenset({"arm"}),
            "single_arm_object",
            "articulation_change",
            "slide",
            motion_base="Press",
            verifier="postcondition",
            retry_mode="non_retryable",
            contract_resolver_hook=_resolve_articulation_contract,
            allows_target_contact=True,
        ),
        AtomicCapability(
            "PushArticulatedPart",
            Slide,
            SlideOptions,
            frozenset({"articulation_goal"}),
            frozenset({"arm"}),
            "single_arm_object",
            "articulation_change",
            "slide",
            motion_base="Press",
            verifier="postcondition",
            retry_mode="non_retryable",
            contract_resolver_hook=_resolve_articulation_contract,
            allows_target_contact=True,
        ),
        AtomicCapability(
            "TurnKnob",
            Twist,
            TwistOptions,
            frozenset({"articulation_goal"}),
            frozenset({"arm"}),
            "single_arm_object",
            "articulation_change",
            "twist",
            motion_base="Press",
            verifier="postcondition",
            retry_mode="non_retryable",
            contract_resolver_hook=_resolve_articulation_contract,
            allows_target_contact=True,
        ),
        AtomicCapability(
            "Press",
            Press,
            PressOptions,
            frozenset({"object", "semantic_goal"}),
            frozenset({"arm"}),
            "single_arm_object",
            "preserve",
            "press",
            verifier="pressed",
            allows_target_contact=True,
        ),
        AtomicCapability(
            "CoordinatedPickment",
            CoordinatedPickment,
            CoordinatedPickmentOptions,
            frozenset({"object", "coordinated_goal"}),
            frozenset({"coordinated"}),
            "coordinated_object",
            "coordinated_hold",
            "coordinated_pickment",
            config_materializer="coordinated_pickment",
            verifier="coordinated_hold",
            failure_classifier="grasp",
        ),
        AtomicCapability(
            "CoordinatedPlacement",
            CoordinatedPlacement,
            CoordinatedPlacementOptions,
            frozenset({"coordinated_placement_goal"}),
            frozenset({"coordinated"}),
            "coordinated_object",
            "coordinated_release",
            "coordinated_placement",
            config_materializer="coordinated_placement",
        ),
        AtomicCapability(
            "HandOver",
            HeldObjectHandOver,
            HeldObjectHandOverOptions,
            frozenset({"handover_goal"}),
            frozenset({"coordinated"}),
            "coordinated_object",
            "transfer_hold",
            "handover",
            config_materializer="handover",
            verifier="receiver_holds",
            failure_classifier="handover",
            retry_mode="recover_then_retry",
        ),
    )
    for capability in definitions:
        registry.register(capability)

    return registry


def capability_precondition(
    capability: AtomicCapability,
    *,
    object_uid: str,
    actor: Mapping[str, Any],
    target_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the generic live precondition used to authorize a retry."""
    if target_binding.get("coordinated_release_role") is not None:
        # Opening a gripper is idempotent. A retry must remain legal when one
        # hand opened on the first attempt and the physical dual-hold predicate
        # therefore no longer holds.
        return {}
    if capability.state_effect == "coordinated_release":
        return {"type": "held_by_both_grippers", "object": object_uid}
    if capability.state_effect in {"preserve_hold", "release", "transfer_hold"}:
        result = {"type": "object_held", "object": object_uid}
        arm = target_binding.get("transfer_arm")
        if arm is None and actor.get("mode") in {"required", "preferred"}:
            arm = actor.get("arm")
        if isinstance(arm, str) and arm:
            result["arm"] = arm
        return result
    return {}


def _resolve_default_contract(
    capability: AtomicCapability, node: Mapping[str, Any]
) -> ResolvedActionContract:
    object_uid = _required_string(node.get("object_uid"), "node.object_uid")
    actor = node.get("actor", {})
    if not isinstance(actor, Mapping):
        raise ValueError("Action Contract resolution requires a mapping actor.")
    binding = node.get("target_binding", {})
    if not isinstance(binding, Mapping):
        raise ValueError("Action Contract resolution requires a target_binding.")
    arms = _actor_arms(actor)
    arm = arms[0] if len(arms) == 1 else None
    arm_claims = tuple(ResourceClaim(f"arm:{item}") for item in arms)
    object_claim = ResourceClaim(f"object:{object_uid}")
    payload_claims = _payload_resource_claims(binding, object_uid)

    if capability.name == "PickUp":
        required_arm = _required_arm(arm, capability.name)
        return ResolvedActionContract(
            requires=(
                StateAtom("arm_free", arm=required_arm),
                StateAtom("object_free", object_uid=object_uid),
            ),
            effects=(
                StateEffect("delete", StateAtom("arm_free", arm=required_arm)),
                StateEffect("delete", StateAtom("object_free", object_uid=object_uid)),
                StateEffect(
                    "add",
                    StateAtom("object_held", object_uid=object_uid, arm=required_arm),
                ),
            ),
            claims=(
                ResourceClaim(f"arm:{required_arm}", lifetime="until_release"),
                ResourceClaim(f"object:{object_uid}", lifetime="until_release"),
            )
            + payload_claims,
        )
    if capability.name == "MoveHeldObject":
        required_arm = _required_arm(arm, capability.name)
        return ResolvedActionContract(
            requires=(
                StateAtom("object_held", object_uid=object_uid, arm=required_arm),
            ),
            claims=(
                ResourceClaim(f"arm:{required_arm}", lifetime="until_release"),
                ResourceClaim(f"object:{object_uid}", lifetime="until_release"),
            )
            + payload_claims,
        )
    if capability.name == "Place":
        required_arm = _required_arm(arm, capability.name)
        return ResolvedActionContract(
            requires=(
                StateAtom("object_held", object_uid=object_uid, arm=required_arm),
            ),
            effects=(
                StateEffect(
                    "delete",
                    StateAtom("object_held", object_uid=object_uid, arm=required_arm),
                ),
                StateEffect("add", StateAtom("arm_free", arm=required_arm)),
                StateEffect("add", StateAtom("object_free", object_uid=object_uid)),
            ),
            claims=arm_claims + (object_claim,) + payload_claims,
        )
    if capability.name == "HandOver":
        transfer = _required_string(
            binding.get("transfer_arm"), "target_binding.transfer_arm"
        )
        receive = _required_string(
            binding.get("receive_arm"), "target_binding.receive_arm"
        )
        if transfer == receive:
            raise ValueError("HandOver requires distinct transfer and receive arms.")
        return ResolvedActionContract(
            requires=(
                StateAtom("object_held", object_uid=object_uid, arm=transfer),
                StateAtom("arm_free", arm=receive),
            ),
            effects=(
                StateEffect(
                    "delete",
                    StateAtom("object_held", object_uid=object_uid, arm=transfer),
                ),
                StateEffect("delete", StateAtom("arm_free", arm=receive)),
                StateEffect("add", StateAtom("arm_free", arm=transfer)),
                StateEffect(
                    "add",
                    StateAtom("object_held", object_uid=object_uid, arm=receive),
                ),
                StateEffect(
                    "add", StateAtom("handover_complete", object_uid=object_uid)
                ),
            ),
            claims=(
                ResourceClaim(f"arm:{transfer}"),
                ResourceClaim(f"arm:{receive}", lifetime="until_release"),
                ResourceClaim(f"object:{object_uid}", lifetime="until_release"),
            ),
        )
    if capability.name == "CoordinatedPickment":
        coordinated_arms = _coordinated_arms(arms, capability.name)
        requires = tuple(StateAtom("arm_free", arm=item) for item in coordinated_arms)
        effects = tuple(
            StateEffect("delete", StateAtom("arm_free", arm=item))
            for item in coordinated_arms
        ) + (
            StateEffect("delete", StateAtom("object_free", object_uid=object_uid)),
            StateEffect(
                "add", StateAtom("object_coordinated_held", object_uid=object_uid)
            ),
        )
        claims = tuple(
            ResourceClaim(f"arm:{item}", lifetime="until_release")
            for item in coordinated_arms
        ) + (
            ResourceClaim(f"object:{object_uid}", lifetime="until_release"),
        )
        return ResolvedActionContract(
            requires=requires + (StateAtom("object_free", object_uid=object_uid),),
            effects=effects,
            claims=claims + payload_claims,
        )
    if capability.name == "CoordinatedPlacement":
        coordinated_arms = _coordinated_arms(arms, capability.name)
        effects = (
            StateEffect(
                "delete", StateAtom("object_coordinated_held", object_uid=object_uid)
            ),
            StateEffect("add", StateAtom("object_free", object_uid=object_uid)),
        ) + tuple(
            StateEffect("add", StateAtom("arm_free", arm=item))
            for item in coordinated_arms
        )
        claims = tuple(ResourceClaim(f"arm:{item}") for item in coordinated_arms) + (
            object_claim,
        )
        return ResolvedActionContract(
            requires=(StateAtom("object_coordinated_held", object_uid=object_uid),),
            effects=effects,
            claims=claims + payload_claims,
        )

    requirements: tuple[StateAtom, ...] = ()
    if capability.state_effect == "coordinated_hold":
        requirements = (StateAtom("object_free", object_uid=object_uid),)
    elif capability.state_effect == "coordinated_release":
        requirements = (StateAtom("object_coordinated_held", object_uid=object_uid),)
    elif capability.resource_mode in {"single_arm", "single_arm_object"}:
        required_arm = _required_arm(arm, capability.name)
        requirements = (StateAtom("arm_free", arm=required_arm),)
    claims = arm_claims
    if "object" in capability.resource_mode:
        claims += (object_claim,)
    return ResolvedActionContract(
        requires=requirements,
        claims=claims + payload_claims,
    )


def _payload_resource_claims(
    binding: Mapping[str, Any], object_uid: str
) -> tuple[ResourceClaim, ...]:
    """Resolve exclusive claims for objects physically carried by a carrier."""
    raw_payloads = binding.get("payloads", ())
    if not isinstance(raw_payloads, Sequence) or isinstance(
        raw_payloads, (str, bytes, bytearray)
    ):
        raise ValueError("target_binding.payloads must be a list.")
    payload_uids: list[str] = []
    for index, raw_payload in enumerate(raw_payloads):
        value = (
            raw_payload.get("object")
            if isinstance(raw_payload, Mapping)
            else raw_payload
        )
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"target_binding.payloads[{index}] requires an object UID."
            )
        if value == object_uid:
            raise ValueError("An AtomicAction carrier cannot be its own payload.")
        payload_uids.append(value)
    if len(payload_uids) != len(set(payload_uids)):
        raise ValueError("target_binding payload objects must be unique.")
    return tuple(ResourceClaim(f"object:{uid}") for uid in payload_uids)


def _resolve_end_effector_contract(
    node: Mapping[str, Any],
) -> ResolvedActionContract:
    object_uid = _required_string(node.get("object_uid"), "node.object_uid")
    actor = node.get("actor", {})
    binding = node.get("target_binding", {})
    if not isinstance(actor, Mapping) or not isinstance(binding, Mapping):
        raise ValueError("MoveEndEffector contract requires actor and target_binding.")
    arm = _required_arm(_actor_arms(actor)[0], "MoveEndEffector")
    if binding.get("operation") == "retreat" or node.get("role") == "cleanup":
        requires = [
            StateAtom(
                (
                    "arm_clear"
                    if binding.get("operation") == "retreat_after_lift"
                    else "arm_free"
                ),
                arm=arm,
            )
        ]
        if binding.get("source") == "handover":
            requires.append(StateAtom("handover_complete", object_uid=object_uid))
        return ResolvedActionContract(
            requires=tuple(requires),
            effects=(StateEffect("add", StateAtom("arm_clear", arm=arm)),),
            claims=(ResourceClaim(f"arm:{arm}"),),
            completion="cleanup",
            failure_policy="safety_required",
        )
    return ResolvedActionContract(
        requires=(StateAtom("arm_free", arm=arm),),
        claims=(ResourceClaim(f"arm:{arm}"),),
    )


def _resolve_axis_align_contract(node: Mapping[str, Any]) -> ResolvedActionContract:
    """Keep the object free while enforcing one verified E2 terminal barrier."""
    object_uid = _required_string(node.get("object_uid"), "node.object_uid")
    actor = node.get("actor", {})
    if not isinstance(actor, Mapping):
        raise ValueError("AxisAlign contract requires an actor mapping.")
    arm = _required_arm(_actor_arms(actor)[0], "AxisAlign")
    return ResolvedActionContract(
        requires=(
            StateAtom("arm_free", arm=arm),
            StateAtom("object_free", object_uid=object_uid),
        ),
        effects=(
            StateEffect("add", StateAtom("arm_free", arm=arm)),
            StateEffect("add", StateAtom("object_free", object_uid=object_uid)),
        ),
        claims=(
            ResourceClaim(f"arm:{arm}"),
            ResourceClaim(f"object:{object_uid}"),
        ),
        completion="terminal_barrier",
        failure_policy="task_required",
    )


def _verify_axis_alignment(
    *,
    executor: Any,
    step: Any,
    arm: str,
    outcome: Any,
    attempted: torch.Tensor,
) -> torch.Tensor:
    """Reuse the E2 live predicate before committing AxisAlign completion."""
    del arm, outcome
    verified_failed, success, _ = executor._verify_step(step, ~attempted)
    return attempted & success & ~verified_failed


def _resolve_pour_contract(node: Mapping[str, Any]) -> ResolvedActionContract:
    """Retain one verified holder until the E3 action chain completes."""
    object_uid = _required_string(node.get("object_uid"), "node.object_uid")
    actor = node.get("actor", {})
    binding = node.get("target_binding", {})
    if not isinstance(actor, Mapping) or not isinstance(binding, Mapping):
        raise ValueError("Pour contract requires actor and target_binding mappings.")
    arm = _required_arm(_actor_arms(actor)[0], "Pour")
    return ResolvedActionContract(
        requires=(StateAtom("object_held", object_uid=object_uid, arm=arm),),
        claims=(
            ResourceClaim(f"arm:{arm}", lifetime="until_release"),
            ResourceClaim(f"object:{object_uid}", lifetime="until_release"),
        )
        + _payload_resource_claims(binding, object_uid),
        completion="terminal_barrier",
        failure_policy="task_required",
    )


def _resolve_articulation_contract(
    node: Mapping[str, Any],
) -> ResolvedActionContract:
    """Require one free arm and verify the observed articulation terminal state."""
    object_uid = _required_string(node.get("object_uid"), "node.object_uid")
    actor = node.get("actor", {})
    if not isinstance(actor, Mapping):
        raise ValueError("Articulation action contract requires an actor mapping.")
    arm = _required_arm(_actor_arms(actor)[0], "articulation action")
    return ResolvedActionContract(
        requires=(StateAtom("arm_free", arm=arm),),
        claims=(
            ResourceClaim(f"arm:{arm}"),
            ResourceClaim(f"object:{object_uid}"),
        ),
        completion="terminal_barrier",
        failure_policy="task_required",
    )


def _resolve_joints_contract(node: Mapping[str, Any]) -> ResolvedActionContract:
    object_uid = _required_string(node.get("object_uid"), "node.object_uid")
    actor = node.get("actor", {})
    if not isinstance(actor, Mapping):
        raise ValueError("MoveJoints contract requires an actor mapping.")
    arm = _required_arm(_actor_arms(actor)[0], "MoveJoints")
    binding = node.get("target_binding", {})
    if not isinstance(binding, Mapping):
        raise ValueError("MoveJoints contract requires a target_binding mapping.")
    release_role = binding.get("coordinated_release_role")
    if release_role is not None:
        if (
            node.get("control") != "hand"
            or binding.get("source") != "gripper_open"
            or not node.get("sync_group")
        ):
            raise ValueError(
                "Coordinated MoveJoints release requires a synchronized "
                "hand action targeting gripper_open."
            )
        if release_role not in {"participant", "commit"}:
            raise ValueError(
                "coordinated_release_role must be 'participant' or 'commit'."
            )
        claims = (
            ResourceClaim(f"arm:{arm}", lifetime="until_release"),
            ResourceClaim(f"object:{object_uid}", lifetime="until_release"),
        )
        if release_role == "participant":
            return ResolvedActionContract(
                requires=(StateAtom("object_coordinated_held", object_uid=object_uid),),
                claims=claims,
            )
        return ResolvedActionContract(
            requires=(StateAtom("object_coordinated_held", object_uid=object_uid),),
            effects=(
                StateEffect(
                    "delete",
                    StateAtom("object_coordinated_held", object_uid=object_uid),
                ),
                StateEffect("add", StateAtom("object_free", object_uid=object_uid)),
                StateEffect("add", StateAtom("arm_free", arm="left_arm")),
                StateEffect("add", StateAtom("arm_free", arm="right_arm")),
            ),
            claims=claims,
        )
    if node.get("control") == "hand":
        return ResolvedActionContract(
            requires=(StateAtom("object_held", object_uid=object_uid, arm=arm),),
            claims=(
                ResourceClaim(f"arm:{arm}", lifetime="until_release"),
                ResourceClaim(f"object:{object_uid}", lifetime="until_release"),
            ),
        )
    if node.get("role") == "cleanup":
        required_home = binding.get("required_home", False)
        if not isinstance(required_home, bool):
            raise TypeError("joint_state required_home must be a boolean.")
        return ResolvedActionContract(
            requires=(StateAtom("arm_clear", arm=arm),),
            effects=(
                StateEffect("add", StateAtom("arm_home", arm=arm)),
                StateEffect("add", StateAtom("arm_free", arm=arm)),
            ),
            claims=(ResourceClaim(f"arm:{arm}"),),
            completion="terminal_barrier",
            failure_policy="safety_required" if required_home else "best_effort",
        )
    return ResolvedActionContract(
        requires=(StateAtom("arm_free", arm=arm),),
        claims=(ResourceClaim(f"arm:{arm}"),),
    )


def _verify_arm_clearance(
    *,
    executor: Any,
    step: Any,
    arm: str,
    outcome: Any,
    attempted: torch.Tensor,
) -> torch.Tensor:
    """Verify a released TCP is clear, plus the transfer side for handover."""
    policy = outcome.grounded.motion_policy
    object_uid = policy.get("clearance_object_uid")
    if not isinstance(object_uid, str) or not object_uid:
        return attempted
    transfer_arm = str(policy.get("transfer_arm", arm))
    if transfer_arm not in {"left_arm", "right_arm"}:
        return torch.zeros_like(attempted)
    entity = executor.env.sim.get_rigid_object(object_uid)
    getter = getattr(executor.env, "get_current_xpos_agent", None)
    if entity is None or not callable(getter):
        return torch.zeros_like(attempted)
    left, right = getter()
    eef = torch.as_tensor(
        left if transfer_arm == "left_arm" else right,
        dtype=torch.float32,
        device=executor.env.device,
    )
    if eef.ndim == 2:
        eef = eef.unsqueeze(0).repeat(int(executor.env.num_envs), 1, 1)
    object_pose = torch.as_tensor(
        entity.get_local_pose(to_matrix=True),
        dtype=torch.float32,
        device=executor.env.device,
    )
    if object_pose.ndim == 2:
        object_pose = object_pose.unsqueeze(0).repeat(int(executor.env.num_envs), 1, 1)
    offset = eef[:, :3, 3] - object_pose[:, :3, 3]
    distance = torch.linalg.vector_norm(offset, dim=1)
    minimum_clearance = policy.get(
        "minimum_clearance",
        policy.get("minimum_transfer_clearance", 0.10),
    )
    clear = distance >= float(minimum_clearance)
    role_axis = policy.get("transfer_role_axis")
    if role_axis is None:
        return attempted & clear
    role_axis = torch.as_tensor(
        role_axis,
        dtype=offset.dtype,
        device=offset.device,
    )
    if role_axis.ndim == 1:
        role_axis = role_axis.unsqueeze(0).repeat(int(executor.env.num_envs), 1)
    lateral = torch.sum(offset * role_axis, dim=1)
    clear &= lateral >= float(policy.get("minimum_transfer_lateral_clearance", 0.06))
    return attempted & clear


def _actor_arms(actor: Mapping[str, Any]) -> tuple[str, ...]:
    mode = str(actor.get("mode", "auto"))
    if mode == "coordinated":
        arms = actor.get("arms", ())
        if not isinstance(arms, (list, tuple)):
            raise ValueError("Coordinated actor arms must be a sequence.")
        result = tuple(str(item) for item in arms)
        if len(result) < 2 or any(not item for item in result):
            raise ValueError("Coordinated actor requires at least two named arms.")
        return result
    if mode in {"required", "preferred"}:
        return (_required_string(actor.get("arm"), "actor.arm"),)
    return ("auto",)


def _coordinated_arms(arms: tuple[str, ...], action: str) -> tuple[str, ...]:
    if len(arms) < 2:
        raise ValueError(f"{action} requires a coordinated actor.")
    return arms


def _required_arm(arm: str | None, action: str) -> str:
    if arm is None:
        raise ValueError(f"{action} requires exactly one arm.")
    return arm


def _required_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _callable_name(value: Callable[..., Any] | None) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", "")
    name = getattr(
        value, "__qualname__", getattr(value, "__name__", type(value).__name__)
    )
    return f"{module}.{name}" if module else str(name)
