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

"""Provider-free planning declarations shared with configured call lowerers.

Only JSON metadata lives here. Runtime factories own metric calibration, scene
resolution, policy and Atomic Skill construction.
"""

from __future__ import annotations
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "SemanticCallContract",
    "semantic_planning_catalog",
    "validate_semantic_call_payload",
]


@dataclass(frozen=True, slots=True)
class SemanticCallContract:
    """Immutable, simulator-free argument and reference contract.

    Args:
        required: Required argument names and their reference/type rules.
        optional: Optional argument names and their reference/type rules.
        resource_slots: Permitted logical resource slots.
        preconditions: Declared symbolic input conditions.
        effects: Declared symbolic output conditions.
    """

    required: Mapping[str, str | tuple[str, ...]]
    optional: Mapping[str, str | tuple[str, ...]]
    resource_slots: tuple[str, ...] = ("primary",)
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for declarations in (self.required, self.optional):
            for name, rule in declarations.items():
                if type(name) is not str or not name or name != name.strip():
                    raise ValueError(
                        "Contract argument names must be nonempty identifiers."
                    )
                if type(rule) is str:
                    if not rule:
                        raise ValueError("Contract rules must be nonempty.")
                elif (
                    type(rule) not in (list, tuple)
                    or not rule
                    or any(type(value) is not str or not value for value in rule)
                ):
                    raise TypeError(
                        "Contract rules must be symbolic type names or finite string enumerations."
                    )
        if set(self.required) & set(self.optional):
            raise ValueError("Required and optional arguments must be disjoint.")
        object.__setattr__(
            self,
            "required",
            MappingProxyType(
                {
                    key: tuple(rule) if isinstance(rule, (list, tuple)) else rule
                    for key, rule in self.required.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "optional",
            MappingProxyType(
                {
                    key: tuple(rule) if isinstance(rule, (list, tuple)) else rule
                    for key, rule in self.optional.items()
                }
            ),
        )
        for name in ("resource_slots", "preconditions", "effects"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_metadata(self) -> dict[str, Any]:
        """Return independently owned planning metadata."""
        return dict(
            required={
                key: list(rule) if isinstance(rule, tuple) else rule
                for key, rule in self.required.items()
            },
            optional={
                key: list(rule) if isinstance(rule, tuple) else rule
                for key, rule in self.optional.items()
            },
            resource_slots=list(self.resource_slots),
            preconditions=list(self.preconditions),
            effects=list(self.effects),
        )


_RELATIONS = ["on", "above", "behind", "front_of", "left_of", "right_of"]


def semantic_planning_catalog(
    contracts: Mapping[str, SemanticCallContract] | None = None,
) -> dict[str, Any]:
    """Return detached symbolic call contracts without importing a simulator.

    Returns:
        JSON catalog of actual built-ins and configured lowerer families.
    """
    definitions = [
        (
            "pick",
            "pick_up",
            {"object": "entity"},
            {"grasp": "affordance"},
            ["primary"],
            "acquire",
            ["grasp_affordance"],
        ),
        (
            "place",
            "place",
            {"object": "entity"},
            {"on": "affordance", "inside": "affordance", "at": "target_ref"},
            ["primary"],
            "release",
            ["destination_affordance", "placement_policy"],
        ),
        (
            "hand_over",
            "hand_over",
            {"object": "entity"},
            {"final_target": "target_ref"},
            ["source", "destination"],
            "transfer",
            ["handover_geometry"],
        ),
        (
            "simulation.axis_align",
            "axis_align",
            {"object": "entity"},
            {},
            ["primary"],
            "acquire_align",
            ["axis_alignment_policy"],
        ),
        (
            "simulation.articulation_link_slide",
            "slide",
            {"handle": "part"},
            {},
            ["primary"],
            "control",
            ["articulation_joint", "operation_policy"],
        ),
        (
            "simulation.open_door",
            "open_door",
            {"object": "part", "state": ["open", "closed"]},
            {},
            ["primary"],
            "control",
            ["articulation_joint"],
        ),
        (
            "simulation.press",
            "press",
            {"object": "part"},
            {},
            ["primary"],
            "control",
            ["press_affordance", "operation_policy"],
        ),
        (
            "simulation.twist",
            "twist",
            {"object": "part"},
            {},
            ["primary"],
            "control",
            ["twist_affordance", "operation_policy"],
        ),
        (
            "simulation.pour",
            "pour",
            {"object": "entity"},
            {},
            ["primary"],
            "held",
            ["pour_target", "pour_policy"],
        ),
        (
            "simulation.move_held_object",
            "move_held_object",
            {"target": "target"},
            {"object": "entity", "reference": "entity"},
            ["primary"],
            "held",
            ["relative_target_pose"],
        ),
        (
            "simulation.place_relative",
            "place",
            {"object": "entity", "reference": "entity", "relation": _RELATIONS},
            {"stage_id": "identifier"},
            ["primary"],
            "release",
            ["relative_placement_route"],
        ),
        (
            "simulation.coordinated_transport",
            "coordinated_pickment",
            {"object": "entity", "target": "target"},
            {},
            ["left", "right"],
            "coordinated_release",
            ["paired_grasp", "transport_target"],
        ),
        (
            "simulation.push_object",
            "push_object",
            {"object": "entity", "target": "target"},
            {},
            ["primary"],
            "control",
            ["push_target"],
        ),
        (
            "simulation.park",
            "move_joints",
            {},
            {},
            ["primary"],
            "preserve",
            ["park_posture"],
        ),
    ]
    calls = {}
    for cid, skill, required, optional, slots, effect, runtime in definitions:
        calls[cid] = dict(
            skill=skill,
            required=required,
            optional=optional,
            resource_slots=slots,
            effect=effect,
            runtime_bindings=runtime + ["resource_profile"],
        )
    capabilities = {
        "pick": ["graspable"],
        "place": ["placeable"],
        "hand_over": ["handover"],
        "simulation.axis_align": ["orientable"],
        "simulation.articulation_link_slide": ["slideable"],
        "simulation.open_door": ["openable"],
        "simulation.press": ["pressable"],
        "simulation.twist": ["turnable"],
        "simulation.pour": ["pourable"],
        "simulation.place_relative": ["placeable"],
        "simulation.coordinated_transport": ["dual_graspable"],
        "simulation.push_object": ["pushable"],
        "simulation.move_held_object": ["transportable"],
        "simulation.park": [],
    }
    for cid, value in calls.items():
        value["capabilities"] = capabilities[cid]
    calls["hand_over"]["required_policy"] = {"release_at_target": False}
    calls["hand_over"]["planning_restrictions"] = {
        "explicit_resources": ["source", "destination"]
    }
    calls["hand_over"]["runtime_bindings"].append("handover_policy")
    preconditions = {
        "acquire": ["object_unheld", "primary_free"],
        "acquire_align": ["object_unheld", "primary_free"],
        "release": ["object_held_by_primary"],
        "held": ["object_held_by_primary"],
        "transfer": ["object_held_by_source", "destination_free"],
        "control": ["primary_free"],
        "coordinated_release": ["both_arms_free"],
        "preserve": [],
    }
    for value in calls.values():
        value["preconditions"] = preconditions[value["effect"]]
    calls["simulation.pour"][
        "description"
    ] = "Tilt and restore source orientation while preserving its grasp. Never use axis_align on this held source. Original position must be restored separately via a source-pose binding."
    calls["simulation.axis_align"][
        "description"
    ] = "Acquire an UNHELD object and align its axis. Not usable as in-hand rotation after pouring."
    calls["simulation.coordinated_transport"][
        "description"
    ] = "Both arms acquire, transport and RELEASE the object. No separate placement afterward."
    calls["move_held_object"] = dict(
        skill="move_held_object",
        required={"object": "entity", "at": "target_ref"},
        optional={},
        resource_slots=["primary"],
        effect="held",
        preconditions=["object_held_by_primary"],
        runtime_bindings=["resource_profile"],
        capabilities=["transportable"],
    )
    if contracts is not None:
        calls = {
            cid: {**calls.get(cid, {}), **contract.to_metadata()}
            for cid, contract in contracts.items()
        }
    return {"schema_version": "semantic_planning_catalog/v1", "calls": deepcopy(calls)}


def validate_semantic_call_payload(
    call: dict[str, Any],
    *,
    contracts: Mapping[str, SemanticCallContract] | None = None,
) -> dict[str, Any]:
    """Validate canonical arguments against the shared static call contract.

    Args:
        call: Canonical built-in or registered Semantic Call JSON.
    Returns:
        A detached selected call declaration.
    Raises:
        ValueError: Unknown calls, extra/missing arguments or invalid types.
    """
    cid = call.get("call_id") if call.get("kind") == "registered" else call.get("kind")
    spec = semantic_planning_catalog(contracts)["calls"].get(cid)
    if spec is None:
        raise ValueError(f"Unknown semantic call: {cid}")
    if call.get("kind") == "registered":
        if set(call) - {"kind", "call_id", "arguments", "resources"}:
            raise ValueError(f"{cid}: unexpected envelope fields")
        args = call.get("arguments", {})
    else:
        args = {k: v for k, v in call.items() if k not in {"kind", "resources"}}
    if (
        type(args) is not dict
        or set(spec["required"]) - set(args)
        or set(args) - set(spec["required"]) - set(spec["optional"])
    ):
        raise ValueError(
            f"{cid} arguments must contain only {', '.join(repr(k) for k in spec['required'])}; optional keys {list(spec['optional'])}. Received keys: {list(args) if type(args) is dict else type(args).__name__}"
        )
    for key, value in args.items():
        if (
            value is None
            and cid in {"pick", "place", "hand_over"}
            and key in {"grasp", "final_target", "at", "on", "inside"}
            and key in spec["optional"]
        ):
            continue
        rule = {**spec["required"], **spec["optional"]}[key]
        if rule == "target_ref":
            valid = (
                type(value) is dict
                and set(value) == {"kind", "target"}
                and value["kind"] == "target_ref"
                and isinstance(value["target"], str)
                and bool(value["target"])
                and value["target"] == value["target"].strip()
            )
        elif isinstance(rule, list):
            valid = value in rule
        else:
            valid = type(value) is str and bool(value) and value == value.strip()
        if not valid:
            raise ValueError(f"{cid}.{key}: invalid argument {value!r}")
    if (
        cid == "place"
        and sum(args.get(key) is not None for key in ("on", "inside", "at")) != 1
    ):
        raise ValueError("place requires exactly one destination")
    resources = call.get("resources", {})
    if (
        type(resources) is not dict
        or set(resources) - set(spec["resource_slots"])
        or any(
            type(v) is not str or not v or v != v.strip() for v in resources.values()
        )
    ):
        raise ValueError(
            f"{cid}: invalid resources {resources}; permitted slot keys: {spec['resource_slots']}. Use primary:left for single-arm calls, never a key named slots."
        )
    if cid == "hand_over" and (
        (contracts is None and set(resources) != {"source", "destination"})
        or (
            set(resources) == {"source", "destination"}
            and resources["source"] == resources["destination"]
        )
    ):
        raise ValueError("hand_over requires distinct source and destination")
    return spec
