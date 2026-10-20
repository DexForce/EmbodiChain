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

"""Explicit GenSim experiment policy, independent of task and scene identities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__: list[str] = []

CONTACT_PROFILE = "contact_experiment"
COMPOSITION_PROFILE = "composition_experiment"
AUTO_PROFILE = "auto"


def validate_profile(value: str) -> None:
    if value not in {"default", AUTO_PROFILE, CONTACT_PROFILE, COMPOSITION_PROFILE}:
        raise ValueError(
            "runtime_profile must be default, auto, contact_experiment or composition_experiment."
        )


def resolve_profile(value: str, graph: Mapping[str, Any]) -> str:
    """Select one bundle-owned policy from recipe semantics, never object identities."""
    validate_profile(value)
    if value != AUTO_PROFILE:
        return value
    pouring = any(
        node.get("task_type") == "E3"
        or node.get("call", {}).get("call_id") == "simulation.pour"
        for node in graph["nodes"]
    )
    return COMPOSITION_PROFILE if pouring else CONTACT_PROFILE


def experimental_release_enabled(profile: str, override: bool | None) -> bool:
    """Keep historical bundles readable while allowing an explicit release gate."""
    validate_profile(profile)
    if override is not None and type(override) is not bool:
        raise TypeError("experimental_open_release must be a boolean or None.")
    return (
        profile in {CONTACT_PROFILE, COMPOSITION_PROFILE}
        if override is None
        else override
    )


def merge_contact_revisions(
    selected: Mapping[str, int] | None, generated: Mapping[str, int]
) -> dict[str, int]:
    """Compose independent occurrence permissions without weakening either source."""
    from ._task_program.adaptive_grasp import contact_revision_declarations

    if selected is not None and not isinstance(selected, Mapping):
        raise ValueError("Selected Pick contact revisions must be a mapping.")
    result = contact_revision_declarations(dict(selected or {}))
    additional = contact_revision_declarations(dict(generated))
    for identity, revision in additional.items():
        if identity in result and result[identity] != revision:
            raise ValueError(
                f"Conflicting Pick contact revisions for {identity}: "
                f"selected {result[identity]} versus generated {revision}."
            )
        result[identity] = revision
    return dict(sorted(result.items()))


def program_completion_mask(
    runtime: Mapping[str, Any] | None, count: int
) -> list[bool]:
    """Require complete measured execution, not a TaskSpec certificate."""
    if not runtime or count != 1:
        return [False] * count
    segments = runtime.get("segments") or []
    accepted = (
        runtime.get("completed") is True
        and runtime.get("completed_by_env") == [True]
        and runtime.get("success") == [True]
        and len(runtime.get("lengths", [])) == 1
        and type(runtime["lengths"][0]) is int
        and runtime["lengths"][0] > 0
        and bool(segments)
        and all(
            segment.get("success") is True
            and segment.get("metadata", {}).get("segment_count") == len(segments)
            and not segment.get("metadata", {}).get("runtime", {}).get("failures")
            and segment.get("metadata", {}).get("validation", {}).get("accepted_mask")
            == [True]
            for segment in segments
        )
    )
    return [bool(accepted)]


def program_completion_summary(
    runtime: Mapping[str, Any] | None,
    strict_report: Mapping[str, Any],
    experiment_report: Mapping[str, Any],
    *,
    acceptance_mode: str,
) -> dict[str, Any]:
    """Report both outcomes from one execution without altering its selected verdict."""
    environments = strict_report["environments"]
    count = len(environments)
    completed = program_completion_mask(runtime, count)
    experiment_rows = experiment_report["environments"]
    certificates = {
        row.get("env_id", index): row.get("certificate_ready") is True
        and row.get("status") == "satisfied"
        for index, row in enumerate(
            ((runtime or {}).get("task_spec_evaluation") or {}).get("environments", [])
        )
    }
    return {
        "schema_version": "gen_sim.program-completion-summary/v1",
        "selected_acceptance_mode": acceptance_mode,
        "experimental_program_completed_by_env": [
            value
            and experiment_report.get("failure") is None
            and row["success"] is True
            and all(flag is True for flag in row.get("semantic_success", {}).values())
            for value, row in zip(completed, experiment_rows, strict=True)
        ],
        "strict_execution_success_by_env": [row["success"] for row in environments],
        "task_spec_certificate_ready_by_env": [
            certificates.get(row.get("env_id", index), False)
            for index, row in enumerate(environments)
        ],
        "affects_selected_success": False,
    }


def apply_contact_embodiment(embodiment: dict[str, Any]) -> None:
    """Use Spawn's material-only overlay; never rewrite mass or inertial data."""
    simulation = embodiment["simulation"]
    overlays = simulation.setdefault("link_attrs", {})
    if "gen_sim_contact_pads" in overlays:
        raise ValueError("The contact pad material overlay is already declared.")
    overlays["gen_sim_contact_pads"] = {
        "link_names_expr": [".*inner_finger_pad"],
        "attrs": {
            "material_props": {"static_friction": 2.85, "dynamic_friction": 2.70}
        },
    }
    simulation["asset_physics_mode"] = "overlay"


def apply_contact_options(integration: dict[str, Any]) -> None:
    """Retain the qualified slow-close contact experiment, not composition defaults."""
    options = integration["profile"]["action_options"]
    for call_id, value in options.items():
        if call_id == "pick" and value.get("kind") == "pick_up":
            value["hand_interp_steps"] = max(24, value.get("hand_interp_steps", 0))


def apply_placement_tolerances(
    constraints: dict[str, Any], program: dict[str, Any]
) -> None:
    """Apply the declared experimental tolerances to all supported on placements."""
    for cfg in constraints["presets"].values():
        if cfg.get("kind") == "supported_placement":
            cfg.update(
                position_tolerance=0.10, support_tolerance=0.03, com_tolerance=0.10
            )
    for item in program["program"]["items"]:
        for validator in item.get("validators", []):
            if validator.get("kind") == "object_near_relative_target":
                validator["position_tolerance"] = 0.10


def horizontal_handover_permissions(
    graph: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    """Derive permissions from the end-grasp recipe, never node or object names."""
    objects = {
        n["call"]["arguments"]["object"]
        for n in graph["nodes"]
        if n["call"]
        .get("call_id", "")
        .startswith("gen_sim.pick.handover_horizontal_source.")
    }
    revisions = {
        n["id"]: 8
        for n in graph["nodes"]
        if n["call"]
        .get("call_id", "")
        .startswith("gen_sim.pick.handover_horizontal_source.")
    }
    places = []
    for previous, node in zip(graph["nodes"], graph["nodes"][1:]):
        before, call = previous["call"], node["call"]
        args = call.get("arguments", {})
        if (
            call.get("call_id") == "simulation.place_relative"
            and args.get("object") in objects
            and args.get("relation") == "on"
            and before["kind"] == "hand_over"
            and before.get("object") == args["object"]
            and before.get("resources", {}).get("destination")
            == call.get("resources", {}).get("primary")
        ):
            places.append(node["id"])
    return places, revisions


def add_reacquisition_clearance(
    graph: dict[str, Any], embodiment: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Clear an otherwise unused arm before a terminal same-object reacquisition."""
    from copy import deepcopy

    from ._task_program.idle_clearance import CALL_ID

    result = deepcopy(graph)
    nodes = result["nodes"]
    group = result["task_groups"][-1]
    members = [node for node in nodes if node["id"] in group["node_ids"]]
    first, last = members[0], members[-1]
    call = first["call"]
    if group["task_type"] != "E1" or call["kind"] != "pick":
        return graph, []
    prefix = nodes[: nodes.index(first)]
    if not any(
        n["call"].get("object") == call["object"] and n["call"]["kind"] == "pick"
        for n in prefix
    ):
        return graph, []
    resources = {r["resource_id"]: r for r in embodiment["skill_profile"]["resources"]}
    active = call["resources"]["primary"]
    unused = set(resources) - {active}
    if len(unused) != 1:
        return graph, []
    idle = unused.pop()
    last_use = next(
        (
            node
            for node in reversed(prefix)
            if idle in node["call"].get("resources", {}).values()
        ),
        None,
    )
    if last_use is not None and last_use["call"].get("call_id") != "simulation.park":
        return graph, []

    def motion(resource):
        return next(
            e["control_part"]
            for e in resources[resource]["endpoints"]
            if e["endpoint_id"] == "motion"
        )

    clear_id, restore_id = (
        first["id"] + "__idle_clearance",
        last["id"] + "__idle_restore",
    )
    route_id = "clear_" + group["id"] + "_" + idle
    clear, restore = deepcopy(first), deepcopy(last)
    clear.update(
        id=clear_id,
        role="idle_clearance",
        call={
            "kind": "registered",
            "call_id": CALL_ID,
            "arguments": {"route": route_id},
            "resources": {"primary": idle},
        },
    )
    restore.update(
        id=restore_id,
        role="idle_restore",
        depends_on=[last["id"]],
        call={
            "kind": "registered",
            "call_id": "simulation.park",
            "arguments": {},
            "resources": {"primary": idle},
        },
    )
    first["depends_on"] = [clear_id]
    start, end = nodes.index(first), nodes.index(last)
    result["nodes"] = [
        *nodes[:start],
        clear,
        *nodes[start : end + 1],
        restore,
        *nodes[end + 1 :],
    ]
    group["node_ids"] = [clear_id, *group["node_ids"], restore_id]
    route = {
        "route_id": route_id,
        "clear_segment": clear_id,
        "restore_segment": restore_id,
        "resource": idle,
        "joint_name": embodiment["simulation"]["control_parts"][motion(idle)][0],
        "offset_rad": 0.6,
        "search": True,
        "protected_control_part": motion(active),
    }
    return result, [route]
