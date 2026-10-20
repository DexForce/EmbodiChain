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

"""Derive invocation-local rigid manipulation permissions from semantic recipes."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import TYPE_CHECKING, Any

from .drawer_binding import inside_parts

if TYPE_CHECKING:
    from ..orchestration.source_scene import PreparedScene

__all__: list[str] = []


def _bounds(item: Mapping[str, Any]) -> tuple | None:
    value = item.get("attributes", {}).get("final_world_aabb", {})
    low, high = value.get("min", ()), value.get("max", ())
    if (
        len(low) != 3
        or len(high) != 3
        or not all(
            math.isfinite(a) and math.isfinite(b) and a < b
            for a, b in zip(low, high, strict=True)
        )
    ):
        return None
    return low, high


def _confined_source(subject: Mapping[str, Any], objects: tuple) -> bool:
    """Request planning qualification, not infer containment or contact state."""
    bounds = _bounds(subject)
    if bounds is None:
        return False
    for item in objects:
        if item.get("role") != "articulation":
            continue
        enclosing = _bounds(item)
        if enclosing is not None and all(
            enclosing[0][i] <= bounds[0][i] and bounds[1][i] <= enclosing[1][i]
            for i in range(3)
        ):
            return True
    return False


def select_manipulation_policy(
    graph: Mapping[str, Any],
    candidate: Mapping[str, Any],
    scene: PreparedScene,
) -> dict[str, Any]:
    """Select existing capabilities only for unconstrained ordinary rigid-inside pairs.

    Permissions come from the original typed step, not task names or the mere
    presence of another recipe. Runtime native geometry and all physical gates
    still qualify each proposal. Direct bundle callers keep explicit control.
    """
    steps = {step["id"]: step for step in candidate["draft"]["steps"]}
    by_id = {node["id"]: node for node in graph["nodes"]}

    def ancestors(node: Mapping[str, Any]) -> set[str]:
        seen, pending = set(), list(node["depends_on"])
        while pending:
            identity = pending.pop()
            if identity not in seen:
                seen.add(identity)
                pending.extend(by_id.get(identity, {}).get("depends_on", ()))
        return seen

    rigids = {
        item.get("runtime_uid", item.get("uid")): item
        for item in scene.planner_objects
        if item.get("role") == "rigid_object"
    }
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for node in graph["nodes"]:
        if node["task_type"] == "E1":
            groups.setdefault(node["task_instance_id"], []).append(node)
    pairs = []
    for group, nodes in sorted(groups.items()):
        step = steps.get(group)
        if (
            step is None
            or step["task_type"] != "E1"
            or any(
                step.get(key, "none") != "none"
                for key in ("orientation_goal", "axis", "direction", "layout")
            )
        ):
            continue
        picks = [node for node in nodes if node["call"]["kind"] == "pick"]
        places = [
            node
            for node in nodes
            if node["call"]["kind"] == "place" and "inside" in node["call"]
        ]
        if len(picks) != 1 or len(places) != 1:
            continue
        pick, place = picks[0], places[0]
        source, release = pick["call"], place["call"]
        if source.get("grasp") or source["object"] not in rigids:
            continue
        receiver, subject, part = inside_parts(release["inside"])
        if (
            part
            or receiver not in rigids
            or subject != source["object"]
            or release["object"] != subject
            or source["resources"] != release["resources"]
            or set(source["resources"]) != {"primary"}
            or pick["id"] not in place["depends_on"]
        ):
            continue
        origin = step.get("object", {})
        previous = steps.get(origin.get("step_id"), {})
        upright_result = (
            origin.get("kind") == "step_result"
            and previous.get("task_type") == "E2"
            and previous.get("orientation_goal") == "upright"
        )
        source_untouched = not any(
            by_id[identity]["call"].get(
                "object", by_id[identity]["call"].get("arguments", {}).get("object")
            )
            == subject
            for identity in ancestors(pick)
            if identity in by_id
        )
        confined = (
            not upright_result
            and source_untouched
            and _confined_source(rigids[subject], tuple(scene.planner_objects))
        )
        pairs.append((pick, place, receiver, upright_result, confined))
    contact, yaw, measured = {}, [], []
    for pick, place, receiver, upright_result, confined in pairs:
        yaw.append(place["id"])
        prior_nodes = ancestors(place)
        prior_confined_transfer = any(
            prior_confined
            and prior_receiver == receiver
            and prior_place["id"] in prior_nodes
            for _, prior_place, prior_receiver, _, prior_confined in pairs
        )
        if confined or prior_confined_transfer:
            measured.append(place["id"])
        if upright_result:
            # A released upright result retains the qualified contact-halo screen;
            # do not combine its revision-2 closing with revision-4 preshape.
            contact[pick["id"]] = 2
        elif confined:
            measured.append(pick["id"])
    result: dict[str, Any] = {}
    if contact:
        result["pick_contact_revisions"] = dict(sorted(contact.items()))
    if yaw:
        result["world_yaw_free_places"] = sorted(yaw)
    if measured:
        result["measured_rigid_calls"] = sorted(measured)
    return result
