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

"""Declare source Pour routes without planning or native geometry queries."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
from copy import deepcopy
from embodichain.lab.sim.scene.declaration_placement import source_pour_target_pose
from embodichain.lab.task_program.integrations.hand_calibration import (
    TaskHandCalibration,
)

__all__ = ["resolve_pour_lowerer_declarations", "supports_ordinary_pour"]


def _pour_return_call_declarations(graph: Mapping[str, Any]) -> list[str]:
    """Select the first absolute Place after the same source recipe's Pour."""
    pending: set[tuple[str, str]] = set()
    result = []
    for node in graph["nodes"]:
        if node["task_type"] != "E3":
            continue
        call = node["call"]
        if call.get("call_id") == "simulation.pour":
            pending.add((node["task_instance_id"], call["arguments"]["object"]))
        elif call["kind"] == "place" and call.get("at") is not None:
            key = (node["task_instance_id"], call["object"])
            if key in pending:
                result.append(node["id"])
                pending.remove(key)
    return result


def resolve_pour_lowerer_declarations(
    graph: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    embodiment: Mapping[str, Any],
    hands: Sequence[TaskHandCalibration],
    *,
    geometry_revision: int | None = None,
    proxy_position_tolerance_m: float | None = None,
) -> list[dict[str, Any]]:
    """Preserve original held transport and static/retained Pour declarations.

    Args:
        graph: Canonical ordered semantic graph with frozen return targets.
        objects: Source physical declarations by exact semantic identity.
        embodiment: Selected source embodiment owning grasp model openings.
        hands: Model-owned explicit resource and nominal geometry bindings.
        geometry_revision: None retains static source staging; 1 and 2 enable
            source candidate-dependent geometry and receiver-region selection.
        proxy_position_tolerance_m: Explicit revision-2 proxy budget.

    Returns:
        Closed shared transport/Pour lowerer declarations. Source object and
        target identities, call order and world-relative poses are unchanged.
    """
    by_resource = {hand.resource_id: hand for hand in hands}
    routes, staging, poured = [], [], []
    for node in graph["nodes"]:
        call = node["call"]
        identity = call.get("call_id")
        if identity == "simulation.pour":
            object_id = call["arguments"]["object"]
            if object_id not in poured:
                poured.append(object_id)
        if identity != "simulation.move_held_object":
            continue
        args = call["arguments"]
        object_id, target_id = args["object"], args["target"]
        if node["task_type"] == "E2" or "reference" not in args:
            target = graph["targets"][target_id]
            if target["kind"] != "cyclic_pose" or len(target["values"]) != 1:
                raise ValueError(
                    "Source held transport requires one frozen target pose."
                )
            pose = {"kind": "pose", **deepcopy(target["values"][0])}
        else:
            receiver = args["reference"]
            hand = by_resource[call.get("resources", {}).get("primary", "left")]
            model = embodiment["skill_profile"]["runtime_services"][
                "grasp_pose_generators"
            ][hand.grasp_target_id]["model"]
            from embodichain.toolkits.graspkit import resolve_parallel_jaw_gripper_model

            radius = hand.nominal_pour_radius(
                resolve_parallel_jaw_gripper_model(model).max_opening_width
            )
            pose = {"kind": "scene_entity", "entity_id": receiver}
            if geometry_revision in (1, 2):
                if node["task_type"] != "E3":
                    raise ValueError("Canonical Pour staging must belong to E3.")
                staging.append(
                    dict(
                        object_id=object_id,
                        target_id=target_id,
                        receiver_id=receiver,
                        clearance_m=radius + 0.015,
                    )
                )
            else:
                value, _ = source_pour_target_pose(
                    objects[object_id], objects[receiver], finger_radius=radius
                )
                pose.update(value)
        routes.append(dict(object_id=object_id, target_id=target_id, pose=pose))
    if len(poured) > 1:
        raise ValueError("The source bundle supports at most one Pour object.")
    result = []
    if routes:
        transport = dict(
            kind="move_held_object",
            routes=routes,
            effect_contract="terminal_attachment",
        )
        if staging:
            transport.update(
                kind=(
                    "pour_region_transport"
                    if geometry_revision == 2
                    else "pour_transport"
                ),
                staging_routes=staging,
            )
            if geometry_revision == 2:
                transport["proxy_position_tolerance_m"] = proxy_position_tolerance_m
        result.append(transport)
    if poured:
        result.append(
            dict(
                kind="retained_pour" if geometry_revision in (1, 2) else "jaw_pour",
                object_id=poured[0],
            )
        )
    return result


def supports_ordinary_pour(
    graph: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    """Select only independent, unconstrained E3 recipes before opt-in.

    Inputs are validated planner outputs. This capability check owns the recipe
    shape, not grounding: public configured preflight validates exact typed fields.
    Already-held continuations stay on their existing legacy route.
    """
    pours = [step for step in candidate["draft"]["steps"] if step["task_type"] == "E3"]
    groups = [group for group in graph["task_groups"] if group["task_type"] == "E3"]
    if (
        not pours
        or not groups
        or any(
            (
                step.get(field, "none") != "none"
                for step in pours
                for field in ("orientation_goal", "axis", "direction")
            )
        )
    ):
        return False
    nodes = {node["id"]: node for node in graph["nodes"]}
    positions = {node["id"]: index for (index, node) in enumerate(graph["nodes"])}
    expected = (
        "pick",
        "simulation.move_held_object",
        "simulation.pour",
        "simulation.move_held_object",
        "place",
    )
    for group in groups:
        selected = [nodes[identifier] for identifier in group["node_ids"]]
        while selected and selected[-1]["call"].get("call_id") == "simulation.park":
            selected.pop()
        calls = [node["call"] for node in selected]
        if (
            len(calls) != 5
            or tuple((call.get("call_id", call["kind"]) for call in calls)) != expected
            or positions[selected[-1]["id"]] - positions[selected[0]["id"]] != 4
        ):
            return False
        resources = calls[0].get("resources", {})
        subject = calls[0]["object"]
        if (
            set(resources) != {"primary"}
            or any((call.get("resources", {}) != resources for call in calls[1:]))
            or any((call["arguments"].get("object") != subject for call in calls[1:4]))
            or (calls[4]["object"] != subject)
            or (not calls[1]["arguments"].get("reference"))
            or ("reference" in calls[3]["arguments"])
            or (calls[4].get("at") is None)
        ):
            return False
    return True
