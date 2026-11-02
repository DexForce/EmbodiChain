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

"""Invocation-local native opening clearance, never measured release evidence."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import trimesh

from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.task_program.semantics import (
    Place,
    RegisteredSemanticCall,
    SceneAffordanceRef,
    SceneObjectRef,
)
from .adaptive_grasp import ADAPTIVE_GRASP

if TYPE_CHECKING:
    from .pick_contact import ClosingContactScreen

__all__: list[str] = []


def _pose(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (4, 4)
        or not np.isfinite(result).all()
        or not np.allclose(result[3], (0, 0, 0, 1))
        or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=2e-5)
        or not np.isclose(np.linalg.det(result[:3, :3]), 1.0, atol=2e-5)
    ):
        raise ValueError("Opening requires a finite rigid pose.")
    return result.copy()


class OpeningContactScreen:
    """Check the qualified source-contact-to-open grid against one native support."""

    def __init__(
        self,
        closing: ClosingContactScreen,
        reference_meshes: tuple[Any, ...],
        reference_pose: np.ndarray,
        object_target_pose: np.ndarray,
        reference_rest_offset: float,
        reference_contact_offset: float,
        *,
        identity: dict[str, Any],
    ) -> None:
        if not reference_meshes or not closing.parts:
            raise ValueError("Opening requires complete native hand/support meshes.")
        offsets = [(reference_rest_offset, reference_contact_offset)] + [
            (part.rest_offset, part.contact_offset) for part in closing.parts
        ]
        if any(
            contact is None
            or not math.isfinite(rest)
            or not math.isfinite(contact)
            or not 0 <= rest <= contact
            for rest, contact in offsets
        ):
            raise ValueError("Opening requires original native rest/contact offsets.")
        self.closing = closing
        self.object_target_pose = _pose(object_target_pose)
        self.reference_pose = _pose(reference_pose)
        self.identity = deepcopy(identity)
        self.reference = trimesh.collision.CollisionManager()
        for index, mesh in enumerate(reference_meshes):
            self.reference.add_object(str(index), mesh, transform=self.reference_pose)
        self.reference_shape_count = len(reference_meshes)
        self.managers = []
        for part in closing.parts:
            manager = trimesh.collision.CollisionManager()
            for index, mesh in enumerate(part.meshes):
                manager.add_object(str(index), mesh)
            self.managers.append(manager)
        self.rest_offset = reference_rest_offset
        self.contact_offset = reference_contact_offset
        self.cache: dict[bytes, dict[str, Any]] = {}

    def inspect(
        self, grasp: np.ndarray, closing_record: dict[str, Any]
    ) -> dict[str, Any]:
        """Keep every original candidate; reject only its necessary opening geometry."""
        if not closing_record.get("valid"):
            return {"valid": False, "reason": "source_closing_unqualified"}
        stop = closing_record.get("checked_samples", 0) - 1
        if (
            type(stop) is not int
            or not 0 <= stop < self.closing.steps
            or closing_record.get("reason") != "nominal_bilateral_pad_rest_first"
        ):
            raise ValueError("Opening requires this source's first bilateral sample.")
        grasp = _pose(grasp)
        key = grasp.tobytes() + stop.to_bytes(8, "little")
        if key not in self.cache:
            terminal = self.object_target_pose @ grasp
            result: dict[str, Any] = {
                "valid": True,
                "reason": "nominal_opening_rest_clear",
                "source_bilateral_grid_index": stop,
                "terminal_tcp": terminal.tolist(),
                "checked_queries": 0,
                "contact_halo_links": [],
                "minimum_distance_m": None,
            }
            for step in range(stop, -1, -1):
                for part, manager in zip(
                    self.closing.parts, self.managers, strict=True
                ):
                    for index in range(len(part.meshes)):
                        manager.set_transform(
                            str(index), terminal @ part.tcp_poses[step]
                        )
                    distance = float(manager.min_distance_other(self.reference))
                    if not math.isfinite(distance):
                        raise ValueError("Opening native mesh distance is unavailable.")
                    result["checked_queries"] += 1
                    previous = result["minimum_distance_m"]
                    result["minimum_distance_m"] = (
                        distance if previous is None else min(previous, distance)
                    )
                    rest = self.rest_offset + part.rest_offset
                    contact = self.contact_offset + part.contact_offset
                    if (
                        distance <= contact
                        and part.name not in result["contact_halo_links"]
                    ):
                        result["contact_halo_links"].append(part.name)
                    if distance <= rest:
                        result.update(
                            valid=False,
                            reason="hand_support_rest_envelope_overlap",
                            link=part.name,
                            grid_index=step,
                            native_pair_rest_offset_m=rest,
                            native_pair_contact_offset_m=contact,
                        )
                        self.cache[key] = result
                        return deepcopy(result)
            self.cache[key] = result
        return deepcopy(self.cache[key])

    def report(self) -> dict[str, Any]:
        records = list(self.cache.values())
        return {
            "scope": "discrete_native_opening_prefix_not_physical_release",
            "physical_evidence": False,
            "valid": (
                None if not records else any(record["valid"] for record in records)
            ),
            "aggregation": "any_candidate_passed_not_physical_release",
            "identity": deepcopy(self.identity),
            "native_reference_shapes": self.reference_shape_count,
            "native_reference_rest_offset_m": self.rest_offset,
            "native_reference_contact_offset_m": self.contact_offset,
            "reference_pose": self.reference_pose.tolist(),
            "object_target_pose": self.object_target_pose.tolist(),
            "unique_queries": len(records),
            "accepted_queries": sum(record["valid"] for record in records),
            "rejected_queries": sum(not record["valid"] for record in records),
            "reasons": deepcopy(records),
        }


def build_opening_screen(
    closing_screen: ClosingContactScreen,
    robot: Any,
    simulation: Any,
    scene_binding: Any,
    request: Any,
    context: Any,
    preview: Any,
    *,
    purpose: str | None = None,
) -> OpeningContactScreen | None:
    """Bind only an ordinary same-object on-placement's final downstream pose."""
    if purpose != "ordinary" or preview is None:
        return None
    affordance = getattr(request.goal.semantics, "affordance", None)
    if affordance is not None and affordance.get_custom_config(ADAPTIVE_GRASP) in {
        "pour",
        "ends",
    }:
        return None
    route, subject = preview.route, request.goal.semantics.entity_id
    call = route.place_call
    if type(call) is Place:
        if call.on is None:
            return None
        target = call.on
        if type(target) is SceneAffordanceRef:
            matches = [
                b
                for b in scene_binding.support_surfaces
                if b.entity_id == target.entity_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Opening support affordance is unavailable or ambiguous."
                )
            parent = matches[0].parent_id
            if target.entity_id != f"on__{parent}__{subject}":
                return None
        elif type(target) is SceneObjectRef:
            parent = target.entity_id
        else:
            return None
        placed_subject = call.object.entity_id
    elif type(call) is RegisteredSemanticCall:
        arguments = call.arguments
        if (
            call.call_id != "simulation.place_relative"
            or arguments.get("relation") != "on"
            or arguments.get("reference_part") is not None
        ):
            return None
        parent, placed_subject = arguments.get("reference"), arguments.get("object")
    else:
        return None
    if placed_subject != subject or dict(call.resources) != dict(
        route.pick_call.resources
    ):
        return None
    if (
        set(call.resources) != {"primary"}
        or route.pick_id != request.invocation_id
        or route.pick_call.object.entity_id != subject
    ):
        raise ValueError("Opening continuation identity does not match this Pick.")
    if context is None or context.batch_size != 1 or context.env_ids.tolist() != [0]:
        raise ValueError("Opening requires the current B1/env0 context.")
    if (
        simulation is None
        or not simulation.is_default_backend
        or simulation.num_envs != 1
    ):
        raise ValueError("Opening requires the current B1 Default simulation.")
    references = [b for b in scene_binding.rigid_objects if b.entity_id == parent]
    if len(references) != 1:
        raise ValueError("Opening requires one exact rigid support parent.")
    targets = request.skill_options.downstream_object_target_poses
    if not targets:
        raise ValueError("Opening lacks the final Place object target.")
    target_pose = resolve_pose_goal(
        targets[-1], context, name="opening_final_place_target"
    )
    if target_pose.shape == (4, 4):
        target_pose = target_pose[None]
    reference_pose = context.scene.entities[parent].pose
    if (
        target_pose.shape != (1, 4, 4)
        or reference_pose is None
        or reference_pose.shape != (1, 4, 4)
    ):
        raise ValueError("Opening target/reference pose must match the current batch.")
    scene = simulation.spawn_result
    native_robot = scene.get(closing_screen.identity["native_robot_path"])
    if native_robot is None or native_robot.get_name() != robot.uid:
        raise ValueError("Opening native robot identity is unavailable.")
    handles = [scene.get(path) for path in scene.get_paths()]
    matches = [
        h
        for h in handles
        if h.get_name() == references[0].simulation_uid
        and h.arena_name == native_robot.arena_name
        and callable(getattr(h, "native", None))
    ]
    if len(matches) != 1:
        raise ValueError("Opening native support identity is unavailable or ambiguous.")
    from .pick_contact import _native_meshes

    body = matches[0].native().get_physical_body()
    return OpeningContactScreen(
        closing_screen,
        _native_meshes(body),
        reference_pose[0].detach().cpu().numpy(),
        target_pose[0].detach().cpu().numpy(),
        float(body.get_rest_offset()),
        float(body.get_contact_offset()),
        identity={
            "reference_entity_id": parent,
            "reference_simulation_uid": references[0].simulation_uid,
            "native_reference_path": matches[0].path,
            "native_robot_path": native_robot.path,
            "place_invocation_id": route.place_id,
            "terminal_target_index": len(targets) - 1,
        },
    )
