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

"""Measured E1 planar landing-region and release evidence, never an executor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry
import torch
import trimesh

from .container_targets import _floor_region

__all__: list[str] = []

SUPPORT_TOLERANCE_M = 0.05
PENETRATION_TOLERANCE_M = 0.01


def _support_gap_valid(gap: float) -> bool:
    """Relax positive support clearance, not penetration into the floor."""
    return -PENETRATION_TOLERANCE_M <= gap <= SUPPORT_TOLERANCE_M


def _native_mesh(
    simulation: Any, binding: Any, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    native = simulation.get_rigid_object(binding.simulation_uid)
    if native is None:
        raise ValueError("Placement requires its exact bound native rigid object.")
    vertices = native.get_vertices(scale=True).detach().cpu()
    faces = native.get_triangles().detach().cpu()
    if (
        vertices.ndim != 3
        or vertices.shape[0] != batch_size
        or vertices.shape[-1] != 3
        or not vertices.shape[1]
        or not torch.isfinite(vertices).all()
        or faces.ndim != 3
        or faces.shape[0] != batch_size
        or faces.shape[-1] != 3
        or not faces.shape[1]
        or faces.dtype not in (torch.int32, torch.int64)
        or (faces < 0).any()
        or (faces >= vertices.shape[1]).any()
    ):
        raise ValueError("Placement requires finite scaled native mesh geometry.")
    return vertices, faces


def _valid_pose(pose: torch.Tensor, batch_size: int) -> None:
    if pose.shape != (batch_size, 4, 4) or not torch.isfinite(pose).all():
        raise ValueError("Placement pose does not match the observed batch.")
    rotation = pose[:, :3, :3]
    if (
        not torch.allclose(
            rotation.transpose(-1, -2) @ rotation,
            pose.new_tensor(np.eye(3)).expand_as(rotation),
            atol=1e-4,
        )
        or not torch.allclose(
            torch.linalg.det(rotation), pose.new_ones(batch_size), atol=1e-4
        )
        or not torch.allclose(
            pose[:, 3],
            pose.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(batch_size, -1),
            atol=1e-5,
        )
    ):
        raise ValueError("Placement requires rigid, non-reflected native transforms.")


@dataclass(frozen=True)
class _PlacementGeometry:
    simulation: Any
    object_uid: str
    reference_uid: str
    object_native: Any
    reference_native: Any
    vertices: torch.Tensor
    floors: tuple[tuple[BaseGeometry, float], ...]


def _prepare_placement_geometry(
    simulation: Any, object_binding: Any, reference_binding: Any, batch_size: int
) -> _PlacementGeometry:
    """Cache immutable native geometry, never dynamic poses or acceptance."""
    vertices, _ = _native_mesh(simulation, object_binding, batch_size)
    # Exact duplicate payload points do not affect any projected hull or min-Z.
    vertices = torch.unique(vertices, dim=1)
    reference_vertices, faces = _native_mesh(simulation, reference_binding, batch_size)
    floors = []
    for row in range(batch_size):
        floor = _floor_region(
            trimesh.Trimesh(reference_vertices[row].numpy(), faces[row].numpy())
        )
        if floor is None:
            raise ValueError(
                "Placement reference has no measured planar landing region."
            )
        floors.append(floor)
    return _PlacementGeometry(
        simulation,
        object_binding.simulation_uid,
        reference_binding.simulation_uid,
        simulation.get_rigid_object(object_binding.simulation_uid),
        simulation.get_rigid_object(reference_binding.simulation_uid),
        vertices.clone(),
        tuple(floors),
    )


def measure_planar_placement(
    simulation: Any,
    object_binding: Any,
    reference_binding: Any,
    object_pose: torch.Tensor,
    reference_pose: torch.Tensor,
    *,
    _geometry: _PlacementGeometry | None = None,
) -> tuple[list[bool], dict[str, Any]]:
    """Check the full footprint against actual floor triangles, including holes.

    This qualifies resting within a measured planar landing region, not contact
    force closure or arbitrary curved/three-dimensional cavity containment.
    """
    batch_size = object_pose.shape[0]
    _valid_pose(object_pose, batch_size)
    _valid_pose(reference_pose, batch_size)
    geometry = _geometry or _prepare_placement_geometry(
        simulation, object_binding, reference_binding, batch_size
    )
    if (
        geometry.simulation is not simulation
        or geometry.object_uid != object_binding.simulation_uid
        or geometry.reference_uid != reference_binding.simulation_uid
        or geometry.object_native
        is not simulation.get_rigid_object(object_binding.simulation_uid)
        or geometry.reference_native
        is not simulation.get_rigid_object(reference_binding.simulation_uid)
        or geometry.vertices.shape[0] != batch_size
        or len(geometry.floors) != batch_size
    ):
        raise ValueError(
            "Placement geometry cache differs from its native bindings or batch."
        )
    vertices = geometry.vertices
    relative = torch.linalg.solve(reference_pose, object_pose)
    local = (
        vertices.to(relative) @ relative[:, :3, :3].transpose(-1, -2)
        + relative[:, None, :3, 3]
    )
    accepted, footprints, gaps, areas = [], [], [], []
    for row in range(batch_size):
        if reference_pose[row, 2, 2] < math.cos(math.radians(5.0)):
            raise ValueError("Placement reference lacks a qualified near-level floor.")
        polygon, height = geometry.floors[row]
        points = local[row].numpy()
        footprint = MultiPoint(points[:, :2]).convex_hull
        if footprint.is_empty or footprint.area <= 0:
            raise ValueError("Placement object has no finite area footprint.")
        inside = bool(polygon.covers(footprint))
        gap = float(points[:, 2].min() - height)
        accepted.append(inside and _support_gap_valid(gap))
        footprints.append(inside)
        gaps.append(gap)
        areas.append(float(footprint.area))
    return accepted, {
        "scope": "measured_planar_landing_region",
        "object_pose": object_pose.tolist(),
        "reference_pose": reference_pose.tolist(),
        "full_footprint_inside": footprints,
        "support_gap_m": gaps,
        "footprint_area_m2": areas,
        "support_tolerance_m": SUPPORT_TOLERANCE_M,
        "penetration_tolerance_m": PENETRATION_TOLERANCE_M,
    }


def placement_node(
    group: Mapping[str, Any],
    nodes: Mapping[str, Any],
    predicate: Any,
    member: Mapping[str, Any],
    roles: Mapping[str, Any],
    scene_binding: Any,
    *,
    support_contract_revision: int | None = None,
) -> tuple[Mapping[str, Any], str]:
    """Resolve only the graph's exact object/reference and declared E1 relation."""
    if (
        predicate.name != "semantic_goal"
        or predicate.arguments.get("task_type") != "E1"
    ):
        raise ValueError("This observer supports only E1 placement predicates.")
    reference = roles.get(predicate.arguments.get("reference"), [])
    if len(reference) != 1 or reference[0].get("part_id") is not None:
        raise ValueError("Placement needs one explicitly bound rigid reference.")
    reference_uid = reference[0]["uid"]
    relation = predicate.arguments.get("relation")
    if relation not in {"inside", "on"}:
        raise ValueError("This placement relation needs a different observer.")
    placements = []
    for node_id in group["node_ids"]:
        node = nodes[node_id]
        call = node["call"]
        if call.get("kind") == "place" and call.get("object") == member["object_uid"]:
            affordance = call.get(relation)
            collection = (
                scene_binding.containers
                if relation == "inside"
                else scene_binding.support_surfaces
            )
            if any(
                v.entity_id == affordance and v.parent_id == reference_uid
                for v in collection
            ):
                placements.append(node)
        elif (
            relation == "on"
            and call.get("call_id")
            in (
                {
                    "simulation.place_relative",
                    "gen_sim.place_upright",
                    "gen_sim.stack_place",
                }
                if support_contract_revision == 1
                else {"simulation.place_relative"}
            )
            and call["arguments"].get("object") == member["object_uid"]
            and call["arguments"].get("reference") == reference_uid
            and call["arguments"].get("relation") == "on"
        ):
            placements.append(node)
    if len(placements) != 1:
        raise ValueError("Placement predicate does not match one bound graph call.")
    return placements[0], reference_uid


def measured_release(
    node: Mapping[str, Any],
    object_uid: str,
    segments: Mapping[str, Any],
    boundary: Mapping[str, Any],
    num_envs: int,
    *,
    with_receipt: bool = False,
) -> list[bool] | tuple[list[bool], Mapping[str, Any], Mapping[str, Any]]:
    """Require exact terminal release receipts and no held state at the boundary."""
    runtime = segments[node["id"]]["metadata"]["runtime"]
    identities = runtime["env_ids"]
    if (
        not isinstance(identities, list)
        or any(type(v) is not int for v in identities)
        or identities != list(range(num_envs))
    ):
        raise ValueError(
            "Placement release evidence has unsupported environment identity."
        )
    resource = node["call"]["resources"]["primary"]
    result = [False] * num_envs
    found = False
    verified_receipt = verified_call = None
    for call in runtime.get("calls", []):
        if call.get("status") != "completed":
            continue
        if type(call.get("invocation_revision")) is not int:
            continue
        for receipt in call.get("effects", []):
            spec = receipt["effect_spec"]
            if (
                receipt.get("boundary", {}).get("kind") != "terminal"
                or spec.get("effect_kind") != "release"
                or spec.get("invocation_id") != call["invocation_id"]
                or spec.get("invocation_revision") != call.get("invocation_revision")
                or spec.get("env_ids") != identities
            ):
                continue
            expectations = [
                v
                for v in spec["state_expectations"]
                if v.get("object_id") == object_uid
                and v.get("resource_id") == resource
                and v.get("relation") == "detached"
            ]
            if len(expectations) != 1:
                continue
            clauses = [
                v
                for v in spec["clauses"]
                if v["expectation_id"] == expectations[0]["expectation_id"]
            ]
            constraint = next(
                (
                    v
                    for v in clauses
                    if v.get("evidence_kind") == "constraint"
                    and v.get("expected") is False
                ),
                None,
            )
            pose_clause = next(
                (v for v in clauses if v.get("expectation") == "separated"), None
            )
            if constraint is None or pose_clause is None:
                continue
            samples = [
                receipt["evidence"][v["clause_id"]] for v in (constraint, pose_clause)
            ]
            success = receipt["decision"]["success_mask"]
            if (
                len(success) != num_envs
                or any(type(v) is not bool for v in success)
                or any(
                    s.get("env_ids") != identities
                    or s.get("timestamp") != receipt["timestamp"]
                    or s.get("observation_revision") != receipt["observation_revision"]
                    or len(s.get("valid_mask", [])) != num_envs
                    or any(type(v) is not bool for v in s["valid_mask"])
                    for s in samples
                )
                or len(samples[0].get("values", [])) != num_envs
                or any(type(v) is not bool for v in samples[0]["values"])
            ):
                continue
            relative = torch.tensor(samples[1]["object_to_endpoint"])
            _valid_pose(relative, num_envs)
            found = True
            verified_receipt, verified_call = receipt, call
            result = [
                ok and a and b and value is False
                for ok, a, b, value in zip(
                    success,
                    samples[0]["valid_mask"],
                    samples[1]["valid_mask"],
                    samples[0]["values"],
                    strict=True,
                )
            ]
    if not found:
        raise ValueError("No exact measured terminal release receipt is available.")
    current = boundary["metadata"]["runtime"]
    state = current["task_state"]
    if (
        current["env_ids"] != identities
        or any(type(v) is not int for v in current["env_ids"])
        or type(state["batch_size"]) is not int
        or state["batch_size"] != num_envs
    ):
        raise ValueError(
            "Placement held-state evidence has different environment identity."
        )
    for held in [*state["held_objects"], *state["coordinated_held_objects"]]:
        if held["object_id"] != object_uid:
            continue
        mask = held["active_mask"]
        if len(mask) != num_envs or any(type(v) is not bool for v in mask):
            raise ValueError("Placement held-state mask is invalid.")
        result = [ok and not active for ok, active in zip(result, mask, strict=True)]
    if with_receipt:
        assert verified_receipt is not None and verified_call is not None
        return result, verified_receipt, verified_call
    return result
