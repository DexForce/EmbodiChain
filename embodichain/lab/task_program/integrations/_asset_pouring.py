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

"""Explicit asset-calibrated adaptation for the four-segment PourWater task.

The existing Pour skill rotates about the held object's origin and returns it
upright. This adapter places the origin so the tilted mouth is over the cup;
it does not simulate liquid or claim transfer-volume success.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from embodichain.utils import configclass
from embodichain.lab.task_program.semantics.asset_requirements import AssetRequirements
from embodichain.lab.task_program.semantics.asset_matching import ResolvedAssetSet

__all__: list[str] = []


@configclass
class AssetPourAdaptationCfg:
    """Explicit role selection for calibrated bottle-to-cup pouring.

    Geometry/calibration evidence is read from each selected descriptor's
    ``provenance.pouring`` record. Bottle calibration is tied to the exact
    selected embodiment ``profile_id``.
    """

    bottle_role: str = "bottle"
    cup_role: str = "cup"
    support_uid: str = "table"
    staging_target_id: str = "cup_pour_pose"
    release_clearance_m: float = 0.007
    mouth_clearance_m: float = 0.17


@dataclass
class _PreparedPour:
    cfg: AssetPourAdaptationCfg
    bottle_id: str
    cup_id: str
    bottle_mouth: np.ndarray
    cup_mouth: np.ndarray
    tilt_axis: np.ndarray
    grasp_transform: np.ndarray
    bottle_radius: float
    cup_radius: float
    angle: float
    grasp: dict[str, Any]
    pick_options: dict[str, Any]
    move_service: dict[str, Any]
    program: dict[str, Any]
    segments: list[dict[str, Any]]
    return_affordance: str
    release_target_id: str


def _array(value: object, size: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Pour annotation {label} must contain finite numbers"
        ) from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Pour annotation {label} must contain {size} finite numbers")
    return result


def _positive(value: object, label: str, *, zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError(
            f"Pour annotation {label} must be finite and {'nonnegative' if zero else 'positive'}"
        )
    return float(value)


def _record(descriptor: object, required: set[str], name: str) -> dict[str, Any]:
    value = descriptor.provenance.get("pouring")
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError(
            f"{name} requires explicit provenance.pouring annotations: {sorted(required)}"
        )
    if not isinstance(value.get("evidence"), str) or not value["evidence"].strip():
        raise ValueError(f"{name} pouring annotation requires evidence")
    return dict(value)


def _check_mouth(
    mouth: np.ndarray, descriptor: object, radius: float, label: str
) -> None:
    bounds = descriptor.bounds_m
    lower, upper = np.array(bounds.lower), np.array(bounds.upper)
    if np.any(mouth < lower - 1e-6) or np.any(mouth > upper + 1e-6):
        raise ValueError(f"{label} mouth is outside normalized asset bounds")
    if mouth[2] < upper[2] - max(0.005, 0.1 * (upper[2] - lower[2])):
        raise ValueError(f"{label} mouth must be near the +Z upper rim")
    if any(
        mouth[i] - radius < lower[i] - 1e-6 or mouth[i] + radius > upper[i] + 1e-6
        for i in (0, 1)
    ):
        raise ValueError(f"{label} mouth radius exceeds normalized asset bounds")


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def _pose(offset: np.ndarray) -> list[float]:
    result = np.eye(4)
    result[:3, 3] = offset
    return result.reshape(-1).tolist()


def prepare_pour_adaptation(
    cfg: AssetPourAdaptationCfg,
    requirements: AssetRequirements,
    resolved: ResolvedAssetSet,
    profile: Mapping[str, object],
    integration: dict[str, Any],
    program: dict[str, Any],
) -> _PreparedPour:
    """Validate selected evidence and remove only the calibration being replaced."""
    cfg.validate()
    for value in (
        cfg.bottle_role,
        cfg.cup_role,
        cfg.support_uid,
        cfg.staging_target_id,
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Pour adaptation identifiers must be nonempty")
    if (
        cfg.bottle_role == cfg.cup_role
        or cfg.bottle_role not in requirements.roles
        or cfg.cup_role not in requirements.roles
    ):
        raise ValueError(
            "Pour adaptation requires distinct selected bottle and cup roles"
        )
    _positive(cfg.release_clearance_m, "release_clearance_m", zero=True)
    _positive(cfg.mouth_clearance_m, "mouth_clearance_m")
    relations = [
        relation for relation in requirements.relations if relation.kind == "pours_into"
    ]
    if (
        len(relations) != 1
        or relations[0].object_role != cfg.bottle_role
        or relations[0].target_role != cfg.cup_role
    ):
        raise ValueError(
            "Pour adaptation requires exactly its bottle-to-cup pours_into relation"
        )
    bottle = resolved.assets[cfg.bottle_role].descriptor
    cup = resolved.assets[cfg.cup_role].descriptor
    bottle_record = _record(
        bottle,
        {
            "mouth_center_m",
            "mouth_radius_m",
            "tilt_axis",
            "object_to_eef",
            "profile_id",
            "evidence",
        },
        "bottle",
    )
    cup_record = _record(cup, {"mouth_center_m", "opening_radius_m", "evidence"}, "cup")
    if bottle_record["profile_id"] != profile.get("profile_id"):
        raise ValueError(
            "Bottle grasp calibration profile_id does not match selected embodiment"
        )
    bottle_mouth = _array(bottle_record["mouth_center_m"], 3, "bottle mouth_center_m")
    cup_mouth = _array(cup_record["mouth_center_m"], 3, "cup mouth_center_m")
    bottle_radius = _positive(bottle_record["mouth_radius_m"], "bottle mouth_radius_m")
    cup_radius = _positive(cup_record["opening_radius_m"], "cup opening_radius_m")
    if bottle_radius >= cup_radius:
        raise ValueError("Bottle mouth must fit strictly inside the cup opening")
    _check_mouth(bottle_mouth, bottle, bottle_radius, "bottle")
    _check_mouth(cup_mouth, cup, cup_radius, "cup")
    axis = _array(bottle_record["tilt_axis"], 3, "tilt_axis")
    magnitude = np.linalg.norm(axis)
    if magnitude < 1e-8:
        raise ValueError("Bottle tilt_axis must be nonzero")
    axis /= magnitude
    if abs(axis[2]) >= 0.5:
        raise ValueError("Bottle tilt_axis must be transverse to the upright +Z mouth")
    transform = _array(bottle_record["object_to_eef"], 16, "object_to_eef").reshape(
        4, 4
    )
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("Bottle object_to_eef must be a proper finite SE(3) transform")
    bottle_id = requirements.roles[cfg.bottle_role].entity_id
    cup_id = requirements.roles[cfg.cup_role].entity_id
    roots = {
        node["entity_id"]: node
        for node in integration["scene_binding"].get("rigid_objects", [])
    }
    if bottle_id not in roots or cup_id not in roots:
        raise ValueError("Pour roles must name configured rigid object roots")
    grasps = [
        node
        for node in roots[bottle_id].get("affordances", [])
        if node["kind"] == "antipodal_grasp"
    ]
    if len(grasps) != 1:
        raise ValueError("Pour bottle needs exactly one antipodal grasp")
    options = integration["profile"]["action_options"]
    pick_options = options.get("pick", {})
    pour_options = options.get("simulation.pour", {})
    angle = pour_options.get("rotate_angle")
    if (
        not isinstance(angle, (int, float))
        or isinstance(angle, bool)
        or not math.isfinite(angle)
        or not 0 < abs(angle) < math.pi / 2
    ):
        raise ValueError(
            "Calibrated pouring requires an explicit tilt angle between zero and pi/2"
        )
    services = integration.get("runtime_services", {}).get(
        "registered_semantic_lowerers", []
    )
    moves = [
        node
        for node in services
        if node.get("kind") == "move_held_object"
        and node.get("target_id") == cfg.staging_target_id
    ]
    pours = [
        node
        for node in services
        if node.get("kind") == "pour" and node.get("object_id") == bottle_id
    ]
    if (
        len(moves) != 1
        or moves[0].get("reference_entity_id") != cup_id
        or len(pours) != 1
    ):
        raise ValueError(
            "Pour adaptation requires one cup-relative move service and one bottle Pour service"
        )
    tree = program.get("program", {})
    segments = tree.get("items", [])
    if (
        tree.get("kind") != "sequence"
        or len(segments) != 4
        or any(
            node.get("kind") != "segment"
            or node.get("steps", {}).get("kind") != "invoke"
            for node in segments
        )
    ):
        raise ValueError(
            "Pour adapter supports the four-segment Pick/Move/Pour/Place program"
        )
    calls = [node["steps"]["call"] for node in segments]
    if (
        calls[0].get("kind") != "pick"
        or calls[0].get("object") != bottle_id
        or calls[1].get("call_id") != "simulation.move_held_object"
        or calls[1].get("arguments") != {"target": cfg.staging_target_id}
        or calls[2].get("call_id") != "simulation.pour"
        or calls[2].get("arguments") != {"object": bottle_id}
        or calls[3].get("kind") != "place"
        or calls[3].get("object") != bottle_id
    ):
        raise ValueError(
            "Pour adapter requires the selected bottle/cup program call sequence"
        )
    if calls[3].get("at", {}).get("kind") != "target_ref":
        raise ValueError(
            "Pour return adaptation requires an explicit authored return target"
        )
    return_affordance = f"{bottle_id}_asset_return_support"
    if any(
        affordance.get("entity_id") == return_affordance
        for node in roots.values()
        for affordance in node.get("affordances", [])
    ):
        raise ValueError("Generated pour return affordance ID already exists")
    pick_options.pop("fixed_object_to_eef", None)
    grasps[0].pop("internal_axis", None)
    release_target_id = calls[3].pop("at")["target"]
    calls[3]["on"] = return_affordance
    return _PreparedPour(
        cfg,
        bottle_id,
        cup_id,
        bottle_mouth,
        cup_mouth,
        axis,
        transform,
        bottle_radius,
        cup_radius,
        float(angle),
        grasps[0],
        pick_options,
        moves[0],
        program,
        segments,
        return_affordance,
        release_target_id,
    )


def finish_pour_adaptation(
    prepared: _PreparedPour,
    integration: dict[str, Any],
    entities: dict[str, dict[str, Any]],
    physical_objects: dict[str, dict[str, Any]],
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], list[float]]]:
    """Adapt the calibrated task using the selected assets' new initial poses."""
    p = prepared
    support = physical_objects.get(p.cfg.support_uid)
    if (
        support is None
        or support.get("body_type", "dynamic") == "dynamic"
        or support.get("init_local_pose") is not None
        or any(float(v) != 0 for v in support.get("init_rot", [0, 0, 0]))
    ):
        raise ValueError("Pour return requires an upright static/kinematic support UID")
    support_position = _array(
        support.get("init_pos", [0, 0, 0]), 3, "support initial position"
    )
    bottle_position = _array(
        physical_objects[entities[p.bottle_id]["simulation_uid"]]["init_pos"],
        3,
        "bottle initial position",
    )
    cup_position = _array(
        physical_objects[entities[p.cup_id]["simulation_uid"]]["init_pos"],
        3,
        "cup initial position",
    )
    desired_mouth = p.cup_mouth + np.array(
        [0.0, 0.0, p.cfg.mouth_clearance_m + p.bottle_radius]
    )
    staging = desired_mouth - _rotation(p.tilt_axis, p.angle) @ p.bottle_mouth
    # Bound the intended fixed-axis mouth sweep; actual joint-interpolated
    # execution is independently sampled by the acceptance monitor.
    # Rodrigues yields z(t)=c+a*cos(t)+b*sin(t), so test endpoints and stationary
    # points analytically rather than assuming sampled waypoints certify clearance.
    cross = np.cross(p.tilt_axis, p.bottle_mouth)
    axial = p.tilt_axis * np.dot(p.tilt_axis, p.bottle_mouth)
    a, b = (p.bottle_mouth - axial)[2], cross[2]
    candidates = [0.0, p.angle]
    critical = math.atan2(b, a)
    for k in range(-2, 3):
        angle = critical + k * math.pi
        if min(0.0, p.angle) <= angle <= max(0.0, p.angle):
            candidates.append(angle)
    if (
        min(
            (staging + _rotation(p.tilt_axis, angle) @ p.bottle_mouth)[2]
            - p.bottle_radius
            for angle in candidates
        )
        < p.cup_mouth[2] + p.cfg.mouth_clearance_m - 1e-8
    ):
        raise ValueError("Selected mouth sweep violates cup rim clearance")
    p.grasp["internal_axis"] = p.tilt_axis.tolist()
    p.pick_options["fixed_object_to_eef"] = p.grasp_transform.reshape(-1).tolist()
    p.move_service["relative_pose"] = _pose(staging)
    scene_roots = integration["scene_binding"]["rigid_objects"]
    support_roots = [
        node for node in scene_roots if node["simulation_uid"] == p.cfg.support_uid
    ]
    if len(support_roots) > 1:
        raise ValueError(
            "Pour support simulation UID maps to multiple semantic objects"
        )
    if support_roots:
        support_root = support_roots[0]
    else:
        support_id = f"{p.bottle_id}_asset_return_table"
        if support_id in entities:
            raise ValueError("Generated pour support entity ID already exists")
        support_root = {
            "entity_id": support_id,
            "simulation_uid": p.cfg.support_uid,
            "dynamics": "static",
            "semantic_type": "support",
            "affordances": [],
        }
        scene_roots.append(support_root)
        entities[support_id] = support_root
    support_id = support_root["entity_id"]
    release = (
        bottle_position
        - support_position
        + np.array([0.0, 0.0, p.cfg.release_clearance_m])
    )
    support_root.setdefault("affordances", []).append(
        {
            "entity_id": p.return_affordance,
            "kind": "support_surface",
            "native_name": p.return_affordance,
            "object_target_frame": "parent",
            "object_target_pose": _pose(release),
        }
    )
    p.program.setdefault("targets", {})[p.release_target_id] = {
        "kind": "cyclic_pose",
        "values": [
            {
                "position": (
                    bottle_position + np.array([0.0, 0.0, p.cfg.release_clearance_m])
                ).tolist(),
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        ],
    }
    offsets = {
        (p.bottle_id, p.cup_id): staging.tolist(),
        (p.bottle_id, support_id): (bottle_position - support_position).tolist(),
        (p.cup_id, support_id): (cup_position - support_position).tolist(),
    }
    for segment_index, segment in enumerate(p.segments):
        for validator in segment.get("validators", []):
            if validator.get("kind") != "object_near_target":
                continue
            object_id = validator.get("object")
            if object_id not in {p.bottle_id, p.cup_id}:
                continue
            reference = (
                p.cup_id
                if segment_index == 1 and object_id == p.bottle_id
                else support_id
            )
            offset = offsets[(object_id, reference)]
            tolerance = validator.get("position_tolerance", 0.01)
            target_id = validator.get("target")
            targets = p.program.get("targets", {})
            if target_id in targets:
                world_position = np.array(offset) + (
                    cup_position if reference == p.cup_id else support_position
                )
                targets[target_id] = {
                    "kind": "cyclic_pose",
                    "values": [
                        {
                            "position": world_position.tolist(),
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        }
                    ],
                }
            # Inscribe the XYZ acceptance box in the original Euclidean
            # tolerance sphere so changing reference frames cannot loosen it.
            validator.clear()
            validator.update(
                kind="object_relative_position",
                object=object_id,
                reference=reference,
                reference_frame="parent",
                position_offset=offset,
                position_tolerance=(
                    [tolerance / math.sqrt(3)] * 3
                    if isinstance(tolerance, (int, float))
                    else tolerance
                ),
            )
    p.segments[2].setdefault("validators", []).append(
        {"kind": "object_upright", "object": p.bottle_id, "maximum_tilt_degrees": 15.0}
    )
    return {(p.bottle_id, p.return_affordance)}, offsets
