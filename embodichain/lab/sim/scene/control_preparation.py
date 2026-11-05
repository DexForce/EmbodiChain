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

"""Explicit physical source control scale/support preparation without semantic task data."""

from __future__ import annotations
from collections.abc import Collection
from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Any, Literal
import numpy as np
from embodichain.compute.geometry.control_resize import (
    control_bounds_overlap,
    control_clearance_translations,
)
from .press_inputs import PressBinding
from .revolute_inputs import RevoluteControlBinding

__all__ = [
    "ControlSourceScene",
    "ControlSourceResizeRequest",
    "prepare_control_source_scene",
]


@dataclass(frozen=True, slots=True)
class ControlSourceScene:
    """Ordered physical source sections; no semantic planner records or robot state."""

    background: tuple[dict[str, Any], ...]
    rigid_objects: tuple[dict[str, Any], ...]
    articulations: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ControlSourceResizeRequest:
    """Explicit authored control binding selected by a host declaration."""

    binding: PressBinding | RevoluteControlBinding


def prepare_control_source_scene(
    scene: ControlSourceScene,
    routes: tuple[ControlSourceResizeRequest, ...],
    *,
    body_scale: float | None = None,
    interaction_scale: float | None = None,
    interaction_scale_qualified: bool = False,
    interaction_kind: Literal["revolute", "prismatic"] = "prismatic",
    relayout_margin_m: float = 0.02,
    relayout_uids: Collection[str] = (),
    source_payloads: Collection[tuple[str, str]] = (),
    support_lift_m: float | None = None,
) -> tuple[ControlSourceScene, list[dict[str, Any]]]:
    """Adapt selected controls and optionally resolve bounded scene overlaps.

    E9 retains its historical unit-scale deployment when ``body_scale`` is not
    provided. E8 is source-scale by default; an explicit interaction scale is
    rejected unless the caller supplies an independent asset/gripper
    dimensional qualification; this does not certify physical success. With
    ``support_lift_m=None``, direct E8 callers retain conservative contact-offset
    clearance. An explicit 1--4 mm lift is only a geometrically screened runtime
    candidate. Relayout is deliberately opt-in and accepts only a
    caller-provided set of unreferenced rigid distractors. Candidate motion is
    XY-only, preserving support height/orientation, and is accepted only when
    the moved object remains inside the table footprint and clear of all
    measured geometry.
    Source-declared on/inside payload pairs may retain an already-overlapping
    assembly AABB only for source-scale E8. This defers a coarse screen to native
    physics; it is not collision or containment qualification, and new scale
    candidates cannot acquire it.

    Args:
        scene: Ordered neutral physical source sections.
        routes: Explicit authored target bindings.
        body_scale: Optional prismatic unit-scale override.
        interaction_scale: Explicit dimensionally qualified revolute scale.
        interaction_scale_qualified: Whether the caller supplied independent
            asset/gripper dimensional qualification for that scale.
        interaction_kind: Physical revolute or prismatic control family.
        relayout_margin_m: Explicit original layout collision reserve.
        relayout_uids: Caller-qualified unreferenced rigid objects eligible to move.
        source_payloads: Explicit inherited source parent/object relationships.
        support_lift_m: Optional original bounded total support lift candidate.

    Returns:
        Independent physical sections and original numeric adaptation receipts.
        No simulator, task program, semantic planner or physical certificate is made.
    """
    interaction = "E8" if interaction_kind == "revolute" else "E9"
    if interaction_kind not in {"revolute", "prismatic"}:
        raise ValueError("Unknown physical control adaptation kind.")
    if not math.isfinite(relayout_margin_m) or relayout_margin_m < 0:
        raise ValueError("Control relayout margin must be finite and nonnegative.")
    if not routes:
        return scene, []
    if interaction not in {"E8", "E9"}:
        raise ValueError(f"Unknown interaction {interaction!r}.")
    if type(interaction_scale_qualified) is not bool:
        raise TypeError("interaction_scale_qualified must be a boolean.")
    if interaction_scale is not None:
        if interaction != "E8":
            raise ValueError("interaction_scale is only valid for E8.")
        if not interaction_scale_qualified:
            raise ValueError(
                "E8 interaction_scale requires an independent asset/gripper qualification."
            )
        if (
            isinstance(interaction_scale, bool)
            or not math.isfinite(float(interaction_scale))
            or float(interaction_scale) <= 0
        ):
            raise ValueError("E8 interaction_scale must be finite and positive.")
    if interaction == "E8" and body_scale is not None:
        raise ValueError("E8 uses source scale; pass interaction_scale when qualified.")
    if body_scale is not None and (
        isinstance(body_scale, bool)
        or not math.isfinite(float(body_scale))
        or float(body_scale) <= 0
    ):
        raise ValueError("body_scale must be finite and positive.")
    if relayout_uids and interaction != "E8":
        raise ValueError("relayout_uids is only valid for E8.")
    if source_payloads and interaction != "E8":
        raise ValueError("source_payloads is only valid for E8.")
    if support_lift_m is not None and interaction != "E8":
        raise ValueError("support_lift_m is only valid for E8.")
    requested_relayout = {str(uid) for uid in relayout_uids}
    inherited_payloads = set(source_payloads)
    from .articulation_geometry import _root_pose, read_articulation_geometry
    from .source_measurement import measure_source_geometry as _measure_geometry

    # Adaptation and optional relayout must not mutate the source PreparedScene.
    articulations = [deepcopy(item) for item in scene.articulations]
    rigid_objects = [deepcopy(item) for item in scene.rigid_objects]
    background = [deepcopy(item) for item in scene.background]
    records = []
    relayout_records: list[dict[str, Any]] = []
    route_uids = {str(route.binding.object_id) for route in routes}
    if requested_relayout:
        rigid_uids = {str(item["uid"]) for item in rigid_objects}
        unknown = sorted(requested_relayout - rigid_uids)
        if unknown:
            raise ValueError(
                f"E8 relayout requires declared rigid distractors, got {unknown}."
            )
        forbidden = sorted(requested_relayout & route_uids)
        if forbidden:
            raise ValueError(f"E8 relayout cannot move target objects: {forbidden}.")
        if "table" in requested_relayout:
            raise ValueError("E8 relayout cannot move the table support.")
    runtime_objects = [*background, *rigid_objects, *articulations]
    table = next((item for item in runtime_objects if item.get("uid") == "table"), None)
    table_measurement = (
        None
        if table is None
        else _measure_geometry(
            table, convert_y_up=table.get("shape", {}).get("shape_type") == "Mesh"
        )
    )

    def _set_xy(item: dict[str, Any], translation: np.ndarray) -> None:
        position = np.asarray(item.get("init_pos", [0.0, 0.0, 0.0]), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(
                f"E8 relayout object {item.get('uid')!r} has invalid pose."
            )
        position[:2] += translation
        item["init_pos"] = position.tolist()
        local_pose = item.get("init_local_pose")
        if local_pose is not None:
            pose = np.asarray(local_pose, dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(
                    f"E8 relayout object {item.get('uid')!r} has invalid local pose."
                )
            pose = pose.copy()
            pose[:2, 3] += translation
            item["init_local_pose"] = pose.tolist()

    def _bounds(item: dict[str, Any]) -> np.ndarray:
        measured = _measure_geometry(
            item, convert_y_up=item.get("shape", {}).get("shape_type") == "Mesh"
        )
        if measured is None:
            raise ValueError(f"E8 scale clearance is unmeasured for {item['uid']!r}.")
        return np.asarray(measured["bounds"], dtype=float)

    _overlap = control_bounds_overlap

    def _try_relayout(
        target_uid: str, target_bounds: np.ndarray, candidate_uid: str
    ) -> dict[str, Any] | None:
        candidate = next(item for item in rigid_objects if item["uid"] == candidate_uid)
        original_bounds = _bounds(candidate)
        if not _overlap(target_bounds, original_bounds):
            return None
        margin = relayout_margin_m
        proposals = control_clearance_translations(
            target_bounds, original_bounds, margin
        )
        protected = [
            item
            for item in runtime_objects
            if item["uid"] not in {candidate_uid, "table"}
        ]
        for translation, direction in proposals:
            moved_bounds = original_bounds.copy()
            moved_bounds[:, :2] += translation
            if table_measurement is not None and (
                np.any(moved_bounds[0, :2] < table_measurement["bounds"][0, :2])
                or np.any(moved_bounds[1, :2] > table_measurement["bounds"][1, :2])
            ):
                continue
            if any(_overlap(moved_bounds, _bounds(item)) for item in protected):
                continue
            old_position = list(candidate.get("init_pos", [0.0, 0.0, 0.0]))
            _set_xy(candidate, translation)
            return {
                "object_id": candidate_uid,
                "target_id": target_uid,
                "translation_xy": translation.tolist(),
                "direction": direction,
                "original_init_pos": old_position,
                "adapted_init_pos": list(candidate["init_pos"]),
                "support_policy": "preserve_z_and_orientation",
                "clearance_margin": margin,
            }
        return None

    for route in routes:
        uid = route.binding.object_id
        index = next(i for i, item in enumerate(articulations) if item["uid"] == uid)
        original = articulations[index]
        adapted = deepcopy(original)
        geometry = read_articulation_geometry(original["fpath"])
        before = geometry.world_vertices(original)
        original_bounds = np.stack((before.min(0), before.max(0)))
        source_scale = np.asarray(
            original.get("body_scale", [1.0, 1.0, 1.0]), dtype=float
        )
        if (
            source_scale.shape != (3,)
            or not np.isfinite(source_scale).all()
            or (source_scale <= 0).any()
            or not np.allclose(source_scale, source_scale[0])
        ):
            raise ValueError(
                f"{interaction} source body_scale must be positive and uniform."
            )
        if interaction == "E8" and interaction_scale is not None:
            scale = float(interaction_scale)
        elif interaction == "E8":
            scale = float(source_scale[0])
            binding_scale = float(route.binding.scale)
            if not math.isclose(scale, binding_scale, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError("E8 source scale differs from its calibrated route.")
        else:
            scale = 1.0 if body_scale is None else float(body_scale)
        adapted["body_scale"] = [scale] * 3
        if interaction == "E8":
            adapted["body_scale_mass_policy"] = "fixed_mass"
        after = geometry.world_vertices(adapted)
        pose = _root_pose(original).copy()
        pose[2, 3] += float(before[:, 2].min() - after[:, 2].min())
        adapted["init_pos"] = pose[:3, 3].tolist()
        adapted["init_local_pose"] = pose.tolist()
        support_audit = None
        if interaction == "E8":
            from .control_support import adjust_revolute_source_support

            if table is None or table_measurement is None:
                raise ValueError("E8 support clearance requires a measured table.")
            adapted, support_audit = adjust_revolute_source_support(
                adapted,
                route.binding,
                table,
                table_top_z=float(table_measurement["bounds"][1, 2]),
                requested_total_lift_m=support_lift_m,
            )
        vertices = geometry.world_vertices(adapted)
        bounds = np.stack((vertices.min(0), vertices.max(0)))
        deferred_payloads = []
        for other in (*background, *rigid_objects, *articulations):
            if other["uid"] == uid:
                continue
            # Normalized GLBs retain Y-up; primitive sizes already use runtime XYZ.
            measured = _measure_geometry(
                other, convert_y_up=other.get("shape", {}).get("shape_type") == "Mesh"
            )
            if measured is None:
                raise ValueError(
                    f"{interaction} scale clearance is unmeasured for {other['uid']!r}."
                )
            other_bounds = measured["bounds"]
            if other["uid"] == "table":
                if np.any(bounds[0, :2] < other_bounds[0, :2]) or np.any(
                    bounds[1, :2] > other_bounds[1, :2]
                ):
                    raise ValueError(
                        f"{interaction} scaled control exceeds the table footprint."
                    )
                continue
            overlap = np.minimum(bounds[1], other_bounds[1]) - np.maximum(
                bounds[0], other_bounds[0]
            )
            if np.all(overlap > 1e-6):
                inherited_overlap = np.minimum(
                    original_bounds[1], other_bounds[1]
                ) - np.maximum(original_bounds[0], other_bounds[0])
                if (
                    interaction == "E8"
                    and interaction_scale is None
                    and (uid, other["uid"]) in inherited_payloads
                    and np.all(inherited_overlap > 1e-6)
                ):
                    deferred_payloads.append(
                        {
                            "uid": other["uid"],
                            "scope": "source_declared_payload_aabb_overlap_deferred_to_native_not_collision_certificate",
                            "original_overlap_m": inherited_overlap.tolist(),
                            "adapted_overlap_m": overlap.tolist(),
                        }
                    )
                    continue
                if interaction != "E8" or other["uid"] not in requested_relayout:
                    raise ValueError(
                        f"{interaction} scaled control overlaps {other['uid']!r}."
                    )
                moved = _try_relayout(uid, bounds, str(other["uid"]))
                if moved is None:
                    raise ValueError(
                        f"E8 scaled control overlaps {other['uid']!r}; no valid relayout."
                    )
                relayout_records.append(moved)
                continue
        articulations[index] = adapted
        records.append(
            {
                "object_id": uid,
                "source_sha256": route.binding.source_sha256,
                "original_body_scale": list(original["body_scale"]),
                "adapted_body_scale": adapted["body_scale"],
                "original_init_pos": _root_pose(original)[:3, 3].tolist(),
                "adapted_init_pos": adapted["init_pos"],
                "support_bottom_z": float(before[:, 2].min()),
                "source_edited": False,
                "physics_policy": (
                    "fixed_mass_source_to_scaled_actor_frame"
                    if interaction == "E8"
                    else "preserve_pre_resize_calibration"
                ),
                "scale_policy": (
                    "preserve_source"
                    if interaction == "E8" and interaction_scale is None
                    else (
                        "qualified_interaction_scale"
                        if interaction == "E8"
                        else "unit_scale"
                    )
                ),
                "relayout": [
                    item for item in relayout_records if item["target_id"] == uid
                ],
                **(
                    {"source_payload_overlap_deferred": deferred_payloads}
                    if deferred_payloads
                    else {}
                ),
                **({"support_clearance": support_audit} if support_audit else {}),
            }
        )
    return (
        replace(
            scene,
            articulations=tuple(articulations),
            rigid_objects=tuple(rigid_objects),
        ),
        records,
    )
