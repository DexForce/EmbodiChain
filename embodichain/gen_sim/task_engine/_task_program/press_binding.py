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

"""Source-qualified button routes and explicit generated E9 physics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import math
import re
from typing import Any

import numpy as np

from .press import PressBinding, inspect_press
from .articulation_binding import PARK_CALL

__all__: list[str] = []

PREPARE_CALL = "gen_sim.press_prepare"
PRESS_CALL = "gen_sim.press"
PRESS_REVISION = "15"


@dataclass(frozen=True, slots=True)
class PressRoute:
    binding: PressBinding
    moving_mass: float
    arm: str = "right"
    variant: str = "calibrated"
    extra_press_distance: float = 0.010

    def __post_init__(self) -> None:
        if self.arm not in {"left", "right"}:
            raise ValueError("E9 requires a declared arm.")
        if self.variant not in {
            "original",
            "limits_only",
            "calibrated",
            "approach_only",
        }:
            raise ValueError("Unknown E9 qualification variant.")
        if not math.isfinite(self.moving_mass) or self.moving_mass <= 0:
            raise ValueError("E9 requires positive measured source mass.")
        if (
            type(self.extra_press_distance) not in (int, float)
            or not math.isfinite(self.extra_press_distance)
            or not 0.0 <= self.extra_press_distance <= 0.010
        ):
            raise ValueError("E9 extra_press_distance must be between 0 and 0.010 m.")

    @property
    def link_id(self) -> str:
        return f"{self.binding.object_id}__{self.binding.link}"

    def preset(self, phase: str) -> str:
        if phase not in {"ready", "pressed"}:
            raise ValueError("Unknown E9 acceptance phase.")
        return f"gen_sim.press.{self.binding.object_id}.{phase}"

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["binding"].pop("source_path")
        return result

    @classmethod
    def decode(cls, value: Any) -> PressRoute:
        required = {
            "binding",
            "moving_mass",
            "arm",
            "variant",
        }
        if type(value) is not dict or not required <= set(value) <= required | {
            "extra_press_distance"
        }:
            raise ValueError("E9 route has missing or unexpected fields.")
        raw = value["binding"]
        names = {f.name for f in fields(PressBinding)} - {"source_path"}
        if type(raw) is not dict or set(raw) != names:
            raise ValueError("E9 binding has missing or unexpected fields.")
        data = dict(raw, source_path="")
        for name in ("object_id", "joint", "link", "parent", "collision_mesh"):
            if (
                type(data[name]) is not str
                or not data[name]
                or data[name].strip() != data[name]
            ):
                raise ValueError("E9 identities must be exact non-empty strings.")
        if not data["collision_mesh"].startswith("/") or data["link"] == data["parent"]:
            raise ValueError("E9 requires distinct links and an absolute mesh path.")
        for name in ("scale", "released_position", "pressed_position"):
            if type(data[name]) not in (int, float) or not math.isfinite(data[name]):
                raise ValueError(
                    "E9 coordinates and scale must be finite real numbers."
                )
        for key in ("axis", "limits", "press_position"):
            size = 2 if key == "limits" else 3
            if (
                type(data[key]) not in (list, tuple)
                or len(data[key]) != size
                or any(type(v) not in (int, float) for v in data[key])
            ):
                raise ValueError(
                    "E9 geometry arrays have invalid dimensions or values."
                )
            data[key] = tuple(data[key])
        if type(value["moving_mass"]) not in (int, float):
            raise ValueError("E9 moving mass must be a real number.")
        binding = PressBinding(**data)
        if (
            not all(
                math.isfinite(float(v))
                for v in (*binding.axis, *binding.limits, *binding.press_position)
            )
            or binding.scale <= 0
            or binding.stroke <= 1e-5
            or binding.limits[0] >= binding.limits[1]
            or set((binding.released_position, binding.pressed_position))
            != set(binding.limits)
            or not np.isclose(np.linalg.norm(binding.axis), 1.0)
            or not re.fullmatch("[a-f0-9]{64}", binding.source_sha256)
        ):
            raise ValueError("Invalid E9 geometry or source identity.")
        return cls(
            binding,
            float(value["moving_mass"]),
            value["arm"],
            value["variant"],
            value.get("extra_press_distance", 0.010),
        )


def discover_press(config: dict[str, Any]) -> PressRoute:
    """Resolve one named button and its unique outer collision surface."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(config["fpath"])
    if stage is None:
        raise ValueError("Cannot open the declared button USD.")
    joints = [
        p
        for p in stage.Traverse()
        if p.IsA(UsdPhysics.PrismaticJoint)
        and UsdPhysics.Joint(p).GetJointEnabledAttr().Get() is not False
        and re.search(r"button|press|plunger|switch", p.GetName(), re.I)
    ]
    if len(joints) != 1:
        raise ValueError(
            "E9 needs one unambiguous named button joint, not a substituted part."
        )
    joint = UsdPhysics.PrismaticJoint(joints[0])
    children = joint.GetBody1Rel().GetTargets()
    if len(children) != 1:
        raise ValueError("E9 button joint must own one link.")
    body = stage.GetPrimAtPath(children[0])
    candidates = []
    for prim in Usd.PrimRange(body):
        name = prim.GetName().lower()
        if (
            prim.IsA(UsdGeom.Mesh)
            and prim.HasAPI(UsdPhysics.CollisionAPI)
            and re.search(r"button|cap|plunger", name)
            and not re.search(
                r"label|text|flange|guide|stem", name.replace("and_stem", "")
            )
        ):
            candidates.append(
                inspect_press(
                    config,
                    joint=joints[0].GetName(),
                    link=body.GetName(),
                    collision_mesh=str(prim.GetPath()),
                )
            )
    if not candidates:
        raise ValueError("E9 needs an explicitly named button collision surface.")
    scores = [float(np.dot(b.press_position, b.axis)) for b in candidates]
    outer = [
        b for b, score in zip(candidates, scores) if abs(score - min(scores)) <= 1e-6
    ]
    if len(outer) != 1:
        raise ValueError("E9 button has ambiguous outer collision surfaces.")
    source_mass = UsdPhysics.MassAPI(body).GetMassAttr().Get()
    if source_mass is None or not math.isfinite(source_mass) or source_mass <= 0:
        raise ValueError("E9 calibration requires authored moving-link mass.")
    return PressRoute(outer[0], float(source_mass))


def recipe(object_id: str, arm: str) -> list[dict[str, Any]]:
    if arm not in {"left", "right"}:
        raise ValueError("E9 requires one explicit arm.")
    return [
        {
            "kind": "registered",
            "call_id": call,
            "arguments": {"object": object_id},
            "resources": {"primary": arm},
        }
        for call in (PREPARE_CALL, PRESS_CALL)
    ] + [
        {
            "kind": "registered",
            "call_id": PARK_CALL,
            "arguments": {},
            "resources": {"primary": arm},
        }
    ]


def graph_routes(graph: dict, scene: Any) -> tuple[PressRoute, ...]:
    groups = [g for g in graph.get("task_groups", ()) if g["task_type"] == "E9"]
    if not groups:
        return ()
    if len(groups) != 1:
        raise ValueError("E9 MVP supports one press event per deployment.")
    if len(graph["task_groups"]) != 1:
        raise ValueError(
            "E9 MVP requires a standalone press task; mixed recipes are not qualified."
        )
    nodes = {n["id"]: n for n in graph["nodes"]}
    configs = {c["uid"]: c for c in scene.articulations}
    routes = {}
    for group in groups:
        sequence = [nodes[key] for key in group["node_ids"]]
        if len(sequence) != 3 or [n["role"] for n in sequence] != [
            "primary",
            "primary",
            "cleanup",
        ]:
            raise ValueError(
                "E9 requires prepare/Press/Park with explicit primary/cleanup roles."
            )
        call = sequence[0]["call"]
        if set(call.get("arguments", {})) != {"object"}:
            raise ValueError("E9 calls require exactly one object reference.")
        uid = call["arguments"]["object"]
        if [n["call"] for n in sequence] != recipe(uid, call["resources"]["primary"]):
            raise ValueError("E9 requires prepare/Press/Park in order.")
        if any(b["depends_on"] != [a["id"]] for a, b in zip(sequence, sequence[1:])):
            raise ValueError("E9 recipe dependencies are incomplete.")
        if uid not in configs:
            raise ValueError("E9 object is not a declared articulation.")
        routes[uid] = replace(
            discover_press(configs[uid]), arm=call["resources"]["primary"]
        )
    if len(routes) != 1:
        raise ValueError("E9 MVP supports one button per deployment.")
    return tuple(routes.values())


def prepare_press_scene(
    scene: Any,
    routes: tuple[PressRoute, ...],
    *,
    body_scale: float = 1.0,
    interaction: str = "E9",
) -> tuple[Any, list[dict]]:
    """Deploy selected E9 geometry at unit scale without editing source assets."""
    if not routes:
        return scene, []
    from ..scene.articulation_geometry import _root_pose, read_articulation_geometry
    from ..scene.final_inspection import _measure_geometry

    articulations = list(scene.articulations)
    planner = list(scene.planner_objects)
    records = []
    for route in routes:
        uid = route.binding.object_id
        index = next(i for i, item in enumerate(articulations) if item["uid"] == uid)
        original = articulations[index]
        adapted = deepcopy(original)
        geometry = read_articulation_geometry(original["fpath"])
        before = geometry.world_vertices(original)
        adapted["body_scale"] = [body_scale] * 3
        after = geometry.world_vertices(adapted)
        pose = _root_pose(original).copy()
        pose[2, 3] += float(before[:, 2].min() - after[:, 2].min())
        adapted["init_pos"] = pose[:3, 3].tolist()
        adapted["init_local_pose"] = pose.tolist()
        vertices = geometry.world_vertices(adapted)
        bounds = np.stack((vertices.min(0), vertices.max(0)))
        for other in (*scene.background, *scene.rigid_objects, *articulations):
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
                raise ValueError(
                    f"{interaction} scaled control overlaps {other['uid']!r}."
                )
        articulations[index] = adapted
        for i, item in enumerate(planner):
            if item.get("runtime_uid") == uid:
                planner[i] = {
                    **deepcopy(item),
                    "body_scale": adapted["body_scale"].copy(),
                    "init_pos": adapted["init_pos"].copy(),
                    "init_local_pose": deepcopy(adapted["init_local_pose"]),
                }
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
                "physics_policy": "preserve_pre_resize_calibration",
            }
        )
    return (
        replace(
            scene, articulations=tuple(articulations), planner_objects=tuple(planner)
        ),
        records,
    )


def calibrate_scene(
    payload: dict,
    routes: tuple[PressRoute, ...],
    *,
    source_routes: tuple[PressRoute, ...] | None = None,
) -> list[dict]:
    """Apply only selected button physics in the generated scene, never the source."""
    audits = []
    sources = {
        r.binding.object_id: r
        for r in (routes if source_routes is None else source_routes)
    }
    for route in routes:
        b = route.binding
        source = sources[b.object_id]
        if source.binding.source_sha256 != b.source_sha256:
            raise ValueError("E9 source changed during unit-scale deployment.")
        art = next(
            a for a in payload["simulation"]["articulation"] if a["uid"] == b.object_id
        )
        if art.get("joint_drive_props") or art.get("qpos_limits"):
            raise ValueError(
                "E9 cannot silently replace authored runtime drive/limit overrides."
            )
        mass = source.moving_mass * source.binding.scale**3
        stiffness = max(500.0, mass * 9.81 / (0.04 * source.binding.stroke))
        damping = 2.0 * math.sqrt(stiffness * mass)
        art["asset_physics_mode"] = "overlay"
        if route.variant == "original":
            audits.append(
                {
                    **route.payload(),
                    "source_edited": False,
                    "model_origin": "source_physics_with_exported_scale",
                }
            )
            continue
        art["qpos_limits"] = {b.joint: list(b.limits)}
        if route.variant == "limits_only":
            audits.append(
                {
                    **route.payload(),
                    "source_edited": False,
                    "model_origin": "scaled_limits_only",
                }
            )
            continue
        art["joint_drive_props"] = {
            "target_mode": {b.joint: "position_velocity"},
            "stiffness": {b.joint: stiffness},
            "damping": {b.joint: damping},
            "max_effort": {b.joint: max(5.0, 2.0 * stiffness * source.binding.stroke)},
        }
        art.setdefault("link_attrs", {})["gen_sim_press_density"] = {
            "link_names_expr": [b.link],
            "attrs": {"mass_props": {"mass": mass, "recompute_inertia": True}},
        }
        audits.append(
            {
                **route.payload(),
                "schema": "gen_sim.press_calibration/v1",
                "model_origin": "generated_release_bias_not_source_mechanism",
                "success_criterion": "contact_press_0.05mm_not_activation",
                "mass_policy": "preserve_pre_resize_calibration",
                "calibration_scale": source.binding.scale,
                "source_mass": route.moving_mass,
                "runtime_mass": mass,
                "stiffness": stiffness,
                "damping": damping,
                "runtime_limits": list(b.limits),
                "contact_policy": "native_contact_envelope",
                "source_edited": False,
            }
        )
    return audits


def validate_program(program: dict[str, Any], route: PressRoute) -> None:
    """Reject deployments that remove the mandatory event and retreat checks."""
    items = program["program"]["items"]
    if len(items) != 3:
        raise ValueError("E9 requires its complete three-segment program.")
    calls = recipe(route.binding.object_id, route.arm)
    for index, (item, call) in enumerate(zip(items, calls)):
        expected = [
            {
                "kind": "wait_stable",
                "entity": route.binding.object_id,
                "preset": route.preset("ready" if index == 0 else "pressed"),
            }
        ]
        if (
            item.get("steps") != {"kind": "invoke", "call": call}
            or item.get("post") != expected
        ):
            raise ValueError(
                "E9 program must retain its declared calls and physical post-policies."
            )
