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
"""Source-bound E8 label calibration; setting numbers are never joint angles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .articulation_binding import PARK_CALL, _fixed_base
from .twist_semantics import MissingTwistSemanticsError, resolve_twist_semantics

__all__: list[str] = []
PREPARE_CALL = "gen_sim.twist_prepare"
TWIST_CALL = "gen_sim.twist"
TWIST_REVISION = "14"
TWIST_CHUNK_COUNT = 8
TWIST_CHUNK_ANGLE = math.radians(5.0)
TWIST_SETTLE_STEPS = 5
TWIST_FALLBACK_CHUNKS = 3
TWIST_MAX_CHUNK_COUNT = 32


def required_chunk_count(target_qpos: float) -> int:
    """Cover the calibrated zero-to-setting angle plus bounded contact retries."""
    if type(target_qpos) not in (int, float) or not math.isfinite(target_qpos):
        raise ValueError("E8 chunk coverage requires a finite calibrated angle.")
    count = max(
        TWIST_CHUNK_COUNT,
        math.ceil(abs(target_qpos) / TWIST_CHUNK_ANGLE - 1e-12) + TWIST_FALLBACK_CHUNKS,
    )
    if count > TWIST_MAX_CHUNK_COUNT:
        raise ValueError("E8 calibrated turn exceeds the bounded chunk budget.")
    # A finite feedback budget covers contact slip; verified coarse contact travel
    # turns the unused semantic calls into stationary holds.
    return TWIST_CHUNK_COUNT if target_qpos == 0.0 else TWIST_MAX_CHUNK_COUNT


@dataclass(frozen=True, slots=True)
class KnobBinding:
    object_id: str
    joint: str
    link: str
    parent: str
    source_sha256: str
    scale: float
    axis: tuple[float, float, float]
    axis_sign: float
    origin: tuple[float, float, float]
    outer_point: tuple[float, float, float]
    grip_depth: float
    grip_width: float
    limits: tuple[float, float]
    target_setting: int
    target_qpos: float
    calibration_sha256: str = ""
    mass_lineage_sha256: str = ""


@dataclass(frozen=True, slots=True)
class TwistRoute:
    binding: KnobBinding
    arm: str = "right"
    variant: str = "source"
    grasp_depth: float = 0.008
    settle_steps: int = TWIST_SETTLE_STEPS
    chunk_count: int = TWIST_CHUNK_COUNT
    chunk_angle: float = TWIST_CHUNK_ANGLE
    fixed_roll: float | None = None
    feedback_enabled: bool = False
    grip_extra_m: float = 0.0

    def __post_init__(self) -> None:
        if self.arm not in {"left", "right"}:
            raise ValueError("E8 requires one explicit arm.")
        if self.variant not in {
            "source",
            "no_twist",
            "centroid",
            "closed_tip",
        }:
            raise ValueError("Unknown E8 qualification variant.")
        if (
            type(self.grasp_depth) not in (int, float)
            or not math.isfinite(self.grasp_depth)
            or not 0 < self.grasp_depth <= 0.01
        ):
            raise ValueError("E8 grasp_depth must be finite and within (0, 0.01] m.")
        if type(self.settle_steps) is not int or not 0 <= self.settle_steps <= 100:
            raise ValueError("E8 settle_steps must be an integer within [0, 100].")
        if (
            type(self.chunk_count) is not int
            or not 1 <= self.chunk_count <= TWIST_MAX_CHUNK_COUNT
        ):
            raise ValueError("E8 chunk_count must be an integer within [1, 32].")
        if (
            type(self.chunk_angle) not in (int, float)
            or not math.isfinite(self.chunk_angle)
            or not 0 < self.chunk_angle <= math.pi / 2
        ):
            raise ValueError("E8 chunk_angle must be finite and within (0, pi/2].")
        if self.fixed_roll is not None and (
            type(self.fixed_roll) not in (int, float)
            or not math.isfinite(self.fixed_roll)
            or abs(self.fixed_roll) > math.pi
        ):
            raise ValueError("E8 fixed_roll must be finite and within [-pi, pi].")
        if type(self.feedback_enabled) is not bool:
            raise ValueError("E8 feedback_enabled must be an explicit boolean.")
        if (
            type(self.grip_extra_m) not in (int, float)
            or not math.isfinite(self.grip_extra_m)
            or self.grip_extra_m not in (0.0, 0.002)
            or (not self.feedback_enabled and self.grip_extra_m != 0)
        ):
            raise ValueError("E8 extra grip requires explicit bounded feedback.")

    @property
    def link_id(self) -> str:
        return f"{self.binding.object_id}::{self.binding.link}"

    def preset(self, phase: str) -> str:
        if phase not in {"ready", "chunk", "turned"}:
            raise ValueError("Unknown E8 acceptance phase.")
        return f"gen_sim.twist.{self.binding.object_id}.{phase}"

    def payload(self) -> dict:
        return asdict(self)

    @classmethod
    def decode(cls, value: Any) -> TwistRoute:
        if type(value) is not dict or set(value) != {
            "binding",
            "arm",
            "variant",
            "grasp_depth",
            "settle_steps",
            "chunk_count",
            "chunk_angle",
            "fixed_roll",
            "feedback_enabled",
            "grip_extra_m",
        }:
            raise ValueError("E8 route has missing or unexpected fields.")
        data = dict(value["binding"])
        if set(data) != {f.name for f in fields(KnobBinding)}:
            raise ValueError("E8 binding has missing or unexpected fields.")
        for key in (
            "object_id",
            "joint",
            "link",
            "parent",
            "source_sha256",
            "calibration_sha256",
        ):
            if (
                type(data[key]) is not str
                or not data[key]
                or data[key].strip() != data[key]
            ):
                raise ValueError("E8 identities must be exact nonempty strings.")
        for key in ("source_sha256", "calibration_sha256"):
            if len(data[key]) != 64 or any(
                c not in "0123456789abcdef" for c in data[key]
            ):
                raise ValueError("E8 source and semantics require SHA-256 identities.")
        lineage = data["mass_lineage_sha256"]
        if type(lineage) is not str or (
            lineage
            and (
                len(lineage) != 64 or any(c not in "0123456789abcdef" for c in lineage)
            )
        ):
            raise ValueError("E8 mass lineage must be empty or a SHA-256 identity.")
        for key, size in (
            ("axis", 3),
            ("origin", 3),
            ("outer_point", 3),
            ("limits", 2),
        ):
            if type(data[key]) not in (list, tuple) or len(data[key]) != size:
                raise ValueError("Invalid E8 geometry dimension.")
            data[key] = tuple(data[key])
        numbers = [
            data[k]
            for k in ("scale", "axis_sign", "grip_depth", "grip_width", "target_qpos")
        ]
        numbers += [
            v for k in ("axis", "origin", "outer_point", "limits") for v in data[k]
        ]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers):
            raise ValueError("E8 geometry must contain finite real numbers.")
        if (
            data["scale"] <= 0
            or data["grip_depth"] <= 0
            or data["grip_width"] <= 0
            or data["axis_sign"] not in (-1, 1)
            or not math.isclose(np.linalg.norm(data["axis"]), 1.0, abs_tol=1e-6)
            or data["limits"][0] >= data["limits"][1]
            or not data["limits"][0] <= data["target_qpos"] <= data["limits"][1]
            or type(data["target_setting"]) is not int
        ):
            raise ValueError("Invalid E8 calibrated target.")
        return cls(
            KnobBinding(**data),
            value["arm"],
            value["variant"],
            value["grasp_depth"],
            value["settle_steps"],
            value["chunk_count"],
            value["chunk_angle"],
            value["fixed_roll"],
            value["feedback_enabled"],
            value["grip_extra_m"],
        )


def discover_twist(config: dict, setting: int) -> TwistRoute:
    """Resolve source-declared roles and measure the printed-label target."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    if not _fixed_base(config):
        raise ValueError("E8 requires a fixed base.")
    scale = np.asarray(config.get("body_scale", [1.0, 1.0, 1.0]))
    if (
        scale.shape != (3,)
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.allclose(scale, scale[0])
    ):
        raise ValueError("E8 requires a positive uniform scale.")
    path = Path(config["fpath"])
    semantics = resolve_twist_semantics(path)
    digest = semantics.source_sha256
    from .twist_mass_source import adjacent_manifest_path, qualify_mass_source_copy

    mass_copy = (
        qualify_mass_source_copy(path)
        if adjacent_manifest_path(path).exists()
        else None
    )
    if mass_copy is not None and float(scale[0]) != mass_copy.scale:
        raise ValueError("E8 deployment mass scale differs from its source lineage.")
    labels = dict(semantics.settings)
    if type(setting) is not int or setting not in labels:
        raise ValueError(
            f"E8 setting {setting!r} is not calibrated; available: {sorted(labels)}."
        )
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    stage = None if layer is None else Usd.Stage.Open(layer)
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("E8 requires metre-authored USD.")
    selected_joint = stage.GetPrimAtPath(semantics.joint_path)
    if not selected_joint or not selected_joint.IsA(UsdPhysics.RevoluteJoint):
        raise ValueError("E8 semantic joint changed while binding.")
    joint_name = selected_joint.GetName()
    joints = [
        p
        for p in stage.Traverse()
        if p.IsA(UsdPhysics.Joint) and p.GetName() == joint_name
    ]
    if len(joints) != 1 or not joints[0].IsA(UsdPhysics.RevoluteJoint):
        raise ValueError("E8 requires an unambiguous calibrated revolute joint.")
    joint = UsdPhysics.RevoluteJoint(joints[0])
    parent_paths, child_paths = (
        joint.GetBody0Rel().GetTargets(),
        joint.GetBody1Rel().GetTargets(),
    )
    if (
        len(parent_paths) != 1
        or len(child_paths) != 1
        or joint.GetJointEnabledAttr().Get() is False
    ):
        raise ValueError("E8 needs one enabled parent/child joint.")
    parent, child = stage.GetPrimAtPath(parent_paths[0]), stage.GetPrimAtPath(
        child_paths[0]
    )
    if any(
        p.IsA(UsdPhysics.Joint)
        and p != joints[0]
        and any(
            t in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()
            for t in (*parent_paths, *child_paths)
        )
        for p in stage.Traverse()
    ):
        raise ValueError("E8 requires a root parent and a directly owned knob.")
    cache = UsdGeom.XformCache()
    inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(child)).T)

    def points(prim_path: str) -> np.ndarray:
        prim = stage.GetPrimAtPath(prim_path)
        transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        values = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        return values @ transform[:3, :3].T + transform[:3, 3]

    q = joint.GetLocalRot1Attr().Get()
    axis = (
        np.asarray(Gf.Matrix3d(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())))).T
        @ np.eye(3)["XYZ".index(str(joint.GetAxisAttr().Get()))]
    )
    axis /= np.linalg.norm(axis)
    origin = np.asarray(joint.GetLocalPos1Attr().Get(), dtype=float)
    pointer = points(semantics.pointer_path).mean(0) - origin
    label_points = np.concatenate([points(p) for p in labels[setting]])
    label = (label_points.min(0) + label_points.max(0)) / 2 - origin
    pointer -= axis * np.dot(pointer, axis)
    label -= axis * np.dot(label, axis)
    if min(np.linalg.norm(pointer), np.linalg.norm(label)) < 1e-5:
        raise ValueError("E8 pointer or label has no measurable angular direction.")
    angle = float(
        np.arctan2(np.dot(axis, np.cross(pointer, label)), np.dot(pointer, label))
    )
    limits = tuple(
        math.radians(float(v))
        for v in (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
    )
    targets = [
        angle + k * 2 * math.pi
        for k in (-1, 0, 1)
        if limits[0] - 1e-6 <= angle + k * 2 * math.pi <= limits[1] + 1e-6
    ]
    if len(targets) != 1:
        raise ValueError("E8 label does not identify one reachable joint coordinate.")
    grip = points(semantics.grip_path)
    radial = grip - origin
    radial -= (radial @ axis)[:, None] * axis
    outward_sign = math.copysign(1.0, np.dot(grip.mean(0) - origin, axis))
    inward = -outward_sign * axis
    projection = grip @ inward
    outer = grip[np.isclose(projection, projection.min(), atol=1e-6)].mean(0)
    binding = KnobBinding(
        str(config["uid"]),
        joint_name,
        child.GetName(),
        parent.GetName(),
        digest,
        float(scale[0]),
        tuple(map(float, inward)),
        -outward_sign,
        tuple(map(float, origin * scale[0])),
        tuple(map(float, outer * scale[0])),
        float(np.ptp(projection) * scale[0]),
        float(2 * np.linalg.norm(radial, axis=1).max() * scale[0]),
        limits,
        setting,
        min(limits[1], max(limits[0], targets[0])),
        semantics.semantics_sha256,
        "" if mass_copy is None else mass_copy.manifest_sha256,
    )
    if (
        hashlib.sha256(path.read_bytes()).hexdigest() != digest
        or resolve_twist_semantics(path) != semantics
    ):
        raise ValueError("E8 source or semantics changed while binding.")
    return TwistRoute.decode(
        TwistRoute(
            binding, chunk_count=required_chunk_count(binding.target_qpos)
        ).payload()
    )


def enrich_twist_inventory(objects: list[dict], articulations: list[dict]) -> None:
    """Expose only source-qualified label evidence to ordinary scene binding."""
    configs = {a["uid"]: a for a in articulations}
    for item in objects:
        if item.get("role") != "articulation":
            continue
        config = configs[item["runtime_uid"]]
        if not _fixed_base(config):
            continue
        try:
            semantics = resolve_twist_semantics(config["fpath"])
        except MissingTwistSemanticsError:
            continue
        routes = {k: discover_twist(config, k) for k, _ in semantics.settings}
        values = {k: r.binding.target_qpos for k, r in routes.items()}
        attributes = item.setdefault("attributes", {})
        joint = next(iter(routes.values())).binding.joint
        attributes["joint_settings"] = {joint: list(values.values())}
        attributes["twist_setting_labels"] = {str(k): v for k, v in values.items()}
        attributes["twist_source_sha256"] = semantics.source_sha256
        attributes["twist_semantics_sha256"] = semantics.semantics_sha256
        attributes["twist_control_joint"] = joint


def recipe(
    object_id: str, arm: str, setting: int, *, chunk_count: int = TWIST_CHUNK_COUNT
) -> list[dict]:
    return (
        [
            {
                "kind": "registered",
                "call_id": call,
                "arguments": {"object": object_id, "setting": setting},
                "resources": {"primary": arm},
            }
            for call in (PREPARE_CALL,)
        ]
        + [
            {
                "kind": "registered",
                "call_id": TWIST_CALL,
                "arguments": {"object": object_id, "setting": setting},
                "resources": {"primary": arm},
            }
            for _ in range(chunk_count)
        ]
        + [
            {
                "kind": "registered",
                "call_id": PARK_CALL,
                "arguments": {},
                "resources": {"primary": arm},
            }
        ]
    )


def graph_routes(graph: dict, scene: Any) -> tuple[TwistRoute, ...]:
    groups = [g for g in graph.get("task_groups", ()) if g["task_type"] == "E8"]
    if not groups:
        return ()
    if len(groups) != 1 or len(graph["task_groups"]) != 1:
        raise ValueError(
            "E8 MVP requires one standalone calibrated knob operation; "
            "split E1 transport and E8 twist into separate qualified deployments."
        )
    nodes = {n["id"]: n for n in graph["nodes"]}
    sequence = [nodes[key] for key in groups[0]["node_ids"]]
    call = sequence[0]["call"]
    if set(call["arguments"]) != {"object", "setting"}:
        raise ValueError("E8 requires an exact object and calibrated setting.")
    uid, setting, arm = (
        call["arguments"]["object"],
        call["arguments"]["setting"],
        call["resources"]["primary"],
    )
    configs = {a["uid"]: a for a in scene.articulations}
    if uid not in configs:
        raise ValueError("E8 must bind an articulation.")
    route = replace(discover_twist(configs[uid], setting), arm=arm)
    if (
        [n["call"] for n in sequence]
        != recipe(uid, arm, setting, chunk_count=route.chunk_count)
        or [n["role"] for n in sequence]
        != ["primary"] * (route.chunk_count + 1) + ["cleanup"]
        or any(b["depends_on"] != [a["id"]] for a, b in zip(sequence, sequence[1:]))
    ):
        raise ValueError("E8 requires prepare/bounded-Twist/Park in order.")
    return (route,)


def validate_program(program: dict, route: TwistRoute) -> None:
    items = program["program"]["items"]
    expected_recipe = recipe(
        route.binding.object_id,
        route.arm,
        route.binding.target_setting,
        chunk_count=route.chunk_count,
    )
    if len(items) != len(expected_recipe):
        raise ValueError("E8 requires the complete bounded-twist program.")
    for index, (item, expected) in enumerate(zip(items, expected_recipe)):
        phase = (
            "ready"
            if index == 0
            else (
                "turned"
                if index == len(items) - 1 or index == len(items) - 2
                else "chunk"
            )
        )
        post = [
            {
                "kind": "wait_stable",
                "entity": route.binding.object_id,
                "preset": route.preset(phase),
            }
        ]
        if item.get("steps", {}).get("call") != expected or item.get("post") != post:
            raise ValueError("E8 calls and physical post-policies cannot be removed.")
