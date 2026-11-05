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
"""Source-declared revolute controls and calibrated absolute joint targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from embodichain.lab.sim.spawn.source_geometry import UsdSourceSnapshot, read_usd_source

from .articulation_inputs import _fixed_base
from .control_semantics import resolve_twist_semantics

__all__ = ["RevoluteControlBinding", "inspect_revolute_control"]


@dataclass(frozen=True, slots=True)
class RevoluteControlBinding:
    """Explicit authored control roles and their calibrated physical target.

    Execution chunk limits and acceptance tolerances are supplied separately by
    semantic integration and action policy, never inferred from this geometry.
    """

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

    def __post_init__(self) -> None:
        data = asdict(self)
        _validate_binding(data)
        for name in ("axis", "origin", "outer_point", "limits"):
            object.__setattr__(self, name, data[name])


def _validate_binding(data: dict[str, Any]) -> None:
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
        if len(data[key]) != 64 or any(c not in "0123456789abcdef" for c in data[key]):
            raise ValueError("E8 source and semantics require SHA-256 identities.")
    lineage = data["mass_lineage_sha256"]
    if type(lineage) is not str or (
        lineage
        and (len(lineage) != 64 or any(c not in "0123456789abcdef" for c in lineage))
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
    numbers += [v for k in ("axis", "origin", "outer_point", "limits") for v in data[k]]
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


def inspect_revolute_control(
    config: dict[str, Any], setting: int, *, source: UsdSourceSnapshot | None = None
) -> RevoluteControlBinding:
    """Resolve source-declared roles and measure the printed-label target.

    Args:
        config: Fixed-base asset declaration with positive uniform scale.
        setting: An explicitly annotated printed setting label.
        source: Optional shared static read matching the declared asset.

    Returns:
        Source-qualified geometry and absolute joint target; execution budgets remain separate.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

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
    source = read_usd_source(config["fpath"] if source is None else source)
    source.assert_current(config["fpath"])
    path = source.path
    semantics = resolve_twist_semantics(source)
    digest = semantics.source_sha256
    from .control_mass_source import adjacent_manifest_path, qualify_mass_source_copy

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
    stage = source.stage
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
    binding = RevoluteControlBinding(
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
    semantics.assert_current(source)
    return binding
