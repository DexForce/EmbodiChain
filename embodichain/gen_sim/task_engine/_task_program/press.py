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

"""Source-qualified E9 bindings and contact-backed shallow press events."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "PressBinding",
    "PressSample",
    "evaluate_press_event",
    "inspect_press",
]


@dataclass(frozen=True, slots=True)
class PressBinding:
    """One explicitly selected, source-qualified prismatic button."""

    object_id: str
    joint: str
    link: str
    parent: str
    source_path: str
    source_sha256: str
    collision_mesh: str
    scale: float
    axis: tuple[float, float, float]
    limits: tuple[float, float]
    press_position: tuple[float, float, float]
    released_position: float
    pressed_position: float

    @property
    def stroke(self) -> float:
        """Return the full authored travel in runtime metres."""
        return abs(self.pressed_position - self.released_position)


def inspect_press(
    config: dict[str, Any],
    *,
    joint: str,
    link: str,
    released_position: float = 0.0,
    collision_mesh: str | None = None,
) -> PressBinding:
    """Inspect an explicit button without guessing identities or changing physics.

    The released coordinate is expressed in runtime metres and must identify a
    joint endpoint. The default supports only buttons authored released at zero.
    It is not evidence that the loaded button is actually released.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from .articulation_binding import _fixed_base

    if not _fixed_base(config):
        raise ValueError("E9 requires a declared fixed base.")
    scale = np.asarray(config.get("body_scale", (1.0, 1.0, 1.0)), dtype=float)
    if (
        scale.shape != (3,)
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.allclose(scale, scale[0], atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("E9 requires positive uniform body_scale.")
    for value in (config.get("uid"), joint, link):
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError("E9 requires exact object, joint and link identities.")
    path = Path(config["fpath"]).resolve()
    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("E9 requires a self-contained USD asset.")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    stage = Usd.Stage.Open(str(path))
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("E9 requires metre-authored USD geometry.")
    if any(
        not layer.anonymous and layer != stage.GetRootLayer()
        for layer in stage.GetUsedLayers()
    ):
        raise ValueError("E9 requires a self-contained USD layer.")
    selected = [p for p in stage.Traverse() if p.GetName() == joint]
    bodies = [p for p in stage.Traverse() if p.GetName() == link]
    if len(selected) != 1 or len(bodies) != 1:
        raise ValueError("E9 joint/link selection is missing or ambiguous.")
    prim, body = selected[0], bodies[0]
    if not prim.IsA(UsdPhysics.PrismaticJoint):
        raise ValueError("E9 requires an enabled prismatic joint.")
    slider = UsdPhysics.PrismaticJoint(prim)
    if slider.GetJointEnabledAttr().Get() is False:
        raise ValueError("E9 requires an enabled prismatic joint.")
    parents, children = (
        slider.GetBody0Rel().GetTargets(),
        slider.GetBody1Rel().GetTargets(),
    )
    if len(parents) != 1 or children != [body.GetPath()]:
        raise ValueError("E9 joint must directly own its selected link.")
    parent = stage.GetPrimAtPath(parents[0])
    if not parent.HasAPI(UsdPhysics.RigidBodyAPI) or not body.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        raise ValueError("E9 requires rigid parent and button bodies.")
    incoming_joints = [
        p
        for p in stage.Traverse()
        if p.IsA(UsdPhysics.Joint)
        and p != prim
        and UsdPhysics.Joint(p).GetJointEnabledAttr().Get() is not False
        and any(
            target in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()
            for target in (parents[0], body.GetPath())
        )
    ]
    if incoming_joints:
        raise ValueError("E9 requires a root-body parent and unique button ownership.")
    limits = tuple(
        float(v) * scale[0]
        for v in (slider.GetLowerLimitAttr().Get(), slider.GetUpperLimitAttr().Get())
    )
    if not np.isfinite(limits).all() or limits[1] - limits[0] <= 1e-5:
        raise ValueError("E9 requires finite, distinguishable joint endpoints.")
    if not math.isfinite(released_position) or not any(
        math.isclose(released_position, v, abs_tol=1e-7) for v in limits
    ):
        raise ValueError("E9 released_position must be a joint endpoint.")
    released = min(limits, key=lambda v: abs(v - released_position))
    pressed = max(limits, key=lambda v: abs(v - released))
    q = slider.GetLocalRot1Attr().Get()
    axis = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[str(slider.GetAxisAttr().Get())]]
    axis = (
        np.asarray(Gf.Matrix3d(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())))).T
        @ axis
    )
    axis = axis / np.linalg.norm(axis) * math.copysign(1.0, pressed - released)

    def owned_mesh(p: Any) -> bool:
        owner = p.GetParent()
        while (
            owner
            and not owner.IsPseudoRoot()
            and not owner.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            owner = owner.GetParent()
        return owner == body

    meshes = [
        p
        for p in Usd.PrimRange(body)
        if p.IsA(UsdGeom.Mesh)
        and owned_mesh(p)
        and p.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
        and (collision_mesh is None or str(p.GetPath()) == collision_mesh)
    ]
    if len(meshes) != 1:
        raise ValueError("E9 requires one explicit, unambiguous collision mesh.")
    mesh = UsdGeom.Mesh(meshes[0])
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
        or not len(indices)
        or (counts < 3).any()
        or counts.sum() != len(indices)
        or indices.min() < 0
        or indices.max() >= len(points)
    ):
        raise ValueError("E9 collision mesh topology is invalid.")
    cache = UsdGeom.XformCache()
    body_pose = np.asarray(cache.GetLocalToWorldTransform(body)).T
    parent_pose = np.asarray(cache.GetLocalToWorldTransform(parent)).T
    for pose in (body_pose, parent_pose):
        if not np.isfinite(pose).all() or not np.allclose(
            pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5
        ):
            raise ValueError("E9 requires unscaled rigid-body frames.")
    transform = (
        np.linalg.inv(body_pose)
        @ np.asarray(cache.GetLocalToWorldTransform(meshes[0])).T
    )
    points = (points @ transform[:3, :3].T + transform[:3, 3]) * scale[0]
    projections = points @ axis
    surface = points[np.isclose(projections, projections.min(), atol=1e-6, rtol=0)]
    if not np.isfinite(surface).all() or not len(surface):
        raise ValueError("E9 needs a finite outer contact point.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise ValueError("E9 source changed during inspection.")
    return PressBinding(
        config["uid"],
        joint,
        link,
        parent.GetName(),
        str(path),
        source_hash,
        str(meshes[0].GetPath()),
        float(scale[0]),
        tuple(map(float, axis)),
        tuple(map(float, limits)),
        tuple(map(float, surface.mean(axis=0))),
        float(released),
        float(pressed),
    )


@dataclass(frozen=True, slots=True)
class PressSample:
    """Measured state after the preceding command; contact is exact target/hand."""

    timestamp: float
    qpos: float
    phase: str
    target_contact: bool | None
    valid: bool = True


def evaluate_press_event(
    binding: PressBinding, samples: Iterable[PressSample]
) -> dict[str, Any]:
    """Require a released onset and continuous contact-backed press-phase travel.

    Callers must mark missing, stale, ambiguous or saturated contact observations
    invalid. This records a transient press event, not persistent activation or
    a terminal Task Program success certificate.
    """
    trace = tuple(samples)
    tolerance = min(0.0005, binding.stroke * 0.1)
    direction = math.copysign(1.0, binding.pressed_position - binding.released_position)
    result: dict[str, Any] = {
        "accepted": False,
        "criterion": "contact_press_0.25mm",
        "minimum_contact_stroke_m": 0.00025,
        "activation_verified": False,
        "reason": "missing_samples",
        "max_stroke": 0.0,
        "contact_stroke": 0.0,
        "contact_peak": 0.0,
    }
    if not trace:
        return result
    times = [s.timestamp for s in trace]
    if any(
        not s.valid
        or not math.isfinite(s.qpos)
        or not math.isfinite(s.timestamp)
        or s.timestamp < 0
        or type(s.target_contact) is not bool
        for s in trace
    ) or any(b <= a for a, b in zip(times, times[1:])):
        return {**result, "reason": "invalid_observation"}
    travel = [(s.qpos - binding.released_position) * direction for s in trace]
    result["max_stroke"] = max(travel)
    if any(p < -tolerance or p > binding.stroke + tolerance for p in travel):
        return {**result, "reason": "joint_out_of_bounds"}
    if (
        abs(travel[0]) > tolerance
        or trace[0].phase == "press"
        or trace[0].target_contact
    ):
        return {**result, "reason": "not_initially_released"}
    started = False
    anchor: float | None = None
    previous = 0.0
    for sample, progress in zip(trace, travel):
        if sample.phase != "press":
            if not started and progress > tolerance:
                return {**result, "reason": "passive_before_press"}
            anchor = None
            continue
        started = True
        if not sample.target_contact:
            anchor = None
            continue
        if anchor is None:
            if progress > tolerance:
                continue
            anchor = progress
        if progress < previous - tolerance:
            anchor = None
        elif anchor is not None:
            result["contact_stroke"] = max(result["contact_stroke"], progress - anchor)
            result["contact_peak"] = max(result["contact_peak"], progress)
        previous = progress
    if result["contact_stroke"] >= result["minimum_contact_stroke_m"]:
        return {**result, "accepted": True, "reason": "contact_press_event"}
    return {**result, "reason": "insufficient_contact_stroke"}
