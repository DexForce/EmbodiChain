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

"""Source-qualified prismatic control geometry without runtime acceptance."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from embodichain.lab.sim.spawn.source_geometry import UsdSourceSnapshot, read_usd_source

__all__ = ["PressBinding", "inspect_press", "discover_press_source"]


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
    source: UsdSourceSnapshot | None = None,
) -> PressBinding:
    """Inspect an explicit button without guessing identities or changing physics.

    The released coordinate is expressed in runtime metres and must identify a
    joint endpoint. The default supports only buttons authored released at zero.
    It is not evidence that the loaded button is actually released.

    Args:
        config: Fixed-base physical asset declaration.
        joint: Exact authored prismatic joint name.
        link: Exact authored child link name.
        released_position: Authored endpoint in runtime metres.
        collision_mesh: Optional absolute collider path; omission requires one unambiguous mesh.
        source: Optional shared static read matching the declared asset.

    Returns:
        Stroke, direction and outer contact point without evidence of execution.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from .articulation_inputs import _fixed_base

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
    source = read_usd_source(config["fpath"] if source is None else source)
    source.assert_current(config["fpath"])
    source.require_geometry()
    path = source.path
    source_hash = source.source_sha256
    stage = source.stage
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
    points, _, _ = source.mesh_arrays(str(meshes[0].GetPath()))
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
    source.assert_current()
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


def discover_press_source(config: dict[str, Any]) -> tuple[PressBinding, float]:
    """Resolve one explicitly authored button surface and moving-link mass.

    Args:
        config: Source articulation declaration consumed by ``inspect_press``.

    Returns:
        Qualified Press binding and authored moving-link mass, preserving the
        original named-joint/surface disambiguation and source geometry order.
        It creates no control operation, robot, sensor or simulation.
    """
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
    return outer[0], float(source_mass)
