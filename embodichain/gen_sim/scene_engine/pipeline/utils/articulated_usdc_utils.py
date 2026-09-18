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


from __future__ import annotations

from pathlib import Path
import math
import re


def _read_revolute_qpos_limits(usdc_path: str | Path) -> dict[str, list[float]]:
    """Preserve USD angular limits when a backend omits degree conversion."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usdc_path))
    if stage is None:
        raise ValueError(f"Cannot open articulated USDC: {usdc_path}")
    limits = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        name = re.escape(prim.GetName())
        if name in limits:
            raise ValueError(f"Ambiguous revolute joint name: {prim.GetName()}")
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        if (
            lower is None
            or upper is None
            or math.isnan(lower)
            or math.isnan(upper)
            or lower >= upper
        ):
            raise ValueError(f"Invalid revolute limits: {prim.GetPath()}")
        if math.isfinite(lower) and math.isfinite(upper):
            limits[name] = [math.radians(lower), math.radians(upper)]
    return limits


import numpy as np
from scipy.spatial.transform import Rotation


def _articulation_root_bottom_z(
    usdc_path: str | Path, body_scale: list[float], rotation: list[float]
) -> float:
    """Measure authored collision meshes relative to the native root rigid body.

    Root placement transforms are intentionally removed: runtime reset replaces
    that pose. This measure is for the authored rest state, not a motion sweep.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usdc_path))
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError(f"Missing USD stage/default prim: {usdc_path}")
    prims = list(Usd.PrimRange(stage.GetDefaultPrim()))
    bodies = {p.GetPath(): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    children = set()
    for p in prims:
        if p.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(p)
            if joint.GetBody0Rel().GetTargets():
                children.update(joint.GetBody1Rel().GetTargets())
    roots = set(bodies) - children
    if len(roots) != 1:
        raise ValueError(f"Expected one root rigid body in {usdc_path}: {roots}")
    cache = UsdGeom.XformCache()
    root_world = np.array(cache.GetLocalToWorldTransform(bodies[roots.pop()])).T
    root_from_world = np.linalg.inv(root_world)
    axis = UsdGeom.GetStageUpAxis(stage)
    if axis not in {UsdGeom.Tokens.y, UsdGeom.Tokens.z}:
        raise ValueError(f"Unsupported USD up axis: {axis}")
    basis = (
        np.eye(3)
        if axis == UsdGeom.Tokens.z
        else Rotation.from_euler("x", 90, degrees=True).as_matrix()
    )
    scale = np.asarray(body_scale, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Articulation scale must be finite and positive.")
    world_rotation = Rotation.from_euler("XYZ", rotation, degrees=True).as_matrix()
    minimum = float("inf")
    for p in prims:
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False:
            continue
        if not p.IsA(UsdGeom.Mesh):
            raise ValueError(
                f"Height placement requires collision meshes: {p.GetPath()}"
            )
        points = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=float)
        if not points.size:
            continue
        t = root_from_world @ np.array(cache.GetLocalToWorldTransform(p)).T
        points = (points @ t[:3, :3].T + t[:3, 3]) * scale
        points = points @ basis.T @ world_rotation.T
        if not np.isfinite(points).all():
            raise ValueError(f"Nonfinite collision vertices: {p.GetPath()}")
        minimum = min(minimum, float(points[:, 2].min()))
    if not np.isfinite(minimum):
        raise ValueError(f"No enabled collision mesh in {usdc_path}")
    return minimum


def _canonicalize_articulated_usdc_bottom_center(usdc_path: str | Path) -> Path:
    """Translate the authored hierarchy to its declared-axis bottom center.

    The translation is authored on the default articulation root, so every link,
    collider, and joint frame moves together without changing their relative
    articulation structure. This does not rebase the runtime root rigid body;
    the exporter separately computes its collision-based deployment height.
    """
    resolved_usdc_path = Path(usdc_path).expanduser().resolve()
    if not resolved_usdc_path.is_file() or resolved_usdc_path.suffix.lower() != ".usdc":
        raise FileNotFoundError(
            "Articulated USDC canonicalization requires an existing .usdc file: "
            f"{resolved_usdc_path}"
        )
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "Articulated USDC canonicalization requires the USD Python bindings."
        ) from exc

    stage = Usd.Stage.Open(str(resolved_usdc_path))
    if stage is None:
        raise ValueError(f"Cannot open articulated USDC: {resolved_usdc_path}")
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        raise ValueError(
            f"Articulated USDC has no valid default prim: {resolved_usdc_path}"
        )
    if not root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError(
            "Articulated USDC default prim must have ArticulationRootAPI: "
            f"{root_prim.GetPath()}"
        )

    root_xformable = UsdGeom.Xformable(root_prim)
    if not root_xformable:
        raise ValueError(
            "Articulated USDC default prim must be transformable: "
            f"{root_prim.GetPath()}"
        )
    op_name = "xformOp:translate:scene_engine_bottom_center"
    if any(op.GetOpName() == op_name for op in root_xformable.GetOrderedXformOps()):
        raise ValueError(
            "Articulated USDC is already canonicalized at its bottom center: "
            f"{resolved_usdc_path}"
        )

    bounds = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        .ComputeLocalBound(root_prim)
        .ComputeAlignedBox()
    )
    if bounds.IsEmpty():
        raise ValueError(
            f"Articulated USDC default prim has an empty bound: {root_prim.GetPath()}"
        )
    minimum, maximum = bounds.GetMin(), bounds.GetMax()
    up_index = 1 if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y else 2
    bottom_center = (minimum + maximum) / 2.0
    bottom_center[up_index] = minimum[up_index]
    # Shift the complete articulation hierarchy into the shared SimReady origin.
    root_xformable.AddTranslateOp(opSuffix="scene_engine_bottom_center").Set(
        -bottom_center
    )
    stage.GetRootLayer().Save()
    return resolved_usdc_path
