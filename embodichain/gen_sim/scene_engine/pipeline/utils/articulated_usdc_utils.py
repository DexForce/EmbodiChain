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


def _canonicalize_articulated_usdc_bottom_center(usdc_path: str | Path) -> Path:
    """Place one y-up articulation's local origin at its bottom AABB center.

    The translation is authored on the default articulation root, so every link,
    collider, and joint frame moves together without changing their relative
    articulation structure.
    """
    resolved_usdc_path = Path(usdc_path).expanduser().resolve()
    if not resolved_usdc_path.is_file() or resolved_usdc_path.suffix.lower() != ".usdc":
        raise FileNotFoundError(
            "Articulated USDC canonicalization requires an existing .usdc file: "
            f"{resolved_usdc_path}"
        )
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
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
    bottom_center = Gf.Vec3d(
        (minimum[0] + maximum[0]) / 2.0,
        minimum[1],
        (minimum[2] + maximum[2]) / 2.0,
    )
    # Shift the complete articulation hierarchy into the shared SimReady origin.
    root_xformable.AddTranslateOp(opSuffix="scene_engine_bottom_center").Set(
        -bottom_center
    )
    stage.GetRootLayer().Save()
    return resolved_usdc_path
