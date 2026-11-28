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

"""Load a single-arena DexUni scene from the published demo asset bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RenderCfg,
    RigidObjectCfg,
    VolumeDeformableObjectCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.objects.deformable import DeformableObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import VisualizationCfg

if TYPE_CHECKING:
    from pxr import Usd

__all__ = ["DemoScene", "open_view", "preview_material"]


class DemoScene:
    """Read the authored geometry, material points, and solver settings once."""

    def __init__(self, name: str) -> None:
        from pxr import Usd, UsdGeom

        manifest_path = Path(get_data_path("DexUniMpmDemoAssets/manifest.json"))
        self.root = manifest_path.resolve().parent
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest["schema_version"] != 1:
            raise ValueError(f"Unsupported DexUni MPM asset manifest: {manifest_path}")
        self.spec = manifest["scenes"][name]
        self.stage = Usd.Stage.Open(str(self.path("scene")))
        if (
            UsdGeom.GetStageMetersPerUnit(self.stage) != 1
            or UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z
        ):
            raise ValueError("DexUni MPM demo scenes must use metres and Z-up axes.")

        cloud = UsdGeom.Points.Get(self.stage, self.spec["particle_prim"])
        if not cloud:
            raise ValueError(
                f"Missing material point prim: {self.spec['particle_prim']}"
            )
        self.particle_uid = cloud.GetPrim().GetName()
        with np.load(self.path("material_points"), allow_pickle=False) as data:
            self.positions = data["positions"]
            self.masses = data["particle_masses"]
            self.radius = float(data["particle_radius"])
        count = self.spec["count"]
        if self.positions.shape != (count, 3) or self.masses.shape != (count,):
            raise ValueError(
                f"Inconsistent particle counts in {self.path('material_points')}"
            )

        self.pose = np.asarray(self.spec["pose"], dtype=np.float32)
        self.material: dict[str, float] = self.spec["material"]
        settings = dict(self.spec["simulation"])
        self.steps: int = settings.pop("steps")
        self.enable_ground_plane: bool = settings.pop("enable_ground_plane")
        self.physics = NewtonPhysicsCfg(device="cuda:0", **settings)

    def path(self, key: str) -> Path:
        """Resolve a manifest file relative to this asset's extraction root."""
        path = (self.root / self.spec[key]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Invalid bundle path for {key}: {self.spec[key]!r}")
        return path

    def create_simulation(
        self,
        *,
        headless: bool,
        render_cfg: RenderCfg | None = None,
        visualization: VisualizationCfg | None = None,
    ) -> SimulationManager:
        """Create a manager using the bundle's solver and step settings."""
        visualization = visualization or VisualizationCfg()
        renderer = (
            "no-render" if headless and visualization.backend != "dexsim-web" else "rt"
        )
        return SimulationManager(
            SimulationManagerCfg(
                physics_cfg=self.physics,
                headless=headless,
                enable_ground_plane=self.enable_ground_plane,
                startup_summary="off",
                render_cfg=render_cfg or RenderCfg(renderer=renderer),
                visualization=visualization,
            )
        )

    def add_environment(self, sim: SimulationManager) -> None:
        """Load the authored static boxes with their primitive collision shapes."""
        from pxr import UsdGeom

        transforms = UsdGeom.XformCache()
        env = self.stage.GetDefaultPrim().GetChild("Environment")
        for prim in env.GetChildren():
            if prim.GetTypeName() == "Plane":
                # The manifest enables the manager's shared ground plane.
                continue
            pose = np.asarray(
                transforms.GetLocalToWorldTransform(prim), dtype=np.float32
            ).T.copy()
            if prim.IsA(UsdGeom.Cube):
                scale = np.linalg.norm(pose[:3, :3], axis=0)
                size = UsdGeom.Cube(prim).GetSizeAttr().Get() * scale
                pose[:3, :3] /= scale
            elif prim.IsA(UsdGeom.Mesh):
                # The textured board mesh is authored as a collision box.
                low, high = np.asarray(UsdGeom.Mesh(prim).GetExtentAttr().Get())
                size = high - low
                pose[:3, 3] += pose[:3, :3] @ ((low + high) * 0.5)
            else:
                raise ValueError(f"Unsupported MPM demo collider: {prim.GetPath()}")
            material = (
                preview_material(prim, prim.GetName())
                if sim.has_native_renderer
                else None
            )
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=prim.GetName(),
                    body_type="static",
                    init_local_pose=pose,
                    shape=CubeCfg(size=size.tolist(), visual_material=material),
                )
            )

    def add_material(
        self, sim: SimulationManager, **overrides: float
    ) -> DeformableObject:
        """Bind the original NPZ point order and per-point masses."""
        return sim.add_deformable_object(
            VolumeDeformableObjectCfg.from_dict(
                {
                    "uid": self.particle_uid,
                    "particle_radius": self.radius,
                    "init_local_pose": self.pose,
                    "model": {
                        "kind": "material_point",
                        "sampling": {"kind": "points", "points": self.positions},
                        "physics": {
                            "particle_masses": self.masses,
                            **self.material,
                            **overrides,
                        },
                    },
                }
            )
        )


def preview_material(prim: Usd.Prim, uid: str) -> VisualMaterialCfg:
    """Read this bundle's USD Preview Surface color or relative texture."""
    from pxr import UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    shader, _, _ = material.ComputeSurfaceSource()
    diffuse = shader.GetInput("diffuseColor")
    connection = diffuse.GetConnectedSource()
    texture = None
    rgb = list(diffuse.Get())
    if connection is not None:
        image = UsdShade.Shader(connection[0].GetPrim())
        texture = image.GetInput("file").Get().resolvedPath
        if not texture:
            raise FileNotFoundError(
                f"Texture for {prim.GetPath()} could not be resolved."
            )
        rgb = [1.0, 1.0, 1.0]
    return VisualMaterialCfg(
        uid=uid,
        base_color=[*rgb, 1.0],
        base_color_texture=texture,
        metallic=shader.GetInput("metallic").Get(),
        roughness=shader.GetInput("roughness").Get(),
    )


def open_view(
    sim: SimulationManager,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    """Frame this small tabletop scene in the native viewer."""
    sim.open_window()
    sim.viewer.set_look_at(
        eye=np.array(eye), look_at=np.array(target), up=np.array([0.0, 0.0, 1.0])
    )
