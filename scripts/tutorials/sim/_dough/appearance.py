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

"""Native rendering of the packaged loaf meshes and particle skin bindings."""

from __future__ import annotations

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager
from examples.sim.mpm._scene import DemoScene, preview_material

__all__ = ["LoafSkin"]

_SKIN_NAMES = ("intact_crust", "slice_crust", "loaf_crust", "slice_crumb", "loaf_crumb")


class LoafSkin:
    """Bind the authored crust and cut faces to their material points."""

    def __init__(self, sim: SimulationManager, scene: DemoScene) -> None:
        from pxr import UsdGeom

        self.world = sim._world
        self.ids = list(range(len(_SKIN_NAMES)))
        meshes = [
            UsdGeom.Mesh.Get(scene.stage, scene.spec["shell_prims"][name])
            for name in _SKIN_NAMES
        ]

        arrays = [[] for _ in range(6)]
        with np.load(scene.path("skin_bindings"), allow_pickle=False) as bindings:
            for name, mesh in zip(_SKIN_NAMES, meshes, strict=True):
                tri = np.asarray(
                    mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32
                ).reshape(-1, 3)
                vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                uv = np.asarray(
                    UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get(),
                    dtype=np.float32,
                )
                for dest, values in zip(
                    arrays,
                    (
                        tri,
                        vertices,
                        uv,
                        bindings[f"{name}_particle_ids"],
                        bindings[f"{name}_weights"],
                        bindings[f"{name}_rest_offsets"],
                    ),
                    strict=True,
                ):
                    dest.append(
                        torch.as_tensor(
                            np.require(values, requirements=["C", "W"]),
                            device=sim.device,
                        )
                    )
        triangles, vertices, texcoords, indices, weights, offsets = arrays
        for name, start, end in (("crust", 0, 3), ("crumb", 3, 5)):
            uid = "bread_" + name
            config = preview_material(meshes[start].GetPrim(), uid)
            mat = sim.create_visual_material(config).get_instance(uid).mat
            material_id = mat.get_material_id()
            section = slice(start, end)
            result = self.world.register_soft_meshes_to_renderer(
                self.ids[section],
                material_id,
                material_id,
                triangles[section],
                vertices[section],
                texcoords[section],
                indices[section],
                weights[section],
                visible_flags=[i == 0 for i in self.ids[section]],
                rest_offsets_list=offsets[section],
            )
            if any(r < 0 for r in result):
                raise RuntimeError(f"Native bread mesh registration failed: {result}")

    def update(self, q: torch.Tensor, split: bool) -> None:
        """Reconstruct the intact loaf or the separated pieces from live points."""
        flags = [not split, split, split, split, split]
        result = self.world.sync_soft_meshes_to_renderer(
            self.ids, q.contiguous(), visible_flags=flags, update_flags=flags
        )
        if any(r < 0 for r in result):
            raise RuntimeError(f"Native bread mesh sync failed: {result}")
