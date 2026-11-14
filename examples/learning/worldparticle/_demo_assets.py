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
"""Local official samples and display topology; no dynamics or reference replay."""

from __future__ import annotations
from pathlib import Path

import numpy as np

PROFILES = {
    "fancy_fluid": dict(
        checkpoint="fancy_fluid",
        sample="data/fancy_fluid/66/66.npz",
        steps=199,
        title="WorldParticle / Fancy Fluid",
        up="y",
        reference=True,
        stride=1,
    ),
    "cloth_on_ball": dict(
        checkpoint="cloth_on_ball",
        sample="data/cloth_on_ball/10/10.npz",
        steps=80,
        title="WorldParticle / Cloth on Ball",
        up="y",
        reference=True,
        stride=1,
    ),
    "inverse_design": dict(
        checkpoint="inverse_design",
        sample="data/inverse_design/0/0.npz",
        steps=15,
        title="WorldParticle / Duck deformation (forward inference)",
        up="z",
        reference=True,
        stride=5,
    ),
    "deep_fluid": dict(
        checkpoint="deep_lagrange_dataset",
        sample="data/deep_lagrange_demo/initial.npz",
        steps=199,
        title="WorldParticle / Deep Fluid (custom initial state)",
        up="y",
        reference=False,
        stride=1,
    ),
}


def add_boundary_visuals(
    builder, points, normals, rotation, scene_kind, particle_radius, show_cloud=False
):
    """Add display-only boundary geometry through Spawn; no collision shapes."""
    from dexsim.spawn import ObjectDesc, RenderDesc, GeometryDesc, MaterialDesc

    material = MaterialDesc(
        name="neural_boundary",
        base_color=(0.45, 0.49, 0.55, 1.0),
        roughness=0.7,
        double_sided=True,
    )
    if scene_kind == "cloth_on_ball":
        floor = np.abs(points[:, 1] - points[:, 1].min()) < 1e-5
        sphere = points[~floor].astype(np.float64)
        fit = np.linalg.lstsq(
            np.c_[2 * sphere, np.ones(len(sphere))],
            np.sum(sphere**2, axis=1),
            rcond=None,
        )[0]
        radii = np.linalg.norm(sphere - fit[:3], axis=1)
        radius = radii.mean()
        if np.max(np.abs(radii - radius)) < max(1e-5, radius * 0.001):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = fit[:3] @ rotation.T
            builder.add_sphere(
                "boundary_ball", float(radius), pose=pose, material=material
            )
            points, normals = points[floor], normals[floor]
    if not show_cloud:
        world_points = points @ rotation.T
        low, high = world_points.min(0), world_points.max(0)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2] - 0.015]
        builder.add_cube(
            "display_floor",
            (max(high[0] - low[0], 0.5), max(high[1] - low[1], 0.5), 0.03),
            pose=pose,
            material=material,
        )
        return
    # Sparse surfels visualize the exact input boundary cloud. They are not
    # numerical collision geometry and do not replace the full network input.
    stride = max(1, int(np.ceil(len(points) / 4000)))
    p = points[::stride] @ rotation.T
    n = normals[::stride] @ rotation.T
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    axis = np.tile([0.0, 0.0, 1.0], (len(p), 1))
    axis[np.abs(n[:, 2]) > 0.9] = [1.0, 0.0, 0.0]
    u = np.cross(n, axis)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-8)
    v = np.cross(n, u)
    radius = max(float(particle_radius) * 2, 0.008)
    vertices = np.stack(
        (
            p - radius * u - radius * v,
            p + radius * u - radius * v,
            p + radius * u + radius * v,
            p - radius * u + radius * v,
        ),
        axis=1,
    ).reshape(-1, 3)
    indices = (
        np.arange(len(p))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])
    ).reshape(-1, 3)
    builder.add_object(
        ObjectDesc(
            name="boundary_cloud",
            physics=None,
            renders=[
                RenderDesc.from_geometry(
                    GeometryDesc.mesh(
                        vertices=vertices.astype(np.float32),
                        triangles=indices.astype(np.int32),
                    ),
                    material=material,
                )
            ],
        )
    )
