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

"""Particle-driven crust and crumb surfaces rendered by DexSim's native soft mesh API."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch

from embodichain.lab.sim.material import VisualMaterialCfg


def grain(rng, n, scale):
    small = rng.normal(size=(scale, scale)).astype(np.float32)
    raw = np.asarray(Image.fromarray(small).resize((n, n), Image.Resampling.BICUBIC))
    return (raw - raw.mean()) / (raw.std() + 1e-6)


def textures(directory: Path) -> None:
    """Generate a deterministic crust, porous crumb and board without asset downloads."""
    rng = np.random.default_rng(36)
    n = 768
    v, u = np.mgrid[0:n, 0:n] / (n - 1)
    coarse = grain(rng, n, 10)
    medium = grain(rng, n, 43)
    fine = grain(rng, n, 155)
    tone = (
        0.54
        + 0.085 * coarse
        + 0.060 * medium
        + 0.055 * fine
        + 0.035 * rng.normal(size=(n, n))
    )
    # Angular UV runs bottom -> crown -> bottom. Shallow diagonal flour-dusted scores.
    upper = np.maximum(-np.cos(2 * np.pi * v), 0.0) ** 3
    for center in (0.24, 0.50, 0.76):
        stripe = (u - center - 0.108 * np.sin(2 * np.pi * v)) / 0.028
        score = np.exp(-(stripe**2)) * upper
        tone += 0.56 * score
        tone -= 0.30 * np.exp(-(((stripe - 1.08) / 0.50) ** 2)) * upper
    flour = np.clip((medium + 0.40 * coarse + 0.5 * fine - 1.05) * 0.23, 0, 0.44)
    t = np.clip(tone, 0, 1)[:, :, None]
    crust = np.array([124, 58, 24]) * (1 - t) + np.array([242, 177, 93]) * t
    crust += flour[:, :, None] * (np.array([242, 222, 183]) - crust)
    Image.fromarray(crust.clip(0, 255).astype("uint8")).save(directory / "crust.png")
    base = np.clip(
        np.array([232, 196, 136])[None, None, :]
        + (medium * 5 + fine * 4 + rng.normal(0, 3, (n, n)))[:, :, None],
        0,
        255,
    ).astype("uint8")
    crumb = Image.fromarray(base)
    draw = ImageDraw.Draw(crumb)
    for _ in range(3900):
        x, y = rng.integers(0, n, 2)
        r = min(15, 1.0 + rng.exponential(2.5))
        ry = r * rng.uniform(0.5, 1.1)
        angle = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        j = rng.uniform(0.73, 1.15, len(angle))
        pts = np.column_stack([x + r * j * np.cos(angle), y + ry * j * np.sin(angle)])
        shade = rng.uniform(0.65, 0.94)
        draw.polygon(
            [tuple(p) for p in pts],
            fill=tuple((np.array([226, 185, 120]) * shade).astype(int)),
        )
        draw.line([tuple(p) for p in pts[2:6]], fill=(244, 218, 170), width=1)
    crumb.save(directory / "crumb.png")
    wood = (
        0.50
        + 0.16
        * np.sin(
            2
            * np.pi
            * (
                v * 21
                + 0.55 * np.sin(2 * np.pi * u * 0.8)
                + 0.28 * np.sin(2 * np.pi * (u * 0.6 + v * 2))
            )
        )
        + 0.055 * np.sin(2 * np.pi * (v * 125 + 0.8 * np.sin(4 * u)))
        + 0.03 * fine
    )
    t = np.clip(wood, 0, 1)[:, :, None]
    Image.fromarray(
        (np.array([103, 62, 36]) * (1 - t) + np.array([174, 119, 71]) * t).astype(
            "uint8"
        )
    ).save(directory / "board.png")


def ring(y, theta):
    end = np.sqrt(np.clip(1 - (max(abs(y) - 0.060, 0) / 0.040) ** 2, 0, 1))
    x = 0.050 * end * np.sin(theta)
    z = np.maximum(0.002, 0.027 - 0.038 * end * np.cos(theta))
    noise = 0.0007 * np.sin(73 * y + 3 * theta) * np.cos(37 * y - 5 * theta)
    crown = np.maximum(0, -np.cos(theta)) ** 3
    z += noise * crown
    for center in (0.24, 0.5, 0.76):
        stripe = ((y + 0.1) / 0.2 - center - 0.108 * np.sin(theta)) / 0.028
        z -= 0.0038 * np.exp(-(stripe**2)) * crown
        z += 0.0013 * np.exp(-(((stripe - 1.1) / 0.55) ** 2)) * crown
    return np.column_stack((x, np.full_like(x, y), z))


def shell(lo, hi, board):
    n = 96
    angles = np.linspace(0, 2 * np.pi, n + 1)
    ys = np.linspace(lo, hi, max(3, int((hi - lo) / 0.0025) + 1))
    vertices = []
    uv = []
    faces = []
    for y in ys:
        vertices.extend(ring(y, angles))
        uv.extend(
            np.column_stack(
                (np.full_like(angles, (y + 0.1) / 0.2), angles / (2 * np.pi))
            )
        )
    for j in range(len(ys) - 1):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + n + 1
            faces.extend([(a, b, b + 1), (a, b + 1, a + 1)])
    verts = np.asarray(vertices, dtype=np.float32)
    verts[:, 2] += board
    return verts, np.asarray(faces, np.int32), np.asarray(uv, np.float32)


def cap(y, board, positive):
    n = 96
    rim = ring(y, np.arange(n) * 2 * np.pi / n)
    center = rim.mean(axis=0)
    verts = [center]
    faces = []
    for k in range(1, 14):
        verts.extend(center + (rim - center) * k / 13)
        for i in range(n):
            a = 1 + (k - 1) * n + i
            b = 1 + (k - 1) * n + (i + 1) % n
            if k == 1:
                faces.append((0, a, b))
            else:
                faces.extend([(a - n, a, b), (a - n, b, b - n)])
    v = np.asarray(verts, np.float32)
    uv = np.column_stack((v[:, 0] / 0.10 + 0.5, v[:, 2] / 0.075)).astype(np.float32)
    v[:, 2] += board
    tri = np.asarray(faces, np.int32)
    if positive:
        tri = tri[:, ::-1].copy()
    return v, tri, uv


class LoafSkin:
    """Bind a closed loaf and two cut faces using native eight-particle skinning."""

    def __init__(self, sim, rest, board, seam, directory):
        self.world = sim._world
        self.materials = []
        for name in ("crust", "crumb"):
            mat = sim.create_visual_material(
                VisualMaterialCfg(
                    uid="bread_" + name,
                    base_color_texture=str(directory / (name + ".png")),
                    base_color=[1.0, 1.0, 1.0, 1.0],
                    metallic=0.0,
                    roughness=0.89,
                )
            )
            self.materials.append(
                mat.get_instance("bread_" + name).mat.get_material_id()
            )
        self.ids = list(range(5))
        all_ids = np.arange(len(rest))
        front = all_ids[rest[:, 1] < seam]
        back = all_ids[rest[:, 1] > seam]
        descriptions = [
            (shell(-0.09998, 0.09998, board), all_ids),
            (shell(-0.09998, seam, board), front),
            (shell(seam, 0.09998, board), back),
            (cap(seam, board, True), front),
            (cap(seam, board, False), back),
        ]
        arrays = [[] for _ in range(6)]
        for (v, tri, uv), pool in descriptions:
            distances, local = cKDTree(rest[pool]).query(v, k=8)
            ids = pool[local].astype(np.int32)
            weights = 1 / np.maximum(distances, 0.0004) ** 2
            weights = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
            offset = (v - (rest[ids] * weights[:, :, None]).sum(axis=1)).astype(
                np.float32
            )
            for dest, a in zip(arrays, (tri, v, uv, ids, weights, offset)):
                dest.append(torch.as_tensor(np.ascontiguousarray(a), device=sim.device))
        triangles, verts, texcoords, indices, weights, offsets = arrays
        # Independent registers let crust and crumb both use ordinary weighted reconstruction.
        for start, end, material in (
            (0, 3, self.materials[0]),
            (3, 5, self.materials[1]),
        ):
            sl = slice(start, end)
            result = self.world.register_soft_meshes_to_renderer(
                self.ids[sl],
                material,
                material,
                triangles[sl],
                verts[sl],
                texcoords[sl],
                indices[sl],
                weights[sl],
                visible_flags=[i == 0 for i in self.ids[sl]],
                rest_offsets_list=offsets[sl],
            )
            if any(r < 0 for r in result):
                raise RuntimeError(f"Native bread mesh registration failed: {result}")

    def update(self, q, split):
        flags = [not split, split, split, split, split]
        result = self.world.sync_soft_meshes_to_renderer(
            self.ids, q.contiguous(), visible_flags=flags, update_flags=flags
        )
        if any(r < 0 for r in result):
            raise RuntimeError(f"Native bread mesh sync failed: {result}")
