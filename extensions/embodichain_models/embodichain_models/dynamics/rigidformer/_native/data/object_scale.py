# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Derived from RigidFormer at 160fe0852c51cffae7b7440452b58d0b49db6df6.
# See the model family's THIRD_PARTY_LICENSES for the complete license.

"""Per-object metadata scale for converted MOVi vertex data (data.object_scale_file).

data_generation/movi_to_sim_data.py wrote every object's vertices as R v_asset + p and never applied the metadata
`size`, so stored meshes do not have the simulated dimensions (MOVi-Sphere: all spheres ~1.0 m across, simulated radii
0.35 / 0.7 m; resting mesh bottoms at z ~ -0.15 / +0.2 m instead of the floor). With a scale file from
data_generation/build_object_scales.py, every array read from <scene>/frame_<t>/obj_<i>_pos.npy is scaled about the
object origin p_t (vis_traj.npy[0, t, i, :3], equal to the metadata positions):

    v' = p_t + size_i (v - p_t)

after which resting objects of every size have their lowest vertex at z ~ 0.002 m. Anchor and FPS-order caches are
unchanged (farthest-point sampling is invariant to a uniform scaling about any point). Without a scale file nothing
changes.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from .scene_frames import load_frame
except ImportError:  # imported as src.data.object_scale
    from .scene_frames import load_frame


class ObjectScale:
    def __init__(self, scale_file: str):
        data = json.load(open(scale_file))
        self.scale_file = str(scale_file)
        # converted dataset the sizes belong to (scene names movi_<n> repeat across MOVi-A/B/Sphere)
        self.converted = data.get("converted")
        self.scales: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.float64) for k, v in data["scales"].items()
        }
        self.fingerprint = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in (data.get("fingerprint") or {}).items()
        }
        if "radius0" not in data:
            raise ValueError(
                f"{scale_file} has no radius0 entries (the guard against vertex data that already carries the "
                f"size); rebuild it with data_generation/build_object_scales.py"
            )
        self.radius0 = {
            k: np.asarray(v, dtype=np.float64) for k, v in data["radius0"].items()
        }
        for name, table in (
            ("scales", self.scales),
            ("fingerprint", self.fingerprint),
            ("radius0", self.radius0),
        ):
            bad = [
                k
                for k, v in table.items()
                if not np.isfinite(v).all()
                or (name != "fingerprint" and not (v > 0).all())
            ]
            if bad:
                raise ValueError(
                    f"{scale_file}: non-finite or non-positive {name} for {len(bad)} scene(s), e.g. {bad[:3]}"
                )
        self._origins: Dict[str, np.ndarray] = {}

    def origins(self, scene_dir: Path) -> np.ndarray:
        key = scene_dir.name
        if key not in self._origins:
            o = np.load(scene_dir / "vis_traj.npy")[0, :, :, :3].astype(
                np.float64
            )  # (T, Nobj, 3)
            if not np.isfinite(o).all():
                raise ValueError(
                    f'ObjectScale: non-finite object origins in {scene_dir / "vis_traj.npy"}'
                )
            self._origins[key] = o
        return self._origins[key]

    def apply(self, vertex_path: Path, V: np.ndarray) -> np.ndarray:
        """Scaled copy of V, the array stored at <scene>/frame_<t>/obj_<i>_pos.npy (V itself is never modified)."""
        vertex_path = Path(vertex_path)
        frame_dir = vertex_path.parent
        scene_dir = frame_dir.parent
        if not frame_dir.name.startswith("frame_") or not vertex_path.name.startswith(
            "obj_"
        ):
            raise ValueError(f"ObjectScale: not a per-frame vertex file: {vertex_path}")
        if scene_dir.name not in self.scales:
            raise KeyError(
                f"ObjectScale: scene {scene_dir.name} missing from {self.scale_file}"
            )
        t = int(frame_dir.name.split("_")[1])
        i = int(vertex_path.name.split("_")[1])
        s = self.scales[scene_dir.name][i]
        p = self.origins(scene_dir)[t, i]
        return (p + s * (V.astype(np.float64) - p)).astype(V.dtype)


def from_config(config: dict) -> Optional[ObjectScale]:
    f = (config.get("data", {}) or {}).get("object_scale_file")
    return ObjectScale(f) if f else None


def check_object_counts(object_scale: "ObjectScale", data_path, scenes) -> None:
    """For every scene: as many sizes as objects in frame_00000; object 0's frame-0 origin equal to the sidecar's
    fingerprint (MOVi-A and MOVi-Sphere share seeds and hence object counts, so the counts alone cannot tell them
    apart); and every object's largest frame-0 vertex distance from its origin equal to the sidecar's radius0, i.e. the
    stored vertices are still the unscaled ones the sizes apply to (data re-converted with the size would otherwise be
    scaled twice)."""
    import os

    bad = []
    for sc in scenes:
        if sc not in object_scale.scales:
            bad.append(f"{sc}: missing")
            continue
        frame0 = load_frame(
            Path(data_path) / sc, 0
        )  # either layout (src/data/scene_frames.py)
        n = len(frame0)
        if n != len(object_scale.scales[sc]):
            bad.append(f"{sc}: {n} objects vs {len(object_scale.scales[sc])} sizes")
            continue
        if sc not in object_scale.fingerprint or sc not in object_scale.radius0:
            bad.append(f"{sc}: no fingerprint / radius0 in the sidecar")
            continue
        p0 = np.load(os.path.join(str(data_path), sc, "vis_traj.npy"))[
            0, 0, :, :3
        ].astype(
            np.float64
        )  # (Nobj, 3)
        if (
            not np.abs(p0[0] - object_scale.fingerprint[sc]).max() <= 1e-4
        ):  # (a NaN fails this test too)
            bad.append(
                f"{sc}: frame-0 origin {p0[0].round(4).tolist()} vs fingerprint {object_scale.fingerprint[sc].tolist()}"
            )
            continue
        r = np.array(
            [
                np.linalg.norm(frame0[o].astype(np.float64) - p0[o], axis=1).max()
                for o in range(n)
            ]
        )
        if (
            not np.abs(r - object_scale.radius0[sc]).max() <= 1e-4
        ):  # (a NaN fails this test too)
            bad.append(
                f"{sc}: frame-0 vertex radii {r.round(4).tolist()} vs radius0 {object_scale.radius0[sc].tolist()} "
                f"(sizes {object_scale.scales[sc].round(4).tolist()})"
            )
    if bad:
        raise ValueError(
            f"{object_scale.scale_file}: {len(bad)} scene(s) do not match the data, e.g. {bad[:3]}"
        )
