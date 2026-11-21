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
"""Check soft-goods surface contact from unmodified recorded states.

Run: python check_pouch_film_clearance.py --run <run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

__all__ = ["main"]


def main() -> None:
    """Report discrete surface intersections and clearance after milk lowering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/tmp/m05_fcl")
    import trimesh
    from trimesh.collision import CollisionManager

    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    phases = {name: end for end, name in manifest["scene"]["phases"]}
    with np.load(run / "preview/states.npz") as data:
        time, bag, pouch, bread = (data[k] for k in ("time", "bag", "pouch", "bread"))
    with np.load(run / "inputs/goods_meshes.npz") as data:
        bag_faces, pouch_faces = (data[k] for k in ("bag_triangles", "pouch_triangles"))
        tets = data["bread_tetrahedra"]
    faces = np.concatenate(
        [tets[:, ids] for ids in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
    )
    _, indices, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    assert np.isin(counts, (1, 2)).all(), "Nonmanifold tetrahedral face topology"
    bread_faces = faces[indices[counts == 1]]
    frames = np.flatnonzero(time >= phases["pouch lower"] - 1e-8)
    distances, hits, bread_distances, bread_hits, goods_distances, goods_hits = (
        [] for _ in range(6)
    )
    for frame in frames:
        manager = CollisionManager()
        manager.add_object("bag", trimesh.Trimesh(bag[frame], bag_faces, process=False))
        milk = trimesh.Trimesh(pouch[frame], pouch_faces, process=False)
        loaf = trimesh.Trimesh(bread[frame], bread_faces, process=False)
        distances.append(float(manager.min_distance_single(milk)))
        bread_distances.append(float(manager.min_distance_single(loaf)))
        if manager.in_collision_single(milk):
            hits.append({"frame": int(frame), "time_s": float(time[frame])})
        if manager.in_collision_single(loaf):
            bread_hits.append({"frame": int(frame), "time_s": float(time[frame])})
        manager = CollisionManager()
        manager.add_object("bread", loaf)
        goods_distances.append(float(manager.min_distance_single(milk)))
        if manager.in_collision_single(milk):
            goods_hits.append({"frame": int(frame), "time_s": float(time[frame])})
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    np.savez_compressed(
        output / "pouch_film_clearance.npz",
        time=time[frames],
        distance_m=distances,
        bread_film_distance_m=bread_distances,
        pouch_bread_distance_m=goods_distances,
    )
    result = dict(
        method="FCL triangles of both recorded deforming surfaces, every outer pose after milk lowering; no contact skins, unrecorded substeps or continuous-motion guarantee",
        sample_count=len(frames),
        time_range_s=[float(time[frames[0]]), float(time[frames[-1]])],
        minimum_surface_distance_m=min(distances),
        collision_samples=hits,
        bread_film=dict(
            minimum_surface_distance_m=min(bread_distances),
            collision_samples=bread_hits,
        ),
        pouch_bread=dict(
            minimum_surface_distance_m=min(goods_distances),
            collision_samples=goods_hits,
        ),
        states_sha256=hashlib.sha256(
            (run / "preview/states.npz").read_bytes()
        ).hexdigest(),
    )
    (output / "pouch_film_clearance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
