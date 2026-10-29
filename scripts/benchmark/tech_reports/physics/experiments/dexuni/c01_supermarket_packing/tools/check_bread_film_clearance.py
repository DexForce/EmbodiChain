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
"""Check recorded single-bread/film surfaces without stepping simulation.

Run: python check_bread_film_clearance.py --run <single-bread run>.
FCL checks discrete surfaces after lowering, not continuous motion/containment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

__all__ = ["main"]


def main() -> None:
    """Check actual deforming triangle surfaces at each post-lowering pose."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/tmp/m05_fcl")
    import trimesh
    from trimesh.collision import CollisionManager

    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["scene"]["goods"] == ["bread"]
    lower = next(t for t, name in manifest["scene"]["phases"] if name == "bread lower")
    with np.load(run / "preview/states.npz") as d:
        time, bag, bread = (d[k] for k in ("time", "bag", "bread"))
    with np.load(run / "inputs/goods_meshes.npz") as d:
        bag_faces, tets = d["bag_triangles"], d["bread_tetrahedra"]
    faces = np.concatenate(
        [tets[:, ids] for ids in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
    )
    _, indices, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    assert np.isin(counts, (1, 2)).all()
    bread_faces = faces[indices[counts == 1]]
    frames = np.flatnonzero(time >= lower - 1e-8)
    distances, hits = [], []
    for frame in frames:
        manager = CollisionManager()
        manager.add_object("bag", trimesh.Trimesh(bag[frame], bag_faces, process=False))
        loaf = trimesh.Trimesh(bread[frame], bread_faces, process=False)
        distances.append(float(manager.min_distance_single(loaf)))
        if manager.in_collision_single(loaf):
            hits.append(dict(frame=int(frame), time_s=float(time[frame])))
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    np.savez_compressed(
        output / "bread_film_clearance.npz",
        frame=frames,
        time_s=time[frames],
        distance_m=distances,
    )
    result = dict(
        method="FCL actual deforming zero-thickness triangle surfaces; discrete outer poses only; no continuous/containment guarantee",
        sample_count=len(frames),
        time_range_s=[float(time[frames[0]]), float(time[frames[-1]])],
        minimum_surface_distance_m=min(distances),
        collision_samples=hits,
        states_sha256=hashlib.sha256(
            (run / "preview/states.npz").read_bytes()
        ).hexdigest(),
    )
    (output / "bread_film_clearance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
