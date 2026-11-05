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
"""Create a new S03 cloth and IK snapshot inside outputs/benchmarks/tech_reports/dexuni/.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/s03_cloth_corner_lift/tools/prepare_inputs.py --out-dir PATH
"""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np

REPO = Path(__file__).resolve().parents[8]
TECH = REPO / "scripts/benchmark/tech_reports"
__all__ = ["cloth_mesh", "main"]


def cloth_mesh(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a flat 31x31 sheet and its fixed 10mm corner material group.

    Args:
        config: Shared S03 configuration.

    Returns:
        World-space nodes, oriented triangles and fixed corner node ids.
    """
    cloth = config["cloth"]
    n = cloth["vertices_per_side"]
    size = np.array(cloth["size"])
    center = np.array(cloth["center"])
    xy = np.stack(
        np.meshgrid(
            np.linspace(-size[0] / 2, size[0] / 2, n),
            np.linspace(-size[1] / 2, size[1] / 2, n),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    vertices = np.c_[xy, np.zeros(len(xy))] + center
    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            triangles.extend(((a, a + n, a + 1), (a + n, a + n + 1, a + 1)))
    corner = np.flatnonzero(
        (xy[:, 0] <= -size[0] / 2 + cloth["corner_half_extent"] + 1e-8)
        & (xy[:, 1] <= -size[1] / 2 + cloth["corner_half_extent"] + 1e-8)
    )
    return vertices.astype(np.float32), np.array(triangles, np.int32), corner


def main() -> None:
    """Generate physical cloth geometry and the shared arm path without pinning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=TECH / "assets/physics/dexuni/s03/s03_config.json",
    )
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni" not in out.parents:
        raise ValueError("Inputs must be inside outputs/benchmarks/tech_reports/dexuni")
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    (out / "s03_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if visual := config["cloth"].get("visual"):
        shutil.copy2(visual["texture_source"], out / visual["texture_file"])
    vertices, triangles, corner = cloth_mesh(config)
    np.savez(
        out / "s03_cloth_mesh.npz",
        vertices=vertices,
        triangles=triangles,
        corner_ids=corner,
    )
    path = (
        TECH / "physics/experiments/dexuni/s02_soft_pick_place/tools/prepare_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("s02_input_geometry", path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.INPUTS = out
    result = base.write_ik_path(config, "cuda:0")
    (out / "s02_ik_path.npz").rename(out / "s03_ik_path.npz")
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.is_file()
    }
    (out / "inputs_manifest.json").write_text(
        json.dumps(
            {
                "files_sha256": files,
                "ik": result,
                "nodes": len(vertices),
                "triangles": len(triangles),
                "corner_ids": corner.tolist(),
                "total_mass_kg": float(
                    np.prod(config["cloth"]["size"]) * config["cloth"]["density"]
                ),
                "all_particles_dynamic": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
