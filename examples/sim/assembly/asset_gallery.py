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
"""Generate inspectable precision-assembly manifests and matching USD files."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

from embodichain.gen_sim.assembly import build_assembly, sample_assemblies


def main() -> None:
    """Generate a named gallery, optionally adding stable seeded random pegs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", default=["circle", "star", "usb_c", "m10", "spur_pair"]
    )
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--collision-colors",
        action="store_true",
        help="Also write a USD inspection view colored by convex collision piece.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/assembly_gallery"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = [build_assembly(name, seed=args.seed) for name in args.configs]
    assets.extend(sample_assemblies(args.samples, seed=args.seed))
    for asset in assets:
        stem = asset.content_hash
        asset.save(args.output / f"{stem}.json")
        path = asset.write_usd(args.output / f"{stem}.usda")
        print(
            f"{path}: "
            + ", ".join(
                f"{body.name}: {sum(g.collision for g in body.geometries)} colliders"
                for body in asset.bodies
            )
        )
        print("  generator parameters: " + json.dumps(asset.config, sort_keys=True))
        if args.collision_colors:
            inspection = asset.copy()
            piece = 0
            for body in inspection.bodies:
                for geometry in body.geometries:
                    geometry.visible = geometry.collision
                    if geometry.collision:
                        rgb = colorsys.hsv_to_rgb(
                            (piece * 0.61803398875) % 1, 0.65, 0.9
                        )
                        geometry.rgba = [*rgb, 1.0]
                        piece += 1
            inspection_path = inspection.write_usd(args.output / f"{stem}.pieces.usda")
            print(f"  collision-piece inspection: {inspection_path}")


if __name__ == "__main__":
    main()
