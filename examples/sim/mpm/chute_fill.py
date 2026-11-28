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

"""Drain a material-point liquid down a stationary chute into an open cup."""

from __future__ import annotations
import argparse
import json
import numpy as np
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from ._scene import DT, SPACING, create_scene, add_lattice, open_view


def add_rig(sim: SimulationManager) -> None:
    """Create the tilted chute and the five rigid cup panels."""

    def box(uid, size, pos, angle=0.0):
        sim.add_rigid_object(
            RigidObjectCfg(
                uid=uid,
                body_type="static",
                shape=CubeCfg(size=size),
                init_pos=pos,
                init_rot=(0.0, angle, 0.0),
            )
        )

    box("chute", (0.30, 0.126, 0.016), (0.0, 0.0, 0.20), 35.0)
    for side, y in (("left", -0.058), ("right", 0.058)):
        box(side, (0.30, 0.010, 0.055), (0.0, y, 0.225), 35.0)
    box("cup_base", (0.36, 0.30, 0.012), (0.25, 0.0, 0.006))
    for side, x in (("back", 0.07), ("front", 0.43)):
        box("cup_" + side, (0.012, 0.31, 0.105), (x, 0.0, 0.0525))
    for side, y in (("left", -0.15), ("right", 0.15)):
        box("cup_" + side, (0.37, 0.012, 0.105), (0.25, y, 0.0525))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viscosity", type=float, default=0.001)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.viscosity) or args.viscosity < 0:
        parser.error("viscosity must be finite and nonnegative")
    sim = create_scene(args.headless)
    try:
        add_rig(sim)
        water = add_lattice(
            sim,
            "liquid",
            (8, 8, 8),
            (-0.099, -0.028, 0.285),
            1000.0,
            friction=0.0,
            tensile_yield_ratio=1.0,
            viscosity=args.viscosity,
        )
        sim.prepare()
        if not args.headless:
            open_view(sim, (0.43, -0.70, 0.43), (0.11, 0, 0.14))
        arrival = half = None
        for step in range(1, 181):
            sim.update(step=1)
            q = water.data.nodal_pos_w[0].cpu().numpy()
            if not np.isfinite(q).all():
                raise RuntimeError("Nonfinite liquid state")
            x, y, z = q.T
            inside = (
                (x > 0.076) & (x < 0.424) & (abs(y) < 0.144) & (z > 0.006) & (z < 0.105)
            )
            fraction = float(inside.mean())  # All samples have the same mass.
            if arrival is None and inside.any():
                arrival = step * DT
            if half is None and fraction >= 0.5:
                half = step * DT
        angle = np.deg2rad(35.0)
        along = x * np.cos(angle) - (z - 0.20) * np.sin(angle)
        above = x * np.sin(angle) + (z - 0.20) * np.cos(angle)
        on_chute = (
            ~inside
            & (abs(along) <= 0.15)
            & (abs(y) <= 0.063)
            & (above >= 0)
            & (above <= 0.09)
        )
        m = 1000.0 * SPACING**3
        print(
            json.dumps(
                dict(
                    viscosity=args.viscosity,
                    first_arrival_s=arrival,
                    half_collected_s=half,
                    collected_fraction=fraction,
                    cup_kg=float(inside.sum() * m),
                    chute_kg=float(on_chute.sum() * m),
                    elsewhere_kg=float((~(inside | on_chute)).sum() * m),
                )
            ),
            flush=True,
        )
        if arrival is None or half is None:
            raise RuntimeError("Less than half the material reached the cup.")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
