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

from ._scene import DemoScene, open_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viscosity", type=float, help="Override the viscosity in the asset manifest."
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.viscosity is not None and (
        not np.isfinite(args.viscosity) or args.viscosity < 0
    ):
        parser.error("viscosity must be finite and nonnegative")
    scene = DemoScene("chute_fill")
    viscosity = (
        scene.material["viscosity"] if args.viscosity is None else args.viscosity
    )
    total_mass = scene.masses.sum(dtype=np.float64)
    sim = scene.create_simulation(headless=args.headless)
    try:
        scene.add_environment(sim)
        water = scene.add_material(sim, viscosity=viscosity)
        sim.prepare()
        if not args.headless:
            open_view(sim, (0.43, -0.70, 0.43), (0.11, 0, 0.14))
        arrival = half = None
        for step in range(1, scene.steps + 1):
            sim.update(step=1)
            q = water.data.nodal_pos_w[0].cpu().numpy()
            if not np.isfinite(q).all():
                raise RuntimeError("Nonfinite liquid state")
            x, y, z = q.T
            inside = (
                (x > 0.076) & (x < 0.424) & (abs(y) < 0.144) & (z > 0.006) & (z < 0.105)
            )
            fraction = float(scene.masses[inside].sum(dtype=np.float64) / total_mass)
            if arrival is None and inside.any():
                arrival = step * scene.physics.physics_dt
            if half is None and fraction >= 0.5:
                half = step * scene.physics.physics_dt
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
        print(
            json.dumps(
                dict(
                    viscosity=viscosity,
                    first_arrival_s=arrival,
                    half_collected_s=half,
                    collected_fraction=fraction,
                    cup_kg=float(scene.masses[inside].sum(dtype=np.float64)),
                    chute_kg=float(scene.masses[on_chute].sum(dtype=np.float64)),
                    elsewhere_kg=float(
                        scene.masses[~(inside | on_chute)].sum(dtype=np.float64)
                    ),
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
