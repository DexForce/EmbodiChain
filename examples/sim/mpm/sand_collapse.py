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

"""Release a tall granular column and measure its robust height and footprint."""

from __future__ import annotations

import argparse
import json

import numpy as np

from embodichain.lab.sim import SimulationManager
from embodichain.lab.visualization import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)

from ._scene import DemoScene, open_view


def dimensions(q: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(q).all():
        raise ValueError("Nonfinite particle position")
    low, high = np.quantile(q[:, 0], [0.05, 0.95])
    return float(np.quantile(q[:, 2], 0.95)), float(high - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--friction", type=float, help="Override the friction in the asset manifest."
    )
    parser.add_argument("--headless", action="store_true")
    add_viser_args_to_parser(parser)
    args = parser.parse_args()
    if args.friction is not None and (
        not np.isfinite(args.friction) or args.friction < 0
    ):
        parser.error("friction must be finite and nonnegative")
    scene = DemoScene("sand_collapse")
    friction = scene.material["friction"] if args.friction is None else args.friction
    visualization = visualization_cfg_from_args(args)
    headless = args.headless or visualization.backend != "none"
    sim = scene.create_simulation(headless=headless, visualization=visualization)
    try:
        scene.add_environment(sim)
        sand = scene.add_material(sim, friction=friction)
        sim.prepare()
        if not headless:
            open_view(sim, (0.21, -0.4, 0.20), (0, 0, 0.065))
        initial = dimensions(sand.data.nodal_pos_w[0].cpu().numpy())
        for _ in range(scene.steps):
            sim.update(step=1)
        state = sand.data.nodal_state_w[0].cpu().numpy()
        height, width = dimensions(state[:, :3])
        speed = float(np.quantile(np.linalg.norm(state[:, 3:], axis=1), 0.99))
        print(
            json.dumps(
                dict(
                    friction=friction,
                    height95_m=height,
                    width90_m=width,
                    initial_height95_m=initial[0],
                    initial_width90_m=initial[1],
                    speed99_m_s=speed,
                )
            ),
            flush=True,
        )
        if not (height < initial[0] and width > initial[1] and speed < 0.05):
            raise RuntimeError("Expected a settled, shorter and wider column.")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
