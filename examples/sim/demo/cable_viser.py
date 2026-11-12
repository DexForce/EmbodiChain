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
"""Preview replicated, anchored cables in Viser until Ctrl+C.

Requires CUDA and a DexSim build with cable support. Viser displays segment
capsules without a native renderer. Gravity bends each cable; the scene resets
every six simulated seconds so newly connected browsers can see the motion.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    CableObjectCfg,
    CablePhysicsCfg,
    NewtonPhysicsCfg,
    RenderCfg,
)
from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg


def main() -> None:
    """Run a paced cable-only scene and release its resources on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Simulated seconds; 0 runs until Ctrl+C.",
    )
    args = parser.parse_args()
    if args.num_envs < 1 or args.duration < 0:
        parser.error("--num-envs must be positive and --duration must be nonnegative")

    fps = 60
    dt = 1.0 / fps
    cfg = SimulationManagerCfg(
        headless=True,
        device="cuda:0",
        num_envs=args.num_envs,
        arena_space=2.0,
        physics_dt=dt,
        enable_entity_gizmo=False,
        render_cfg=RenderCfg(renderer="no-render"),
        physics_cfg=NewtonPhysicsCfg(
            solver_cfg={"solver_type": "dexuni"},
            num_substeps=8,
            sync_to_renderer=False,
        ),
        visualization=VisualizationCfg(
            backend="viser",
            env_ids=None,
            scene_fps=30.0,
            viser_server=ViserServerCfg(host="127.0.0.1", port=args.port),
        ),
    )
    sim = SimulationManager(cfg)
    try:
        points = np.zeros((25, 3), dtype=np.float32)
        points[:, 0] = np.linspace(0.0, 0.8, len(points))
        sim.add_cable_object(
            CableObjectCfg(
                uid="preview_cable",
                points=points,
                radius=0.01,
                init_pos=(-0.4, 0.0, 1.2),
                fixed_segments=(0,),
                render_mode="capsules",
                attrs=CablePhysicsCfg(bend_stiffness=0.01, twist_stiffness=0.01),
            )
        )
        runtime = sim.start_visualization()
        sim.capture_visualization(force=True)
        print(
            f"Open {runtime.endpoint} — {args.num_envs} cables; Ctrl+C to stop.",
            flush=True,
        )
        frame = 0
        while args.duration == 0.0 or frame * dt < args.duration:
            started = time.perf_counter()
            if frame and frame % (6 * fps) == 0:
                sim.reset_objects_state()
                sim.capture_visualization(force=True)
            sim.update(step=1)
            frame += 1
            time.sleep(max(0.0, dt - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
