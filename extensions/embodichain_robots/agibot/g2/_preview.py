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

"""Render the official G2 using the W1 example's native preview renderer."""

from __future__ import annotations

__all__ = []


def _main() -> None:
    """Render one full-body image to the caller-selected output path."""
    import os

    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    from pathlib import Path
    import argparse
    from PIL import Image
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DenoisingCfg, RenderCfg, LightCfg
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.robot_registry import resolve_robot_cfg

    parser = argparse.ArgumentParser(
        description="Render AgiBot G2 with the DexForce W1 native preview renderer."
    )
    parser.add_argument("--output", type=Path, default=Path("g2-render.png"))
    args = parser.parse_args()
    out = args.output.parent
    out.mkdir(parents=True, exist_ok=True)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            render_cfg=RenderCfg(
                renderer="hybrid",
                spp=64,
                denoising=DenoisingCfg(offscreen="optix"),
            ),
            startup_summary="off",
            robot_ik_gizmo=None,
        )
    )
    try:
        cfg = resolve_robot_cfg("AgibotG2")
        sim.add_robot(cfg)
        # W1's native preview uses Hybrid on RTX GPUs. Add front/side lighting
        # to separate G2's dark chassis and head from the shared default floor.
        for uid, position, intensity in (
            ("preview_key", (3.0, -4.0, 4.5), 60.0),
            ("preview_fill", (-3.0, -2.0, 3.0), 30.0),
            ("preview_rim", (0.0, 3.0, 4.0), 40.0),
        ):
            sim.add_light(
                LightCfg(
                    uid=uid,
                    intensity=intensity,
                    radius=10.0,
                    init_pos=position,
                )
            )
        camera = sim.add_sensor(
            CameraCfg(
                uid="preview_camera",
                width=1400,
                height=1200,
                intrinsics=(1800.0, 1800.0, 700.0, 600.0),
                enable_color=True,
                enable_depth=False,
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=(2.3, -2.7, 1.8), target=(0.0, 0.0, 0.9), up=(0.0, 0.0, 1.0)
                ),
            )
        )
        sim.prepare()
        sim.enable_physics(False)
        sim.update(step=1)
        camera.update()
        frame = camera.get_data()["color"][0].cpu().numpy()
        Image.fromarray(frame[:, :, :3]).save(args.output)
        print(args.output, flush=True)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    _main()
