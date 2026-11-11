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

"""Capture per-camera ground visibility, mixed Arena policies and selective reset.

Run ``embodichain run-example sim/camera_ground_visibility --help``.
Requires a DexSim build with Mesh render categories and camera exclusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg, RigidObjectCfg, VisualObjectCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_viser_args_to_parser(parser)
    parser.add_argument("--renderer", choices=("fast-rt", "hybrid"), default="fast-rt")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/camera_ground_visibility")
    )
    add_viser_args_to_parser(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=4,
            render_cfg=RenderCfg(renderer=args.renderer),
            visualization=visualization_cfg_from_args(args),
        )
    )
    records = []
    try:
        sim.add_visual_object(
            VisualObjectCfg(
                uid="below_floor",
                shape=CubeCfg(size=(2, 2, 0.1)),
                init_pos=(0, 0, -1),
            )
        )
        box = sim.add_rigid_object(
            RigidObjectCfg(
                uid="falling_box",
                shape=CubeCfg(size=(0.3, 0.3, 0.3)),
                init_pos=(0, 0, 1),
            )
        )
        cameras = [
            sim.add_sensor(
                CameraCfg(
                    uid=name,
                    width=192,
                    height=144,
                    intrinsics=(170, 170, 96, 72),
                    enable_depth=True,
                    ground_plane_visible=visible,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(2, -3, 2), target=(0, 0, 0)
                    ),
                )
            )
            for name, visible in (("training", False), ("debug", True))
        ]
        sim.prepare()
        for _ in range(120):
            sim.update()
        sim.enable_physics(False)

        for stage in ("hidden", "mixed", "reset_env1"):
            if stage == "mixed":
                cameras[0].set_ground_plane_visibility(True, [1, 3])
            elif stage == "reset_env1":
                cameras[0].reset([1])
            sim.render_camera_group([camera.group_id for camera in cameras])
            mosaic = Image.new("RGB", (192 * 4, (144 + 24) * 2), "#1e2329")
            draw = ImageDraw.Draw(mosaic)
            metadata = []
            for row, camera in enumerate(cameras):
                camera.update(fetch_only=True)
                data = camera.get_data()
                rgb = data["color"].cpu().numpy()
                depth = data["depth"].cpu().numpy()
                np.save(args.output / f"{stage}_{camera.cfg.uid}_depth.npy", depth)
                capture = camera.get_capture_metadata()
                metadata.append(capture)
                for env in range(4):
                    x, y = env * 192, row * 168
                    draw.text(
                        (x + 5, y + 5),
                        f"{camera.cfg.uid} env {env} ground={capture['ground_plane_visible'][env]}",
                        fill="white",
                    )
                    mosaic.paste(Image.fromarray(rgb[env, ..., :3]), (x, y + 24))
            mosaic.save(args.output / f"{stage}.png")
            records.append(
                {
                    "stage": stage,
                    "cameras": metadata,
                    "box_poses": box.get_local_pose().cpu().tolist(),
                }
            )
        (args.output / "captures.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            f"Saved RGB mosaics, depth and capture metadata to {args.output.resolve()}"
        )
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
