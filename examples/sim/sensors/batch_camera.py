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

"""Capture a batch of mono or stereo camera images."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.sensors import (
    Camera,
    StereoCamera,
    CameraCfg,
    StereoCameraCfg,
)
from embodichain.data import get_data_path
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_init_gpu_physics,
    maybe_open_window,
    setup_print_options,
    shutdown_sim,
)


def run(args: argparse.Namespace) -> None:
    """Render and save or display one batch of camera frames."""
    sim = create_default_sim(
        args,
        num_envs=args.num_envs,
        arena_space=2,
        add_default_light=False,
    )
    try:
        sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="obj",
                shape=MeshCfg(fpath=get_data_path("Chair/chair.glb")),
                init_pos=(0, 0, 0.2),
            )
        )
        maybe_init_gpu_physics(sim)
        maybe_open_window(sim, args)
        setup_print_options()

        eye = (0.0, 0, 2.0)
        target = (0.0, 0.0, 0.0)
        if args.sensor_type == "stereo":
            camera: Camera | StereoCamera = sim.add_sensor(
                sensor_cfg=StereoCameraCfg(
                    width=640,
                    height=480,
                    extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target),
                )
            )
        else:
            camera = sim.add_sensor(
                sensor_cfg=CameraCfg(
                    width=640,
                    height=480,
                    extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target),
                )
            )

        sim.reset_objects_state()

        started_at = time.perf_counter()
        camera.update()
        print(f"Camera update time: {time.perf_counter() - started_at:.4f} seconds")

        data_frame = camera.get_data()
        rgba = data_frame["color"].cpu().numpy()
        rgba_right = (
            data_frame["color_right"].cpu().numpy()
            if args.sensor_type == "stereo"
            else None
        )

        grid_x = int(np.ceil(np.sqrt(args.num_envs)))
        grid_y = int(np.ceil(args.num_envs / grid_x))
        fig, axs = plt.subplots(grid_x, grid_y, figsize=(12, 6), squeeze=False)
        for env_id, axis in enumerate(axs.flatten()):
            axis.axis("off")
            if env_id >= args.num_envs:
                continue
            image = (
                np.concatenate((rgba[env_id], rgba_right[env_id]), axis=1)
                if rgba_right is not None
                else rgba[env_id]
            )
            axis.imshow(image)
            axis.set_title(f"Env {env_id}")

        if args.headless:
            fig.savefig("camera_data.png")
        else:
            plt.show()
        plt.close(fig)
    finally:
        shutdown_sim(sim)


def main() -> None:
    """Parse command-line arguments and capture a camera batch."""
    parser = argparse.ArgumentParser(description="Run the batch robot simulation.")
    add_demo_args(parser)
    parser.add_argument(
        "--sensor_type",
        "--sensor-type",
        type=str,
        default="camera",
        choices=["stereo", "camera"],
        help="Type of camera sensor to use.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
