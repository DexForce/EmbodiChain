# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import time
import numpy as np
import matplotlib.pyplot as plt

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidObjectCfg, LightCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.objects import RigidObject, Light
from embodichain.lab.sim.sensors import (
    Camera,
    StereoCamera,
    CameraCfg,
    StereoCameraCfg,
)
from embodichain.data import get_data_path


def main(args):
    config = SimulationManagerCfg(
        headless=True,
        sim_device=args.device,
        num_envs=args.num_envs,
        arena_space=2,
        enable_rt=args.enable_rt,
    )
    sim = SimulationManager(config)

    rigid_obj: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="obj",
            shape=MeshCfg(fpath=get_data_path("Chair/chair.glb")),
            init_pos=(0, 0, 0.2),
        )
    )
    light: Light = sim.add_light(
        cfg=LightCfg(light_type="point", init_pos=(0, 0, 2), intensity=50)
    )

    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    if args.headless is False:
        sim.open_window()

    import torch

    torch.set_printoptions(precision=4, sci_mode=False)

    eye = (0.0, 0, 2.0)
    target = (0.0, 0.0, 0.0)
    if args.sensor_type == "stereo":
        camera: StereoCamera = sim.add_sensor(
            sensor_cfg=StereoCameraCfg(
                width=640,
                height=480,
                extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target),
            )
        )
    else:
        camera: Camera = sim.add_sensor(
            sensor_cfg=CameraCfg(
                width=640,
                height=480,
                extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target),
            )
        )

    # TODO: To be removed
    sim.reset_objects_state()

    t0 = time.time()
    camera.update()
    print(f"Camera update time: {time.time() - t0:.4f} seconds")

    data_frame = camera.get_data()

    t0 = time.time()
    rgba = data_frame["color"].cpu().numpy()
    if args.sensor_type == "stereo":
        rgba_right = data_frame["color_right"].cpu().numpy()

    # plot rgba into a grid of images
    grid_x = np.ceil(np.sqrt(args.num_envs)).astype(int)
    grid_y = np.ceil(args.num_envs / grid_x).astype(int)
    fig, axs = plt.subplots(grid_x, grid_y, figsize=(12, 6))
    axs = axs.flatten()
    for i in range(args.num_envs):

        if args.sensor_type == "stereo":
            image = np.concatenate((rgba[i], rgba_right[i]), axis=1)
        else:
            image = rgba[i]
        axs[i].imshow(image)
        axs[i].axis("off")
        axs[i].set_title(f"Env {i}")

    if args.headless:
        plt.savefig(f"camera_data.png")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the batch robot simulation.")
    parser.add_argument(
        "--num_envs", type=int, default=4, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the simulation on.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run the simulation in headless mode."
    )
    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering."
    )
    parser.add_argument(
        "--sensor_type",
        type=str,
        default="camera",
        choices=["stereo", "camera"],
        help="Type of camera sensor to use.",
    )

    args = parser.parse_args()
    main(args)
