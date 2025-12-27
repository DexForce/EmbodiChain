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
        headless=True, sim_device=args.device, arena_space=2, enable_rt=args.enable_rt,
    )
    sim = SimulationManager(config)
    sim.build_multiple_arenas(args.num_envs)

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

    count = 0
    while True:
        t0 = time.time()
        camera.update()
        # print(f"Camera update time: {time.time() - t0:.4f} seconds")
        t1 = time.time()
        print(f"Rendering fps: {args.num_envs / (t1 - t0):.2f}")

        count += 1

        if count==20:
            color = camera.get_data()["color"].cpu().numpy()
            plt.imshow(color[10])
            plt.show()
            break

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
