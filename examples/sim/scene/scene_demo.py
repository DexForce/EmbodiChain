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
"""
This script demonstrates how to create a simulation scene using SimulationManager.
It supports loading kitchen/factory/office scenes via EmbodiChainDataset.
"""

import argparse
import time
from pathlib import Path
import math
import embodichain.utils.logger as logger
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, LightCfg, RobotCfg, URDFCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg, Robot
from embodichain.data.assets.scene_assets import SceneData
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT


def resolve_asset_path(scene_name: str) -> str:
    """
    Resolve the asset path for a given scene (.glb/.gltf),
    downloading if needed using EmbodiChainData.
    """

    current_dir = Path(__file__).parent
    local_glb = current_dir / f"{scene_name}.glb"
    local_gltf = current_dir / f"{scene_name}.gltf"
    if local_glb.exists():
        logger.log_info(f"Using local asset: {local_glb}")
        return str(local_glb)
    if local_gltf.exists():
        logger.log_info(f"Using local asset: {local_gltf}")
        return str(local_gltf)

    scene_data = SceneData()

    extracted_dir = Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "extract" / "SceneData"
    glb_path = extracted_dir / f"{scene_name}.glb"
    gltf_path = extracted_dir / f"{scene_name}.gltf"

    if glb_path.exists():
        logger.log_info(f"Using downloaded asset: {glb_path}")
        return str(glb_path)
    if gltf_path.exists():
        logger.log_info(f"Using downloaded asset: {gltf_path}")
        return str(gltf_path)

    raise FileNotFoundError(
        f"No .glb or .gltf found in extracted folder: {extracted_dir}"
    )


def run_simulation(sim: SimulationManager):
    """Run the simulation loop."""
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.log_info("\n Stopping simulation...")
    finally:
        sim.destroy()
        logger.log_info("Simulation terminated successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="kitchen",
        choices=["kitchen", "factory", "office", "local"],
        help="Choose which scene to load",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Simulation device (cuda or cpu)"
    )
    parser.add_argument(
        "--disable_rt",
        action="store_true",
        default=False,
        help="Disable ray tracing for better visuals",
    )
    args = parser.parse_args()

    logger.log_info(f"Initializing scene '{args.scene}'")

    logger.log_info(f"Resolving and downloading scene '{args.scene}' if needed...")
    try:
        asset_path = resolve_asset_path(args.scene)
        logger.log_info(f"Scene asset ready at: {asset_path}")
    except Exception as e:
        print(f"Failed to download or resolve scene asset: {e}")
        return

    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        physics_dt=1.0 / 100.0,
        sim_device=args.device,
        enable_rt=not args.disable_rt,
    )
    sim = SimulationManager(sim_cfg)

    if args.num_envs > 1:
        sim.build_multiple_arenas(args.num_envs, space=10.0)

    num_lights = 8
    radius = 5
    height = 8
    intensity = 200
    lights = []

    for i in range(num_lights):
        angle = 2 * math.pi * i / num_lights
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        uid = f"l{i+1}"
        cfg = LightCfg(uid=uid, intensity=intensity, radius=600, init_pos=[x, y, z])
        lights.append(sim.add_light(cfg))

    physics_attrs = RigidBodyAttributesCfg(
        mass=10,
        dynamic_friction=0.5,
        static_friction=0.5,
        restitution=0.1,
    )

    try:
        logger.log_info(f"Loading scene asset into simulation: {asset_path}")
        scene_obj: RigidObject = sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid=args.scene,
                shape=MeshCfg(fpath=asset_path),
                body_type="static",
                init_pos=[0.0, 0.0, 0.1],
                init_rot=[90, 180, 0],
                attrs=physics_attrs,
            )
        )
        if args.scene == "factory":
            from embodichain.lab.sim.robots.dexforce_w1.cfg import DexforceW1Cfg

            w1_robot: Robot = sim.add_robot(
                cfg=DexforceW1Cfg.from_dict(
                    {
                        "uid": "dexforce_w1",
                        "version": "v021",
                        "arm_kind": "anthropomorphic",
                        "init_pos": [-1, -0.5, 0],
                        "init_rot": [0, 0, 90],
                    }
                ),
            )

    except Exception as e:
        logger.log_info(f"Failed to load scene asset: {e}")
        return

    logger.log_info(f"Scene '{args.scene}' setup complete!")
    logger.log_info(f"Running simulation with {args.num_envs} environment(s)")
    logger.log_info("Press Ctrl+C to stop the simulation")

    sim.open_window()

    run_simulation(sim)


if __name__ == "__main__":
    main()
