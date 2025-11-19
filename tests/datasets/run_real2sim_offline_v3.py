# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import os
import sys
import tempfile
import shutil
import gymnasium as gym

from embodichain import embodichain_dir
from embodichain.data import get_data_path
from embodichain.utils.utility import dict2args, save_json
from embodichain.lab.scripts.run_env import main
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.envs.generators.reality_parser import RealityParser
from embodichain.lab.gym.envs.generators.real2sim_v3_generator import (
    Real2SimV3EnvGenerator,
)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def test_real2sim_env_offline():
    """Run Real2Sim offline using the V3 generator (based on EmbodiedEnv).

    - Backward compatible with legacy IDs via `RealityParser.get_env_name()` (e.g., pour_water_single_real2sim)
    - Uses V3 `Real2SimEnv` (does not use V2/TablewareEnv)
    - Keeps the same data preparation and saving logic as the offline script
    """
    reality_config_list = [
        os.path.join("configs", "gym/real2sim/reality_config_PourWaterW1Single_v3.json")
    ]

    for i, reality_config in enumerate(reality_config_list):
        print(f"Testing real2sim env V3 (generator) {i} from {reality_config}")
        with tempfile.TemporaryDirectory(prefix=f"real2sim_env_v3_gen_{i}") as temp_dir:
            rp = RealityParser(reality_config)
            # Read from env.dataset (V3 format; no longer read from top-level dataset)
            env_config = rp.config.get("env", {})
            dataset_config = env_config.get("dataset", {})
            full_data_dir = dataset_config.get("dir_path", None)
            if not full_data_dir:
                # TODO: hardcode this as it's only test
                full_data_dir = os.path.join(
                    get_data_path("Real2SimTestData/"), "PourWaterW1Single_v3"
                )
                dataset_config["dir_path"] = full_data_dir

            # Convert relative paths to absolute paths based on full_data_dir
            # This is needed because config_to_cfg uses get_data_path() which expects absolute paths
            def _resolve_path(obj_dict: dict):
                """Convert relative paths to absolute paths based on full_data_dir."""
                if "shape" in obj_dict and "fpath" in obj_dict["shape"]:
                    fpath = obj_dict["shape"]["fpath"]
                    if isinstance(fpath, str) and not os.path.isabs(fpath):
                        obj_dict["shape"]["fpath"] = os.path.join(full_data_dir, fpath)

            # Resolve paths for all objects
            for obj_list in [
                rp.config.get("rigid_object", []),
                rp.config.get("background", []),
                rp.config.get("rigid_object_group", []),
            ]:
                for obj in obj_list:
                    _resolve_path(obj)
                    if "rigid_objects" in obj:
                        for rigid_obj in obj["rigid_objects"].values():
                            _resolve_path(rigid_obj)

            # Complete each trajectory's actual hdf5 file path (pick the first one)
            data = rp.config["task"]["data"]
            for key, val in data.items():
                traj_dir = os.path.join(full_data_dir, val["trajectory"]["path"])
                for filename in sorted(os.listdir(traj_dir)):
                    if filename.endswith(".hdf5"):
                        val["trajectory"]["path"] = os.path.join(traj_dir, filename)
                        break

            obj_path = os.path.join(temp_dir, "object_config.json")
            gym_path = os.path.join(temp_dir, "gym_conf_path.json")
            act_path = os.path.join(temp_dir, "action_conf_path.json")
            save_json(obj_path, rp.get_object_config())
            save_json(gym_path, rp.compose_gym_config_v3())
            # save_json(act_path, rp.get_action_config()[0])

            Real2SimV3EnvGenerator.build_env(rp)
            env_id = rp.get_env_name()

            input_dict = {
                "render_backend": "egl",
                "debug_mode": False,
                "gpu_id": 0,
                "obj_config": obj_path,
                "gym_config": gym_path,
                "headless": True,
                "device": "cpu",
                "enable_rt": False,
                "task_type": "",
                "robot_name": "",
                "save_path": temp_dir,
                "save_video": False,
                "online_config": "",
                "local_dir": full_data_dir,
                "warehouse_dir": get_data_path("Real2SimWareHouse/"),
                "env_name": env_id,
            }
            args = dict2args(input_dict)

            gym_config = rp.compose_gym_config_v3()
            gym_config["env"]["dataset"]["save_path"] = temp_dir
            gym_config["max_episodes"] = 1
            gym_config["env_name"] = env_id
            gym_config["record"] = {"enable": False}

            gym_config["robot"]["robot_type"] = "DexforceW1"

            cfg = config_to_cfg(gym_config)
            cfg.sim_cfg = SimulationManagerCfg(
                headless=args.headless,
                sim_device=args.device,
                enable_rt=args.enable_rt,
                gpu_id=args.gpu_id,
            )
            env = gym.make(env_id, cfg=cfg)
            main(args, env, gym_config)


if __name__ == "__main__":
    test_real2sim_env_offline()
