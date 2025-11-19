# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import os
from embodichain.data import get_data_path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def test_real2sim_env_offline():
    from embodichain.utils.utility import dict2args
    from embodichain import embodichain_dir
    from embodichain.lab.scripts.run_env import (
        make_env,
        main,
        load_gym_config,
        load_objs,
    )
    from embodichain.utils.utility import save_json
    from embodichain.lab.gym.envs.generators.reality_parser import (
        RealityParser,
    )
    import tempfile
    from embodichain.lab.gym.envs.generators.real2sim_generator import (
        Real2SimEnvGenerator,
    )

    real2sim_test_data_dir = get_data_path("Real2SimTestData/")
    reality_config_list = [
        os.path.join(real2sim_test_data_dir, "PourWaterW1Single/reality_config.json"),
        os.path.join(
            real2sim_test_data_dir, "IROS_catch_cup_simple/reality_config.json"
        ),
        os.path.join(real2sim_test_data_dir, "PourWaterW1Dual/reality_config.json"),
    ]

    for i, reality_config in enumerate(reality_config_list):
        print(f"Testing real2sim env {i} from {reality_config}")
        with tempfile.TemporaryDirectory(
            prefix="real2sim_env_{}".format(i)
        ) as temp_dir:
            rp = RealityParser(reality_config)
            full_data_dir = rp.config["dataset"].get("dir_path", None)
            if not full_data_dir:
                full_data_dir = os.path.dirname(reality_config)
                rp.config["dataset"]["dir_path"] = full_data_dir

            data = rp.config["task"]["data"]

            for key, val in data.items():
                traj_dir = os.path.join(full_data_dir, val["trajectory"]["path"])
                for filename in sorted(os.listdir(traj_dir)):
                    if filename.endswith(".hdf5"):
                        val["trajectory"]["path"] = os.path.join(traj_dir, filename)
                        break

            new_class = Real2SimEnvGenerator.build_env(rp)
            Real2SimEnvGenerator.save_task_env(
                new_class, os.path.join(temp_dir, "whatever.pkl")
            )
            save_json(
                os.path.join(temp_dir, "object_config.json"), rp.get_object_config()
            )
            save_json(
                os.path.join(temp_dir, "gym_conf_path.json"), rp.compose_gym_config()
            )
            save_json(
                os.path.join(temp_dir, "action_conf_path.json"),
                rp.get_action_config()[0],
            )
            input_dict = {
                "render_backend": "egl",
                "debug_mode": False,
                "gpu_id": 0,
                "obj_config": os.path.join(temp_dir, "object_config.json"),
                "gym_config": os.path.join(temp_dir, "gym_conf_path.json"),
                "action_config": os.path.join(temp_dir, "action_conf_path.json"),
                "headless": True,
                "enable_rt": False,
                "task_type": "",
                "robot_name": "",
                "save_path": temp_dir,
                "save_video": False,
                "online_config": "",
                "local_dir": full_data_dir,
                "warehouse_dir": get_data_path("Real2SimWareHouse/"),
            }
            args = dict2args(input_dict)
            gym_config = load_gym_config(args)
            gym_config["dataset"]["save_path"] = temp_dir
            gym_config["max_episodes"] = 1
            env = make_env(args, load_objs(args), gym_config)
            main(args, env, gym_config)


if __name__ == "__main__":
    test_real2sim_env_offline()
