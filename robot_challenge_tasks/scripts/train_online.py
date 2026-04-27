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

import os
import argparse
import time
import gymnasium as gym
import torch
import numpy as np
import robot_challenge_tasks

from torch import multiprocessing as mp
from multiprocessing.managers import ListProxy
from tensordict import TensorDict

from embodichain.agents.engine import OnlineDataEngine, OnlineDataEngineCfg
from embodichain.lab.gym.utils.gym_utils import (
    merge_args_with_gym_config,
    add_env_launcher_args_to_parser,
)
from embodichain.utils.utility import load_json
from embodichain.utils import logger


def main():
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()

    # Build config

    args.headless = True  # Force headless for simulation process
    args.enable_rt = True
    args.filter_dataset_saving = True

    gym_config = load_json(args.gym_config)
    gym_config = merge_args_with_gym_config(args, gym_config)

    cfg = OnlineDataEngineCfg(
        buffer_size=2,
        max_episode_steps=250,
        state_dim=14,
        gym_config=gym_config,
        refill_threshold=20,
    )
    data_engine = OnlineDataEngine(cfg=cfg)
    data_engine.start()

    # --- 3. Training / Consumption Loop ---
    print(f"[Main Process] Listening for data...")
    try:
        while True:
            # Wait for availability
            time.sleep(
                1
            )  # Polling interval, can be optimized with event or condition variable

            batch_data = data_engine.sample_batch(
                batch_size=4, chunk_size=4
            )  # Sample a batch of episodes

            # Read data
            # Handle wrap-around reading if necessary
            # For simplicity, let's just print stats

            # If wrap-around

            logger.log_info(
                f"[Main Process] Sampled batch of {len(batch_data)} data.",
                color="orange",
            )

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        data_engine.stop()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
