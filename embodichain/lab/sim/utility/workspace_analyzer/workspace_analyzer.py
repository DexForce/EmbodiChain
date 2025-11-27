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

import gc
import os
import time
import numpy as np
import open3d as o3d
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Sequence, Callable, Protocol
from itertools import product, islice
from tqdm import tqdm
from contextlib import contextmanager
from pathlib import Path
from enum import Enum

from embodichain.utils import logger
from embodichain.lab.sim.objects.robot import Robot
from scipy.spatial.transform import Rotation as R

all = [
    "workspace_analyzer",
]


class workspace_analyzer:
    def __init__(self, robot: Robot, device: torch.device = torch.device("cpu")):
        self.robot = robot
        self.device = device

        # Extract joint limits
        self.qpos_limits = robot._entities[0].get_joint_limits()


if __name__ == "__main__":
    # Example usage
    from IPython import embed
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.robots.dexforce_w1.types import (
        DexforceW1HandBrand,
        DexforceW1ArmSide,
        DexforceW1ArmKind,
        DexforceW1Version,
    )
    from embodichain.lab.sim.robots.dexforce_w1.utils import build_dexforce_w1_cfg

    config = SimulationManagerCfg(headless=False, sim_device="cpu")
    sim = SimulationManager(config)
    sim.build_multiple_arenas(1)
    sim.set_manual_update(False)

    from embodichain.lab.sim.robots import DexforceW1Cfg

    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "anthropomorphic"}
    )
    robot = sim.add_robot(cfg=cfg)
    print("DexforceW1 robot added to the simulation.")

    wa = workspace_analyzer(robot=robot)

    embed(header="Workspace Analyzer Test Environment")
