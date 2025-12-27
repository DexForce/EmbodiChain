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

import torch
import numpy as np
from IPython import embed

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import DexforceW1Cfg

from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
    WorkspaceAnalyzer,
)

if __name__ == "__main__":
    # Example usage
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    config = SimulationManagerCfg(headless=False, sim_device="cpu")
    sim_manager = SimulationManager(config)
    sim_manager.build_multiple_arenas(1)
    sim_manager.set_manual_update(False)

    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "industrial"}
    )
    robot = sim_manager.add_robot(cfg=cfg)
    print("DexforceW1 robot added to the simulation.")

    print("Example: Joint Space Analysis")

    wa_joint = WorkspaceAnalyzer(robot=robot, sim_manager=sim_manager)
    results_joint = wa_joint.analyze(num_samples=3000, visualize=True)

    print(f"\nJoint Space Results:")
    print(
        f"  Valid points: {results_joint['num_valid']} / {results_joint['num_samples']}"
    )
    print(f"  Analysis time: {results_joint['analysis_time']:.2f}s")
    print(f"  Metrics: {results_joint['metrics']}")

    embed(header="End of Joint Space Analysis Example")
