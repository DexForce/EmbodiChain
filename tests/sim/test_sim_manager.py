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
import dexsim.environment
import numpy as np
import dexsim
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from dexsim.utility.path import get_resources_data_path


def test_sim_init():
    config = SimulationManagerCfg()
    config.headless = True

    sim = SimulationManager(config)
    sim.get_env().clean()

    assert isinstance(sim.get_env(), dexsim.environment.Env)
    assert isinstance(sim.get_world(), dexsim.World)

    # test add_sensor
    intrinsic = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    cam1 = sim.add_sensor(
        "MonocularCam", sensor_uid="cam1", resolution=(640, 480), intrinsic=intrinsic
    )
    assert sim.get_sensor("cam1") == cam1
    assert len(sim.get_sensor_uid_list()) == 1
    assert sim.get_sensor_uid_list()[0] == "cam1"

    # TODO: test add_robot

    # test_add_fixed_actor.
    model_path = get_resources_data_path("Model", "lego", "lego.ply")

    actor = sim.add_fixed_actor(fpath=model_path, init_pose=np.eye(4))
    assert sim.get_fixed_actor_uid_list() == ["lego.ply"]
    assert sim.get_fixed_actor("lego.ply") == actor

    sim.remove_fixed_actor("lego.ply")
    assert sim.get_fixed_actor_uid_list() == []

    # test add_dynamic_actor
    actor = sim.add_dynamic_actor(fpath=model_path, init_pose=np.eye(4))
    assert sim.get_dynamic_actor_uid_list() == ["lego.ply"]
    assert sim.get_dynamic_actor("lego.ply") == actor


if __name__ == "__main__":
    test_sim_init()
