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

from typing import List, Dict, Any

import embodichain.lab.gym.envs.managers.randomization as rand
import embodichain.lab.gym.envs.managers.events as events
import embodichain.lab.gym.envs.managers.observations as obs

from embodichain.lab.gym.envs.managers import (
    EventCfg,
    SceneEntityCfg,
    ObservationCfg,
)
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.sensors import StereoCameraCfg, SensorCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.cfg import (
    LightCfg,
    ArticulationCfg,
    RobotCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
)
from embodichain.data import get_data_path
from embodichain.utils import configclass


@configclass
class ExampleEventCfg:

    replace_obj: EventCfg = EventCfg(
        func=events.replace_assets_from_group,
        mode="reset",
        params={
            "entity_cfg": SceneEntityCfg(
                uid="fork",
            ),
            "folder_path": get_data_path("TableWare/tableware/fork/"),
        },
    )

    randomize_fork_mass: EventCfg = EventCfg(
        func=rand.randomize_rigid_object_mass,
        mode="reset",
        params={
            "entity_cfg": SceneEntityCfg(
                uid="fork",
            ),
            "mass_range": (0.1, 2.0),
        },
    )

    randomize_light: EventCfg = EventCfg(
        func=rand.randomize_light,
        mode="interval",
        interval_step=5,
        params={
            "entity_cfg": SceneEntityCfg(
                uid="point",
            ),
            "position_range": [[-0.5, -0.5, 2], [0.5, 0.5, 2]],
            "color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
            "intensity_range": [50.0, 100.0],
        },
    )

    randomize_table_mat: EventCfg = EventCfg(
        func=rand.randomize_visual_material,
        mode="interval",
        interval_step=10,
        params={
            "entity_cfg": SceneEntityCfg(
                uid="table",
            ),
            "random_texture_prob": 0.5,
            "texture_path": get_data_path("CocoBackground/coco"),
            "base_color_range": [[0.2, 0.2, 0.2], [1.0, 1.0, 1.0]],
        },
    )


@configclass
class ObsCfg:

    obj_pose: ObservationCfg = ObservationCfg(
        func=obs.get_rigid_object_pose,
        mode="add",
        name="fork_pose",
        params={"entity_cfg": SceneEntityCfg(uid="fork")},
    )


@configclass
class ExampleCfg(EmbodiedEnvCfg):

    # Define the robot configuration using DexforceW1Cfg
    robot: RobotCfg = DexforceW1Cfg.from_dict(
        {
            "uid": "dexforce_w1",
            "version": "v021",
            "arm_kind": "anthropomorphic",
            "init_pos": [0.0, 0, 0.0],
        }
    )

    # Define the sensor configuration using StereoCameraCfg
    sensor: List[SensorCfg] = [
        StereoCameraCfg(
            uid="eye_in_head",
            width=960,
            height=540,
            enable_mask=True,
            enable_depth=True,
            left_to_right_pos=(0.06, 0, 0),
            intrinsics=(450, 450, 480, 270),
            intrinsics_right=(450, 450, 480, 270),
            extrinsics=StereoCameraCfg.ExtrinsicsCfg(
                parent="eyes",
            ),
        )
    ]

    light: EmbodiedEnvCfg.EnvLightCfg = EmbodiedEnvCfg.EnvLightCfg(
        direct=[
            LightCfg(
                uid="point",
                light_type="point",
                color=(1.0, 1.0, 1.0),
                intensity=50.0,
                init_pos=(0, 0, 2),
            )
        ]
    )

    background: List[RigidObjectCfg] = [
        RigidObjectCfg(
            uid="table",
            shape=MeshCfg(
                fpath=get_data_path("CircleTableSimple/circle_table_simple.ply"),
                compute_uv=True,
            ),
            attrs=RigidBodyAttributesCfg(
                mass=10.0,
                static_friction=0.95,
                dynamic_friction=0.85,
                restitution=0.01,
            ),
            body_type="kinematic",
            init_pos=(0.80, 0, 0.8),
            init_rot=(0, 90, 0),
        ),
    ]

    rigid_object: List[RigidObjectCfg] = [
        RigidObjectCfg(
            uid="fork",
            shape=MeshCfg(
                fpath=get_data_path("TableWare/tableware/fork/standard_fork_scale.ply"),
            ),
            body_scale=(0.75, 0.75, 1.0),
            init_pos=(0.8, 0, 1.0),
        ),
    ]

    articulation_cfg: List[ArticulationCfg] = [
        ArticulationCfg(
            uid="drawer",
            fpath="SlidingBoxDrawer/SlidingBoxDrawer.urdf",
            init_pos=(0.5, 0.0, 0.85),
        )
    ]

    events = ExampleEventCfg()

    observations = ObsCfg()


@register_env("ModularEnv-v1", max_episode_steps=100, override=True)
class ModularEnv(EmbodiedEnv):
    """
    An example of a modular environment that inherits from EmbodiedEnv
    and uses custom event and observation managers.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)


if __name__ == "__main__":
    import gymnasium as gym
    import argparse

    from embodichain.lab.sim import SimulationManagerCfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_rt", action="store_true", help="Enable ray tracing")
    args = parser.parse_args()

    env_cfg = ExampleCfg(sim_cfg=SimulationManagerCfg(enable_rt=args.enable_rt))

    # Create the Gym environment
    env = gym.make("ModularEnv-v1", cfg=env_cfg)

    while True:
        obs, info = env.reset()

        for i in range(100):
            action = torch.zeros(env.action_space.shape, dtype=torch.float32)
            obs, reward, done, truncated, info = env.step(action)
