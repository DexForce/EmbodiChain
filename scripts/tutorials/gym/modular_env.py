# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

from __future__ import annotations

import argparse
import random
from copy import deepcopy

from embodichain.cli.sim import (
    add_sim_args_to_parser,
    add_seed_arg_to_parser,
    resolve_seed,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options without initializing simulation resources."""
    parser = argparse.ArgumentParser()
    add_sim_args_to_parser(parser)
    add_seed_arg_to_parser(parser, scope="task environment")
    return parser


if __name__ == "__main__":
    # Parse before importing optional simulation/planning dependencies.
    _cli_args = build_parser().parse_args()


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
    RenderCfg,
    physics_cfg_for_backend,
    LightCfg,
    ArticulationCfg,
    RobotCfg,
    RigidObjectCfg,
    RigidBodyPhysicsCfg,
)
from embodichain.data import get_data_path
from embodichain.utils import configclass
from embodichain.utils.file import get_all_files_in_directory
from embodichain.lab.gym.envs.managers.event_manager import _derive_functor_seed


class _ReplacePreselectedFork(events.replace_assets_from_group):
    """Consume the first reset draw without replacing an already selected fork."""

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor | None,
        entity_cfg: SceneEntityCfg,
        folder_path: str,
    ) -> None:
        initial_path = env._initial_fork_path
        env._initial_fork_path = None
        if initial_path is not None:
            # Preview the actual event stream: reset(seed=...) may have changed
            # it since construction. Leave fallback sampling to the base term.
            preview = random.Random()
            preview.setstate(random.getstate())
            if (
                preview.choice(self._asset_group_path) == initial_path
                and self.asset_cfg.shape.fpath == initial_path
            ):
                random.choice(self._asset_group_path)
                # As with a real replacement, keep the template separate from
                # the live object modified by later mass randomization terms.
                self.asset_cfg = deepcopy(self.asset_cfg)
                return
        super().__call__(env, env_ids, entity_cfg, folder_path)


@configclass
class ExampleEventCfg:

    replace_obj: EventCfg = EventCfg(
        func=_ReplacePreselectedFork,
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

    randomize_table_mat: EventCfg = EventCfg(
        func=rand.randomize_visual_material,
        mode="interval",
        interval_step=25,
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

    background: List[RigidObjectCfg] = [
        RigidObjectCfg(
            uid="table",
            shape=MeshCfg(
                fpath=get_data_path("CircleTableSimple/circle_table_simple.ply"),
                compute_uv=True,
            ),
            attrs=RigidBodyPhysicsCfg.from_dict(
                {
                    "mass_props": {"mass": 10.0},
                    "material_props": {
                        "static_friction": 0.95,
                        "dynamic_friction": 0.85,
                        "restitution": 0.01,
                    },
                }
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


def _preselect_initial_fork(
    cfg: EmbodiedEnvCfg,
) -> tuple[EmbodiedEnvCfg, str | None]:
    """Prepare this tutorial's first reset asset before scene materialization.

    Use the same named event stream and sorted asset list as EventManager.
    Keep unseeded runs and customized replacement terms on their normal path.
    """
    term = getattr(cfg.events, "replace_obj", None)
    if (
        cfg.seed is None
        or cfg.seed < 0
        or term is None
        or term.func is not _ReplacePreselectedFork
        or term.mode != "reset"
    ):
        return cfg, None
    folder_path = term.params["folder_path"]
    if not folder_path.endswith("/"):
        return cfg, None
    paths = sorted(get_all_files_in_directory(get_data_path(folder_path)))
    if not paths:
        raise ValueError(f"No fork assets found in {folder_path!r}.")
    stream_seed = _derive_functor_seed(cfg.seed, "call", "reset", "replace_obj", 0)
    path = random.Random(stream_seed).choice(paths)
    prepared = deepcopy(cfg)
    uid = term.params["entity_cfg"].uid
    asset_cfg = next(asset for asset in prepared.rigid_object if asset.uid == uid)
    asset_cfg.shape.fpath = path
    return prepared, path


@register_env("ModularEnv-v1", max_episode_steps=100, override=True)
class ModularEnv(EmbodiedEnv):
    """
    An example of a modular environment that inherits from EmbodiedEnv
    and uses custom event and observation managers.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs):
        cfg, self._initial_fork_path = _preselect_initial_fork(cfg)
        super().__init__(cfg, **kwargs)


if __name__ == "__main__":
    import gymnasium as gym
    import argparse

    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.cli.sim import (
        add_sim_args_to_parser,
        add_seed_arg_to_parser,
        resolve_seed,
    )
    from embodichain.lab.visualization import visualization_cfg_from_args

    parser = build_parser()
    args = _cli_args
    seed = resolve_seed(args.seed)
    print(f"[INFO]: Environment seed: {seed}", flush=True)

    env_cfg = ExampleCfg(
        seed=seed,
        sim_cfg=SimulationManagerCfg(
            render_cfg=RenderCfg(renderer=args.renderer),
            headless=args.headless,
            device=args.device,
            num_envs=args.num_envs,
            physics_cfg=physics_cfg_for_backend(args.physics),
            visualization=visualization_cfg_from_args(args),
        ),
        num_envs=args.num_envs,
    )

    # Create the Gym environment
    env = gym.make("ModularEnv-v1", cfg=env_cfg)

    for i in range(5):
        obs, info = env.reset()

        for i in range(100):
            action = torch.zeros(env.action_space.shape, dtype=torch.float32)
            obs, reward, done, truncated, info = env.step(action)
