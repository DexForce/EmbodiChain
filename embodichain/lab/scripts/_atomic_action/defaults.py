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

"""Internal sample defaults for the atomic-action launcher."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from typing import Any

from embodichain.cli.sim import add_sim_args_to_parser

SCENES = (
    "pickup",
    "place",
    "pour",
    "press",
    "slide",
    "twist",
    "handover",
    "move_joint",
    "axis_align",
    "open_door",
)
ROBOTS = ("ur5", "franka", "ur10")
PLANNERS = ("trapezoidal", "toppra", "curobo")
MULTI_ROBOT_SCENES = frozenset({"pickup", "place", "move_joint", "handover"})

OPTIONS: dict[str, dict[str, Any]] = {
    "pickup": {"approach": "top", "custom_approach_direction": None},
    "place": {},
    "pour": {"rotate_angle": math.pi / 4},
    "press": {"press_distance": 0.03, "press_position": None, "rigid_object": False},
    "slide": {"translation_distance": 0.18, "approach_distance": 0.10},
    "twist": {"twist_angle": -0.7853981634, "rigid_object": False},
    "handover": {"is_horizontal": False},
    "move_joint": {},
    "axis_align": {"alignment": "upright"},
    "open_door": {
        "open_angle": math.radians(60),
        "approach_distance": 0.10,
        "retract_distance": 0.10,
        "door_waypoint_count": 50,
    },
}


def _sample_defaults(
    skill: str,
    *,
    physics: str = "default",
    robot: str = "ur5",
    planner: str = "trapezoidal",
) -> dict[str, Any]:
    """Build an independent sample recipe without loading external resources.

    Scene implementations own layouts; these values retain the original sample
    options, grasp sampling and expansion policy. Compatibility is checked here
    before the launcher imports any scene or constructs a simulator.
    """
    if skill not in SCENES:
        raise ValueError(f"skill must be one of {SCENES}")
    if physics not in ("default", "newton"):
        raise ValueError("physics must be default or newton")
    if robot not in ROBOTS:
        raise ValueError(f"robot must be one of {ROBOTS}")
    if planner not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}")
    if skill not in MULTI_ROBOT_SCENES and robot != "ur5":
        raise ValueError(
            f"Skill {skill!r} requires robot 'ur5'; the 'franka' and 'ur10' "
            "presets are supported by pickup, place, move_joint and handover"
        )
    has_affordance = skill != "move_joint"
    return {
        "skill": skill,
        "physics": physics,
        "robot": robot,
        "planner": planner,
        "seed": 0,
        "runtime": {
            "num_envs": 16 if has_affordance else 4,
            "device": "cpu",
            "renderer": "auto",
            "headless": False,
            "arena_space": None,
            "gpu_id": 0,
            "real_time": False,
            "hold_steps": None,
        },
        "options": deepcopy(OPTIONS[skill]),
        "grasp": {"n_sample": 10000, "force_reannotate": False},
        "expansion": {
            "affordance": {
                "enabled": has_affordance,
                "branches_per_family": 4 if has_affordance else 1,
            },
            "trajectory": {
                "variants_per_family": 4,
                "max_proposals": 128,
                "spatial": {
                    "enabled": True,
                    "operators": ["joint_residual", "via_points"],
                    "joint_offset_scale": 0.005,
                    "dedup_tolerance": 0.002,
                    "via_count": 3,
                },
            },
        },
        "output": {
            "directory": f"/tmp/atomic-action/{skill}",
            "save_video": False,
            "save_trajectories": True,
        },
    }


def _recipes(cfg: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    affordance = cfg["expansion"]["affordance"]
    branches = affordance["branches_per_family"] if affordance["enabled"] else 1
    variants = cfg["expansion"]["trajectory"]["variants_per_family"]
    return tuple(
        (branch, variant) for branch in range(branches) for variant in range(variants)
    )


def _scene_args(
    cfg: dict[str, Any], recipe_indices: tuple[int, ...], batch_id: int
) -> argparse.Namespace:
    # Simulation/Viser parser defaults are owned by the common launcher.
    parser = argparse.ArgumentParser()
    add_sim_args_to_parser(parser)
    args = parser.parse_args([])
    vars(args).update(cfg["runtime"])
    vars(args).update(cfg["options"])
    vars(args).update(cfg["grasp"])
    args.physics = cfg["physics"]
    args.robot = cfg["robot"]
    args.planner = cfg["planner"]
    args.auto_play = True
    args.auto_play_video_path = None
    args.no_vis_eef_axis = False
    args.diagnose_plan = False
    args.headless_play = args.headless
    args.sampling_seed = cfg["seed"]
    args.sampling_attempt = 0
    affordance = cfg["expansion"]["affordance"]
    args.affordance_branches = (
        affordance["branches_per_family"] if affordance["enabled"] else 1
    )
    args._recipes = tuple(_recipes(cfg)[index] for index in recipe_indices)
    args._recipe_indices = recipe_indices
    args._batch_id = batch_id
    args._config = cfg
    args._configured_run = True
    args._records = []
    args._replay_index = 0
    return args


__all__: list[str] = []
