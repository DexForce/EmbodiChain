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
from embodichain.utils import configclass

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
    "move_held_object",
    "push_object",
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
    "move_held_object": {},
    "push_object": {},
    "axis_align": {"alignment": "upright"},
    "open_door": {
        "open_angle": math.radians(60),
        "approach_distance": 0.10,
        "retract_distance": 0.10,
        "door_waypoint_count": 50,
    },
}


@configclass
class AtomicSkillRunCfg:
    """Resolved sample selectors, fixed-scene factors and output settings."""

    skill: str = "pickup"
    physics: str = "default"
    robot: str = "ur5"
    planner: str = "trapezoidal"
    variants: int = 1
    expansion: str = "both"
    seed: int = 0
    device: str | None = None
    renderer: str = "auto"
    headless: bool = False
    arena_space: float | None = None
    gpu_id: int = 0
    real_time: bool = False
    output_dir: str = "/tmp/atomic-action/pickup"
    save_video: bool = False
    save_trajectories: bool = True
    grasp_approach_samples: int = 1
    n_sample: int = 10000
    force_reannotate: bool = False
    hold_steps: int | None = None

    @property
    def affordance_enabled(self) -> bool:
        return (
            self.variants > 1
            and self.expansion in ("affordance", "both")
            and self.skill not in {"move_joint", "push_object"}
        )

    @property
    def trajectory_enabled(self) -> bool:
        return self.variants > 1 and self.expansion in ("trajectory", "both")


def _sample_defaults(
    skill: str,
    *,
    physics: str = "default",
    robot: str = "ur5",
    planner: str = "trapezoidal",
) -> AtomicSkillRunCfg:
    """Validate sample compatibility before loading scene or simulator modules."""
    for value, choices, name in (
        (skill, SCENES, "skill"),
        (physics, ("default", "newton"), "physics"),
        (robot, ROBOTS, "robot"),
        (planner, PLANNERS, "planner"),
    ):
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}")
    if skill not in MULTI_ROBOT_SCENES and robot != "ur5":
        raise ValueError(
            f"Skill {skill!r} requires robot 'ur5'; the 'franka' and 'ur10' "
            "presets are supported by pickup, place, move_joint and handover"
        )
    return AtomicSkillRunCfg(
        skill=skill,
        physics=physics,
        robot=robot,
        planner=planner,
        output_dir=f"/tmp/atomic-action/{skill}",
    )


def _scene_args(cfg: AtomicSkillRunCfg) -> argparse.Namespace:
    """Build the compatibility arguments consumed by shared example scenes."""
    parser = argparse.ArgumentParser()
    add_sim_args_to_parser(parser)
    args = parser.parse_args([])
    vars(args).update(cfg.to_dict())
    vars(args).update(deepcopy(OPTIONS[cfg.skill]))
    args.num_envs = cfg.variants
    args.auto_play = True
    args.auto_play_video_path = None
    args.no_vis_eef_axis = False
    args.diagnose_plan = False
    args.headless_play = cfg.headless
    args.sampling_seed = cfg.seed
    args.sampling_attempt = 0
    args.affordance_branches = cfg.variants if cfg.affordance_enabled else 1
    return args


__all__: list[str] = []
