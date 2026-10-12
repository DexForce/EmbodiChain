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

"""Strict, simulator-free configuration for the atomic-action launcher."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml

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
RUNTIME_DEFAULTS = {
    "num_envs": 1,
    "device": None,
    "renderer": "auto",
    "headless": False,
    "arena_space": None,
    "gpu_id": 0,
    "real_time": False,
    "hold_steps": None,
}


def _mapping(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = value.keys() - fields
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {sorted(unknown)}")
    return deepcopy(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(
            f"{name} must be a finite {'positive ' if positive else ''}number"
        )
    return float(value)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid atomic scene YAML: {exc}") from exc
    cfg = _mapping(
        raw,
        "config",
        {
            "skill",
            "physics",
            "robot",
            "planner",
            "seed",
            "runtime",
            "options",
            "grasp",
            "expansion",
            "output",
        },
    )
    if cfg.get("skill") not in SCENES:
        raise ValueError(f"skill must be one of {SCENES}")
    if cfg.get("physics") not in ("default", "newton"):
        raise ValueError("config must declare physics: default|newton")
    cfg.setdefault("robot", "ur5")
    cfg.setdefault("planner", "trapezoidal")
    cfg.setdefault("seed", 0)
    if cfg["robot"] not in ("ur5", "franka", "ur10") or (
        cfg["skill"] not in ("pickup", "place", "move_joint", "handover")
        and cfg["robot"] != "ur5"
    ):
        raise ValueError(
            "This scene requires its example's UR5 robot; robot variants are supported by pickup, place, move_joint and handover"
        )
    if cfg["planner"] not in ("trapezoidal", "toppra", "curobo"):
        raise ValueError("planner must be trapezoidal, toppra or curobo")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    runtime = RUNTIME_DEFAULTS | _mapping(
        cfg.get("runtime", {}), "runtime", set(RUNTIME_DEFAULTS)
    )
    _positive_int(runtime["num_envs"], "runtime.num_envs")
    if type(runtime["gpu_id"]) is not int or runtime["gpu_id"] < 0:
        raise ValueError("runtime.gpu_id must be a non-negative integer")
    for key in ("headless", "real_time"):
        _bool(runtime[key], f"runtime.{key}")
    if runtime["device"] is not None and (
        not isinstance(runtime["device"], str) or not runtime["device"]
    ):
        raise ValueError("runtime.device must be a nonempty device string")
    if runtime["renderer"] not in ("auto", "no-render", "hybrid", "fast-rt", "rt"):
        raise ValueError("Unsupported runtime.renderer")
    if runtime["arena_space"] is not None:
        _number(runtime["arena_space"], "runtime.arena_space", positive=True)
    if runtime["hold_steps"] is not None and (
        type(runtime["hold_steps"]) is not int or runtime["hold_steps"] < 0
    ):
        raise ValueError("runtime.hold_steps must be a non-negative integer")
    cfg["runtime"] = runtime
    defaults = OPTIONS[cfg["skill"]]
    options = defaults | _mapping(cfg.get("options", {}), "options", set(defaults))
    for key, value in options.items():
        if isinstance(defaults[key], bool):
            _bool(value, f"options.{key}")
        elif isinstance(defaults[key], (int, float)):
            _number(value, f"options.{key}")
    if "door_waypoint_count" in options:
        _positive_int(options["door_waypoint_count"], "options.door_waypoint_count")
    for key in ("custom_approach_direction", "press_position"):
        if options.get(key) is not None:
            value = options[key]
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"options.{key} must contain three numbers")
            for number in value:
                _number(number, f"options.{key}")
    if cfg["skill"] == "pickup":
        if options["approach"] not in ("top", "side", "side_y", "custom"):
            raise ValueError("Unsupported pickup approach")
        if options["approach"] == "custom" and (
            options["custom_approach_direction"] is None
            or sum(v * v for v in options["custom_approach_direction"]) == 0
        ):
            raise ValueError(
                "custom approach requires a nonzero custom_approach_direction"
            )
    if cfg["skill"] == "axis_align" and options["alignment"] not in (
        "upright",
        "horizontal_align",
    ):
        raise ValueError("alignment must be upright or horizontal_align")
    if cfg["skill"] == "slide" and not 0 < options["translation_distance"] <= 0.285:
        raise ValueError("translation_distance must be in (0, 0.285]")
    cfg["options"] = options
    grasp = {"n_sample": 10000, "force_reannotate": False} | _mapping(
        cfg.get("grasp", {}), "grasp", {"n_sample", "force_reannotate"}
    )
    _positive_int(grasp["n_sample"], "grasp.n_sample")
    _bool(grasp["force_reannotate"], "grasp.force_reannotate")
    cfg["grasp"] = grasp
    expansion = _mapping(
        cfg.get("expansion", {}), "expansion", {"affordance", "trajectory"}
    )
    affordance = {"enabled": False, "branches_per_family": 1} | _mapping(
        expansion.get("affordance", {}),
        "expansion.affordance",
        {"enabled", "branches_per_family"},
    )
    _bool(affordance["enabled"], "expansion.affordance.enabled")
    _positive_int(
        affordance["branches_per_family"], "expansion.affordance.branches_per_family"
    )
    if cfg["skill"] == "move_joint" and affordance["enabled"]:
        raise ValueError("move_joint has no grasp/contact affordance to expand")
    trajectory = {"variants_per_family": 1, "max_proposals": 128} | _mapping(
        expansion.get("trajectory", {}),
        "expansion.trajectory",
        {"variants_per_family", "max_proposals", "spatial"},
    )
    _positive_int(
        trajectory["variants_per_family"], "expansion.trajectory.variants_per_family"
    )
    _positive_int(trajectory["max_proposals"], "expansion.trajectory.max_proposals")
    if trajectory["max_proposals"] < trajectory["variants_per_family"]:
        raise ValueError("trajectory.max_proposals must cover variants_per_family")
    spatial = {
        "enabled": False,
        "operators": ["joint_residual", "via_points"],
        "joint_offset_scale": 0.005,
        "dedup_tolerance": 0.002,
        "via_count": 3,
    } | _mapping(
        trajectory.get("spatial", {}),
        "expansion.trajectory.spatial",
        {"enabled", "operators", "joint_offset_scale", "dedup_tolerance", "via_count"},
    )
    _bool(spatial["enabled"], "expansion.trajectory.spatial.enabled")
    if (
        not isinstance(spatial["operators"], list)
        or not spatial["operators"]
        or len(set(spatial["operators"])) != len(spatial["operators"])
        or any(
            op not in ("joint_residual", "via_points") for op in spatial["operators"]
        )
    ):
        raise ValueError(
            "spatial.operators must be a unique list of joint_residual/via_points"
        )
    for key in ("joint_offset_scale", "dedup_tolerance"):
        if not 0 < _number(spatial[key], f"spatial.{key}", positive=True) <= 1:
            raise ValueError(f"spatial.{key} must be in (0, 1]")
    _positive_int(spatial["via_count"], "spatial.via_count")
    if (
        spatial["enabled"]
        and spatial["dedup_tolerance"] >= spatial["joint_offset_scale"]
    ):
        raise ValueError("spatial.dedup_tolerance must be below joint_offset_scale")
    if trajectory["variants_per_family"] > 1 and not spatial["enabled"]:
        raise ValueError("trajectory variants require spatial.enabled")
    trajectory["spatial"] = spatial
    cfg["expansion"] = {"affordance": affordance, "trajectory": trajectory}
    output = {
        "directory": f"/tmp/atomic-action/{cfg['skill']}",
        "save_video": False,
        "save_trajectories": True,
    } | _mapping(
        cfg.get("output", {}),
        "output",
        {"directory", "save_video", "save_trajectories"},
    )
    for key in ("save_video", "save_trajectories"):
        _bool(output[key], f"output.{key}")
    if not isinstance(output["directory"], str) or not output["directory"]:
        raise ValueError("output.directory must be a nonempty path")
    directory = Path(output["directory"]).expanduser()
    output["directory"] = str(
        (path.parent / directory).resolve()
        if not directory.is_absolute()
        else directory
    )
    if output["save_video"] and runtime["renderer"] == "no-render":
        raise ValueError("save_video requires a renderer")
    cfg["output"] = output
    return cfg


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
