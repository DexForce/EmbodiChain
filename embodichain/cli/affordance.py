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

"""Prepare object interaction data for simulation skills."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.task_program.semantics.profiles import RobotSkillProfile

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch standalone labeling or deployment cache preparation.

    Args:
        argv: Command-line arguments, or None for the process arguments.
    """
    parser = argparse.ArgumentParser(prog="embodichain affordance", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "label-grasps",
        add_help=False,
        help="Write a standalone .pt dataset from a mesh and hand configuration.",
    )
    commands.add_parser(
        "precompute-grasp-cache",
        add_help=False,
        help="Prepare mounted hand/object pairs for run-task.",
    )
    command, remaining = parser.parse_known_args(argv)
    if command.command == "label-grasps":
        from embodichain.lab.scripts.label_grasps import cli

        cli(remaining)
    else:
        _precompute(remaining)


def _precompute(argv: Sequence[str]) -> None:
    """Warm common deployment hand/affordance pairs before starting a planner."""
    import gymnasium
    from embodichain.lab.gym.envs.managers.record import (
        record_camera_data,
        validation_cameras,
    )
    from embodichain.lab.gym.utils.gym_utils import (
        add_env_launcher_args_to_parser,
        build_env_cfg_from_args,
    )
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.affordance.interactions import AntipodalAffordance
    from embodichain.lab.sim.affordance.grasp.cache import ValidatedGraspCache

    parser = argparse.ArgumentParser(
        prog="embodichain affordance precompute-grasp-cache",
        description="Populate the physical grasp cache for mounted hand/object pairs in a Task Program deployment.",
    )
    parser.add_argument(
        "--gym-config", required=True, help="Task Program deployment YAML"
    )
    parser.add_argument(
        "--physics", choices=("default", "newton"), help="Select the deployment backend"
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("HAND", "OBJECT", "AFFORDANCE"),
        help="Execution control part, object ID and antipodal affordance ID. Repeat for more pairs.",
    )
    args = parser.parse_args(argv)
    launcher = argparse.ArgumentParser(add_help=False)
    add_env_launcher_args_to_parser(launcher, require_gym_config=True)
    launcher.set_defaults(
        headless=True,
        num_envs=1,
        renderer="no-render",
        filter_dataset_saving=True,
        disable_sensor=True,
        physics=args.physics,
    )
    launch_args = launcher.parse_args(["--gym-config", args.gym_config])
    discover_task_packages()
    execute_init_hooks()
    cfg, gym_config, action_config = build_env_cfg_from_args(launch_args)
    # These event functors create their own cameras during world construction,
    # independently of the deployment's sensor acquisition setting.
    for name, event in vars(cfg.events).items():
        func = getattr(event, "func", None)
        if isinstance(func, type) and issubclass(
            func, (record_camera_data, validation_cameras)
        ):
            setattr(cfg.events, name, None)
    env = gymnasium.make(gym_config["id"], cfg=cfg, **action_config)
    try:
        assembly = env.unwrapped.task_program_adapter.assemble_runtime(
            cfg.task_program.integration
        )
        for part, obj, affordance_id in args.pair:
            affordance = assembly.scene_registry.object_semantics(
                obj, affordance=affordance_id
            ).affordance
            generator = assembly.engine.grasp_pose_generators[part]
            cache = getattr(generator, "validated_labels", None)
            if not isinstance(affordance, AntipodalAffordance) or not isinstance(
                cache, ValidatedGraspCache
            ):
                raise ValueError(
                    f"{part}/{affordance_id} requires a validated antipodal hand and affordance"
                )
            commands = assembly.engine.control_profiles[part].commands
            opened, closed = (
                commands[name].positions.reshape(-1) for name in ("open", "grasp")
            )
            for adjustment in _grasp_adjustments(assembly.robot_profile):
                path = cache.precompute(
                    affordance.mesh_vertices,
                    affordance.mesh_triangles,
                    affordance.validation_attrs,
                    hand_commands=(opened, closed),
                    pose_adjustment=adjustment,
                )
                print(f"{part} / {obj} / {affordance_id}: {path}", flush=True)
    finally:
        env.unwrapped.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _grasp_adjustments(profile: RobotSkillProfile) -> list[dict]:
    """Use the same final TCPs as the deployment's skill option templates."""
    from embodichain.lab.sim.atomic_skills import HandOverOptions, PickUpOptions
    from embodichain.lab.sim.atomic_skills._grasp_selection import pickup_adjustment

    adjustments = []
    for preset in profile.presets.values():
        for options in preset.skill_option_templates.values():
            if isinstance(options, PickUpOptions):
                adjustment = pickup_adjustment(options)
            elif isinstance(options, HandOverOptions):
                adjustment = (
                    {"retract": options.grasp_retract_distance}
                    if options.grasp_retract_distance
                    else {}
                )
            else:
                adjustment = {}
            if adjustment not in adjustments:
                adjustments.append(adjustment)
    return adjustments or [{}]
