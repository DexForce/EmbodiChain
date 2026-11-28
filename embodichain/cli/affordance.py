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
    from embodichain.lab.gym.envs.task_program._grasp_preparation import (
        prepare_deployment_grasps,
    )

    for (part, obj, affordance_id), path in prepare_deployment_grasps(
        args.gym_config, args.physics, args.pair
    ):
        print(f"{part} / {obj} / {affordance_id}: {path}", flush=True)
