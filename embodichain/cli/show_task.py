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

"""Inspect a logical task and its concrete deployments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from embodichain.cli._task_catalog import (
    _add_source_argument,
    _catalog_from_args,
    _select_task,
    _capability_label,
    _validation_summary,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Print task details selected by unique key or package-qualified key."""
    parser = argparse.ArgumentParser(prog="embodichain show-task")
    parser.add_argument("task", help="Unique task key or PACKAGE:KEY")
    _add_source_argument(parser)
    args = parser.parse_args(argv)
    try:
        task = _select_task(_catalog_from_args(args), args.task)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.config_root:
        print(
            "Static config catalog. Runtime registrations were not inspected; additional expert or RL capabilities may be available."
        )
    print(f"{task.title} ({task.qualified_key})")
    print(task.summary)
    print(f"Category: {'/'.join(task.task_path[:-1])}")
    print(f"Tags: {', '.join(task.tags) or 'None'}")
    print(f"Default deployment: {task.default_deployment or 'unspecified'}")
    print("Preview: unavailable")
    if task.readme is not None:
        print(f"README: {task.readme}")
    for deployment in task.deployments:
        print(f"\n{deployment.name}: {deployment.env_id}")
        print(f"  Physics: {deployment.physics or 'unavailable'}")
        print(f"  Embodiment: {', '.join(deployment.embodiments) or 'unavailable'}")
        print(f"  Supported uses: {_capability_label(deployment)}")
        print(f"  Config: {deployment.config_ref or 'unavailable'}")
        print(f"  Agents: {', '.join(deployment.agent_refs) or 'None'}")
        print(f"  {_validation_summary(deployment)}")
        if deployment.validation_resource is not None:
            print(f"  Validation report: {deployment.validation_resource}")
        if deployment.command:
            print(f"  {deployment.command}")


__all__: list[str] = []
