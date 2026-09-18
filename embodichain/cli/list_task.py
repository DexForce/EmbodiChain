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

"""List tasks and export a static gallery from the shared catalog."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

from embodichain.cli._task_catalog import (
    _CAPABILITY_ORDER,
    _TASK_PROGRAM,
    _HANDWRITTEN_DEMO,
    _RL,
    _EnvironmentListEntry,
    _config_environment_entries,
    _implements_handwritten_demo,
    _task_path_from_module,
    _task_package_module_names,
    _task_config_roots,
    _discover_catalog,
    _catalog_entries,
    _catalog_from_args,
    _add_source_argument,
    _render_html,
)


def _collect_environment_entries() -> list[_EnvironmentListEntry]:
    return _sort_entries(_catalog_entries(_discover_catalog()).values())


def _sort_entries(entries) -> list[_EnvironmentListEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            tuple(part.casefold() for part in entry.task_path),
            entry.env_id.casefold(),
        ),
    )


def _print_environment_entries(entries: Sequence[_EnvironmentListEntry]) -> None:
    """Print environment entries as a table with a task-directory tree."""
    from prettytable import PrettyTable

    task_groups = [
        (task_path, list(task_entries))
        for task_path, task_entries in groupby(
            entries,
            key=lambda entry: entry.task_path,
        )
    ]
    table = PrettyTable()
    table.title = f"Tasks ({len(task_groups)}) / Environments ({len(entries)})"
    table.field_names = [
        "Task",
        "Environment ID",
        "Embodiment",
        "Capability",
        "Config",
    ]
    table.align = "l"
    active_categories: tuple[str, ...] = ()
    for group_index, (task_path, task_entries) in enumerate(task_groups):
        categories = task_path[:-1]
        shared_depth = 0
        for active, category in zip(active_categories, categories):
            if active != category:
                break
            shared_depth += 1
        for depth in range(shared_depth, len(categories)):
            table.add_row([f"{'  ' * depth}{categories[depth]}/", "", "", "", ""])

        task_name = task_path[-1]
        for entry_index, entry in enumerate(task_entries):
            labels = [
                capability
                for capability in _CAPABILITY_ORDER
                if capability in entry.capabilities
            ]
            if not labels:
                labels.append("Environment Only")
            table.add_row(
                [
                    (
                        f"{'  ' * len(categories)}{task_name}"
                        if entry_index == 0
                        else ""
                    ),
                    entry.env_id,
                    ", ".join(sorted(entry.embodiments, key=str.casefold)) or "-",
                    ", ".join(labels),
                    ", ".join(sorted(entry.config_names, key=str.casefold)) or "-",
                ]
            )
        if group_index < len(task_groups) - 1:
            table.add_divider()
        active_categories = categories
    print(table)


def main(argv: Sequence[str] | None = None) -> None:
    """List discovered tasks or export their static HTML gallery."""
    parser = argparse.ArgumentParser(
        prog="embodichain list-task",
        description="List tasks by category, deployment, and capability.",
        epilog="Environment Only means the task currently exposes neither an Expert Demo entry point nor a supported RL configuration.",
    )
    parser.add_argument(
        "--category", help="Filter by category path, including its subcategories."
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        metavar="PATH",
        help="Write a static HTML task gallery.",
    )
    _add_source_argument(parser)
    args = parser.parse_args(argv)
    try:
        if not args.config_root and not args.export_html:
            from embodichain.lab.gym.utils.registration import discover_task_packages

            discover_task_packages()
            entries = _collect_environment_entries()
        else:
            tasks = _catalog_from_args(args)
            if args.category:
                category = tuple(args.category.strip("/").split("/"))
                tasks = [
                    task
                    for task in tasks
                    if task.task_path[:-1][: len(category)] == category
                ]
            if args.export_html:
                args.export_html.parent.mkdir(parents=True, exist_ok=True)
                args.export_html.write_text(
                    _render_html(tasks, static_only=bool(args.config_root)),
                    encoding="utf-8",
                )
                print(f"Exported task gallery: {args.export_html}")
            entries = _sort_entries(_catalog_entries(tasks).values())
        if args.category:
            category = tuple(args.category.strip("/").split("/"))
            entries = [
                entry
                for entry in entries
                if entry.task_path[:-1][: len(category)] == category
            ]
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.config_root:
        print(
            "Static config catalog. Runtime registrations were not inspected; additional expert or RL capabilities may be available."
        )
    if not entries:
        print("No registered tasks found.")
        return
    _print_environment_entries(entries)


if __name__ == "__main__":
    main()


__all__: list[str] = []
