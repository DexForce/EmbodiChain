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

"""List registered robot configurations without loading their implementations."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from prettytable import PrettyTable, TableStyle

from embodichain.robot_registry import _robot_specs


def _variant_profiles(spec: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Group declared variants that share their complete control summary."""
    variants = spec.get("variants") or {}
    if not variants:
        defaults = ", ".join(
            f"{key}={value!r}"
            for key, value in sorted(spec.get("defaults", {}).items())
        )
        return [(defaults or "-", spec)]
    groups: list[tuple[list[str], Mapping[str, Any]]] = []
    fields = ("dof", "control_parts", "mimic_joints", "control_part_mimics")
    for name, profile in variants.items():
        for names, existing in groups:
            if all(profile.get(field) == existing.get(field) for field in fields):
                names.append(name)
                break
        else:
            groups.append(([name], profile))
    default = spec.get("default_variant")
    groups.sort(key=lambda group: default not in group[0])
    result = []
    for names, profile in groups:
        paths = [name.partition("/") for name in names]
        if (
            all(separator for _prefix, separator, _suffix in paths)
            and len({suffix for _prefix, _separator, suffix in paths}) == 1
        ):
            versions = ", ".join(
                prefix + ("*" if name == default else "")
                for name, (prefix, _separator, _suffix) in zip(names, paths)
            )
            label = f"{versions}\n{paths[0][2]}"
        else:
            label = ", ".join(name + ("*" if name == default else "") for name in names)
        result.append((label, profile))
    return result


def _format_control_parts(spec: Mapping[str, Any]) -> str:
    """Mark each control group's independent and mimic joint counts."""
    part_mimics = spec.get("control_part_mimics")
    lines = []
    for part, count in (spec.get("control_parts") or {}).items():
        if part_mimics is None:
            label = f"{count} total"
        else:
            mimic_count = len(set(part_mimics.get(part, ())))
            label = (
                f"{count - mimic_count}+{mimic_count}m" if mimic_count else str(count)
            )
        lines.append(f"{part} ({label})")
    return "\n".join(lines) or "-"


def _format_robot_entries(
    entries: Mapping[str, Mapping[str, Any]],
    *,
    color: bool | None = None,
    width: int | None = None,
) -> str:
    """Format robot families, model-specific DOFs, control groups, and mimics."""
    if color is None:
        color = "NO_COLOR" not in os.environ and sys.stdout.isatty()
    width = max(80, min(width or shutil.get_terminal_size((120, 24)).columns, 120))
    families = [
        name
        for name, spec in entries.items()
        if not (
            spec.get("variant_of") in entries
            and entries[spec["variant_of"]].get("variants")
        )
    ]
    table = PrettyTable()
    table.set_style(TableStyle.SINGLE_BORDER)
    label = "robot" if len(families) == 1 else "robots"
    table.title = f"EmbodiChain · Robot Catalog · {len(families)} {label}"
    table.field_names = [
        "Robot ID",
        "Variants",
        "Control DOF",
        "Control Parts",
        "Mimic Joints",
    ]
    table.align = "l"
    available_width = width - 16
    robot_id_width = max(12, min(20, max(map(len, families), default=12)))
    parts_width = max(17, int(available_width * 0.24))
    variants_width = max(
        8,
        min(
            int(available_width * 0.23),
            available_width - robot_id_width - 11 - parts_width - 12,
        ),
    )
    table.max_width = {
        "Robot ID": robot_id_width,
        "Variants": variants_width,
        "Control DOF": 11,
        "Control Parts": parts_width,
        "Mimic Joints": available_width
        - robot_id_width
        - 11
        - parts_width
        - variants_width,
    }
    for index, robot_id in enumerate(sorted(families, key=str.casefold)):
        for profile_index, (variants, profile) in enumerate(
            _variant_profiles(entries[robot_id])
        ):
            mimics = profile.get("mimic_joints")
            if mimics is None:
                mimic_text = "-"
            elif mimics:
                mimic_text = f"{len(mimics)} total\n" + "\n".join(mimics)
            else:
                mimic_text = "None"
            dof = profile.get("dof")
            table.add_row(
                [
                    robot_id if profile_index == 0 else "",
                    variants,
                    str(dof) if dof is not None else "-",
                    _format_control_parts(profile),
                    mimic_text,
                ]
            )
        if index < len(families) - 1:
            table.add_divider()
    lines = table.get_string().splitlines()
    if lines:
        lines[0] = f"╭{lines[0][1:-1]}╮"
        lines[-1] = f"╰{lines[-1][1:-1]}╯"
    if color:
        for index, line in enumerate(lines):
            cells = line.split("│")
            if len(cells) == 7:
                for column, code in (
                    (1, "1;36"),
                    (2, "1;32"),
                    (3, "1;32"),
                    (4, "1;33"),
                    (5, "1;33"),
                ):
                    if cells[column].strip():
                        cells[column] = f"\033[{code}m{cells[column]}\033[0m"
                lines[index] = "│".join(cells)
            elif "EmbodiChain ·" in line:
                lines[index] = f"\033[1;36m{line}\033[0m"
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """List robot configurations from installed catalogs.

    Args:
        argv: Arguments excluding the command name. Uses sys.argv when omitted.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain list-robot",
        description="List robot families, variants, independent control DOFs, and mimic joints.",
        epilog=(
            "Control DOF counts independent joints in default control parts. "
            "Part counts use independent+mimic notation; m marks mimic joints. "
            "Mimic Joints lists URDF followers, including passive joints outside "
            "control groups. Overlapping groups count once. * marks the default "
            "variant; '-' means unavailable metadata. DualArmRobot variants "
            "select base_robot; URRobot variants select robot_type. W1 variants "
            "combine body version with left/right hand_types; hand releases "
            "use v021 independently of the body version. bare sets "
            "with_default_eef=False."
        ),
    )
    parser.parse_args(argv)
    try:
        entries = _robot_specs()
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if not entries:
        print("No registered robots found.")
        return
    print(_format_robot_entries(entries))


if __name__ == "__main__":
    main()


__all__: list[str] = []
