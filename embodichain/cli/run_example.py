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

"""Launch installed examples from lightweight entry-point catalogs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from importlib import import_module, metadata
import sys

__all__: list[str] = []


def _example_specs() -> dict[str, dict[str, str]]:
    specs = {}
    for entry_point in metadata.entry_points(group="embodichain.examples"):
        catalog = entry_point.load()
        if not isinstance(catalog, Mapping):
            raise TypeError(f"Example catalog {entry_point.name!r} must be a mapping")
        for name, spec in catalog.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Example names must be nonempty strings")
            if name in specs:
                raise ValueError(f"Duplicate example name {name!r}")
            if not isinstance(spec, Mapping):
                raise TypeError(
                    f"Example {name!r} must declare a target and description"
                )
            target = spec.get("target")
            description = spec.get("description", "")
            if not isinstance(target, str) or target.count(":") != 1:
                raise ValueError(f"Example {name!r} target must be module:callable")
            module, attribute = target.split(":")
            if (
                not all(part.isidentifier() for part in module.split("."))
                or not attribute.isidentifier()
            ):
                raise ValueError(f"Example {name!r} target must be module:callable")
            if not isinstance(description, str):
                raise TypeError(f"Example {name!r} description must be a string")
            specs[name] = {"target": target, "description": description}
    return specs


def main(argv: Sequence[str] | None = None) -> None:
    """List installed examples or forward arguments to the selected example.

    Args:
        argv: Arguments following the command name, or process arguments.
    """
    specs = _example_specs()
    parser = argparse.ArgumentParser(
        prog="embodichain run-example", description=__doc__
    )
    parser.add_argument("--list", action="store_true", help="List installed examples.")
    parser.add_argument("example", nargs="?", choices=sorted(specs))
    parser.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="Example arguments."
    )
    args = parser.parse_args(argv)
    if args.list:
        for name, spec in sorted(specs.items()):
            print(f"{name}: {spec['description']}")
        return
    if args.example is None:
        parser.print_help()
        return
    module, attribute = specs[args.example]["target"].split(":")
    handler = getattr(import_module(module), attribute)
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    handler(arguments)


def atomic_action_cli(argv: Sequence[str] | None = None) -> None:
    """Forward the existing Atomic Skill command to the registered example.

    Args:
        argv: Arguments following the legacy command name.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    main(["atomic-action", *arguments])
