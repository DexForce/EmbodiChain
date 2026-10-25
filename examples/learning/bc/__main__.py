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

"""Lazy commands for the behavior cloning showcase."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
import sys

__all__: list[str] = []

_COMMANDS = {
    "collect": "_collect",
    "train": "_train",
    "evaluate": "_evaluate",
    "reference-parity": "_reference_parity",
    "runtime-oracle": "_runtime_oracle",
    "window-candidate": "_window_candidate",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one showcase command, importing only its selected dependencies."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="embodichain run-example learning/bc", description=__doc__
    )
    parser.add_argument("command", choices=tuple(_COMMANDS))
    args = parser.parse_args(arguments[:1])
    sys.argv = [f"{parser.prog} {args.command}", *arguments[1:]]
    import_module(f"examples.learning.bc._support.{_COMMANDS[args.command]}").main()


if __name__ == "__main__":
    main()
