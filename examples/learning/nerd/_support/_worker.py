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


"""Bootstrap one private showcase worker without importing the SDK CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from types import ModuleType

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker", help="Private module stem in this support directory.")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    name = args.worker
    if (
        not name.isidentifier()
        or not name.startswith("_")
        or name.startswith("__")
        or name == "_worker"
        or not (directory / (name + ".py")).is_file()
    ):
        parser.error("worker must name a private module in the NeRD support directory")
    # Examples are wheel data, so their installed parent need not be on sys.path.
    # Keep this bootstrap compatible with isolated historical Python 3.8 workers.
    examples = directory.parents[2]
    package = ModuleType("examples")
    package.__path__ = [str(examples)]
    sys.modules["examples"] = package
    sys.argv = [str(directory / (name + ".py")), *args.arguments]
    runpy.run_module(
        "examples.learning.nerd._support." + name,
        run_name="__main__",
        alter_sys=True,
    )


if __name__ == "__main__":
    main()
