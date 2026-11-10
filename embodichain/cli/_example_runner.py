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

"""Execute an example with package imports and ordinary script arguments."""

from __future__ import annotations

import importlib.util
from importlib.machinery import PathFinder
from pathlib import Path
import runpy
import sys
from types import ModuleType

__all__: list[str] = []


def _has_module_path(root: Path, relative: Path) -> bool:
    """Check import resolution without executing an unselected parent script."""
    search = [str(root)]
    for index, part in enumerate(relative.parts):
        if not part.isidentifier():
            return False
        # Probe one directory at a time. Qualified namespace specs require
        # imported parents, which this validation deliberately avoids.
        spec = PathFinder.find_spec(part, search)
        if spec is None:
            return False
        if index == len(relative.parts) - 1:
            return spec.origin == str(root / relative.with_suffix(".py"))
        directory = str(root.joinpath(*relative.parts[: index + 1]))
        if directory not in (spec.submodule_search_locations or ()):
            return False
        search = [directory]
    return False


def _run(root: Path, script: Path, arguments: list[str]) -> None:
    # Keep the caller's working directory for relative input/output arguments.
    # Both package imports and legacy bare sibling imports occur in examples.
    sys.path[:0] = [
        str(root.parent),
        str(Path(__file__).resolve().parents[2]),
    ]
    # Bare sibling imports remain available without shadowing installed SDKs
    # with helpers such as ods/simple_bc_demo/lerobot.py.
    sys.path.append(str(script.parent))
    initializer = root / "__init__.py"
    if initializer.is_file():
        spec = importlib.util.spec_from_file_location(
            "examples", initializer, submodule_search_locations=[str(root)]
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        sys.modules["examples"] = package
        spec.loader.exec_module(package)
    else:
        package = ModuleType("examples")
        package.__path__ = [str(root)]
        sys.modules["examples"] = package
    relative = script.relative_to(root).with_suffix("")
    module = ".".join(("examples", *relative.parts))
    sys.argv = [str(script), *arguments]
    if _has_module_path(root, relative):
        # Module execution preserves package-relative imports and spawn workers.
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        # A file may have dots in its name or share a stem with a package.
        # In that case Python's module lookup would select a different file.
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    _run(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
