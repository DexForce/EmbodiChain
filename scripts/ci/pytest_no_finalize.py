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

"""Run resource-lane pytest without DexSim's unsafe native finalizers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

import os
import sys
import gc
import importlib
import importlib.util
from pathlib import Path

import pytest

__all__ = ["main"]

_NATIVE_IMPORTS = (
    "numpy",
    "torch",
    "scipy",
    "trimesh",
    "tensordict",
    "warp",
    "pytorch_kinematics",
    "pinocchio",
    "open3d",
    "newton",
)


class _NativeImportGC:
    """Keep static dependencies out of repeated GC in this owned process."""

    def __init__(self, modules: Sequence[str]) -> None:
        self.modules = modules
        self.enabled = False
        self.freeze_count = 0
        self.disabled_reason = "bootstrap did not run"

    @staticmethod
    def _domain_loaded(root: Path) -> bool:
        domain_roots = {
            "dexsim",
            "embodichain",
            "embodichain_tasks",
            "embodichain_robots",
            "embodichain_models",
            "embodichain_benchmarks",
        }
        test_prefix = str(root / "tests") + os.sep
        return any(
            name.split(".", 1)[0] in domain_roots
            or (
                module is not None
                and str(vars(module).get("__file__", "")).startswith(test_prefix)
            )
            for name, module in list(sys.modules.items())
        )

    @pytest.hookimpl(tryfirst=True)
    def pytest_load_initial_conftests(self, early_config: pytest.Config) -> None:
        if self._domain_loaded(early_config.rootpath):
            self.disabled_reason = "domain or test modules already loaded"
            return
        try:
            for name in self.modules:
                if importlib.util.find_spec(name) is not None:
                    importlib.import_module(name)
        except Exception as error:
            self.disabled_reason = f"dependency preload unavailable: {error!r}"
            return
        if self._domain_loaded(early_config.rootpath):
            self.disabled_reason = "dependency preload imported domain or test code"
            return
        gc.collect()
        gc.freeze()
        self.enabled = True
        self.freeze_count = gc.get_freeze_count()
        # Domain/test modules and every fixture are imported/created afterward,
        # so their cycles still participate in full lifecycle collections.
        # Keep existing permanent roots intact. This process always os._exit()s
        # after pytest hooks, so it never returns a changed GC policy to a host.


def _native_resources_released() -> bool:
    native = sys.modules.get("dexsim")
    manager_module = sys.modules.get("embodichain.lab.sim.sim_manager")
    if native is None and manager_module is None:
        return True
    try:
        manager = manager_module.SimulationManager if manager_module else None
        state = (
            int(native.get_world_num()) if native else 0,
            len(manager._instances) if manager else 0,
            manager._cleanup_queue.qsize() if manager else 0,
        )
    except Exception as error:
        print(f"[CI] cannot verify final native resources: {error!r}", file=sys.stderr)
        return False
    if state != (0, 0, 0):
        print(
            f"[CI] native World/manager/cleanup queue remain: {state}", file=sys.stderr
        )
        return False
    return True


def main(argv: Sequence[str] | None = None) -> NoReturn:
    """Return pytest's result after all hooks without unloading native libraries.

    Args:
        argv: Pytest arguments, or ``sys.argv[1:]`` when omitted. A leading
            ``--freeze-native-imports`` preloads static dependencies before
            domain/test imports and retains them until this process exits.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    freeze_imports = arguments[:1] == ["--freeze-native-imports"]
    if freeze_imports:
        plugin = _NativeImportGC(_NATIVE_IMPORTS)
        exit_code = pytest.main(arguments[1:], plugins=[plugin])
        status = (
            f"{plugin.freeze_count} static import objects"
            if plugin.enabled
            else plugin.disabled_reason
        )
        print(f"[CI] native import GC: {status}")
        if plugin.enabled and not _native_resources_released() and exit_code == 0:
            exit_code = 1
    else:
        exit_code = pytest.main(arguments)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))


if __name__ == "__main__":
    main()
