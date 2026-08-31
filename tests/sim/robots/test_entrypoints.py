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

from __future__ import annotations

from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest

import embodichain.lab.sim as sim_module
from embodichain.lab.sim.cfg import NewtonPhysicsCfg

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ROBOT_ENTRYPOINTS = (
    "embodichain/lab/sim/robots/cobotmagic.py",
    "embodichain/lab/sim/robots/franka_panda.py",
    "embodichain/lab/sim/robots/ur_robot.py",
    "embodichain/lab/sim/robots/dual_arm.py",
    "embodichain/lab/sim/robots/dexforce_w1/cfg.py",
)


@pytest.mark.parametrize("relative_path", _ROBOT_ENTRYPOINTS)
def test_robot_entrypoint_selects_newton_backend(
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    """Each robot smoke program must forward ``--physics newton``."""
    captured: dict[str, object] = {}

    class SimulationManagerCfgSpy:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class SimulationManagerSpy:
        def __init__(self, _cfg: SimulationManagerCfgSpy) -> None:
            pass

        def add_robot(self, *, cfg: object) -> object:
            return cfg

        def prepare(self) -> None:
            pass

        def update(self, *, step: int) -> None:
            pass

        def open_window(self) -> None:
            pass

        def destroy(self) -> None:
            pass

    monkeypatch.setattr(sim_module, "SimulationManagerCfg", SimulationManagerCfgSpy)
    monkeypatch.setattr(sim_module, "SimulationManager", SimulationManagerSpy)
    ipython_module = ModuleType("IPython")
    ipython_module.embed = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [relative_path, "--physics", "newton"],
    )

    original_path_entry = sys.path[0]
    try:
        runpy.run_path(_REPOSITORY_ROOT / relative_path, run_name="__main__")
    finally:
        sys.path[0] = original_path_entry

    assert isinstance(captured["physics_cfg"], NewtonPhysicsCfg)
