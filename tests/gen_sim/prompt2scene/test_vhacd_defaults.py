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
from types import SimpleNamespace

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager import (
    manager as simulation_manager_module,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager.schemas import (
    GravityDropRequest,
)


def test_gravity_drop_request_defaults_to_vhacd() -> None:
    request = GravityDropRequest(glb_path=Path("object.glb"))

    assert request.convex_decomposition_method == "vhacd"


def test_prompt2scene_gravity_manager_passes_vhacd_to_rigid_object_cfg(
    tmp_path: Path,
    monkeypatch,
) -> None:
    glb_path = tmp_path / "object.glb"
    glb_path.write_bytes(b"glb")
    captured = {}

    class FakeRigidObject:
        def get_local_pose(self, to_matrix: bool):
            assert to_matrix is True
            return [SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: self))]

        def numpy(self):
            return np.eye(4)

    class FakeSim:
        def __init__(self, cfg):
            self.cfg = cfg

        def add_rigid_object(self, cfg):
            captured["rigid_object_cfg"] = cfg
            return FakeRigidObject()

        def update(self, step: int):
            captured["step"] = step

        def _deferred_destroy(self):
            captured["destroyed"] = True

    monkeypatch.setattr(
        simulation_manager_module,
        "_EmbodiSimManager",
        FakeSim,
    )
    monkeypatch.setattr(
        simulation_manager_module.SimulationManager,
        "_compute_adaptive_drop_height",
        lambda self, path: 0.2,
    )

    manager = simulation_manager_module.SimulationManager()
    manager.run_gravity_simulation(GravityDropRequest(glb_path=glb_path))

    cfg = captured["rigid_object_cfg"]
    assert cfg.max_convex_hull_num == 32
    assert cfg.convex_decomposition_method == "vhacd"
    assert captured["step"] == 300
    assert captured["destroyed"] is True
