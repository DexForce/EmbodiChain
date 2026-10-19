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

import pytest
import torch

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
    VisualMaterialCfg,
)
from embodichain.lab.sim.cfg import RenderCfg, VisualObjectCfg, physics_cfg_for_backend
from embodichain.lab.sim.shapes import CubeCfg

pytestmark = [pytest.mark.requires_sim, pytest.mark.renderer]


@pytest.mark.parametrize("physics", ["default", "newton"])
def test_render_only_object_native_lifecycle(physics):
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=2,
            scene_node_capacity=128,
            physics_cfg=physics_cfg_for_backend(physics),
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    visual = None
    try:
        visual = sim.add_visual_object(
            VisualObjectCfg(
                uid="prop",
                shape=CubeCfg(size=(0.1, 0.2, 0.3)),
                init_pos=(0.0, 0.0, 1.0),
            )
        )
        sim.prepare()
        assert all(
            h.object_desc.physics is None and not h.object_desc.collisions
            for h in visual._entities
        )
        before = visual.get_local_pose().clone()
        sim.update()
        torch.testing.assert_close(visual.get_local_pose(), before)
        target = before[1:].clone()
        target[:, 0] += 0.3
        visual.set_local_pose(target, [1])
        visual.set_visible(False, [1])
        source_material = visual._entities[1].get_render_body().get_material(0)
        source_name = None if source_material is None else source_material.get_name()
        material = sim.create_visual_material(
            VisualMaterialCfg(uid="alternate", base_color=[0.8, 0.1, 0.2, 1.0])
        )
        visual.set_visual_material(material, [1])
        assigned_name = visual._entities[1].get_render_body().get_material(0).get_name()
        sim.sync_render_state()
        torch.testing.assert_close(visual.get_local_pose()[1:], target)
        torch.testing.assert_close(visual.get_local_pose()[:1], before[:1])
        assert visual.get_visibility().tolist() == [True, False]
        other = sim.add_visual_object(
            VisualObjectCfg(uid="other", shape=CubeCfg(size=(0.1, 0.1, 0.1)))
        )
        torch.testing.assert_close(visual.get_local_pose()[1:], target)
        assert visual.get_visibility().tolist() == [True, False]
        assert (
            visual._entities[1].get_render_body().get_material(0).get_name()
            == assigned_name
        )
        assert sim.remove_asset("other")
        del other
        torch.testing.assert_close(visual.get_local_pose()[1:], target)
        sim.reset_objects_state(torch.tensor([1], dtype=torch.int64))
        torch.testing.assert_close(visual.get_local_pose(), before)
        assert visual.get_visibility().all()
        restored = visual._entities[1].get_render_body().get_material(0)
        assert (None if restored is None else restored.get_name()) == source_name
        assert sim.get_rigid_object("prop") is None
        assert sim.remove_asset("prop")
        assert sim.get_asset("prop") is None
        assert not visual.is_spawn_bound
    finally:
        sim.destroy(exit_process=False)
        del visual
        SimulationManager.flush_cleanup_queue()
