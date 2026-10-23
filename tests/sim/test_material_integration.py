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

"""Native material parity and restore/rebuild qualification for DexSim 0.5.1rc1."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from embodichain.lab.sim import (
    SimulationManager,
    SimulationManagerCfg,
    VisualMaterialCfg,
)
from embodichain.lab.sim.cfg import (
    RenderCfg,
    RigidObjectCfg,
    VisualObjectCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.randomization.visual import (
    randomize_visual_material,
)

pytestmark = [pytest.mark.requires_sim, pytest.mark.renderer]


@pytest.mark.parametrize("physics", ["default", "newton"])
def test_native_material_creation_reset_and_topology_rebuild(physics):
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
    obj = other = runtime = None
    cfg = VisualMaterialCfg(
        base_color=[0.8, 0.1, 0.2, 1.0],
        metallic=0.3,
        roughness=0.2,
        emissive=[0.1, 0.2, 0.3],
        emissive_intensity=2.0,
        ior=1.7,
    )
    try:
        obj = sim.add_rigid_object(
            RigidObjectCfg(
                uid="red",
                body_type="static",
                init_pos=(0.0, 0.0, 0.5),
                shape=CubeCfg(size=(0.1, 0.1, 0.1), visual_material=cfg),
            )
        )
        other = sim.add_visual_object(
            VisualObjectCfg(
                uid="blue",
                shape=CubeCfg(
                    size=(0.1, 0.1, 0.1),
                    visual_material=VisualMaterialCfg(base_color=[0.1, 0.2, 0.8, 1.0]),
                ),
            )
        )
        sim.prepare()
        initial_names = [m.mat.get_name() for m in obj.get_visual_material_inst()]
        assert obj.get_visual_material_inst()[0].base_color == pytest.approx(
            cfg.base_color[:3], abs=1e-5
        )
        assert other.get_visual_material_inst()[0].base_color == pytest.approx(
            [0.1, 0.2, 0.8], abs=1e-5
        )

        runtime = sim.create_visual_material(cfg.replace(uid="runtime"))
        shared_visual = sim.add_visual_object(
            VisualObjectCfg(
                uid="runtime_copy",
                shape=CubeCfg(
                    size=(0.1, 0.1, 0.1), visual_material=cfg.replace(uid="runtime")
                ),
            )
        )
        assert shared_visual.get_visual_material_inst()[0].base_color == pytest.approx(
            cfg.base_color[:3], abs=1e-5
        )
        assert sim.remove_asset("runtime_copy")
        del shared_visual
        borrowed_cfg = VisualMaterialCfg(uid="borrowed_state", roughness=0.2)
        borrowed_visual = sim.add_visual_object(
            VisualObjectCfg(
                uid="borrowed_visual",
                shape=CubeCfg(size=(0.1, 0.1, 0.1), visual_material=borrowed_cfg),
            )
        )
        native_template = sim.get_env().find_material("borrowed_state")
        native_template.get_inst().set_roughness(0.8)
        borrowed = sim.create_visual_material(borrowed_cfg)
        assert borrowed.inst.roughness == pytest.approx(0.8)
        assert native_template.get_inst().get_roughness() == pytest.approx(0.8)
        assert sim.remove_asset("borrowed_visual")
        del borrowed, borrowed_visual, native_template
        parsed = obj.get_visual_material_inst()[0]
        assert parsed.roughness == pytest.approx(runtime.inst.roughness)
        assert parsed.metallic == pytest.approx(runtime.inst.metallic)
        assert parsed.ior == pytest.approx(runtime.inst.ior)
        np.testing.assert_allclose(
            parsed.mat.get_emissive(), runtime.inst.mat.get_emissive(), atol=1e-5
        )
        del parsed

        obj.set_visual_material(runtime, [1])
        selected = obj.get_visual_material_inst([1])[0]
        selected.set_roughness(0.6)
        assert selected.mat.get_roughness() == pytest.approx(0.6)
        assert runtime.get_instance(selected.uid) is selected
        del selected
        obj.restore_visual_material([1])

        # Adding/removing a different asset rebuilds Newton's native render resources.
        extra = sim.add_visual_object(
            VisualObjectCfg(uid="extra", shape=CubeCfg(size=(0.1, 0.1, 0.1)))
        )
        assert [
            m.mat.get_name() for m in obj.get_visual_material_inst()
        ] == initial_names
        assert [
            entity.get_render_body().get_material(0).get_name()
            for entity in obj._entities
        ] == initial_names
        assert sim.remove_asset("extra")
        del extra
        sim.prepare()
        assert [
            entity.get_render_body().get_material(0).get_name()
            for entity in obj._entities
        ] == initial_names

        # A current, intentional replacement must also survive a later rebuild.
        obj.set_visual_material(runtime, [1])
        selected_name = obj.get_visual_material_inst([1])[0].mat.get_name()
        assert sim.remove_asset("blue")
        sim.prepare()
        assert (
            obj._entities[0].get_render_body().get_material(0).get_name()
            == initial_names[0]
        )
        assert (
            obj._entities[1].get_render_body().get_material(0).get_name()
            == selected_name
        )
        obj.reset([1])
        assert (
            obj._entities[1].get_render_body().get_material(0).get_name()
            == initial_names[1]
        )

        # Repeated partial randomization retains native instance capacity and the other row.
        event_cfg = FunctorCfg(func=randomize_visual_material)
        event_cfg.params = {
            "entity_cfg": SceneEntityCfg(uid="red"),
            "p_solid": 1.0,
            "solid_texture_count": 2,
            "metallic_range": (0.9, 0.9),
            "roughness_range": (0.2, 0.2),
            "ior_range": (2.0, 2.0),
        }
        env = SimpleNamespace(sim=sim, num_envs=2, device=torch.device("cpu"))
        randomizer = randomize_visual_material(event_cfg, env)
        selected_ids = torch.tensor([1])
        randomizer(env, selected_ids, **event_cfg.params)
        template_counts = [
            segments[0].working_inst.mat.get_template().inst_count()
            for segments in randomizer._reuse_state
        ]
        for _ in range(1025):
            randomizer(env, selected_ids, **event_cfg.params)
        assert [
            segments[0].working_inst.mat.get_template().inst_count()
            for segments in randomizer._reuse_state
        ] == template_counts
        assert obj.get_visual_material_inst([1])[0].metallic == pytest.approx(0.9)
        assert obj.get_visual_material_inst([1])[0].roughness == pytest.approx(0.2)
        assert obj.get_visual_material_inst([1])[0].ior == pytest.approx(2.0)
        assert (
            obj._entities[0].get_render_body().get_material(0).get_name()
            == initial_names[0]
        )
        del randomizer, env

        replacement_cfg = RigidObjectCfg(
            uid="red",
            body_type="static",
            init_pos=(0.0, 0.0, 0.5),
            shape=CubeCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=VisualMaterialCfg(base_color=[0.1, 0.2, 0.8, 1.0]),
            ),
        )
        obj = sim.replace_rigid_object(replacement_cfg)
        assert obj.get_visual_material_inst()[0].base_color == pytest.approx(
            [0.1, 0.2, 0.8], abs=1e-5
        )
        definition_count = len(sim._spawn_scene.builder.materials)
        obj = sim.replace_rigid_object(replacement_cfg.copy())
        assert len(sim._spawn_scene.builder.materials) == definition_count
        assert sim.remove_asset("red")
        obj = sim.add_rigid_object(
            replacement_cfg.replace(
                shape=CubeCfg(
                    size=(0.1, 0.1, 0.1),
                    visual_material=VisualMaterialCfg(base_color=[0.1, 0.8, 0.2, 1.0]),
                )
            )
        )
        assert obj.get_visual_material_inst()[0].base_color == pytest.approx(
            [0.1, 0.8, 0.2], abs=1e-5
        )
    finally:
        sim.destroy(exit_process=False)
        del obj, other, runtime
        SimulationManager.flush_cleanup_queue()


@pytest.mark.parametrize("physics", ["default", "newton"])
def test_native_preset_identity_reset_and_rebuild(physics):
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
    cfg = VisualMaterialCfg.from_preset(
        "BrickRed", uid="brick", ior=1.6, roughness=0.25
    )
    obj = runtime = other = child = extra = selected = alias = None
    try:
        # Inline first, then runtime creation borrows the exact current handle.
        obj = sim.add_rigid_object(
            RigidObjectCfg(
                uid="brick_shape",
                body_type="static",
                init_pos=(0, 0, 0.5),
                shape=CubeCfg(size=(0.1, 0.1, 0.1), visual_material=cfg),
            )
        )
        sim.prepare()
        original_ids = [m.mat.get_material_id() for m in obj.get_visual_material_inst()]
        assert obj.get_visual_material_inst()[0].ior == pytest.approx(1.6)
        obj.get_visual_material_inst()[0].set_roughness(0.55)
        runtime = sim.create_visual_material(cfg)
        assert runtime.inst.mat.get_material_id() == original_ids[0]
        assert runtime.inst.roughness == pytest.approx(0.55)
        # Equivalent UIDs share immutable native templates, with separate children.
        other = sim.create_visual_material(cfg.replace(uid="other_brick"))
        assert other.inst.mat.get_material_id() != runtime.inst.mat.get_material_id()
        assert other.inst.roughness == pytest.approx(0.25)
        assert runtime.mat.get_inst().get_roughness() == pytest.approx(0.25)
        assert (
            other.mat.get_inst().get_material_id()
            == runtime.mat.get_inst().get_material_id()
        )
        child = runtime.create_instance("other_brick")
        assert child.mat.get_material_id() != other.inst.mat.get_material_id()
        child.set_ior(2.0)
        child.mat.set_transmission(0.75)
        assert child.mat.get_transmission() == pytest.approx(0.75)
        runtime.set_default_properties(child, cfg)
        assert child.ior == pytest.approx(1.6)
        # The authored legacy template leaves transmission unset (native None).
        assert child.mat.get_transmission() is None
        count = runtime.mat.inst_count()
        for _ in range(1025):
            assert runtime.create_instance("other_brick") is child
            assert sim.create_visual_material(cfg) is runtime
        assert runtime.mat.inst_count() == count
        with pytest.raises(ValueError, match="Conflicting visual material"):
            sim.create_visual_material(cfg.replace(ior=1.7))
        # Runtime first, then a Spawn declaration binds that exact child.
        alias = sim.add_visual_object(
            VisualObjectCfg(
                uid="runtime_alias",
                shape=CubeCfg(
                    size=(0.1, 0.1, 0.1), visual_material=cfg.replace(uid="other_brick")
                ),
            )
        )
        assert (
            alias.get_visual_material_inst()[0].mat.get_material_id()
            == other.inst.mat.get_material_id()
        )
        obj.set_visual_material(other, [1])
        selected = obj.get_visual_material_inst([1])[0]
        assert (
            other.get_instance(selected.uid).mat.get_material_id()
            == selected.mat.get_material_id()
        )
        selected_id = selected.mat.get_material_id()
        extra = sim.add_visual_object(
            VisualObjectCfg(uid="extra_preset", shape=CubeCfg(size=(0.1, 0.1, 0.1)))
        )
        assert obj.get_visual_material_inst([1])[0].mat.get_material_id() == selected_id
        obj.reset([1])
        assert [
            m.mat.get_material_id() for m in obj.get_visual_material_inst()
        ] == original_ids
        assert sim.remove_asset("extra_preset")
        sim.prepare()
        assert [
            m.mat.get_material_id() for m in obj.get_visual_material_inst()
        ] == original_ids
        # Native cleanup preserves reserved definitions and live resource state.
        sim.clean_materials()
        runtime = sim.create_visual_material(cfg)
        assert runtime.inst.roughness == pytest.approx(0.55)
        assert runtime.inst.mat.get_material_id() == original_ids[0]
    finally:
        sim.destroy(exit_process=False)
        del obj, runtime, other, child, extra, selected, alias
        SimulationManager.flush_cleanup_queue()
