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

import numpy as np
import pytest

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.visualization.markers import MarkerGroupCfg, MarkerPrototypeCfg

pytestmark = [pytest.mark.gpu, pytest.mark.requires_sim]


def _require_render_actor_capability():
    from dexsim import engine, spawn

    required = (
        (spawn, "create_render_actor"),
        (getattr(engine, "RenderBody", None), "set_raytrace_visible"),
        (getattr(engine, "RenderBody", None), "set_pickable"),
        (getattr(engine, "RenderBody", None), "build"),
        (getattr(engine, "MaterialInst", None), "set_unlit"),
        (getattr(engine, "MaterialInst", None), "set_alpha_mode"),
        (getattr(engine, "AlphaMode", None), "BLEND"),
    )
    if any(not hasattr(owner, name) for owner, name in required):
        pytest.skip("Requires DexSim generic render actor and material capabilities")


def test_marker_group_native_rendering_and_sensor_isolation():
    _require_render_actor_capability()
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(renderer="hybrid"),
            startup_summary="off",
        )
    )
    try:
        _exercise_marker_rendering(sim)
    finally:
        sim.destroy(exit_process=False)
        del sim
        SimulationManager.flush_cleanup_queue()


def _exercise_marker_rendering(sim):
    import dexsim

    arena = sim.get_env()
    world = sim.get_world()
    camera_group = world.create_camera_group([96, 96], 1, True)
    camera = arena.create_camera(
        "markers_contract", 96, 96, True, dexsim.render.ViewFlags.ALL, camera_group
    )
    camera.open_camera()
    camera.look_at([0, 0, 3], [0, 0, 1], [0, 1, 0])

    def render():
        world._flush_render_updates()
        camera.render()
        return np.array(camera.get_rgb_map()[..., :3], copy=True)

    baseline = render()
    count = arena.get_actor_num()
    revision = sim._prepared_spawn_topology_revision
    step = sim._visualization_sim_step
    group = sim.add_marker_group(
        MarkerGroupCfg(
            name="native",
            prototypes={
                "box": MarkerPrototypeCfg(
                    shape="box", scale=(0.5, 0.5, 0.5), color=(1, 0, 0, 1)
                ),
                "sphere": MarkerPrototypeCfg(
                    shape="sphere", scale=(0.5, 0.5, 0.5), color=(0, 1, 0, 0.5)
                ),
            },
        )
    )
    group.update(translations=[[0, 0, 1]])
    # The lit scene has temporal RT/DLSS noise. Detect the marker's saturated
    # color instead of incorrectly requiring identical consecutive beauty frames.
    sensor = render().astype(float)
    background = baseline.astype(float)
    red_mask = lambda image: (image[..., 0] > image[..., 1] + 50)
    assert (
        np.count_nonzero(red_mask(sensor)) <= np.count_nonzero(red_mask(background)) + 2
    )
    camera_group.set_visual_overlay_enabled(True)
    red = render()
    assert (
        np.count_nonzero(red[..., 0].astype(float) > red[..., 1].astype(float) + 50)
        > 20
    )
    group.update(prototype_indices=[1])
    green = render()
    assert (
        np.count_nonzero(green[..., 1].astype(float) > green[..., 0].astype(float) + 20)
        > 20
    )
    for offset in (0.1, 0.2, 0.3):
        group.update(translations=[[offset, 0, 1]])
        assert arena.get_actor_num() == count + 1
    group.clear()
    assert arena.get_actor_num() == count
    assert sim._prepared_spawn_topology_revision == revision
    assert sim._visualization_sim_step == step
    group.remove()
    assert sim.get_marker_overlays() == ()


def test_environment_batched_native_attachment_follows_bound_articulation(tmp_path):
    from dataclasses import fields
    from dexsim.spawn import RigidBodyPhysicsDesc
    from embodichain.lab.sim.cfg import ArticulationCfg

    if "com_quaternion" not in {field.name for field in fields(RigidBodyPhysicsDesc)}:
        pytest.skip(
            "Registered-asset smoke requires compatible DexSim COM descriptors; installed schema lacks com_quaternion"
        )

    _require_render_actor_capability()
    source = tmp_path / "marker_parent.urdf"
    source.write_text("""<robot name="marker_parent">
      <link name="base"><inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
      <link name="tool"><inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
      <joint name="joint" type="revolute"><parent link="base"/><child link="tool"/><origin xyz="0 0 1"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="10" velocity="1"/></joint>
    </robot>""")
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            num_envs=2,
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(renderer="hybrid"),
            startup_summary="off",
        )
    )
    try:
        target = sim.add_articulation(
            ArticulationCfg(uid="parent", fpath=str(source), init_pos=(0.0, 0.0, 2.0))
        )
        group = sim.add_marker_group(
            MarkerGroupCfg(
                name="batched",
                prototypes={"box": MarkerPrototypeCfg(scale=(0.1, 0.1, 0.1))},
            )
        )
        group.update(translations=[[[1, 0, 0]], [[2, 0, 0]]])
        with pytest.raises(RuntimeError, match="prepared"):
            group.attach("parent")
        sim.prepare()
        group.attach("parent", env_ids=[0])
        group.attach("parent", link_name="tool", env_ids=[1])
        origins = sim.arena_offsets.cpu().numpy()
        np.testing.assert_allclose(
            group.snapshot()[0].position, origins[0] + [1, 0, 2], atol=1e-5
        )
        np.testing.assert_allclose(
            group.snapshot()[1].position, origins[1] + [2, 0, 3], atol=1e-5
        )
        count = sim.get_env().get_actor_num()
        step = sim._visualization_sim_step
        # Selected environment growth must leave the first environment untouched.
        group.update(env_ids=[1], translations=[[3, 0, 0], [4, 0, 0]])
        assert group.counts == (1, 2)
        assert sim.get_env().get_actor_num() == count + 1
        group.set_visibility(False, env_ids=[0])
        assert [m.visible for m in group.snapshot()] == [False, True, True]
        group.detach(env_ids=[1])
        np.testing.assert_allclose(
            group.snapshot()[1].position, origins[1] + [3, 0, 3], atol=1e-5
        )
        sim.sync_render_state()
        assert sim._visualization_sim_step == step
        assert sim.remove_asset("parent")
        sim.sync_render_state()
        np.testing.assert_allclose(
            group.snapshot()[0].position, origins[0] + [1, 0, 2], atol=1e-5
        )
        group.clear(env_ids=[1])
        assert group.counts == (1, 0)
    finally:
        sim.destroy(exit_process=False)
        del sim
        SimulationManager.flush_cleanup_queue()


def test_native_environment_batches_selected_mutations_without_preparation():
    _require_render_actor_capability()
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            num_envs=2,
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(renderer="hybrid"),
            startup_summary="off",
        )
    )
    try:
        arena = sim.get_env()
        count = arena.get_actor_num()
        revision = sim._prepared_spawn_topology_revision
        group = sim.add_marker_group(
            MarkerGroupCfg(
                name="native_batches",
                prototypes={"box": MarkerPrototypeCfg(scale=(0.1, 0.1, 0.1))},
            )
        )
        group.update(
            translations=[[[1, 0, 1]], [[2, 0, 1]]],
            colors=[[[1, 0, 0, 0.5]], [[0, 1, 0, 1]]],
        )
        assert arena.get_actor_num() == count + 2
        origins = sim.arena_offsets.cpu().numpy()
        np.testing.assert_allclose(group.snapshot()[0].position, origins[0] + [1, 0, 1])
        np.testing.assert_allclose(group.snapshot()[1].position, origins[1] + [2, 0, 1])
        group.update(env_ids=[1], translations=[[3, 0, 1], [4, 0, 1]])
        assert group.counts == (1, 2)
        assert arena.get_actor_num() == count + 3
        group.set_visibility(False, env_ids=[0])
        assert [m.visible for m in group.snapshot()] == [False, True, True]
        group.clear(env_ids=[1])
        assert arena.get_actor_num() == count + 1
        assert group.counts == (1, 0)
        assert sim._prepared_spawn_topology_revision == revision
        assert sim._visualization_sim_step == 0
        group.remove()
        assert arena.get_actor_num() == count
    finally:
        sim.destroy(exit_process=False)
        del sim
        SimulationManager.flush_cleanup_queue()
