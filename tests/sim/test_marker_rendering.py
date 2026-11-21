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

import os

import numpy as np
import pytest

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.visualization.markers import MarkerGroupCfg, MarkerPrototypeCfg

pytestmark = [pytest.mark.gpu, pytest.mark.requires_sim]

_WINDOW_MODES = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            os.environ.get("SDL_VIDEODRIVER") == "dummy",
            reason="SDL's dummy video driver cannot open a native window.",
        ),
    ),
]


def _require_render_actor_capability():
    from dexsim import engine, spawn
    from dexsim.scene import Scene

    required = (
        (Scene, "add_mesh_object"),
        (Scene, "remove_mesh_object"),
        (getattr(spawn, "RenderDesc", None), "render_mode"),
        (getattr(spawn, "MaterialDesc", None), "unlit"),
        (getattr(spawn, "MaterialDesc", None), "alpha_mode"),
        (getattr(spawn, "MaterialDesc", None), "depth_write"),
        (getattr(engine, "RenderBody", None), "set_raytrace_visible"),
        (getattr(engine, "RenderBody", None), "set_pickable"),
        (getattr(engine, "RenderBody", None), "build"),
        (getattr(engine, "MaterialInst", None), "set_unlit"),
        (getattr(engine, "MaterialInst", None), "set_alpha_mode"),
        (getattr(engine, "AlphaMode", None), "BLEND"),
    )
    if any(not hasattr(owner, name) for owner, name in required):
        pytest.skip("Requires DexSim Scene mesh lifecycle and overlay capabilities")


def test_native_marker_removal_preserves_retained_material():
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
        # Keep borrowed native references inside the helper's lifetime, before
        # the World is destroyed.
        _exercise_repeated_marker_lifecycle(sim)
    finally:
        sim.destroy(exit_process=False)
        del sim
        SimulationManager.flush_cleanup_queue()


def _exercise_repeated_marker_lifecycle(sim):
    scene = sim._spawn_scene.builder.result
    arena = sim.get_env()
    retained_material = None
    try:
        baseline_paths = set(scene.handles)
        baseline_count = arena.get_actor_num()
        revision = sim._prepared_spawn_topology_revision
        group = sim.add_marker_group(
            MarkerGroupCfg(
                name="material_collection",
                prototypes={
                    "box": MarkerPrototypeCfg(shape="box"),
                    "sphere": MarkerPrototypeCfg(shape="sphere"),
                },
            )
        )
        for _ in range(3):
            group.update(translations=[[0, 0, 1], [1, 0, 1]], prototype_indices=[0, 0])
            handles = {
                path: scene.handles[path]
                for path in set(scene.handles) - baseline_paths
            }
            assert len(handles) == 2
            assert arena.get_actor_num() == baseline_count + 2
            retained_material = next(iter(handles.values())).get_material()
            color = [0.25, 0.5, 0.75, 0.5]
            group.update(
                translations=[[0, 1, 1], [1, 1, 1]],
                scales=[[2, 2, 2], [2, 2, 2]],
                colors=[color, color],
            )
            group.set_visibility(False)
            assert set(scene.handles) == baseline_paths | handles.keys()
            for path, handle in handles.items():
                assert scene.handles[path] is handle
                assert handle.is_valid
                assert not handle.is_visible()
                np.testing.assert_allclose(handle.get_world_pose()[:3, 3][1:], [1, 1])
                np.testing.assert_allclose(handle.get_scale(), [2, 2, 2])
            np.testing.assert_allclose(retained_material.get_base_color(), color[:3])

            group.update(prototype_indices=[1, 1])
            replacements = [
                scene.handles[path] for path in set(scene.handles) - baseline_paths
            ]
            assert len(replacements) == 2
            assert arena.get_actor_num() == baseline_count + 2
            assert all(
                not handle.is_valid and handle.native() is None
                for handle in handles.values()
            )
            assert all(handle.is_valid for handle in replacements)
            assert set(scene.handles).isdisjoint(handles)

            group.clear()
            assert arena.get_actor_num() == baseline_count
            assert set(scene.handles) == baseline_paths
            assert all(
                not handle.is_valid and handle.native() is None
                for handle in replacements
            )
            # A separately retained material remains usable after marker removal.
            retained_material.set_base_color([1, 0, 0, 1])
            np.testing.assert_allclose(retained_material.get_base_color(), [1, 0, 0])
            del retained_material
            group.clear()
            assert arena.get_actor_num() == baseline_count
        group.remove()
        group.remove()
        assert sim.get_marker_overlays() == ()
        assert sim._prepared_spawn_topology_revision == revision
        assert sim._visualization_sim_step == 0

    finally:
        # Pytest retains failed frames; drop native wrappers before World teardown.
        retained_material = None
        arena = None


@pytest.mark.parametrize("window", _WINDOW_MODES)
def test_marker_group_native_rendering_and_sensor_isolation(window: bool) -> None:
    _require_render_actor_capability()
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            width=320,
            height=240,
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(renderer="hybrid"),
            startup_summary="off",
        )
    )
    try:
        if window:
            assert sim.open_window()
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

    sim.prepare()

    def render():
        if sim.is_window_opened:
            with sim.render_frame():
                sim.render_camera_group([camera_group.get_group_id()])
        else:
            sim.sync_render_state()
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
    group.update(translations=[[2, 0, 1]])
    moved = render().astype(float)
    assert np.count_nonzero(moved[..., 1] > moved[..., 0] + 20) < 5
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
    from embodichain.lab.sim.cfg import ArticulationCfg

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


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("window", _WINDOW_MODES)
def test_robot_markers_textures_opacity_and_material_switch(
    tmp_path, request: pytest.FixtureRequest, stereo: bool, window: bool
) -> None:
    from PIL import Image
    import trimesh
    from embodichain.lab.sim.cfg import DenoisingCfg

    pixels = np.full((8, 8, 4), [255, 0, 0, 255], dtype=np.uint8)
    pixels[:, 4:] = [0, 0, 255, 255]
    pixels[4:, 4:] = [255, 255, 255, 255]
    mesh = trimesh.Trimesh(
        vertices=[[-0.6, -0.6, 0], [0.6, -0.6, 0], [0.6, 0.6, 0], [-0.6, 0.6, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(pixels),
                baseColorFactor=[255] * 4,
                metallicFactor=0.0,
                roughnessFactor=0.5,
            ),
        ),
    )
    mesh.export(tmp_path / "mesh.glb")
    path = tmp_path / "ghost.urdf"
    path.write_text(
        '<robot name="texture"><link name="base"><visual>'
        '<geometry><mesh filename="mesh.glb"/></geometry>'
        "</visual></link></robot>"
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            width=320,
            height=240,
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(
                renderer=request.config.getoption("--renderer"),
                denoising=DenoisingCfg(window="off", offscreen="off"),
            ),
            startup_summary="off",
        )
    )
    try:
        if window:
            assert sim.open_window()
        _exercise_textured_ghost(sim, path, stereo)
    finally:
        sim.destroy(exit_process=False)
        del sim
        SimulationManager.flush_cleanup_queue()


def _exercise_textured_ghost(sim, path, stereo):
    from embodichain.lab.sim.sensors import CameraCfg, StereoCameraCfg
    from embodichain.lab.visualization.markers import (
        RobotMarkersAppearanceCfg,
        RobotMarkers,
        RobotMarkersCfg,
    )

    # OfflineRT's lit marker preview is gray with either capture route in
    # 0.6.0rc1. Unlit markers provide a color oracle for its synchronization.
    unlit = sim.sim_config.render_cfg.renderer == "rt"
    sim.prepare()
    sim.set_ground_plane_visibility(False)
    scene = sim._spawn_scene.builder.result

    def make_camera(uid, include_overlays):
        intrinsics = (105.0, 105.0, 64.0, 64.0)
        config_type = StereoCameraCfg if stereo else CameraCfg
        return sim.add_sensor(
            config_type(
                uid=uid,
                width=128,
                height=128,
                intrinsics=intrinsics,
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=(0, 0, 3), target=(0, 0, 1), up=(0, 1, 0)
                ),
                include_visual_overlays=include_overlays,
                **({"intrinsics_right": intrinsics} if stereo else {}),
            )
        )

    sensor = make_camera("sensor", False)
    preview = make_camera("preview", True)

    def render(camera):
        with sim.render_frame():
            sim.render_camera_group([camera.group_id])
            camera.update(fetch_only=True)
        data = camera.get_data()
        keys = ("color", "color_right") if stereo else ("color",)
        return np.stack([data[key][0, ..., :3].cpu().numpy() for key in keys]).astype(
            float
        )

    def colored(image, channel):
        others = [i for i in range(3) if i != channel]
        return np.count_nonzero(
            (image[..., channel] > 100)
            & (image[..., channel] > image[..., others].max(-1) + 40),
            axis=(1, 2),
        )

    baseline = render(sensor)
    before = set(scene.handles)
    ghost = RobotMarkers(
        RobotMarkersCfg(
            name="texture",
            asset_path=str(path),
            appearance=RobotMarkersAppearanceCfg(unlit=unlit, mode="asset", opacity=1),
        ),
        sim,
    )
    poses = np.eye(4)[None, None]
    poses[..., 2, 3] = 1
    ghost.visualize_link_poses(poses)
    assert np.all(colored(render(sensor), 0) <= colored(baseline, 0) + 2)
    opaque = render(preview)
    assert np.all(colored(opaque, 0) > 100) and np.all(colored(opaque, 2) > 100)
    for ground_visible in (False, True):
        sensor.set_ground_plane_visibility(ground_visible)
        preview.set_ground_plane_visibility(ground_visible)
        assert np.all(colored(render(sensor), 0) <= colored(baseline, 0) + 2)
        visible = render(preview)
        assert np.all(colored(visible, 0) > 100) and np.all(colored(visible, 2) > 100)
    ghost.set_appearance(
        RobotMarkersAppearanceCfg(
            unlit=unlit, mode="override", color=(0, 1, 0), opacity=1
        )
    )
    opaque_green = render(preview)
    assert np.all(colored(opaque_green, 1) > 100)
    assert np.all(colored(opaque_green, 0) == 0) and np.all(
        colored(opaque_green, 2) == 0
    )
    ghost.set_appearance(
        RobotMarkersAppearanceCfg(unlit=unlit, mode="asset", opacity=1)
    )
    restored = render(preview)
    assert np.all(colored(restored, 0) > 100) and np.all(colored(restored, 2) > 100)
    ghost.set_appearance(
        RobotMarkersAppearanceCfg(unlit=unlit, mode="asset", opacity=0)
    )
    invisible = render(preview)
    assert np.all(colored(invisible, 0) <= colored(baseline, 0) + 2)
    assert np.all(colored(invisible, 2) <= colored(baseline, 2) + 2)
    ghost.set_appearance(
        RobotMarkersAppearanceCfg(
            unlit=unlit, mode="override", color=(0, 1, 0), opacity=0.6
        )
    )
    green = render(preview)
    assert np.all(colored(green, 1) > 100) and np.all(colored(green, 0) == 0)
    ghost.set_appearance(
        RobotMarkersAppearanceCfg(unlit=unlit, mode="asset", opacity=0.6)
    )
    translucent = render(preview)
    assert np.all(colored(translucent, 0) > 100) and np.all(
        colored(translucent, 2) > 100
    )
    assert np.all(colored(render(sensor), 0) <= colored(baseline, 0) + 2)
    ghost.set_appearance(
        RobotMarkersAppearanceCfg(
            unlit=unlit, mode="override", color=(1, 0, 0), opacity=1
        )
    )
    centers = []
    for x in (-0.3, 0.3):
        poses[..., 0, 3] = x
        ghost.visualize_link_poses(poses)
        sim.update(step=1)
        frame = render(preview)
        red = (frame[..., 0] > 100) & (frame[..., 0] > frame[..., 1:].max(-1) + 40)
        assert np.all(red.sum(axis=(1, 2)) > 100)
        centers.append([np.nonzero(eye)[1].mean() for eye in red])
    # A 0.6 m lateral translation at 2 m depth with fx=105 moves 31.5 pixels.
    np.testing.assert_allclose(np.subtract(centers[1], centers[0]), 31.5, atol=3)
    ghost.remove()
    assert set(scene.handles) == before
    removed = render(preview)
    assert np.all(colored(removed, 0) <= colored(baseline, 0) + 2)
