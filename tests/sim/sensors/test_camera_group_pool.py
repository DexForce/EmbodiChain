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

"""Native camera-group sharing across compatible startup sensors."""

from __future__ import annotations

import dexsim
import pytest

from dexsim.types import Renderer

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.sim.cfg import VisualObjectCfg
from embodichain.lab.sim.sensors import CameraCfg, StereoCameraCfg
from embodichain.lab.sim.sensors._camera_group_pool import _CameraGroupPool
from embodichain.lab.sim.shapes import CubeCfg


def _simulation() -> SimulationManager:
    """Create two small offscreen Arenas on the selected native renderer."""
    return SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=2,
            render_cfg=RenderCfg(renderer="fast-rt"),
            startup_summary="off",
        )
    )


@pytest.mark.no_sim
def test_group_plan_requires_matching_resolution_outputs_and_overlays() -> None:
    """Native flags alone cannot prove two sensors have identical outputs."""
    pool = _CameraGroupPool(2)
    pool.plan(
        [
            CameraCfg(uid="rgb", width=32, height=32, enable_color=True),
            CameraCfg(uid="no_rgb", width=32, height=32, enable_color=False),
            CameraCfg(uid="rgb_depth", width=32, height=32, enable_depth=True),
            CameraCfg(uid="overlay", width=32, height=32, include_visual_overlays=True),
            CameraCfg(uid="larger", width=64, height=32),
        ]
    )
    assert len(pool._by_key) == 5


@pytest.mark.gpu
def test_compatible_startup_cameras_share_slots_and_survive_removal() -> None:
    """Only the planned RGB pair shares a target and surviving slots stay put."""
    sim = _simulation()
    try:
        first_cfg = CameraCfg(uid="first", width=32, height=32)
        second_cfg = CameraCfg(uid="second", width=32, height=32)
        depth_cfg = CameraCfg(uid="depth", width=32, height=32, enable_depth=True)
        sim._plan_camera_groups([first_cfg, second_cfg, depth_cfg])
        first = sim.add_sensor(first_cfg)
        second = sim.add_sensor(second_cfg)
        depth = sim.add_sensor(depth_cfg)
        sim.prepare()

        assert first.group_ids == second.group_ids
        assert depth.group_ids != first.group_ids
        group = first._camera_groups[0].native
        assert all(group.get_camera(i) is first._entities[i] for i in range(2))
        assert all(group.get_camera(i + 2) is second._entities[i] for i in range(2))
        assert len(set((*first.group_ids, *second.group_ids, *depth.group_ids))) == 2

        second.update()
        first.update(fetch_only=True)
        assert first.get_capture_metadata() is not None
        assert second.get_capture_metadata() is not None
        assert first.get_data()["color"].shape == (2, 32, 32, 4)
        assert second.get_data()["color"].shape == (2, 32, 32, 4)

        dynamic = sim.add_sensor(CameraCfg(uid="dynamic", width=32, height=32))
        assert dynamic.group_ids != first.group_ids
        sim.remove_asset("first")
        assert all(group.get_camera(i) is None for i in range(2))
        assert all(group.get_camera(i + 2) is second._entities[i] for i in range(2))
        second.update()
        assert second.get_data()["color"].shape == (2, 32, 32, 4)
        readded = sim.add_sensor(CameraCfg(uid="first", width=32, height=32))
        assert readded.group_ids != second.group_ids
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
def test_stereo_and_mono_share_one_compatible_target() -> None:
    """The eye-major stereo batch keeps both logical two-row observations."""
    sim = _simulation()
    try:
        mono_cfg = CameraCfg(uid="mono", width=32, height=32)
        stereo_cfg = StereoCameraCfg(uid="stereo", width=32, height=32)
        sim._plan_camera_groups([mono_cfg, stereo_cfg])
        mono = sim.add_sensor(mono_cfg)
        stereo = sim.add_sensor(stereo_cfg)
        sim.prepare()

        assert mono.group_ids == stereo.group_ids
        group = mono._camera_groups[0].native
        assert all(group.get_camera(i) is mono._entities[i] for i in range(2))
        assert all(
            group.get_camera(i + 2) is stereo._entities[i]._left_view for i in range(2)
        )
        assert all(
            group.get_camera(i + 4) is stereo._entities[i]._right_view for i in range(2)
        )
        sim.render_camera_group(list(mono.group_ids))
        mono.update(fetch_only=True)
        stereo.update(fetch_only=True)
        assert stereo.get_data()["color"].shape == (2, 32, 32, 4)
        assert stereo.get_data()["color_right"].shape == (2, 32, 32, 4)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
def test_shared_target_keeps_per_camera_ground_visibility() -> None:
    """Sharing a target does not merge the two cameras' ground policies."""
    sim = _simulation()
    try:
        sim.add_visual_object(
            VisualObjectCfg(
                uid="under_floor",
                shape=CubeCfg(size=(1, 1, 1)),
                init_pos=(0, 0, -2),
            )
        )
        extrinsics = CameraCfg.ExtrinsicsCfg(
            eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0)
        )
        hidden_cfg = CameraCfg(
            uid="hidden",
            width=32,
            height=32,
            intrinsics=(28, 28, 16, 16),
            enable_depth=True,
            ground_plane_visible=False,
            extrinsics=extrinsics,
        )
        shown_cfg = CameraCfg(
            uid="shown",
            width=32,
            height=32,
            intrinsics=(28, 28, 16, 16),
            enable_depth=True,
            extrinsics=extrinsics,
        )
        sim._plan_camera_groups([hidden_cfg, shown_cfg])
        hidden = sim.add_sensor(hidden_cfg)
        shown = sim.add_sensor(shown_cfg)
        sim.prepare()
        sim.enable_physics(False)

        assert hidden.group_ids == shown.group_ids
        sim.render_camera_group(list(hidden.group_ids))
        hidden.update(fetch_only=True)
        shown.update(fetch_only=True)
        assert hidden.get_capture_metadata()["ground_plane_visible"] == (False, False)
        assert shown.get_capture_metadata()["ground_plane_visible"] == (True, True)
        assert (
            hidden.get_data()["depth"][:, 16, 16] > shown.get_data()["depth"][:, 16, 16]
        ).all()
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
def test_hybrid_packs_across_sensor_boundary() -> None:
    """A second camera spans a shared 32-view target and one tail target."""
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=17,
            render_cfg=RenderCfg(renderer="hybrid"),
            startup_summary="off",
        )
    )
    try:
        if dexsim.get_world_config().renderer != Renderer.HYBRID:
            pytest.skip("The process selected another native renderer.")
        first_cfg = CameraCfg(uid="first", width=32, height=32)
        second_cfg = CameraCfg(uid="second", width=32, height=32)
        sim._plan_camera_groups([first_cfg, second_cfg])
        first = sim.add_sensor(first_cfg)
        second = sim.add_sensor(second_cfg)
        sim.prepare()

        assert len(first.group_ids) == 1
        assert len(second.group_ids) == 2
        assert first.group_ids[0] == second.group_ids[0]
        with pytest.raises(ValueError, match="all of its group IDs"):
            sim.render_camera_group([first.group_ids[0]])
        second.update()
        first.update(fetch_only=True)
        assert first.get_capture_metadata() is not None
        assert second.get_data()["color"].shape == (17, 32, 32, 4)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
