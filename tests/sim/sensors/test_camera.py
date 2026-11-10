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
# ----------------------------------------------------------------------------,

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.sensors import (
    BaseSensor,
    Camera,
    SensorCfg,
    CameraCfg,
    StereoCamera,
    StereoCameraCfg,
)
from embodichain.lab.sim.sensors.stereo import PairCameraView
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.sim.cfg import ArticulationCfg, DenoisingCfg, RenderCfg
from embodichain.data import get_data_path
from scripts.tutorials.sim.create_sensor import create_sensor as create_tutorial_sensor

FULL_NUM_ENVS = 4
FULL_WIDTH = 640
FULL_HEIGHT = 480
SMOKE_NUM_ENVS = 1
SMOKE_WIDTH = 160
SMOKE_HEIGHT = 120
ART_PATH = "SlidingBoxDrawer/SlidingBoxDrawer.urdf"


class CameraTest:
    def setup_simulation(
        self,
        sim_device,
        renderer="hybrid",
        num_envs=FULL_NUM_ENVS,
        width=FULL_WIDTH,
        height=FULL_HEIGHT,
        enable_auxiliary_data=True,
    ):
        # Setup SimulationManager
        config = SimulationManagerCfg(
            headless=True,
            device=sim_device,
            render_cfg=RenderCfg(renderer=renderer),
            num_envs=num_envs,
        )
        self.sim = SimulationManager(config)
        # Create batch of cameras
        cfg_dict = {
            "sensor_type": "Camera",
            "width": width,
            "height": height,
            "enable_mask": enable_auxiliary_data,
            "enable_depth": enable_auxiliary_data,
            "enable_normal": enable_auxiliary_data,
            "enable_position": enable_auxiliary_data,
        }
        cfg = SensorCfg.from_dict(cfg_dict)
        self.camera: Camera = self.sim.add_sensor(cfg)
        self.sim.prepare()

    def test_get_data(self):

        self.camera.update()

        # Get data from the camera
        data = self.camera.get_data()

        # Check if data is a dictionary
        assert isinstance(data, TensorDict), "Camera data should be a TensorDict"

        # Check if all expected keys are present
        for key in self.camera.SUPPORTED_DATA_TYPES:
            assert key in data, f"Missing key in camera data: {key}"

        # Check if the data shape matches the expected shape
        assert data["color"].shape == (
            FULL_NUM_ENVS,
            FULL_HEIGHT,
            FULL_WIDTH,
            4,
        ), "RGB data shape mismatch"
        assert data["depth"].shape == (
            FULL_NUM_ENVS,
            FULL_HEIGHT,
            FULL_WIDTH,
        ), "Depth data shape mismatch"
        assert data["normal"].shape == (
            FULL_NUM_ENVS,
            FULL_HEIGHT,
            FULL_WIDTH,
            3,
        ), "Normal data shape mismatch"
        assert data["position"].shape == (
            FULL_NUM_ENVS,
            FULL_HEIGHT,
            FULL_WIDTH,
            3,
        ), "Position data shape mismatch"
        assert data["mask"].shape == (
            FULL_NUM_ENVS,
            FULL_HEIGHT,
            FULL_WIDTH,
        ), "Mask data shape mismatch"

        # Check if the data types are correct
        assert data["color"].dtype == torch.uint8, "Color data type mismatch"
        assert data["depth"].dtype == torch.float32, "Depth data type mismatch"
        assert data["normal"].dtype == torch.float32, "Normal data type mismatch"
        assert data["position"].dtype == torch.float32, "Position data type mismatch"
        assert data["mask"].dtype == torch.int32, "Mask data type mismatch"

    def test_local_pose_with_env_ids(self):
        env_ids = [0, 1, 2]

        pose = (
            torch.eye(4, device=self.sim.device).unsqueeze(0).repeat(len(env_ids), 1, 1)
        )
        pose[:, 2, 3] = 2.0

        self.camera.set_local_pose(pose, env_ids=env_ids)

        # Verify the local pose for specified env_ids
        assert torch.allclose(self.camera.get_local_pose(to_matrix=True)[env_ids], pose)

    def test_attach_to_parent(self):
        art_path = get_data_path(ART_PATH)
        assert os.path.isfile(art_path)

        cfg_dict = {"fpath": art_path}
        self.art: Articulation = self.sim.add_articulation(
            cfg=ArticulationCfg.from_dict(cfg_dict)
        )
        self.sim.prepare()
        self.camera: Camera = self.sim.add_sensor(
            sensor_cfg=CameraCfg(
                uid="test",
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    parent="handle_xpos", pos=(0.1, 0.2, 0.3)
                ),
            )
        )
        assert self.camera.is_attached
        for view, articulation in zip(
            self.camera._entities, self.art._entities, strict=True
        ):
            parent = articulation.get_render_body("handle_xpos").render_node()
            assert (
                view.get_node().path_name().rsplit("/", maxsplit=1)[0]
                == parent.path_name()
            )
        expected_pose = self.camera.cfg.extrinsics.transformation.unsqueeze(0).repeat(
            self.camera.num_instances, 1, 1
        )
        torch.testing.assert_close(
            self.camera.get_local_pose(to_matrix=True).cpu(), expected_pose
        )

    def test_set_intrinsics(self):
        # Define new intrinsic parameters
        new_intrinsics = (
            torch.tensor(
                [500.0, 500.0, 320.0, 240.0],
                device=self.sim.device,
            )
            .unsqueeze(0)
            .repeat(FULL_NUM_ENVS, 1)
        )

        # Set new intrinsic parameters for all environments
        self.camera.set_intrinsics(new_intrinsics)

    def teardown_method(self):
        """Clean up resources after each test method."""
        if (
            hasattr(self, "camera")
            and getattr(self.camera, "uid", None) is not None
            and hasattr(self, "sim")
        ):
            self.sim.remove_asset(self.camera.uid)
        if hasattr(self, "sim"):
            self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        import gc

        gc.collect()


class TestCameraHybridCUDA(CameraTest):
    def setup_method(self):

        self.setup_simulation("cuda", renderer="hybrid")


def test_create_sensor_tutorial_preserves_attached_camera_view() -> None:
    """Keep the wrist-camera view stable after the xyzw convention migration."""
    sim = MagicMock()

    create_tutorial_sensor(sim, SimpleNamespace(attach_sensor=True))

    cfg = sim.add_sensor.call_args.kwargs["sensor_cfg"]
    expected_rotation = torch.tensor(
        [
            [0.579228, 0.573576, 0.579228],
            [0.405580, -0.819152, 0.405580],
            [0.707107, 0.0, -0.707107],
        ],
        dtype=torch.float32,
    )
    assert cfg.extrinsics.parent == "ee_link"
    torch.testing.assert_close(
        cfg.extrinsics.transformation[:3, :3],
        expected_rotation,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


@pytest.mark.parametrize(
    ("sim_device", "renderer"),
    [("cpu", "hybrid"), ("cpu", "fast-rt"), ("cuda", "fast-rt"), ("cuda", "rt")],
)
def test_camera_backend_smoke(sim_device, renderer):
    """Check that each remaining backend/device pair renders a color frame."""
    test = CameraTest()
    test.setup_simulation(
        sim_device,
        renderer,
        num_envs=SMOKE_NUM_ENVS,
        width=SMOKE_WIDTH,
        height=SMOKE_HEIGHT,
        enable_auxiliary_data=False,
    )
    try:
        test.camera.update()
        data = test.camera.get_data()
        assert data["color"].shape == (SMOKE_NUM_ENVS, SMOKE_HEIGHT, SMOKE_WIDTH, 4)
    finally:
        test.teardown_method()


def test_camera_parent_attachment_cpu() -> None:
    """Attach a camera to a materialized articulation link on the CPU backend."""
    test = CameraTest()
    test.setup_simulation(
        "cpu",
        renderer="hybrid",
        num_envs=SMOKE_NUM_ENVS,
        width=SMOKE_WIDTH,
        height=SMOKE_HEIGHT,
        enable_auxiliary_data=False,
    )
    try:
        test.test_attach_to_parent()
    finally:
        test.teardown_method()


@pytest.mark.no_sim
@pytest.mark.parametrize("stereo", [False, True])
def test_camera_attachment_reapplies_parent_relative_extrinsics(stereo: bool) -> None:
    """Every view attaches to its own arena node before resetting its local pose."""
    cfg_type = StereoCameraCfg if stereo else CameraCfg
    camera_type = StereoCamera if stereo else Camera
    camera = object.__new__(camera_type)
    camera.cfg = cfg_type(
        uid="wrist_camera",
        extrinsics=CameraCfg.ExtrinsicsCfg(parent="wrist", pos=(0.1, 0.2, 0.3)),
    )
    camera.num_instances = 2
    camera._is_attached = False
    views = [
        [
            MagicMock(spec=["attach_node", "set_local_pose"])
            for _ in range(2 if stereo else 1)
        ]
        for _ in range(2)
    ]
    camera._entities = [
        PairCameraView(*pair, camera.cfg.left_to_right.numpy()) if stereo else pair[0]
        for pair in views
    ]
    nodes = [object(), object()]

    assert not camera.is_attached
    camera.attach_to_parent_nodes(nodes)

    assert camera.is_attached
    for node, pair in zip(nodes, views, strict=True):
        for index, view in enumerate(pair):
            view.attach_node.assert_called_once_with(node)
            assert [call[0] for call in view.method_calls] == [
                "attach_node",
                "set_local_pose",
            ]
            expected_pose = camera.cfg.extrinsics.transformation.numpy().copy()
            if stereo:
                expected_pose[0, 3] += (-0.5 if index == 0 else 0.5) * 0.05
            np.testing.assert_allclose(
                view.set_local_pose.call_args.args[0], expected_pose
            )


@pytest.mark.no_sim
@pytest.mark.parametrize("stereo", [False, True])
def test_camera_parent_config_does_not_imply_attachment(
    monkeypatch: pytest.MonkeyPatch, stereo: bool
) -> None:
    """Attachment state reflects actual reparenting, not merely configuration."""
    monkeypatch.setattr(
        BaseSensor,
        "__init__",
        lambda self, config, device, *, num_instances: setattr(self, "cfg", config),
    )
    cfg_type = StereoCameraCfg if stereo else CameraCfg
    camera_type = StereoCamera if stereo else Camera
    monkeypatch.setattr(Camera, "reset", lambda self: None)
    owner = MagicMock(num_envs=1)
    camera = camera_type(
        cfg_type(extrinsics=CameraCfg.ExtrinsicsCfg(parent="wrist")), owner=owner
    )
    owner.get_env.assert_called_once_with(0)
    assert not camera.is_attached


@pytest.mark.no_sim
@pytest.mark.parametrize("parent_count", [0, 1, 3])
def test_camera_attachment_rejects_mismatched_parent_count(parent_count: int) -> None:
    camera = object.__new__(Camera)
    camera.num_instances = 2
    camera._entities = [MagicMock(spec=["attach_node"]) for _ in range(2)]
    camera._is_attached = False
    camera.reset = MagicMock()

    with pytest.raises(RuntimeError, match="parent nodes for 2 camera instances"):
        camera.attach_to_parent_nodes([object() for _ in range(parent_count)])

    assert not camera.is_attached
    camera.reset.assert_not_called()
    for view in camera._entities:
        view.attach_node.assert_not_called()


@pytest.mark.no_sim
def test_camera_attachment_rejects_missing_node_before_reparenting() -> None:
    camera = object.__new__(Camera)
    camera.num_instances = 2
    camera._entities = [MagicMock(spec=["attach_node"]) for _ in range(2)]
    camera._is_attached = False

    with pytest.raises(ValueError, match="parent node in every arena"):
        camera.attach_to_parent_nodes([object(), None])

    assert not camera.is_attached
    for view in camera._entities:
        view.attach_node.assert_not_called()


if __name__ == "__main__":
    test = TestCameraHybridCUDA()
    test.setup_method()
    test.test_attach_to_parent()


@pytest.mark.no_sim
@pytest.mark.parametrize("stereo", [False, True])
def test_camera_detaches_all_views_before_parent_removal(stereo: bool) -> None:
    camera = object.__new__(StereoCamera if stereo else Camera)
    camera._is_attached = True
    views = [MagicMock() for _ in range(4 if stereo else 2)]
    camera._entities = (
        [PairCameraView(*views[i : i + 2], np.eye(4)) for i in (0, 2)]
        if stereo
        else views
    )

    camera._detach_from_parent_nodes()

    assert not camera.is_attached
    for view in views:
        view.get_node.return_value.detach_parent.assert_called_once_with()


@pytest.mark.no_sim
def test_ground_visibility_preflight_and_selected_reset() -> None:
    """Invalid batches cannot partially change camera policy or enqueue writes."""
    camera = object.__new__(Camera)
    owner = MagicMock()
    owner._GROUND_RENDER_CATEGORY = 1
    camera._owner_ref = lambda: owner
    camera._is_destroyed = False
    camera.num_instances = 4
    camera._entities = [MagicMock() for _ in range(4)]
    camera._ground_plane_visibility = [True] * 4
    camera._capture_metadata = None
    camera.cfg = CameraCfg()
    camera._reset_pose = MagicMock()
    for ids in ([0, 4], [0, 0], [1.5]):
        with pytest.raises(ValueError):
            camera.set_ground_plane_visibility(False, ids)
    for entity in camera._entities:
        entity.set_excluded_render_categories.assert_not_called()
    camera.set_ground_plane_visibility(False, torch.tensor([1, 3]))
    assert camera._ground_plane_visibility == [True, False, True, False]
    camera.reset([1])
    assert camera._ground_plane_visibility == [True, True, True, False]
    assert camera._entities[3].set_excluded_render_categories.call_args.args == (1,)
    camera._is_destroyed = True
    with pytest.raises(RuntimeError, match="destroyed"):
        camera.set_ground_plane_visibility(True)


@pytest.mark.no_sim
def test_ground_visibility_missing_sdk_fails_before_allocating_camera(
    monkeypatch,
) -> None:
    owner = MagicMock(num_envs=4)
    owner._require_ground_plane_filtering.side_effect = RuntimeError("SDK capability")
    with pytest.raises(RuntimeError, match="SDK capability"):
        Camera(CameraCfg(ground_plane_visible=False), owner=owner)
    owner.get_world.assert_not_called()


@pytest.mark.gpu
@pytest.mark.subprocess_sim
@pytest.mark.parametrize("renderer", ["fast-rt", "hybrid"])
def test_parallel_camera_ground_visibility_rgb_depth_reset_and_rebuild(
    renderer: str,
) -> None:
    """Qualify each native renderer in its own process, independent of DLSS."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "runpy.run_path(sys.argv[1])['_run_camera_ground_visibility_case'](sys.argv[2])",
            str(Path(__file__).resolve()),
            renderer,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_camera_ground_visibility_case(renderer: str) -> None:
    """Mixed policies within a group and between cameras preserve one physical floor."""
    from dexsim.types import Renderer
    from embodichain.lab.sim.cfg import VisualObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=4,
            # Ground filtering compares raw RGB/depth; NGX availability and
            # reconstruction at this tiny resolution are separate contracts.
            render_cfg=RenderCfg(
                renderer=renderer,
                denoising=DenoisingCfg(window="off", offscreen="off"),
            ),
        )
    )
    try:
        assert (
            sim.get_world().get_renderer_type()
            == {
                "fast-rt": Renderer.FASTRT,
                "hybrid": Renderer.HYBRID,
            }[renderer]
        )
        sim.add_visual_object(
            VisualObjectCfg(
                uid="under_floor",
                shape=CubeCfg(size=(1, 1, 1)),
                init_pos=(0, 0, -2),
            )
        )
        cameras = [
            sim.add_sensor(
                CameraCfg(
                    uid=name,
                    width=32,
                    height=32,
                    intrinsics=(28, 28, 16, 16),
                    enable_depth=True,
                    ground_plane_visible=visible,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(0, 0, 3),
                        target=(0, 0, 0),
                        up=(0, 1, 0),
                    ),
                )
            )
            for name, visible in (("training", False), ("debug", True))
        ]
        sim.prepare()
        sim.enable_physics(False)

        def capture():
            sim.render_camera_group([camera.group_id for camera in cameras])
            for camera in cameras:
                camera.update(fetch_only=True)
            return [camera.get_data()["depth"][:, 16, 16].clone() for camera in cameras]

        hidden, shown = capture()
        torch.testing.assert_close(hidden, torch.full((4,), 4.5), atol=0.04, rtol=0)
        torch.testing.assert_close(shown, torch.full((4,), 3.0), atol=0.04, rtol=0)
        rgb_hidden = cameras[0].get_data()["color"].clone().float()
        rgb_shown = cameras[1].get_data()["color"].clone().float()
        assert (rgb_hidden[..., :3] - rgb_shown[..., :3]).abs().mean() > 1.0
        floor_desc = sim._default_plane.object_desc
        assert floor_desc.physics is not None and floor_desc.collisions

        cameras[0].set_ground_plane_visibility(True, [1, 3])
        mixed, _ = capture()
        torch.testing.assert_close(
            mixed, torch.tensor([4.5, 3.0, 4.5, 3.0]), atol=0.04, rtol=0
        )
        metadata = cameras[0].get_capture_metadata()
        assert metadata["ground_plane_visible"] == (False, True, False, True)
        cameras[0].reset([1])
        reset, _ = capture()
        torch.testing.assert_close(
            reset, torch.tensor([4.5, 4.5, 4.5, 3.0]), atol=0.04, rtol=0
        )
        assert metadata["ground_plane_visible"] == (False, True, False, True)

        sim.add_visual_object(
            VisualObjectCfg(
                uid="rebuild_trigger",
                shape=CubeCfg(),
                init_pos=(5, 0, 2),
            )
        )
        sim.prepare()
        rebuilt, _ = capture()
        torch.testing.assert_close(rebuilt, reset, atol=0.04, rtol=0)
        sim.set_ground_plane_visibility(False)
        globally_hidden = capture()
        for depth in globally_hidden:
            torch.testing.assert_close(depth, torch.full((4,), 4.5), atol=0.04, rtol=0)
        assert cameras[1].get_capture_metadata()["ground_plane_visible"] == (False,) * 4
        assert sim._default_plane.object_desc.physics is not None
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.no_sim
def test_stereo_ground_exclusion_updates_both_eyes() -> None:
    left, right = MagicMock(), MagicMock()
    pair = PairCameraView(left, right, np.eye(4, dtype=np.float32))
    pair.set_excluded_render_categories(1)
    left.set_excluded_render_categories.assert_called_once_with(1)
    right.set_excluded_render_categories.assert_called_once_with(1)


@pytest.mark.no_sim
def test_failed_image_read_invalidates_capture_metadata() -> None:
    camera = object.__new__(Camera)
    camera.cfg = CameraCfg()
    camera._capture_metadata = {"capture_sequence": 1}
    camera._pending_capture_metadata = {"capture_sequence": 2}
    camera._frame_buffer = MagicMock()
    camera._frame_buffer.get_rgb_gpu_buffer.side_effect = RuntimeError("read failed")
    with pytest.raises(RuntimeError, match="read failed"):
        camera.update(fetch_only=True)
    assert camera.get_capture_metadata() is None
