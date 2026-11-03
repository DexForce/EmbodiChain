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

import math
from pathlib import Path
import numpy as np
import pytest
import torch
from dexsim.scene import Scene

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    DenoisingCfg,
    GaussianRenderCfg,
    GaussianSplatCfg,
    RenderCfg,
    RigidObjectCfg,
    VisualElementCfg,
    VisualObjectCfg,
)
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg

pytestmark = [
    pytest.mark.requires_sim,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not hasattr(Scene, "clear_visual_meshes"),
        reason="requires qualified DexSim visual-composition build",
    ),
]


def _write_splat(path: Path) -> None:
    fields = "x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
    values = (0, 0, 0, 1.4, -1.4, -1.4, 4, *([math.log(0.15)] * 3), 1, 0, 0, 0)
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        + "".join(f"property float {name}\n" for name in fields)
        + "end_header\n"
        + " ".join(map(str, values))
        + "\n"
    )


def _write_arm(path: Path) -> None:
    path.write_text("""<robot name="arm">
<link name="base"><inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.2 0.2 0.3"/></geometry></visual><collision><geometry><box size="0.2 0.2 0.3"/></geometry></collision></link>
<link name="tool"><inertial><mass value="0.2"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.1 0.1 0.4"/></geometry></visual><collision><geometry><box size="0.1 0.1 0.4"/></geometry></collision></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="tool"/><origin xyz="0 0 0.3"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="20" velocity="2"/></joint>
</robot>""")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_gaussian_four_environment_rigid_link_capture_reset_and_removal(
    tmp_path, device
):
    splat = tmp_path / "scan.ply"
    _write_splat(splat)
    urdf = tmp_path / "arm.urdf"
    _write_arm(urdf)
    element = VisualElementCfg(
        id="scan", representation=GaussianSplatCfg(file_path=str(splat))
    )
    offset = np.eye(4, dtype=np.float32)
    offset[:3, 3] = (0.1, 0.2, 0.3)
    link = VisualElementCfg(
        id="tool_scan", representation=element.representation, local_pose=offset
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=4,
            device=device,
            enable_entity_gizmo=False,
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer="fast-rt",
                gaussian=GaussianRenderCfg(enabled=True),
                denoising=DenoisingCfg(window="off", offscreen="off"),
            ),
        )
    )
    visual = rigid = arm = camera = None
    try:
        visual = sim.add_visual_object(
            VisualObjectCfg(uid="room", init_pos=(0, 0, 1), visuals=[element])
        )
        rigid = sim.add_rigid_object(
            RigidObjectCfg(
                uid="body",
                init_pos=(1, 0, 1),
                shape=CubeCfg(size=(0.2, 0.3, 0.4)),
                visuals=[element],
            )
        )
        arm = sim.add_articulation(
            ArticulationCfg(
                uid="arm",
                fpath=str(urdf),
                build_pk_chain=False,
                init_pos=(-1, 0, 0.4),
                link_visuals={
                    "tool": [
                        link,
                        VisualElementCfg(
                            id="mesh", representation=CubeCfg(size=(0.1, 0.1, 0.1))
                        ),
                    ]
                },
            )
        )
        camera = sim.add_sensor(
            CameraCfg(
                uid="camera",
                width=160,
                height=120,
                intrinsics=(100, 100, 80, 60),
                extrinsics=CameraCfg.ExtrinsicsCfg(eye=(0, -3, 2), target=(0, 0, 1)),
                enable_depth=True,
            )
        )
        sim.prepare()
        mass = rigid.get_mass().clone()
        inertia = rigid.get_inertia().clone()
        assert len(set(visual.get_visual_elements()[0]["resource_ids"])) == 1
        assert len(set(arm.get_visual_elements()[0]["resource_ids"])) == 1
        semantic_vertices, semantic_faces = rigid.get_semantic_mesh()
        torch.testing.assert_close(
            semantic_vertices.amax(1) - semantic_vertices.amin(1),
            torch.tensor([[0.2, 0.3, 0.4]] * 4, device=semantic_vertices.device),
        )
        assert semantic_faces.shape[0] == 4
        with pytest.raises(ValueError, match="no triangle mesh"):
            rigid.get_vertices()
        arm.set_qpos(
            torch.tensor([[0.2], [0.4], [0.6], [0.8]], device=device), target=False
        )
        sim.sync_render_state()
        binding = arm._visual_binding
        for env, handle in enumerate(binding.handles[0]):
            root = sim.get_env(env).get_root_node().get_world_pose()
            theta = 0.2 * (env + 1)
            owner_pose = np.array(
                [
                    [math.cos(theta), 0, math.sin(theta), -1],
                    [0, 1, 0, 0],
                    [-math.sin(theta), 0, math.cos(theta), 0.7],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
            expected = root @ owner_pose @ offset
            np.testing.assert_allclose(handle.get_world_pose(), expected, atol=1e-5)
        sim.render_camera_group([camera.group_id])
        camera.set_intrinsics(torch.tensor([110.0, 100.0, 80.0, 60.0]))
        camera.update(fetch_only=True)
        assert camera.capture_metadata["intrinsics"][0, 0, 0] == 100
        camera.set_intrinsics(torch.tensor([100.0, 100.0, 80.0, 60.0]))
        before = camera.get_data()["color"].clone()
        assert before.shape == (4, 120, 160, 4) and before.dtype == torch.uint8
        assert int(before[..., :3].max()) > int(before[..., :3].min())
        assert camera.capture_metadata["status"]["camera_count"] == 4
        assert camera.capture_metadata["output_contract"]["depth"]["is_view_z"]
        # Path-traced mesh shading varies even without state changes. Test
        # isolation on the deterministic Gaussian-dominated central patch.
        visual.set_visible(False, [1])
        sim.render_camera_group([camera.group_id])
        camera.update(fetch_only=True)
        after = camera.get_data()["color"].clone()
        before_patch = before[:, 55:65, 75:85, :3].float().mean((1, 2))
        after_patch = after[:, 55:65, 75:85, :3].float().mean((1, 2))
        assert before_patch[1, 0] - after_patch[1, 0] > 50
        torch.testing.assert_close(
            before_patch[[0, 2, 3]], after_patch[[0, 2, 3]], atol=2, rtol=0
        )
        visual.reset([1])
        sim.sync_render_state()
        assert visual._visual_binding.visibility.tolist() == [True] * 4
        torch.testing.assert_close(rigid.get_mass(), mass)
        torch.testing.assert_close(rigid.get_inertia(), inertia)
        # A new topology revision must retain source semantics and current state.
        rigid.set_visible(False, [2])
        sim.add_visual_object(
            VisualObjectCfg(uid="extra", shape=CubeCfg(size=(0.05, 0.05, 0.05)))
        )
        assert not rigid._visual_binding.visibility[2]
        torch.testing.assert_close(rigid.get_semantic_mesh()[0], semantic_vertices)
        assert sim.remove_asset("room")
        sim.render_camera_group([camera.group_id])
        camera.update(fetch_only=True)
        assert camera.capture_metadata["valid"]
        assert sim.remove_asset("body")
        assert sim.remove_asset("arm")
        sim.render_camera_group([camera.group_id])
        camera.update(fetch_only=True)
        assert camera.capture_metadata == {}
    finally:
        visual = rigid = arm = camera = None
        sim.destroy(exit_process=False)
        sim.flush_cleanup_queue()


def test_gaussian_workspace_failure_cannot_publish_a_valid_capture(tmp_path):
    path = tmp_path / "scan.ply"
    _write_splat(path)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            enable_entity_gizmo=False,
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer="fast-rt",
                denoising=DenoisingCfg(window="off", offscreen="off"),
                gaussian=GaussianRenderCfg(enabled=True, workspace_memory_budget_mb=1),
            ),
        )
    )
    camera = None
    try:
        sim.add_visual_object(
            VisualObjectCfg(
                uid="scan",
                visuals=[
                    VisualElementCfg(
                        representation=GaussianSplatCfg(file_path=str(path))
                    )
                ],
            )
        )
        camera = sim.add_sensor(
            CameraCfg(
                uid="camera",
                width=640,
                height=480,
                intrinsics=(400, 400, 320, 240),
                extrinsics=CameraCfg.ExtrinsicsCfg(eye=(0, -3, 1), target=(0, 0, 0)),
            )
        )
        sim.prepare()
        with pytest.raises(RuntimeError, match="Gaussian capture failed"):
            sim.render_camera_group([camera.group_id])
        with pytest.raises(RuntimeError, match="No completed Gaussian capture"):
            camera.update(fetch_only=True)
        assert camera.capture_metadata["valid"] is False
    finally:
        camera = None
        sim.destroy(exit_process=False)
        sim.flush_cleanup_queue()


@pytest.mark.parametrize("renderer", ["fast-rt", "hybrid"])
def test_gaussian_randomization_parallel_cameras_isolation_and_reset(
    tmp_path, renderer
):
    path = tmp_path / "scan.ply"
    _write_splat(path)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=4,
            device="cpu",
            enable_entity_gizmo=False,
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer=renderer,
                gaussian=GaussianRenderCfg(enabled=True),
                denoising=DenoisingCfg(window="off", offscreen="off"),
            ),
        )
    )
    visual = None
    cameras = []
    try:
        visual = sim.add_visual_object(
            VisualObjectCfg(
                uid="scan",
                init_pos=(0, 0, 1),
                visuals=[
                    VisualElementCfg(
                        id="main", representation=GaussianSplatCfg(file_path=str(path))
                    ),
                    VisualElementCfg(
                        id="variant",
                        representation=GaussianSplatCfg(
                            file_path=str(path),
                            point_opacities=np.array([0.9], np.float32),
                            sh_coefficients=np.array([[[-1.4, 1.4, -1.4]]], np.float32),
                        ),
                        visible=False,
                    ),
                ],
            )
        )
        for index, width in enumerate([80, 96]):
            cameras.append(
                sim.add_sensor(
                    CameraCfg(
                        uid=f"camera_{index}",
                        width=width,
                        height=64,
                        intrinsics=(60, 60, width / 2, 32),
                        enable_depth=True,
                        extrinsics=CameraCfg.ExtrinsicsCfg(
                            eye=(0, -2, 1), target=(0, 0, 1)
                        ),
                    )
                )
            )
        sim.prepare()
        assert cameras[0].group_id != cameras[1].group_id
        resource_ids = visual.get_visual_elements()[0]["resource_ids"]
        assert len(set(resource_ids)) == 1
        variant_ids = visual.get_visual_elements()[1]["resource_ids"]
        assert len(set(variant_ids)) == 1 and variant_ids[0] != resource_ids[0]
        source_data = visual._visual_binding.handles[0][0].get_data(
            fields=["sh", "leaf_opacity"]
        )
        variant_data = visual._visual_binding.handles[1][0].get_data(
            fields=["sh", "leaf_opacity"]
        )
        assert source_data["sh"][0, 0, 0] > 0 and variant_data["sh"][0, 0, 0] < 0
        np.testing.assert_allclose(variant_data["leaf_opacity"], [0.9])

        def capture():
            sim.render_camera_group([cam.group_id for cam in cameras])
            result = []
            for cam in cameras:
                cam.update(fetch_only=True)
                assert cam.capture_metadata["valid"]
                assert cam.capture_metadata["status"]["group_id"] == cam.group_id
                result.append(cam.get_data()["color"].clone())
            return result

        before = capture()
        # Change only one environment. Shared payload IDs must remain stable.
        visual.set_gaussian_visual_state("main", [1], color_gain=(0, 0, 0))
        after = capture()
        assert visual.get_visual_elements()[0]["resource_ids"] == resource_ids
        for old, new in zip(before, after):
            torch.testing.assert_close(old[[0, 2, 3]], new[[0, 2, 3]], atol=2, rtol=0)
            assert (old[1].float() - new[1].float()).abs().sum() > 500
        visual.set_gaussian_visual_state("main", [2], opacity_gain=0.0)
        transparent = capture()
        for old, new in zip(after, transparent):
            torch.testing.assert_close(old[[0, 1, 3]], new[[0, 1, 3]], atol=2, rtol=0)
            assert (old[2].float() - new[2].float()).abs().sum() > 500
        visual.set_gaussian_visual_state("main", [1], scale=1.8)
        after = capture()
        np.testing.assert_allclose(
            np.linalg.norm(
                visual._visual_binding.handles[0][1].get_world_pose()[:3, :3], axis=0
            ),
            1.8,
            atol=1e-5,
        )
        # Select another preloaded appearance in a different arena.
        visual.set_gaussian_visual_state("main", [3], visible=False)
        visual.set_gaussian_visual_state("variant", [3], visible=True)
        selected = capture()
        for previous, new in zip(after, selected):
            torch.testing.assert_close(previous[:3], new[:3], atol=2, rtol=0)
            assert (previous[3].float() - new[3].float()).abs().sum() > 500
        # Topology rebuild keeps live randomization; reset uses the independent baseline.
        sim.add_visual_object(
            VisualObjectCfg(
                uid="extra", shape=CubeCfg(size=(0.01, 0.01, 0.01)), init_pos=(0, 0, -5)
            )
        )
        rebuilt = capture()
        for old, new in zip(selected, rebuilt):
            torch.testing.assert_close(old, new, atol=2, rtol=0)
        visual.reset([1])
        restored = capture()
        for initial, selected_frame, new in zip(before, selected, restored):
            torch.testing.assert_close(initial[1], new[1], atol=2, rtol=0)
            torch.testing.assert_close(selected_frame[3], new[3], atol=2, rtol=0)
        with pytest.raises(ValueError):
            visual.set_gaussian_visual_state(
                "main", [0, 2], color_gain=[[1, 1, 1], [-1, 1, 1]]
            )
        unchanged = capture()
        for old, new in zip(restored, unchanged):
            torch.testing.assert_close(old, new, atol=2, rtol=0)
    finally:
        cameras.clear()
        visual = None
        sim.destroy(exit_process=False)
        sim.flush_cleanup_queue()
