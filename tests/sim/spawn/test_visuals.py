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

from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from embodichain.lab.sim.cfg import (
    GaussianSplatCfg,
    GaussianRenderCfg,
    VisualElementCfg,
    VisualObjectCfg,
    RigidObjectCfg,
    ArticulationCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.spawn._visuals import _VisualBinding
from embodichain.lab.sim.spawn.descriptors import (
    visual_desc_from_cfg,
    rigid_desc_from_cfg,
)
from embodichain.lab.sim.cfg.visuals import _visual_elements
from embodichain.lab.sim.sim_manager import SimulationManager

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    "kwargs",
    [
        {"file_path": "scene.usdz"},
        {"file_path": "scene.gsq", "lod_path": "scene.dgslod"},
        {"file_path": "scene.ply", "lod_enabled": True},
        {"file_path": "scene.ply", "lod_detail_scale": float("nan")},
    ],
)
def test_gaussian_resource_rejects_unsupported_contracts(kwargs):
    with pytest.raises(ValueError):
        GaussianSplatCfg(**kwargs)


def test_visual_config_roundtrip_and_explicit_empty_replacement():
    data = {
        "uid": "room",
        "visuals": [
            {
                "id": "scan",
                "representation": {"type": "gaussian_splat", "file_path": "room.ply"},
            }
        ],
    }
    cfg = VisualObjectCfg.from_dict(data)
    assert isinstance(cfg.visuals[0].representation, GaussianSplatCfg)
    desc, _ = visual_desc_from_cfg(cfg)
    assert desc.physics is None and desc.collisions == [] and desc.renders == []
    assert VisualObjectCfg.from_dict({"uid": "empty", "visuals": []}).visuals == []
    assert VisualObjectCfg(uid="legacy", shape=CubeCfg()).visuals is None
    link = ArticulationCfg.from_dict({"link_visuals": {"tool": data["visuals"]}})
    assert isinstance(link.link_visuals["tool"][0], VisualElementCfg)


def test_invalid_binding_transform_and_duplicate_ids_reject():
    with pytest.raises(ValueError, match="unique"):
        _visual_elements([VisualElementCfg(representation=CubeCfg())] * 2)
    pose = np.eye(4)
    pose[0, 1] = 0.2
    with pytest.raises(ValueError, match="rigid"):
        VisualElementCfg(representation=CubeCfg(), local_pose=pose)
    with pytest.raises(ValueError, match="uniform"):
        VisualElementCfg(representation=CubeCfg(), scale=-1)


def test_visual_replacement_does_not_compile_gaussians_as_collision_geometry():
    cfg = RigidObjectCfg(uid="body", shape=CubeCfg(size=(0.2, 0.3, 0.4)))
    before, _ = rigid_desc_from_cfg(cfg)
    cfg.visuals = [
        VisualElementCfg(representation=GaussianSplatCfg(file_path="scan.ply"))
    ]
    after, _ = rigid_desc_from_cfg(cfg)
    assert before.physics == after.physics
    assert before.collisions[0].size == after.collisions[0].size
    assert before.body_scale.tolist() == after.body_scale.tolist()


def _binding(tmp_path, link=None):
    path = tmp_path / "scan.ply"
    path.write_text("fixture")
    element = VisualElementCfg(
        id="scan", representation=GaussianSplatCfg(file_path=str(path))
    )
    element.local_pose[:3, 3] = [0.25, 0.5, 1]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [2, 3, 4]
    cfg = (
        VisualObjectCfg(uid="owner", visuals=[element])
        if link is None
        else ArticulationCfg(uid="owner", link_visuals={link: [element]})
    )
    owner = SimpleNamespace(
        cfg=cfg,
        uid="owner",
        num_instances=4,
        device=torch.device("cpu"),
        link_names=["tool"],
        get_local_pose=lambda **kw: torch.tensor(np.stack([pose] * 4)),
        get_link_pose=lambda *a, **kw: torch.tensor(np.stack([pose] * 4)),
        _entities=[],
    )
    spawn = SimpleNamespace(
        builder=SimpleNamespace(), arena_names=tuple(f"arena_{i}" for i in range(4))
    )
    binding = _VisualBinding(owner, spawn)
    roots = [np.eye(4, dtype=np.float32) for _ in range(4)]
    for i, root in enumerate(roots):
        root[0, 3] = i * 10
    batch = MagicMock()
    handles = {
        f"arena_{i}/{binding.elements[0].name}": SimpleNamespace(
            get_resource_id=lambda: 42
        )
        for i in range(4)
    }
    scene = SimpleNamespace(
        topology_revision=1,
        handles=handles,
        clear_visual_meshes=MagicMock(),
        create_gaussian_splat_batch=lambda h: batch,
        get_arena=lambda name: SimpleNamespace(
            get_root_node=lambda: SimpleNamespace(
                get_world_pose=lambda: roots[int(name[-1])]
            )
        ),
    )
    spawn.builder.result = scene
    return binding, scene, batch, roots, pose


@pytest.mark.parametrize("link", [None, "tool"])
def test_owner_and_link_compose_arena_offset_once_and_selectively_reset(tmp_path, link):
    binding, scene, batch, roots, pose = _binding(tmp_path, link)
    binding.sync()
    expected = np.stack(
        [r @ pose @ binding.elements[0].config.local_pose for r in roots]
    )
    np.testing.assert_allclose(batch.apply.call_args.kwargs["world_poses"], expected)
    assert batch.apply.call_count == 1
    binding.set_visible(False, [1, 3])
    binding.sync()
    assert batch.apply.call_args.kwargs["visible"].tolist() == [
        True,
        False,
        True,
        False,
    ]
    binding.reset([1])
    binding.sync()
    assert batch.apply.call_args.kwargs["visible"].tolist() == [True, True, True, False]
    binding.describe()[0]["local_pose"][0, 3] = 999
    assert binding.elements[0].config.local_pose[0, 3] == 0.25


def test_unknown_link_and_unshared_resources_fail_preparation(tmp_path):
    binding, scene, _, _, _ = _binding(tmp_path, "missing")
    with pytest.raises(ValueError, match="Unknown visual link"):
        binding.bind(scene)
    binding, scene, _, _, _ = _binding(tmp_path)
    scene.handles[f"arena_3/{binding.elements[0].name}"].get_resource_id = lambda: 43
    with pytest.raises(RuntimeError, match="share"):
        binding.bind(scene)


def test_gaussian_capture_rejects_failure_and_wrong_group_without_overwriting_success():
    sim = object.__new__(SimulationManager)
    sim._world = SimpleNamespace(
        get_gaussian_frame_status=lambda: dict(
            valid=True, completed=True, outcome="SUCCESS", group_id=8, frame_index=42
        ),
        get_gaussian_output_contract=lambda: {
            "depth": {"definition": "view-Z"},
            "group_id": 8,
            "frame_index": 42,
            "completed": True,
            "same_frame_identity": True,
        },
        get_renderer_type=lambda: SimpleNamespace(name="FASTRT"),
    )
    sim._visualization_sim_time = 0.1
    sim.sim_config = SimpleNamespace(
        render_cfg=SimpleNamespace(
            tone_mapping_enabled=False,
            tone_mapping_exposure=1,
            denoising=SimpleNamespace(offscreen="off"),
        )
    )
    sim._spawn_scene = SimpleNamespace(_gaussian_capture_state=lambda: {})
    first = sim._capture_gaussian_status(8)
    assert first["renderer"] == "FASTRT"
    assert first["configured_offscreen_denoising"] == "off"
    assert first["camera_pose_convention"] == "opengl"
    with pytest.raises(RuntimeError, match="mismatch"):
        sim._capture_gaussian_status(9)
    original_contract = sim._world.get_gaussian_output_contract
    sim._world.get_gaussian_output_contract = lambda: dict(
        original_contract(), frame_index=43
    )
    with pytest.raises(RuntimeError, match="changed between"):
        sim._capture_gaussian_status(8)
    sim._world.get_gaussian_output_contract = original_contract
    sim._world.get_gaussian_frame_status = lambda: dict(
        valid=True, completed=True, outcome="ALLOCATION_FAILURE", group_id=8
    )
    with pytest.raises(RuntimeError, match="failed"):
        sim._capture_gaussian_status(8)
    assert first["status"]["outcome"] == "SUCCESS"


@pytest.mark.parametrize("failure_stage", ["publication", "render"])
def test_failed_frame_cannot_republish_previous_camera_capture(tmp_path, failure_stage):
    from embodichain.lab.sim.sensors import Camera

    binding, scene, batch, _, _ = _binding(tmp_path)
    sim = object.__new__(SimulationManager)
    sim._spawn_scene = SimpleNamespace(
        builder=SimpleNamespace(is_finalized=True, result=scene),
        has_visual_composition=True,
        has_gaussians=True,
        sync_visuals=binding.sync,
    )
    sim.physics = SimpleNamespace(sync_render_state=MagicMock())
    sim._world = SimpleNamespace(render_camera_group=MagicMock())
    camera = object.__new__(Camera)
    camera.group_id = 8
    camera._owner = lambda: sim
    camera.cfg = SimpleNamespace()
    camera._capture_metadata = {"valid": True, "frame_id": 41}
    sim._sensors = {"camera": camera}
    sim._gaussian_captures = {8: {"frame_id": 41}}
    sim._render_state_published = False
    failing_call = (
        batch.apply
        if failure_stage == "publication"
        else sim._world.render_camera_group
    )
    failing_call.side_effect = RuntimeError("native frame failure")
    with pytest.raises(RuntimeError, match="native frame failure"):
        sim.render_camera_group([8])
    assert camera.capture_metadata["valid"] is False
    with pytest.raises(RuntimeError, match="No completed Gaussian capture"):
        camera.update(fetch_only=True)
    # Clearing the native failure allows a full state publication, but an old
    # image still cannot acquire a new capture identity without rendering.
    failing_call.side_effect = None
    sim.sync_render_state()
    with pytest.raises(RuntimeError, match="No completed Gaussian capture"):
        camera.update(fetch_only=True)
    assert not hasattr(scene, "step")


def test_companion_failure_rolls_back_declaration_without_deleting_existing_assets(
    tmp_path,
):
    binding, scene, batch, roots, pose = _binding(tmp_path)
    builder = binding.spawn.builder
    binding.spawn.handles = lambda uid: ()
    builder.scene_desc = SimpleNamespace(iter_entities=lambda: ())
    builder.add_gaussian_splat = MagicMock(
        side_effect=RuntimeError("invalid Gaussian data")
    )
    builder.remove_gaussian_splat = MagicMock()
    with pytest.raises(RuntimeError, match="invalid Gaussian"):
        binding.declare()
    builder.remove_gaussian_splat.assert_called_once_with(binding.elements[0].name)
    builder.remove_gaussian_splat.reset_mock()
    builder.scene_desc.iter_entities = lambda: (
        SimpleNamespace(name=binding.elements[0].name),
    )
    with pytest.raises(ValueError, match="already exist"):
        binding.declare()
    builder.remove_gaussian_splat.assert_not_called()


def test_declared_visual_metadata_does_not_require_native_resources():
    from embodichain.lab.sim.objects import VisualObject

    obj = VisualObject(
        VisualObjectCfg(
            uid="scan",
            visuals=[
                VisualElementCfg(representation=GaussianSplatCfg(file_path="scan.ply"))
            ],
        )
    )
    metadata = obj.get_visual_elements()
    assert (
        metadata[0]["representation"] == "gaussian_splat"
        and metadata[0]["resource_ids"] == ()
    )
    metadata[0]["local_pose"][0, 3] = 100
    assert obj.cfg.visuals[0].local_pose[0, 3] == 0


def test_gaussian_render_budget_and_missing_sidecar_validate_without_engine():
    with pytest.raises(ValueError, match="nonnegative"):
        GaussianRenderCfg(workspace_memory_budget_mb=-1)
    with pytest.raises(TypeError, match="booleans"):
        GaussianRenderCfg(enabled="yes")
    with pytest.raises(ValueError, match="physical shape"):
        rigid_desc_from_cfg(
            RigidObjectCfg(
                uid="missing",
                visuals=[
                    VisualElementCfg(
                        representation=GaussianSplatCfg(file_path="scan.ply")
                    )
                ],
            )
        )


def test_explicit_semantic_geometry_is_independent_without_visual_replacement():
    from embodichain.lab.sim.objects import RigidObject

    owner = SimpleNamespace(
        cfg=RigidObjectCfg(
            uid="body",
            shape=CubeCfg(size=(1, 1, 1)),
            semantic_shape=CubeCfg(size=(0.2, 0.3, 0.4)),
        ),
        num_instances=2,
        device=torch.device("cpu"),
        get_body_scale=lambda ids: torch.full((len(ids), 3), 2.0),
    )
    vertices, triangles = RigidObject.get_semantic_mesh(owner, [1])
    torch.testing.assert_close(
        vertices.amax(1) - vertices.amin(1), torch.tensor([[0.4, 0.6, 0.8]])
    )
    assert triangles.shape[0] == 1


def test_robot_visual_annotations_resolve_across_configclass_inheritance():
    from typing import get_type_hints
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.sim.robots import DualArmRobotCfg

    def reflected(value):
        pass

    for cls in (ArticulationCfg, RobotCfg, DualArmRobotCfg):
        reflected.__annotations__ = {
            "value": cls.__init__.__annotations__["link_visuals"]
        }
        assert (
            get_type_hints(reflected, globalns={}, localns={})["value"]
            == dict[str, list[VisualElementCfg]]
        )


def test_visual_material_operations_reject_gaussian_only_selection(tmp_path):
    binding, scene, _, _, _ = _binding(tmp_path)
    binding.bind(scene)
    with pytest.raises(TypeError, match="captured SH"):
        binding.set_material(object(), [0])
    with pytest.raises(TypeError, match="captured SH"):
        binding.apply_material_inst(0, object())


def test_gaussian_randomization_is_selective_validated_and_resettable(tmp_path):
    binding, scene, batch, *_ = _binding(tmp_path)
    binding.set_gaussian_state(
        "scan",
        [1, 3],
        color_gain=[[0, 1, 0], [0.5, 0.5, 0.5]],
        opacity_gain=[0.2, 0.8],
        geometry_scales=[2, 0.5],
        visible=[False, True],
    )
    binding.sync()
    values = batch.apply.call_args.kwargs
    np.testing.assert_array_equal(values["geometry_scales"], [1, 2, 1, 0.5])
    np.testing.assert_array_equal(
        values["color_gain"], [[1, 1, 1], [0, 1, 0], [1, 1, 1], [0.5, 0.5, 0.5]]
    )
    with pytest.raises(ValueError):
        binding.set_gaussian_state("scan", [1], color_gain=[2, 2, 2], opacity_gain=-1)
    np.testing.assert_array_equal(
        binding.describe()[0]["instance_state"]["color_gain"][1], [0, 1, 0]
    )
    # Rebinding retains current values; reset restores only selected baseline rows.
    scene.topology_revision += 1
    binding.sync()
    binding.reset([1])
    binding.sync()
    values = batch.apply.call_args.kwargs
    np.testing.assert_array_equal(values["geometry_scales"], [1, 1, 1, 0.5])
    np.testing.assert_array_equal(values["color_gain"][3], [0.5, 0.5, 0.5])
    assert values["visible"].all()
    snapshot = binding.describe()[0]["instance_state"]
    snapshot["geometry_scales"][:] = 99
    assert binding.gaussian_state[0]["geometry_scales"][0] == 1
