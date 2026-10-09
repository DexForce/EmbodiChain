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

"""CPU contracts for the frozen Isaac adapter; no Kit or GPU is initialized."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from scripts.benchmark.rendering.backends.scene_isaaclab import (
    IsaacLabScene,
    _advance_pt_history,
    _apply_carb_overrides,
    _configure_accumulation,
    _geometry_prototypes,
    _instance_geometry,
    _object_matrix,
    _native_lighting,
    _render_settings,
    _resolution_evidence,
    _segmentation_map,
    _selected_cameras,
)
from scripts.benchmark.rendering.formal import render_profile


def _scene() -> dict[str, Any]:
    return {
        "camera_model": {"horizontal_fov_deg": 55, "near_m": 0.01, "far_m": 8},
        "cameras": [
            {"name": name, "eye": [2, -2, 1], "target": [0, 0, 0], "up": [0, 0, 1]}
            for name in ("front", "side")
        ],
        "objects": [
            {
                "uid": "cube",
                "instance_id": 7,
                "geometry": {"kind": "box", "size": [1, 2, 3]},
                "world_from_object": np.eye(4).tolist(),
            }
        ],
    }


def test_camera_selection_preserves_original_scene_and_order() -> None:
    scene = _scene()
    original = copy.deepcopy(scene)
    selected = _selected_cameras(scene, ("side",), None)
    selected[0]["eye"][0] = 9
    assert scene == original
    assert selected[0]["name"] == "side"
    with pytest.raises(ValueError, match="missing or duplicate"):
        _selected_cameras(scene, ("side", "side"), None)


def test_profile_overrides_are_explicit_and_frame_generation_forbidden() -> None:
    assert _render_settings(None)["samples_per_pixel"] == 4
    settings = _render_settings(render_profile("high_quality"))
    assert settings["samples_per_pixel"] == 16
    assert settings["rendering_mode"] == "quality"
    assert settings["enable_dlssg"] is False
    profile = render_profile("realtime")
    profile["isaaclab"]["enable_dlssg"] = True
    with pytest.raises(ValueError, match="Frame generation"):
        _render_settings(profile)


def test_isaac_lighting_controls_resolve_without_changing_scene_or_profile() -> None:
    scene = {
        "lighting": {
            "isaac_directional_intensity": 3000.0,
            "isaac_dome_intensity": 150.0,
        }
    }
    original = copy.deepcopy(scene)
    profile = {
        "lighting_controls": {
            "isaaclab": {
                "environment_intensity": 250.0,
                "environment_color": [0.8, 1.0, 1.2],
            },
            "embodichain": {"environment_intensity": 17.0},
        }
    }
    assert _native_lighting(scene, None) == {
        "directional_intensity": 3000.0,
        "environment_intensity": 150.0,
        "environment_color": [1.0, 1.0, 1.0],
    }
    resolved = _native_lighting(scene, profile)
    assert resolved["environment_intensity"] == 250.0
    resolved["environment_color"][0] = 0
    assert profile["lighting_controls"]["isaaclab"]["environment_color"][0] == 0.8
    assert scene == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"directional_intensity": -1},
        {"environment_intensity": float("inf")},
        {"environment_color": [1, 1]},
        {"environment_color": [1, float("nan"), 1]},
        {"environment_intensity": True},
        {"unexpected": 1},
    ],
)
def test_invalid_isaac_lighting_controls_are_rejected(
    overrides: dict[str, Any],
) -> None:
    scene = {
        "lighting": {
            "isaac_directional_intensity": 3000.0,
            "isaac_dome_intensity": 150.0,
        }
    }
    with pytest.raises(ValueError):
        _native_lighting(scene, {"lighting_controls": {"isaaclab": overrides}})


def test_explicit_post_camera_overrides_preserve_pathtracing_mode() -> None:
    events = ["camera_constructed", "reset_completed"]
    sim = SimpleNamespace(set_setting=lambda path, value: events.append((path, value)))
    settings = {
        "carb_settings": {
            "/rtx/rendermode": "PathTracing",
            "/rtx/pathtracing/spp": 16,
            "/rtx/pathtracing/totalSpp": 0,
        }
    }
    overrides = _apply_carb_overrides(sim, settings)
    assert events[2:] == list(overrides.items())
    assert overrides["/rtx/rendermode"] == "PathTracing"
    assert settings["carb_settings"]["/rtx/pathtracing/totalSpp"] == 0
    profile = render_profile("realtime")
    profile["isaaclab"]["carb_settings"] = {"/rtx-transient/dlssg/enabled": True}
    with pytest.raises(ValueError, match="Frame generation"):
        _render_settings(profile)


def test_internal_resolution_warning_is_not_fabricated_into_an_extent() -> None:
    message = "DLSS increasing input dimensions: Render resolution of (128, 128) is below minimal input resolution of 300."
    result = _resolution_evidence(
        128,
        128,
        "native_output_resolution",
        [128, 128],
        {"/rtx/post/aa/op": 4},
        message,
    )
    assert result["native_resolution_status"] == "unsupported"
    assert result["requested_internal_resolution"] == [128, 128]
    assert result["resolved_internal_resolution"] is None
    assert result["runtime_resize_warnings"][0]["minimum_input_dimension"] == 300
    assert result["observed_output_resolution"] == [128, 128]


def test_no_warning_or_taa_setting_is_not_internal_resolution_proof() -> None:
    result = _resolution_evidence(
        512,
        512,
        "native_output_resolution",
        [512, 512],
        {"/rtx/post/aa/op": 1, "/rtx-transient/dldenoiser/enabled": False},
        "",
    )
    assert result["native_resolution_status"] == "not_verified"
    assert result["resolved_internal_resolution"] is None


def test_pt_history_reset_uses_documented_native_control_without_changing_spp() -> None:
    settings = {
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 16,
        "/rtx/pathtracing/totalSpp": 0,
        "/rtx/externalFrameCounter": 17,
    }
    sim = SimpleNamespace(get_setting=settings.get, set_setting=settings.__setitem__)
    setup = _configure_accumulation(sim, "reset_each_exposure")
    assert setup["external_counter_before_setup"] == 17
    assert setup["reset_control_observed"] is True
    event = _advance_pt_history(sim)
    assert event["before"] == 17
    assert event["observed_after_write"] == 18
    assert "not a measured GPU exposure" in event["role"]
    assert settings["/rtx/pathtracing/spp"] == 16
    assert settings["/rtx/pathtracing/totalSpp"] == 0


def test_accumulation_control_rejects_a_stopping_cap_instead_of_mutating_it() -> None:
    settings = {
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 16,
        "/rtx/pathtracing/totalSpp": 16,
    }
    sim = SimpleNamespace(get_setting=settings.get, set_setting=settings.__setitem__)
    with pytest.raises(RuntimeError, match="unlimited totalSpp=0"):
        _configure_accumulation(sim, "reset_each_exposure")
    assert settings["/rtx/pathtracing/totalSpp"] == 16
    before = dict(settings)
    assert _configure_accumulation(sim, "renderer_default")["control"] == "no_override"
    assert settings == before


def test_mesh_buffers_are_authored_once_and_instances_have_mutable_roots(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    scene = _scene()
    asset = tmp_path / "mesh.npz"
    np.savez_compressed(
        asset,
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        normals=np.array([[0, 0, 1]] * 3, dtype=np.float32),
    )
    mesh = {
        "uid": "mesh",
        "instance_id": 8,
        "world_from_object": np.eye(4).tolist(),
        "geometry": {
            "kind": "mesh",
            "path": asset.name,
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        },
    }
    scene["objects"].extend([mesh, {**mesh, "uid": "same_mesh", "instance_id": 9}])
    assets, prototypes, radii = _geometry_prototypes(scene, tmp_path / "scene.json")
    assert prototypes["mesh"] == prototypes["same_mesh"]
    assert len([prim for prim in assets.Traverse() if prim.IsA(UsdGeom.Mesh)]) == 1
    stage = Usd.Stage.CreateInMemory()
    roots = []
    for index in range(3):
        shape, operation = _instance_geometry(
            stage,
            f"/World/env_{index}/mesh",
            mesh,
            assets,
            prototypes["mesh"],
            [index * 50, 0, 0],
        )
        roots.append(shape.GetPrim())
        np.testing.assert_allclose(
            np.asarray(operation.Get()).T[:3, 3], [index * 50, 0, 0]
        )
    assert all(prim.IsInstance() for prim in roots)
    assert len({str(prim.GetPrototype().GetPath()) for prim in roots}) == 1
    assert stage.GetPrimAtPath("/World/env_1/mesh/Geometry").IsInstanceProxy()


def test_box_scale_is_preserved_under_replayed_rotation_and_translation() -> None:
    obj = _scene()["objects"][0]
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [0.3, 0.4, 0.5]
    matrix = _object_matrix(obj, pose, [50, 0, 0])
    np.testing.assert_allclose(matrix[:3, :3], pose[:3, :3] @ np.diag([1, 2, 3]))
    np.testing.assert_allclose(matrix[:3, 3], [50.3, 0.4, 0.5])


def test_native_mapping_resolves_unique_instance_proxy_paths() -> None:
    paths = {
        "/World/Envs/env_000000/Objects/cube": 7,
        "/World/Envs/env_000001/Objects/cube": 7,
        "/World/Envs/env_000001/Objects/robot_link": 9,
    }
    mapping = _segmentation_map(
        {
            4: next(iter(paths)) + "/Geometry",
            12: {"primPath": "/World/Envs/env_000001/Objects/robot_link/Geometry"},
        },
        paths,
    )
    assert mapping == {0: 0, 4: 7, 12: 9}
    with pytest.raises(RuntimeError, match="missing"):
        _segmentation_map({}, paths)
    with pytest.raises(RuntimeError, match="identifies no"):
        _segmentation_map({1: "UNLABELLED"}, paths)


def _adapter() -> tuple[IsaacLabScene, list[str]]:
    adapter = IsaacLabScene.__new__(IsaacLabScene)
    events = []
    cuda = SimpleNamespace(
        current_stream=lambda device: "torch_stream",
        synchronize=lambda device: events.append("torch_sync:" + device),
    )

    class TorchProxy:
        def __getattr__(self, name: str) -> Any:
            return cuda if name == "cuda" else getattr(torch, name)

    adapter.torch = TorchProxy()
    adapter.wp = SimpleNamespace(
        stream_from_torch=lambda stream: SimpleNamespace(
            wait_stream=lambda other: events.append("warp_to_torch")
        ),
        get_stream=lambda device: "warp_stream",
        synchronize_device=lambda device: events.append("warp_sync:" + device),
    )
    adapter.device = (
        "cpu"  # Deliberately CPU tensors; the real constructor always selects CUDA.
    )
    adapter.width, adapter.height, adapter.num_envs = 3, 2, 1
    adapter.scene = _scene()
    adapter.camera_specs = copy.deepcopy(adapter.scene["cameras"])
    adapter.modalities = ("rgb", "depth", "seg")
    adapter._frame_dt = 1 / 60
    adapter._accumulation_policy = "renderer_default"
    adapter._segmentation_labels = None
    adapter._segmentation_keys = adapter._segmentation_values = None
    adapter.object_paths = {"/World/Envs/env_000000/Objects/cube": 7}
    adapter.metadata = {}
    simulation = SimpleNamespace(
        render_generation=0,
        get_physics_step_count=lambda: 0,
        is_playing=lambda: False,
        set_setting=lambda *args: None,
        get_setting=lambda name: False,
    )

    def render() -> None:
        simulation.render_generation += 1
        events.append("render")

    simulation.render = render
    adapter.sim = simulation
    adapter._kit_app = SimpleNamespace(
        get_update_number=lambda: simulation.render_generation
    )
    adapter._physx = SimpleNamespace(get_simulation_time=lambda: 0.125)
    adapter._timeline = SimpleNamespace(get_current_time=lambda: 0.125)
    adapter._physics_baseline = adapter._physics_snapshot()
    outputs = {
        "rgb": SimpleNamespace(torch=torch.zeros((2, 2, 3, 4), dtype=torch.uint8)),
        "depth": SimpleNamespace(torch=torch.ones((2, 2, 3, 1), dtype=torch.float32)),
        "instance_id_segmentation_fast": SimpleNamespace(
            torch=torch.full((2, 2, 3, 1), 4, dtype=torch.int32)
        ),
    }
    data = SimpleNamespace(
        output=outputs,
        info={
            "instance_id_segmentation_fast": {
                "idToLabels": {4: next(iter(adapter.object_paths))}
            }
        },
    )

    def update(**kwargs: Any) -> None:
        outputs["rgb"].torch.fill_(simulation.render_generation)
        events.append("camera_update")

    adapter.camera = SimpleNamespace(update=update, data=data)
    return adapter, events


def test_device_delivery_is_fresh_and_performs_no_host_image_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, events = _adapter()

    def forbid_cpu(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("render_device must not call Tensor.cpu")

    monkeypatch.setattr(torch.Tensor, "cpu", forbid_cpu)
    first = adapter.render_device()
    second = adapter.render_device()
    assert events == ["render", "camera_update", "warp_to_torch"] * 2
    assert first["rgb"][0, 0, 0, 0] == 1
    assert second["rgb"][0, 0, 0, 0] == 2
    assert second["seg"].dtype == torch.int32
    assert torch.all(second["seg"] == 7)
    assert (
        adapter.metadata["no_step_evidence"]["before"]
        == adapter.metadata["no_step_evidence"]["after"]
    )
    assert adapter.metadata["physics_steps"] == 0


def test_unmapped_pixels_and_absent_mapping_are_errors() -> None:
    adapter, _ = _adapter()
    adapter.camera.data.output["instance_id_segmentation_fast"].torch[0, 0, 0] = 99
    with pytest.raises(RuntimeError, match="Unmapped native"):
        adapter.render_device()
    adapter, _ = _adapter()
    adapter.camera.data.info = {}
    with pytest.raises(RuntimeError, match="mapping is missing"):
        adapter.render_device()


def test_changed_native_mapping_rebuilds_cached_gpu_map() -> None:
    adapter, _ = _adapter()
    adapter.render_device()
    adapter.camera.data.info["instance_id_segmentation_fast"]["idToLabels"] = {
        8: next(iter(adapter.object_paths))
    }
    adapter.camera.data.output["instance_id_segmentation_fast"].torch.fill_(8)
    assert torch.all(adapter.render_device()["seg"] == 7)


def test_observed_physics_time_advancement_is_rejected() -> None:
    adapter, _ = _adapter()
    adapter._physx.get_simulation_time = Mock(side_effect=[0.125, 0.25])
    with pytest.raises(RuntimeError, match="advanced observed"):
        adapter.render_device()


def test_unavailable_native_physx_clock_keeps_real_counter_and_timeline_evidence() -> (
    None
):
    adapter, _ = _adapter()
    adapter._physx = SimpleNamespace()
    adapter._physics_baseline = adapter._physics_snapshot()
    adapter.render_device()
    evidence = adapter.metadata["no_step_evidence"]
    assert evidence["after"]["physx_simulation_time_s"] is None
    assert "does not expose" in evidence["after"]["physx_time_missing_reason"]
    assert evidence["after"]["isaac_step_count"] == 0
    assert evidence["after"]["timeline_time_s"] == 0.125
    assert evidence["after"]["timeline_playing"] is False
    assert evidence["after"]["kit_player_simulations_enabled"] is False


def test_completion_synchronizes_warp_and_selected_cuda_device() -> None:
    adapter, events = _adapter()
    adapter.render_device()
    events.clear()
    adapter.device = "cuda:7"
    adapter.synchronize()
    assert events == ["warp_sync:cuda:7", "torch_sync:cuda:7"]


def test_host_delivery_performs_one_cpu_transfer_without_numpy_copy() -> None:
    adapter, _ = _adapter()
    values, transfer_counts = {}, {}
    for name in ("rgb", "depth", "seg"):
        array = np.zeros((2, 2))
        transfer_counts[name] = 0

        def cpu(key: str = name, result: Any = array) -> Any:
            transfer_counts[key] += 1
            return SimpleNamespace(numpy=lambda: result)

        values[name] = SimpleNamespace(cpu=cpu, expected=array)
    actual = adapter.to_host(values)
    assert transfer_counts == {"rgb": 1, "depth": 1, "seg": 1}
    assert all(actual[name] is value.expected for name, value in values.items())


def test_apply_state_updates_cached_ops_without_physics_or_pose_readback() -> None:
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd

    adapter, _ = _adapter()
    obj = adapter.scene["objects"][0]
    assets, prototypes, radii = _geometry_prototypes(
        adapter.scene, Path("unused/scene.json")
    )
    stage = Usd.Stage.CreateInMemory()
    adapter._objects = {"cube": obj}
    adapter._object_poses = {"cube": np.eye(4)}
    adapter._geometry_radii = radii
    adapter._env_offsets = [np.zeros(3), np.array([50, 0, 0])]
    adapter._object_ops = {
        "cube": [
            _instance_geometry(
                stage,
                f"/World/env_{index}/cube",
                obj,
                assets,
                prototypes["cube"],
                offset,
            )[1]
            for index, offset in enumerate(adapter._env_offsets)
        ]
    }
    adapter.Gf, adapter.Sdf = Gf, Sdf
    adapter._publish_cameras = Mock()
    pose = np.eye(4)
    pose[:3, 3] = [0.2, 0.3, 0.4]
    frame = {
        "objects": {"cube": pose.tolist()},
        "cameras": {spec["name"]: spec for spec in adapter.camera_specs},
        "frame_id": 5,
        "time_s": 5 / 60,
    }
    adapter.apply_state(frame)
    for operation, offset in zip(
        adapter._object_ops["cube"], adapter._env_offsets, strict=True
    ):
        matrix = np.asarray(operation.Get()).T
        np.testing.assert_allclose(matrix[:3, 3], pose[:3, 3] + offset)
        np.testing.assert_allclose(matrix[:3, :3], np.diag([1, 2, 3]))
    assert adapter.metadata["frame_id"] == 5
    adapter._publish_cameras.assert_called_once()
