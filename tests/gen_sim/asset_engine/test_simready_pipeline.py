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

"""Shared asset processing retains replaceable geometry and borrowed services."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import trimesh

from embodichain.gen_sim.asset_engine.utils import simready_utils
from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg


@pytest.mark.parametrize("backend_name", ["simready_utils", "full_blender"])
def test_backend_entry_points_share_scaling_without_render_or_model_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    if backend_name == "full_blender":
        # The shared orchestration accepts geometry adapters without loading
        # Blender's native runtime into the unit-test process.
        monkeypatch.setitem(sys.modules, "bpy", ModuleType("bpy"))
        module_name = "_simready_blender_unit_backend"
        spec = importlib.util.spec_from_file_location(
            module_name, Path(simready_utils.__file__).with_name("full_blender.py")
        )
        backend = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, backend)
        spec.loader.exec_module(backend)
    else:
        backend = simready_utils

    normalize = simready_utils.normalize_to_unit_cube
    dimensions = simready_utils.get_aabb_dims
    scale = simready_utils.scale_mesh_uniform_to_dimensions
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    monkeypatch.setattr(backend, "init_pose_simple", lambda source: mesh.copy())
    monkeypatch.setattr(backend, "normalize_to_unit_cube", normalize)
    monkeypatch.setattr(backend, "get_aabb_dims", dimensions)
    monkeypatch.setattr(backend, "scale_mesh_uniform_to_dimensions", scale)
    monkeypatch.setattr(backend, "render_views", lambda *args: [])
    model_runs = []

    def classify(images, *, extra_text, services):
        model_runs.append(services)
        return {"category": 0, "detected_object": "box"}

    def infer_dimensions(**kwargs):
        model_runs.append(kwargs["services"])
        return {"dimensions_m": {"depth": 0.4, "width": 0.2, "height": 0.1}}

    def semantics(**kwargs):
        model_runs.append(kwargs["services"])
        return {"material": "plastic"}

    def export(mesh, name, directory):
        output = directory / f"{backend_name}.obj"
        mesh.export(output)
        return str(output)

    monkeypatch.setattr(backend, "ask_mllm_detect_and_classify", classify)
    monkeypatch.setattr(backend, "ask_llm_dimension", infer_dimensions)
    monkeypatch.setattr(backend, "ask_llm_semantics_info", semantics)
    monkeypatch.setattr(backend, "export_final_mesh", export)

    with GenSimServices(GenSimServicesCfg()) as services:
        result = backend.process_mesh(mesh, "box", out_dir=tmp_path, services=services)
        assert all(run is services for run in model_runs)
        assert services.selections() == {}

    exported = trimesh.load(result["Path"], force="mesh")
    np.testing.assert_allclose(exported.extents, [0.4, 0.2, 0.1])
    assert result["uniform_scale"] == pytest.approx(0.4)
    assert result["semantics_result"] == {"material": "plastic"}
