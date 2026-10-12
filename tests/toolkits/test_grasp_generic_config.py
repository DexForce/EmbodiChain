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

"""Agent-selected assembly parameters and compact job configuration contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from scripts.tools.assemble._geometry import load_mesh
from scripts.tools.assemble._json_io import read_json, write_json
from scripts.tools.grasp_assemble._config import (
    action_schema,
    load_assembly,
    load_config,
    resolve_scene,
    resolve_task_plan,
)
from scripts.tools.grasp_assemble._layout import resolve_layout


@pytest.fixture
def generic_job(tmp_path: Path) -> Path:
    assets = {}
    for role in ("base", "assemble"):
        mesh = trimesh.convex.convex_hull(
            np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3], [0.1, 0.2, 0.3]])
        )
        mesh.apply_translation([0, 0, -0.03])
        mesh_path = tmp_path / f"{role}.obj"
        mesh.export(mesh_path)
        assets[role] = {
            "path": str(mesh_path),
            "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        }
    write_json(
        tmp_path / "assembly.json",
        {
            "schema": "codex-assemble/v1",
            "success": True,
            "validation": {"accepted": True},
            "assets": assets,
            "T_base_assemble": np.eye(4).tolist(),
        },
    )
    path = tmp_path / "job.json"
    write_json(path, {"assembly_result": "assembly.json", "output_dir": "out"})
    return path


@pytest.fixture
def task_plan() -> dict:
    return {
        "assemble_initial_rpy_degrees": [35, -20, 40],
        "clearance": 0.05,
        "pre_grasp_distance": 0.06,
        "insertion_direction_base": [0, 2, 0],
        "insertion_distance": 0.09,
        "retract_direction_base": [0, -3, 0],
        "retract_distance": 0.07,
        "reason": "Approach the support from its open side.",
    }


def test_minimal_job_resolves_grounded_seed_poses(generic_job: Path) -> None:
    config = load_config(generic_job)
    assembly = load_assembly(config)
    resolved = resolve_scene(config, assembly)
    assert config["layout"]["mode"] == "optimize"
    assert config["codex"]["max_turns"] == 12
    assert config["execution"]["rotation_tolerance_degrees"] == 10
    assert "T_world_base" not in config
    for key, xy in (
        ("T_world_base", [-0.55, -0.1]),
        ("T_world_assemble_initial", [-0.35, -0.3]),
    ):
        transform = np.asarray(resolved[key])
        np.testing.assert_allclose(transform[:3, :3], np.eye(3))
        np.testing.assert_allclose(transform[:2, 3], xy)
        assert transform[2, 3] == pytest.approx(0.031)


def test_agent_orientation_grounds_exact_vertices_and_preserves_scene(
    generic_job: Path, task_plan: dict
) -> None:
    loaded = load_config(generic_job)
    assembly = load_assembly(loaded)
    config = resolve_scene(loaded, assembly)
    before = deepcopy(config)
    result = resolve_task_plan(config, assembly, task_plan)
    transform = np.asarray(result["T_world_assemble_initial"])
    vertices = load_mesh(Path(assembly["assets"]["assemble"]["path"])).vertices
    actual_z = (vertices @ transform[:3, :3].T + transform[:3, 3])[:, 2]
    assert actual_z.min() == pytest.approx(0.001)
    np.testing.assert_allclose(
        transform[:3, :3],
        Rotation.from_euler("xyz", [35, -20, 40], degrees=True).as_matrix(),
    )
    assert result["task_plan"]["insertion_direction_base"] == [0, 1, 0]
    assert result["task_plan"]["retract_direction_base"] == [0, -1, 0]
    assert result["motion"]["clearance"] == 0.05
    assert result["motion"]["pre_grasp_distance"] == 0.06
    assert result["motion"]["retract_distance"] == 0.07
    assert config == before
    assert task_plan["insertion_direction_base"] == [0, 2, 0]


def test_explicit_pose_and_motion_override_agent_proposal(
    generic_job: Path, task_plan: dict
) -> None:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    pose[:3, 3] = [-0.4, -0.3, 0.04]
    write_json(
        generic_job,
        read_json(generic_job)
        | {
            "T_world_base": np.eye(4).tolist(),
            "T_world_assemble_initial": pose.tolist(),
            "motion": {"clearance": 0.2, "pre_grasp_distance": 0.12},
        },
    )
    config = load_config(generic_job)
    assembly = load_assembly(config)
    config = resolve_scene(config, assembly)
    result = resolve_task_plan(config, assembly, task_plan)
    assert result["layout"]["mode"] == "fixed"
    assert result["execution"]["rotation_tolerance_degrees"] == 20
    np.testing.assert_allclose(result["T_world_assemble_initial"], pose)
    np.testing.assert_allclose(
        result["task_plan"]["assemble_initial_rpy_degrees"], [0, 0, 30], atol=1e-12
    )
    assert result["motion"]["clearance"] == 0.2
    assert result["task_plan"]["clearance"] == 0.2
    assert result["motion"]["pre_grasp_distance"] == 0.12
    assert result["motion"]["retract_distance"] == 0.07


def test_layout_yaw_keeps_oriented_object_grounded(
    generic_job: Path, task_plan: dict
) -> None:
    config = load_config(generic_job)
    assembly = load_assembly(config)
    config = resolve_task_plan(resolve_scene(config, assembly), assembly, task_plan)
    layout = {
        "base_xy": [-0.6, 0.1],
        "assemble_xy": [-0.4, -0.3],
        "base_yaw_offset_degrees": 20,
        "assemble_yaw_offset_degrees": -45,
    }
    resolved = resolve_layout(config, layout)
    transform = np.asarray(resolved["T_world_assemble_initial"])
    vertices = load_mesh(Path(assembly["assets"]["assemble"]["path"])).vertices
    assert (vertices @ transform[2, :3] + transform[2, 3]).min() == pytest.approx(0.001)


@pytest.mark.parametrize(
    "change",
    [
        {"insertion_direction_base": [0, 0, 0]},
        {"insertion_direction_base": [True, 0, 0]},
        {"retract_direction_base": [0, float("nan"), 1]},
        {"assemble_initial_rpy_degrees": [0, 0]},
        {"assemble_initial_rpy_degrees": [0, "90", 0]},
        {"clearance": 0.019},
        {"pre_grasp_distance": 0.301},
        {"insertion_distance": float("inf")},
        {"insertion_distance": True},
        {"retract_distance": -1},
        {"reason": " "},
        {"unexpected": 1},
    ],
)
def test_task_plan_rejects_invalid_parameters(
    generic_job: Path, task_plan: dict, change: dict
) -> None:
    config = load_config(generic_job)
    assembly = load_assembly(config)
    with pytest.raises(ValueError, match="task_plan"):
        resolve_task_plan(resolve_scene(config, assembly), assembly, task_plan | change)


def test_task_plan_none_preserves_archived_config() -> None:
    saved = {"motion": {"clearance": 0.12}, "T_world_base": np.eye(4).tolist()}
    result = resolve_task_plan(saved, {}, None)
    assert result == saved
    result["motion"]["clearance"] = 0.2
    assert saved["motion"]["clearance"] == 0.12


def test_action_schema_exposes_required_task_plan(task_plan: dict) -> None:
    schema = action_schema()
    assert "task_plan" in schema["required"]
    contract = schema["properties"]["task_plan"]["anyOf"][0]
    assert set(contract["required"]) == set(task_plan)
    assert contract["additionalProperties"] is False


def test_assembly_path_only_config_uses_repository_output_directory(
    generic_job: Path,
) -> None:
    write_json(generic_job, {"assembly_result": "assembly.json"})
    result = load_config(generic_job)
    repository = Path(__file__).resolve().parents[2]
    assert result["output_dir"] == str(
        repository / "outputs" / "grasp_assemble" / "job"
    )
    assert result["assembly_result"] == str(generic_job.parent / "assembly.json")
