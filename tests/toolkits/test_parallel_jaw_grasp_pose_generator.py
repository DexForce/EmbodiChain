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

"""Tests for the standalone parallel-jaw grasp-pose service."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from embodichain.toolkits.graspkit import pg_grasp
from embodichain.toolkits.graspkit import (
    GraspPoseGenerator,
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    AntipodalSamplerCfg,
    GraspAnnotationCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp._antipodal_backend import (
    _AntipodalMeshBackend,
)
from embodichain.toolkits.graspkit.pg_grasp import pose_generator as module


class _Backend:
    """Small stand-in for the private single-mesh backend."""

    instances: ClassVar[list[_Backend]] = []

    def __init__(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        **options: Any,
    ) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.options = options
        self.device = vertices.device
        self.is_prepared = True
        self._antipodal_pairs = torch.ones(1, 2, 3)
        self.annotate_calls = 0
        self.best_directions: list[torch.Tensor] = []
        type(self).instances.append(self)

    @property
    def antipodal_pairs(self) -> torch.Tensor:
        return self._antipodal_pairs.clone()

    def annotate(self) -> None:
        self.annotate_calls += 1

    def get_grasp_poses(
        self,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[bool, torch.Tensor, float]:
        self.best_directions.append(approach_direction)
        result = object_pose.clone()
        result[0, 3] += 0.1
        return True, result, 0.04

    def get_valid_grasp_poses(self, **_: object) -> tuple[object, ...]:
        return False, torch.eye(4), 0.0, torch.zeros(1)

    def get_dual_arm_valid_grasp_poses(self, **_: object) -> None:
        return None


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> type[_Backend]:
    """Replace the heavyweight antipodal backend for focused service tests."""
    _Backend.instances.clear()
    monkeypatch.setattr(module, "_AntipodalMeshBackend", _Backend)
    return _Backend


def _geometry() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[0, 1, 2]], dtype=torch.long),
    )


def _generator(*, force_refresh: bool = False) -> AntipodalGraspPoseGenerator:
    return AntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(
            model_id="concrete_test_eef",
            min_opening_width=0.004,
            max_opening_width=0.12,
            finger_length=0.09,
            finger_width=0.025,
            finger_thickness=0.008,
            palm_depth=0.07,
        ),
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(
            sample_count=321,
            max_candidates=7,
        ),
        collision_cfg=ParallelJawGraspCollisionCfg(
            point_sample_density=0.02,
            max_decomposition_hulls=5,
            opening_margin=0.003,
            filter_ground_collision=False,
        ),
        annotation_cfg=GraspAnnotationCfg(force_refresh=force_refresh),
    )


def test_generic_hierarchy_contains_no_concrete_eef_name() -> None:
    generator = _generator()

    assert isinstance(generator, GraspPoseGenerator)
    assert isinstance(generator, ParallelJawGraspPoseGenerator)
    assert generator.gripper_model.model_id == "concrete_test_eef"
    assert "pgi" not in type(generator).__name__.lower()
    assert "pgi" not in type(generator.gripper_model).__name__.lower()


def test_package_exposes_only_the_unified_generator_api() -> None:
    assert "AntipodalGraspPoseGenerator" in pg_grasp.__all__
    assert "GraspPoseGenerator" in pg_grasp.__all__
    assert "ParallelJawGripperModelCfg" in pg_grasp.__all__
    assert "GraspGenerator" not in pg_grasp.__all__
    assert "GraspGeneratorCfg" not in pg_grasp.__all__


def test_graspkit_does_not_import_lab() -> None:
    """The standalone toolkit must not import EmbodiChain's lab layer."""
    graspkit_root = (
        Path(__file__).resolve().parents[2] / "embodichain" / "toolkits" / "graspkit"
    )
    forbidden_imports: list[str] = []
    for source_path in graspkit_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module_names = [node.module]
            else:
                continue
            forbidden_imports.extend(
                f"{source_path.relative_to(graspkit_root)}:{module_name}"
                for module_name in module_names
                if module_name == "embodichain.lab"
                or module_name.startswith("embodichain.lab.")
            )

    assert forbidden_imports == []


def test_prepare_mesh_reuses_backend_and_returns_owned_pairs(
    backend: type[_Backend],
) -> None:
    vertices, triangles = _geometry()
    generator = _generator()

    first = generator.prepare_mesh(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
    )
    first.zero_()
    second = generator.prepare_mesh(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
    )

    assert len(backend.instances) == 1
    assert torch.equal(second, torch.ones(1, 2, 3))


def test_direct_best_grasp_configures_and_reuses_private_mesh_backend(
    backend: type[_Backend],
) -> None:
    vertices, triangles = _geometry()
    generator = _generator()
    object_poses = torch.eye(4).repeat(2, 1, 1)

    success, grasp_poses, opening_widths = generator.get_best_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        obj_poses=object_poses,
        approach_direction=torch.tensor([[0, 0, -2], [0, -3, 0]]),
    )
    generator.get_best_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        obj_poses=object_poses,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )

    assert success.tolist() == [True, True]
    assert torch.allclose(grasp_poses[:, 0, 3], torch.full((2,), 0.1))
    assert torch.allclose(opening_widths, torch.full((2,), 0.04))
    assert len(backend.instances) == 1
    configured = backend.instances[0]
    sampler_cfg = configured.options["sampler_cfg"]
    collision_cfg = configured.options["collision_cfg"]
    assert sampler_cfg.n_sample == 321
    assert sampler_cfg.min_length == 0.004
    assert sampler_cfg.max_length == 0.12
    assert configured.options["max_candidates"] == 7
    assert configured.options["filter_ground_collision"] is False
    assert collision_cfg.finger_length == 0.09
    assert collision_cfg.open_check_margin == 0.003
    assert torch.allclose(
        torch.stack(configured.best_directions[:2]),
        torch.tensor([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]),
    )


def test_failed_candidates_receive_infinite_cost_and_refresh_is_service_owned(
    backend: type[_Backend],
) -> None:
    vertices, triangles = _geometry()
    generator = _generator(force_refresh=True)

    results = generator.get_valid_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        obj_poses=torch.eye(4).unsqueeze(0),
        approach_direction=torch.tensor([0, 0, -1]),
    )

    poses, costs = results[0]
    assert poses.shape == (1, 4, 4)
    assert costs.shape == (1,)
    assert torch.isinf(costs).all()
    assert backend.instances[0].annotate_calls == 1


def test_valid_candidates_apply_object_aware_cost_before_return(
    backend: type[_Backend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices, triangles = _geometry()
    generator = _generator()
    generator.prepare_mesh(mesh_vertices=vertices, mesh_triangles=triangles)
    object_pose = torch.eye(4)
    object_pose[0, 3] = 0.25
    grasp_poses = torch.eye(4).repeat(2, 1, 1)

    def get_valid_grasp_poses(**kwargs: object) -> tuple[object, ...]:
        callback = kwargs["pose_cost_fn"]
        assert callable(callback)
        costs = callback(grasp_poses, torch.tensor([0.2, 0.4]))
        return True, grasp_poses, torch.tensor([0.02, 0.03]), costs

    monkeypatch.setattr(
        backend.instances[0],
        "get_valid_grasp_poses",
        get_valid_grasp_poses,
    )

    results = generator.get_valid_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        obj_poses=object_pose.unsqueeze(0),
        approach_direction=torch.tensor([0, 0, -1]),
        pose_cost_fn=lambda obj, _grasps, current: current + obj[0, 3],
    )

    assert torch.allclose(results[0][1], torch.tensor([0.45, 0.65]))


def test_rejects_collision_margin_outside_gripper_opening() -> None:
    with pytest.raises(ValueError, match="opening_margin"):
        AntipodalGraspPoseGenerator(
            ParallelJawGripperModelCfg(max_opening_width=0.05),
            collision_cfg=ParallelJawGraspCollisionCfg(opening_margin=0.05),
        )


def test_backend_cache_key_includes_sampling_and_annotation_policy() -> None:
    vertices, triangles = _geometry()
    generator = object.__new__(_AntipodalMeshBackend)
    generator._sampler_cfg = AntipodalSamplerCfg()
    generator._interactive_annotation = False
    generator._use_largest_connected_component = False
    baseline = generator._get_cache_dir(vertices, triangles)

    generator._sampler_cfg = generator._sampler_cfg.replace(n_sample=123)
    changed_sampler = generator._get_cache_dir(vertices, triangles)
    generator._interactive_annotation = True
    changed_selection = generator._get_cache_dir(vertices, triangles)
    generator._interactive_annotation = False
    generator._use_largest_connected_component = True
    changed_component_policy = generator._get_cache_dir(vertices, triangles)

    assert (
        len(
            {
                baseline,
                changed_sampler,
                changed_selection,
                changed_component_policy,
            }
        )
        == 4
    )
