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

import pytest
import torch

import embodichain.lab.sim.atomic_actions.core as core_module

from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    Affordance,
    AntipodalAffordance,
    AtomicAction,
    InteractionPoints,
    ObjectSemantics,
)
from embodichain.lab.sim.planners import (
    MotionGenOptions,
    MoveType,
    PlanResult,
    PlanState,
)


class DummyRobot:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.qpos = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        self.fail_ik = False
        self.last_ik_call: dict | None = None
        self.last_fk_call: dict | None = None

    def get_qpos(self) -> torch.Tensor:
        return self.qpos.clone()

    def compute_ik(
        self,
        pose: torch.Tensor,
        qpos_seed: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_ik_call = {
            "pose": pose.clone(),
            "qpos_seed": qpos_seed.clone(),
            "name": name,
        }
        success = torch.tensor([not self.fail_ik], dtype=torch.bool, device=self.device)
        if self.fail_ik:
            return success, torch.zeros_like(qpos_seed)
        return success, qpos_seed + 1.0

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str,
        to_matrix: bool,
    ) -> torch.Tensor:
        self.last_fk_call = {
            "qpos": qpos.clone(),
            "name": name,
            "to_matrix": to_matrix,
        }
        batch_size = qpos.shape[0]
        poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        poses[:, 0, 3] = qpos.sum(dim=-1)
        return poses


class DummyMotionGenerator:
    def __init__(self) -> None:
        self.robot = DummyRobot()
        self.last_target_states: list[PlanState] | None = None
        self.last_options: MotionGenOptions | None = None

    def generate(
        self,
        target_states: list[PlanState],
        options: MotionGenOptions,
    ) -> PlanResult:
        self.last_target_states = target_states
        self.last_options = options
        positions = torch.tensor(
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=torch.float32
        )
        return PlanResult(success=True, positions=positions)


class DummyAtomicAction(AtomicAction):
    def execute(
        self,
        target: torch.Tensor | ObjectSemantics,
        start_qpos: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list[float]]:
        return True, torch.empty(0), []

    def validate(
        self,
        target: torch.Tensor | ObjectSemantics,
        start_qpos: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        return True


class DummyGraspGenerator:
    instances: list["DummyGraspGenerator"] = []

    def __init__(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        cfg=None,
        gripper_collision_cfg=None,
    ) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.cfg = cfg
        self.gripper_collision_cfg = gripper_collision_cfg
        self.device = vertices.device
        self._hit_point_pairs: torch.Tensor | None = None
        self.annotate_calls = 0
        self.get_grasp_pose_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        DummyGraspGenerator.instances.append(self)

    def annotate(self) -> None:
        self.annotate_calls += 1
        self._hit_point_pairs = torch.ones(
            (1, 2, 3), dtype=torch.float32, device=self.device
        )

    def get_grasp_poses(
        self,
        obj_pose: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[bool, torch.Tensor, float]:
        self.get_grasp_pose_calls.append((obj_pose.clone(), approach_direction.clone()))
        if float(obj_pose[0, 3]) > 0.5:
            return False, torch.eye(4, dtype=torch.float32, device=self.device), 0.0

        grasp_pose = obj_pose.clone()
        grasp_pose[2, 3] += 0.02
        return True, grasp_pose, 0.04


class TestAffordanceAndSemantics:
    def test_affordance_mesh_properties_and_custom_config(self) -> None:
        vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        triangles = torch.tensor([[0, 1, 1]], dtype=torch.int64)
        affordance = Affordance(
            object_label="mug",
            geometry={"mesh_vertices": vertices, "mesh_triangles": triangles},
        )

        affordance.set_custom_config("score_threshold", 0.8)

        assert torch.equal(affordance.mesh_vertices, vertices)
        assert torch.equal(affordance.mesh_triangles, triangles)
        assert affordance.get_custom_config("score_threshold") == pytest.approx(0.8)
        assert affordance.get_batch_size() == 1

    def test_affordance_mesh_properties_raise_on_invalid_types(self) -> None:
        affordance = Affordance(
            geometry={"mesh_vertices": [[0.0, 0.0, 0.0]], "mesh_triangles": [[0, 1, 2]]}
        )

        with pytest.raises(TypeError):
            _ = affordance.mesh_vertices

        with pytest.raises(TypeError):
            _ = affordance.mesh_triangles

    def test_interaction_points_helpers(self) -> None:
        interaction_points = InteractionPoints(
            points=torch.tensor(
                [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                dtype=torch.float32,
            ),
            normals=torch.tensor(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ),
            point_types=["push", "touch", "push"],
        )

        push_points = interaction_points.get_points_by_type("push")

        assert push_points is not None
        assert push_points.shape == (2, 3)
        assert interaction_points.get_batch_size() == 3
        assert torch.allclose(
            interaction_points.get_approach_direction(1),
            torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32),
        )

        no_normal_points = InteractionPoints(
            points=torch.zeros((1, 3), dtype=torch.float32)
        )
        assert torch.allclose(
            no_normal_points.get_approach_direction(0),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        )

    def test_object_semantics_binds_label_and_geometry_to_affordance(self) -> None:
        geometry = {"mesh_vertices": torch.zeros((4, 3), dtype=torch.float32)}
        affordance = Affordance()

        semantics = ObjectSemantics(
            label="cup",
            affordance=affordance,
            geometry=geometry,
            properties={"mass": 0.1},
        )

        assert semantics.affordance.object_label == "cup"
        assert semantics.affordance.geometry is geometry
        assert semantics.properties["mass"] == pytest.approx(0.1)

    def test_antipodal_affordance_requires_mesh_geometry(self) -> None:
        affordance = AntipodalAffordance(object_label="mug")

        with pytest.raises(RuntimeError):
            affordance.get_best_grasp_poses(
                torch.eye(4, dtype=torch.float32).unsqueeze(0)
            )

    def test_antipodal_affordance_get_best_grasp_poses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        DummyGraspGenerator.instances.clear()
        monkeypatch.setattr(core_module, "GraspGenerator", DummyGraspGenerator)

        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=torch.float32,
        )
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        affordance = AntipodalAffordance(
            object_label="mug",
            geometry={"mesh_vertices": vertices, "mesh_triangles": triangles},
            custom_config={
                "generator_cfg": object(),
                "gripper_collision_cfg": object(),
            },
        )
        object_poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
        object_poses[1, 0, 3] = 1.0

        is_success, grasp_xpos, open_length = affordance.get_best_grasp_poses(
            object_poses
        )
        generator = DummyGraspGenerator.instances[-1]

        assert generator.annotate_calls == 1
        assert torch.equal(is_success, torch.tensor([True, False], dtype=torch.bool))
        assert torch.allclose(
            grasp_xpos[0, :3, 3],
            torch.tensor([0.0, 0.0, 0.02], dtype=torch.float32),
        )
        assert torch.allclose(grasp_xpos[1], torch.eye(4, dtype=torch.float32))
        assert open_length.tolist() == pytest.approx([0.04, 0.0])


class TestAtomicActionHelpers:
    def setup_method(self) -> None:
        self.motion_generator = DummyMotionGenerator()
        self.action = DummyAtomicAction(
            motion_generator=self.motion_generator,
            cfg=ActionCfg(control_part="arm"),
        )

    def test_ik_solve_uses_control_part_and_seed(self) -> None:
        target_pose = torch.eye(4, dtype=torch.float32)
        qpos_seed = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float32)

        result = self.action._ik_solve(target_pose, qpos_seed)

        assert torch.allclose(result, qpos_seed + 1.0)
        assert self.motion_generator.robot.last_ik_call is not None
        assert self.motion_generator.robot.last_ik_call["name"] == "arm"
        assert torch.allclose(
            self.motion_generator.robot.last_ik_call["qpos_seed"],
            qpos_seed.unsqueeze(0),
        )

    def test_ik_solve_raises_when_solver_fails(self) -> None:
        self.motion_generator.robot.fail_ik = True

        with pytest.raises(RuntimeError):
            self.action._ik_solve(torch.eye(4, dtype=torch.float32))

    def test_fk_compute_handles_single_and_batched_qpos(self) -> None:
        single_qpos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        batched_qpos = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )

        single_pose = self.action._fk_compute(single_qpos)
        batched_pose = self.action._fk_compute(batched_qpos)

        assert single_pose.shape == (4, 4)
        assert batched_pose.shape == (2, 4, 4)
        assert single_pose[0, 3] == pytest.approx(6.0)
        assert torch.allclose(
            batched_pose[:, 0, 3],
            torch.tensor([1.0, 1.0], dtype=torch.float32),
        )
        assert self.motion_generator.robot.last_fk_call is not None
        assert self.motion_generator.robot.last_fk_call["to_matrix"] is True
        assert self.motion_generator.robot.last_fk_call["name"] == "arm"

    def test_apply_offset_updates_translation_in_place_copy(self) -> None:
        pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.2]], dtype=torch.float32)

        result = self.action._apply_offset(pose, offset)

        assert torch.allclose(result[:, :3, 3], offset)
        assert torch.allclose(pose[:, :3, 3], torch.zeros((2, 3), dtype=torch.float32))
