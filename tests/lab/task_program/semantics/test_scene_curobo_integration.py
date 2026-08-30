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

"""Cross-layer CPU tests for registry-backed cuRobo obstacle identity."""

from __future__ import annotations

import torch

from embodichain.lab.sim.planners import (
    CuroboPlanOptions,
    CuroboPlanner,
    CuroboPlannerCfg,
    CuroboWorldCfg,
    MotionGenerator,
)
from embodichain.lab.task_program.semantics import (
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneRegistry,
)


class _RigidObject:
    """Minimal live rigid-object geometry and pose surface."""

    def __init__(self) -> None:
        self.uid = "external_cube"
        self.pose = torch.eye(4).repeat(2, 1, 1)

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        assert to_matrix is True
        return self.pose

    def get_vertices(
        self,
        env_ids: list[int],
        *,
        scale: bool,
    ) -> list[torch.Tensor]:
        assert env_ids == [0]
        assert scale is True
        return [torch.zeros(8, 3)]

    def get_triangles(self, env_ids: list[int]) -> list[torch.Tensor]:
        assert env_ids == [0]
        return [torch.zeros(12, 3, dtype=torch.long)]


class _Simulation:
    """Resolve one rigid object through its simulation-native UID."""

    def __init__(self, rigid_object: _RigidObject) -> None:
        self.rigid_object = rigid_object

    def get_rigid_object(self, uid: str) -> _RigidObject | None:
        return self.rigid_object if uid == self.rigid_object.uid else None


def test_registry_id_remains_authoritative_through_curobo_binding() -> None:
    rigid_object = _RigidObject()
    registry = SceneRegistry.from_simulation(
        _Simulation(rigid_object),  # type: ignore[arg-type]
        rigid_objects={"cube": rigid_object.uid},
        collision_roles={"cube": SceneCollisionRole.DYNAMIC},
        collision_world_mode=SceneCollisionWorldMode.PER_ENV,
    )
    mode = registry.resolve_collision_world_mode(batch_size=2)
    geometry = registry.collision_geometry_by_id()
    world_cfg = CuroboWorldCfg(
        rigid_objects=geometry,  # type: ignore[arg-type]
        obstacle_representation="cuboid",
        dynamic_obstacle_names=list(registry.dynamic_collision_entity_ids),
        multi_env=mode is SceneCollisionWorldMode.PER_ENV,
    )
    planner = object.__new__(CuroboPlanner)
    planner.cfg = CuroboPlannerCfg(robot_uid="unused", world=world_cfg)
    motion_generator = object.__new__(MotionGenerator)
    motion_generator.planner = planner
    provider = registry.make_planning_scene_provider(
        motion_generator,
        batch_size=2,
    )

    snapshot = provider.snapshot(
        timestamp=0.0,
        env_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    obstacle_poses = snapshot.collision_obstacle_poses(
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    bound = motion_generator.bind_collision_world(
        CuroboPlanOptions(),
        obstacle_poses=obstacle_poses,
    )

    assert set(world_cfg.rigid_objects or {}) == {"cube"}
    assert set(obstacle_poses) == {"cube"}
    assert set(bound.dynamic_obstacle_poses or {}) == {"cube"}
    assert rigid_object.uid not in obstacle_poses
