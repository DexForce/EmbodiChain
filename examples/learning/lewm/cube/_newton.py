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

"""Lower the Cube's closed gripper linkage on the DexSim 0.6 Newton builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dexsim.spawn.adapters.newton_adapter import NewtonSpawnAdapter

if TYPE_CHECKING:
    from dexsim import Arena
    from dexsim.engine.newton_physics.newton_backend import NewtonBackend
    from dexsim.scene import Scene
    from dexsim.spawn import NewtonEqualityDesc, SceneDesc

__all__ = []


class _CubeNewtonSpawnAdapter(NewtonSpawnAdapter):
    """Author the two CONNECT rows before the per-scene DexUni solver is built."""

    def __init__(self, equalities: tuple[NewtonEqualityDesc, ...]) -> None:
        self.equalities = equalities

    def _build_scene(
        self,
        result: Scene,
        arena: Arena,
        arena_index: int,
        scene: SceneDesc,
        path_prefix: str,
        backend: NewtonBackend,
        world_index: int,
        **kwargs: Any,
    ) -> None:
        super()._build_scene(
            result,
            arena,
            arena_index,
            scene,
            path_prefix,
            backend,
            world_index,
            **kwargs,
        )
        if not any(art.name == "ogbench_robot" for art in scene.articulations):
            return

        import warp as wp

        robot = result.get_articulation(f"{path_prefix}ogbench_robot")._physics_binding
        body_ids = {link.name: link.body_id for link in robot.links}
        joint_ids = {joint.name: joint.joint_id for joint in robot.joints}
        for equality in self.equalities:
            row = {
                "mujoco:equality_constraint_label": equality.name,
                "mujoco:equality_constraint_type": equality.equality_type,
                "mujoco:equality_constraint_enabled": equality.enabled,
                "mujoco:eq_solref": equality.solref,
                "mujoco:eq_solimp": equality.solimp,
            }
            if equality.equality_type == 0:
                row.update(
                    {
                        "mujoco:equality_constraint_objtype": 1,
                        "mujoco:equality_constraint_body1": body_ids[equality.object1],
                        "mujoco:equality_constraint_body2": body_ids[equality.object2],
                        "mujoco:equality_constraint_anchor": wp.vec3(
                            *equality.data[:3]
                        ),
                    }
                )
            elif equality.equality_type == 2:
                row.update(
                    {
                        "mujoco:equality_constraint_objtype": 3,
                        "mujoco:equality_constraint_joint1": joint_ids[
                            equality.object1
                        ],
                        "mujoco:equality_constraint_joint2": joint_ids[
                            equality.object2
                        ],
                        "mujoco:equality_constraint_polycoef": list(equality.data[:5]),
                    }
                )
            else:
                raise ValueError(
                    f"Unexpected Cube equality type {equality.equality_type}"
                )
            backend.builder.add_custom_values(**row)
