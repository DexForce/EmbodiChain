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

"""Physical collision extraction for the joint feasibility simulation adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from embodichain.lab.sim.scene.feasibility import (
    CollisionBody,
    UnsupportedFeasibilityError,
)

__all__ = ["collision_bodies_from_simulation"]


def collision_bodies_from_simulation(env: Any) -> tuple[CollisionBody, ...]:
    """Read materialized robot-link and rigid-body physical collision shapes.

    Args:
        env: Prepared B=1 Gym host with full-link Robot kinematics.
    Returns:
        Link-local physical shapes with IDs ``robot_uid/link/index`` and
        ``object_uid/index``. Render geometry is never substituted.

    Unsupported physical geometry APIs/scales and shapes fail explicitly.
    The caller owns permanent structural and phase-specific contact permissions.
    """
    robot = env.robot
    if env.sim.get_articulation_uid_list():
        raise UnsupportedFeasibilityError(
            "Non-robot articulations require an explicit complete collision adapter"
        )
    if not np.allclose(robot.cfg.body_scale, (1.0, 1.0, 1.0)):
        raise UnsupportedFeasibilityError(
            "Scaled robot collision adaptation requires an explicit geometry provider"
        )

    def convert(shape, collision_id, frame_uid, *, is_robot=False):
        kind = shape.shape_type.name.lower()
        kwargs = {}
        if kind == "box":
            dimensions = tuple((shape.half_extents.cpu().numpy() * 2).tolist())
        elif kind == "sphere":
            dimensions = (float(shape.radius),)
        elif kind in ("capsule", "cylinder"):
            dimensions = (float(shape.radius), 2 * float(shape.half_height))
        elif kind == "convex" and (
            shape.vertices is not None
            and shape.triangles is not None
            and shape.vertices.numel()
            and shape.triangles.numel()
        ):
            dimensions = ()
            kwargs = {
                "vertices": tuple(map(tuple, shape.vertices.cpu().numpy())),
                "triangles": tuple(
                    tuple(int(i) for i in f)
                    for f in shape.triangles.cpu().numpy().reshape(-1, 3)
                ),
            }
        else:
            raise UnsupportedFeasibilityError(
                f"Native physical shape {collision_id}: {kind} is unsupported"
            )
        return CollisionBody(
            collision_id,
            frame_uid,
            kind,
            dimensions,
            shape.local_pose.cpu().tolist(),
            robot=is_robot,
            **kwargs,
        )

    bodies = []
    for link_name in robot.link_names:
        try:
            shapes = robot.get_link_collision_shapes(link_name)
        except RuntimeError as error:
            raise UnsupportedFeasibilityError(str(error)) from error
        frame = f"{robot.uid}/{link_name}"
        bodies.extend(
            convert(shape, f"{frame}/{index}", frame, is_robot=True)
            for index, shape in enumerate(shapes)
        )
    if not bodies:
        raise UnsupportedFeasibilityError(
            "Robot has no accessible physical collision geometry"
        )
    for uid in env.sim.get_rigid_object_uid_list():
        obj = env.sim.get_rigid_object(uid)
        bodies.extend(
            convert(shape, f"{uid}/{index}", uid)
            for index, shape in enumerate(obj.get_collision_shapes())
        )
    if getattr(env.sim, "_default_plane", None) is not None:
        bodies.append(
            CollisionBody(
                "default_plane", "default_plane", "halfspace", (0.0, 0.0, 1.0, 0.0)
            )
        )
    return tuple(bodies)
