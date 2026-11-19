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


def collision_bodies_from_simulation(
    env: Any, *, env_id: int = 0
) -> tuple[CollisionBody, ...]:
    """Read materialized robot-link and rigid-body physical collision shapes.

    Args:
        env: Prepared Gym host with full-link Robot kinematics.
        env_id: Physical arena whose collision resources are inspected.
    Returns:
        Link-local physical shapes with IDs ``robot_uid/link/index`` and
        ``object_uid/index``. Render geometry is never substituted.

    Unsupported physical geometry APIs/scales and shapes fail explicitly.
    The caller owns permanent structural and phase-specific contact permissions.
    """
    return _physical_collision_bodies(env, env_id=env_id)


def _physical_collision_bodies(
    env: Any, *, env_id: int, layout_mesh_boxes: list[str] | None = None
) -> tuple[CollisionBody, ...]:
    """Optionally bound rigid triangle meshes for an isolated layout proposal query."""
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
        local_pose = shape.local_pose.cpu().numpy()
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
        elif kind == "mesh" and not is_robot and layout_mesh_boxes is not None:
            if (
                shape.vertices is None
                or shape.triangles is None
                or not shape.triangles.numel()
            ):
                raise UnsupportedFeasibilityError(
                    f"Missing physical mesh geometry for {collision_id}"
                )
            points = shape.vertices.cpu().numpy()
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or not len(points)
                or not np.isfinite(points).all()
            ):
                raise UnsupportedFeasibilityError(
                    f"Invalid physical mesh vertices for {collision_id}"
                )
            faces = shape.triangles.cpu().numpy()
            if (
                faces.dtype.kind not in "iu"
                or faces.size % 3
                or faces.min() < 0
                or faces.max() >= len(points)
            ):
                raise UnsupportedFeasibilityError(
                    f"Invalid physical mesh triangles for {collision_id}"
                )
            low = points.min(axis=0).astype(float)
            high = points.max(axis=0).astype(float)
            # Descriptors already include body scale. Keep the box in the shape
            # axes and transform its center to the owner frame exactly once.
            local_pose = local_pose.astype(float, copy=True)
            local_pose[:3, 3] += local_pose[:3, :3] @ ((low + high) / 2)
            dimensions = tuple(np.maximum(high - low, 1e-6))
            kind = "box"
            layout_mesh_boxes.append(collision_id)
        else:
            raise UnsupportedFeasibilityError(
                f"Native physical shape {collision_id}: {kind} is unsupported"
            )
        return CollisionBody(
            collision_id,
            frame_uid,
            kind,
            dimensions,
            local_pose.tolist(),
            robot=is_robot,
            **kwargs,
        )

    bodies = []
    for link_name in robot.link_names:
        try:
            shapes = robot.get_link_collision_shapes(
                link_name, **({"env_id": env_id} if env_id else {})
            )
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
            for index, shape in enumerate(
                obj.get_collision_shapes(**({"env_id": env_id} if env_id else {}))
            )
        )
    if getattr(env.sim, "_default_plane", None) is not None:
        bodies.append(
            CollisionBody(
                "default_plane", "default_plane", "halfspace", (0.0, 0.0, 1.0, 0.0)
            )
        )
    return tuple(bodies)
