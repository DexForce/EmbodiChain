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

"""Headless static visualization of the grasp poses selected by E5."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import AntipodalAffordance

from .models import ActionOutcome

__all__ = [
    "CoordinatedGraspPoseScene",
    "grasp_pose_image_path",
    "render_coordinated_grasp_pose_png",
    "selected_coordinated_grasp_scene",
]


_SAFE_NAME = re.compile(r"[^0-9A-Za-z._-]+")
_IMAGE_WIDTH = 960
_IMAGE_HEIGHT = 720


@dataclass(frozen=True, slots=True)
class CoordinatedGraspPoseScene:
    """Detached env-zero inputs for one selected coordinated grasp image."""

    object_label: str
    mesh_vertices: torch.Tensor
    mesh_triangles: torch.Tensor
    object_pose: torch.Tensor
    left_grasp_pose: torch.Tensor
    right_grasp_pose: torch.Tensor


def _pose_row(value: torch.Tensor, env_id: int, *, name: str) -> torch.Tensor:
    pose = torch.as_tensor(value, dtype=torch.float32)
    if pose.shape == (4, 4):
        if env_id != 0:
            raise ValueError(f"{name} has no row for environment {env_id}.")
        result = pose
    elif pose.ndim == 3 and pose.shape[1:] == (4, 4) and pose.shape[0] > env_id:
        result = pose[env_id]
    else:
        raise ValueError(f"{name} must contain a (4, 4) pose for environment {env_id}.")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return result.detach().cpu().clone()


def selected_coordinated_grasp_scene(
    outcome: ActionOutcome,
    *,
    left_control_part: str,
    right_control_part: str,
    env_id: int = 0,
) -> CoordinatedGraspPoseScene | None:
    """Extract the exact valid grasp pair selected by E5 for one environment."""
    if not isinstance(outcome, ActionOutcome):
        raise TypeError("outcome must be an ActionOutcome.")
    if type(env_id) is not int or env_id < 0:
        raise ValueError("env_id must be a non-negative integer.")
    success = torch.as_tensor(outcome.success, dtype=torch.bool).reshape(-1)
    if success.numel() <= env_id or not bool(success[env_id].item()):
        return None
    object_pose_value = outcome.grounded.object_pose
    if object_pose_value is None:
        raise ValueError("Selected E5 outcome does not retain its live object pose.")
    object_pose = _pose_row(object_pose_value, env_id, name="live object pose")
    left_held = outcome.next_state.get_held_object(left_control_part)
    right_held = outcome.next_state.get_held_object(right_control_part)
    if left_held is None or right_held is None:
        raise ValueError("Selected E5 outcome does not retain both held-object states.")
    if left_held.semantics.entity_id != right_held.semantics.entity_id:
        raise ValueError("Selected E5 grasps must refer to the same object.")
    affordance = left_held.semantics.affordance
    if not isinstance(affordance, AntipodalAffordance):
        raise TypeError("Selected E5 object must retain an AntipodalAffordance.")
    left_relation = _pose_row(
        left_held.object_to_eef,
        env_id,
        name="left object-to-EEF pose",
    )
    right_relation = _pose_row(
        right_held.object_to_eef,
        env_id,
        name="right object-to-EEF pose",
    )
    return CoordinatedGraspPoseScene(
        object_label=left_held.semantics.label,
        mesh_vertices=torch.as_tensor(
            affordance.mesh_vertices,
            dtype=torch.float32,
        )
        .detach()
        .cpu()
        .clone(),
        mesh_triangles=torch.as_tensor(
            affordance.mesh_triangles,
            dtype=torch.int64,
        )
        .detach()
        .cpu()
        .clone(),
        object_pose=object_pose,
        left_grasp_pose=object_pose @ left_relation,
        right_grasp_pose=object_pose @ right_relation,
    )


def grasp_pose_image_path(output_root: str | Path, step_id: str) -> Path:
    """Return the stable env-zero image path for one E5 semantic step."""
    safe_step = _SAFE_NAME.sub("_", str(step_id)).strip("._")
    if not safe_step:
        raise ValueError("step_id must contain a usable path component.")
    return Path(output_root) / "env_0000" / "grasp_poses" / f"{safe_step}.png"


def _material(rendering: object, color: tuple[float, float, float, float]) -> object:
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = color
    return material


def render_coordinated_grasp_pose_png(
    scene: CoordinatedGraspPoseScene,
    output_path: str | Path,
) -> Path:
    """Render one object mesh and its selected left/right grasp frames to PNG."""
    if not isinstance(scene, CoordinatedGraspPoseScene):
        raise TypeError("scene must be a CoordinatedGraspPoseScene.")
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".png":
        raise ValueError("Grasp pose visualization output must use a .png suffix.")

    import open3d as o3d

    renderer = o3d.visualization.rendering.OffscreenRenderer(
        _IMAGE_WIDTH,
        _IMAGE_HEIGHT,
    )
    renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(scene.mesh_vertices.numpy()),
        triangles=o3d.utility.Vector3iVector(scene.mesh_triangles.numpy()),
    )
    mesh.compute_vertex_normals()
    mesh.transform(scene.object_pose.numpy())
    renderer.scene.add_geometry(
        "object",
        mesh,
        _material(o3d.visualization.rendering, (0.25, 0.62, 0.34, 1.0)),
    )

    world_vertices = np.asarray(mesh.vertices)
    mesh_extent = float(np.ptp(world_vertices, axis=0).max())
    frame_size = max(0.035, min(0.12, 0.35 * mesh_extent))
    center_radius = max(0.006, 0.075 * frame_size)
    for name, pose, color in (
        ("left", scene.left_grasp_pose, (0.82, 0.16, 0.58, 1.0)),
        ("right", scene.right_grasp_pose, (0.05, 0.55, 0.80, 1.0)),
    ):
        pose_np = pose.numpy()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(pose_np)
        renderer.scene.add_geometry(
            f"{name}_frame",
            frame,
            _material(o3d.visualization.rendering, (1.0, 1.0, 1.0, 1.0)),
        )
        center = o3d.geometry.TriangleMesh.create_sphere(radius=center_radius)
        center.compute_vertex_normals()
        center.translate(pose_np[:3, 3])
        renderer.scene.add_geometry(
            f"{name}_center",
            center,
            _material(o3d.visualization.rendering, color),
        )

    bounds = renderer.scene.bounding_box
    center = np.asarray(bounds.get_center(), dtype=np.float64)
    extent = max(float(np.asarray(bounds.get_extent()).max()), 0.10)
    view = np.array([1.4, -1.8, 1.2], dtype=np.float64)
    view /= np.linalg.norm(view)
    renderer.setup_camera(
        50.0,
        center,
        center + view * (2.2 * extent),
        np.array([0.0, 0.0, 1.0]),
    )
    image = renderer.render_to_image()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".png",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        if not o3d.io.write_image(temporary_path.as_posix(), image):
            raise OSError(f"Open3D could not write {temporary_path}.")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
