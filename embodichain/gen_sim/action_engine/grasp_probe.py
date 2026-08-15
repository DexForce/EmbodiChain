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

"""Capability-specific static probe for coordinated antipodal grasps."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_engine.config import default_runtime_policy
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPolicy,
    GraspCandidateProvider,
)

__all__ = ["probe_coordinated_grasp_policy"]


def probe_coordinated_grasp_policy(
    action_graph: Mapping[str, Any],
    static_scene_manifest: Mapping[str, Any],
    *,
    robot_profile: str,
) -> list[dict[str, Any]]:
    """Probe whether the current finite policy finds dual candidates.

    This is deliberately not a proof of physical infeasibility. An empty
    finite candidate set is reported as ``grasp_policy_unsatisfied`` so callers
    can stop generation without converting that result into a scene
    contradiction.
    """
    object_by_uid = {
        str(item.get("uid")): item
        for item in static_scene_manifest.get("objects", ())
        if isinstance(item, Mapping) and item.get("uid")
    }
    targets = {
        str(node.get("object_uid"))
        for node in action_graph.get("nodes", ())
        if isinstance(node, Mapping)
        and node.get("atomic_action") == "CoordinatedPickment"
        and node.get("object_uid")
    }
    if not targets:
        return []

    runtime_policy = default_runtime_policy(robot_profile)
    grasp = runtime_policy.grasp
    sampling_policy = AntipodalGraspPolicy(
        n_sample=int(grasp["antipodal_n_sample"]),
        max_angle=float(grasp["antipodal_max_angle"]),
        min_contact_span=float(grasp["min_contact_span"]),
        max_contact_span=(
            None
            if grasp["max_contact_span"] is None
            else float(grasp["max_contact_span"])
        ),
        max_deviation_angle=float(grasp["max_deviation_angle"]),
        n_deviated_approach_directions=int(
            grasp["n_deviated_approach_directions"]
        ),
        n_top_grasps=int(grasp["n_top_grasps"]),
        viser_port=int(grasp["viser_port"]),
        max_decomposition_hulls=int(grasp["max_decomposition_hulls"]),
        filter_support_collision=bool(grasp["filter_support_collision"]),
    )
    middle_empty_ratio = float(
        runtime_policy.motion_defaults["CoordinatedPickment"].get(
            "middle_empty_ratio", 0.4
        )
    )
    return [
        _probe_object(
            uid,
            object_by_uid.get(uid),
            eef_profile=runtime_policy.end_effector_profile,
            sampling_policy=sampling_policy,
            middle_empty_ratio=middle_empty_ratio,
        )
        for uid in sorted(targets)
    ]


def _probe_object(
    uid: str,
    manifest_object: Any,
    *,
    eef_profile: Any,
    sampling_policy: AntipodalGraspPolicy,
    middle_empty_ratio: float,
) -> dict[str, Any]:
    subject = f"CoordinatedPickment.object:{uid}"
    try:
        if not isinstance(manifest_object, Mapping):
            raise ValueError("Static scene object is missing.")
        vertices, triangles, object_pose = _load_probe_mesh(manifest_object)
        import warp as wp

        wp.init()
        provider = GraspCandidateProvider(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            eef_profile=eef_profile,
            sampling_policy=sampling_policy,
        )
        result = provider.get_dual_arm_valid_grasp_poses(
            object_pose=object_pose,
            approach_direction=torch.tensor([0.0, 0.0, -1.0]),
            left_to_right_arm_direction=torch.tensor([0.0, 1.0, 0.0]),
            middle_empty_ratio=middle_empty_ratio,
            approach_attempt_id=0,
        )
        left = bool(result is not None and result["left"]["is_success"])
        right = bool(result is not None and result["right"]["is_success"])
        satisfied = left and right
        return {
            "kind": "grasp_policy_probe",
            "subject": subject,
            "status": "proven" if satisfied else "runtime_probe",
            "reason": (
                "Current EEF and finite grasp policy found candidates on both sides."
                if satisfied
                else "Current finite grasp policy found no complete left/right candidate set."
            ),
            "evidence": {
                "outcome": (
                    "grasp_policy_satisfied"
                    if satisfied
                    else "grasp_policy_unsatisfied"
                ),
                "end_effector_profile_id": eef_profile.profile_id,
                "left_candidate_found": left,
                "right_candidate_found": right,
                "diagnostics": provider.diagnostics,
            },
        }
    except Exception as exc:
        return {
            "kind": "grasp_policy_probe",
            "subject": subject,
            "status": "runtime_probe",
            "reason": "Static grasp probe could not run; live runtime validation is required.",
            "evidence": {
                "outcome": "runtime_probe_required",
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def _load_probe_mesh(
    manifest_object: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import trimesh
    from scipy.spatial.transform import Rotation

    geometry = manifest_object.get("geometry", {})
    shape = geometry.get("shape", {}) if isinstance(geometry, Mapping) else {}
    mesh_path = Path(str(shape.get("fpath", ""))).expanduser().resolve()
    if not mesh_path.is_file():
        raise ValueError(f"Static mesh is unavailable: {mesh_path}")
    loaded = trimesh.load(mesh_path.as_posix(), force="scene")
    mesh = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError("Static mesh contains no triangles.")

    pose = manifest_object.get("initial_pose", {})
    scale = np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    sim_vertices = np.column_stack(
        (vertices[:, 0], -vertices[:, 2], vertices[:, 1])
    )
    sim_vertices *= np.asarray([scale[0], scale[2], scale[1]])
    rotation = Rotation.from_euler(
        "XYZ", pose.get("rotation", [0.0, 0.0, 0.0]), degrees=True
    ).as_matrix()
    object_pose = torch.eye(4, dtype=torch.float32)
    object_pose[:3, :3] = torch.as_tensor(rotation, dtype=torch.float32)
    object_pose[:3, 3] = torch.as_tensor(
        pose.get("position", [0.0, 0.0, 0.0]), dtype=torch.float32
    )
    return (
        torch.as_tensor(sim_vertices.copy(), dtype=torch.float32),
        torch.as_tensor(triangles.copy(), dtype=torch.int64),
        object_pose,
    )
