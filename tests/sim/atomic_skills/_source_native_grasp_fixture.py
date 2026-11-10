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

"""Frozen source query inputs and backend-independent decision projections for the explicit source-policy adapter."""

from __future__ import annotations

from scene._native_hand_fixture import decisions, recording_manager

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from copy import deepcopy


def query_inputs(case, part_type):
    parts = []
    distances = {
        "already_touching": [0.105, 0.08, 0.06],
        "never_touching": [0.15, 0.14, 0.13],
    }.get(case.get("stroke"), [0.15, 0.105, 0.08])
    for sign in (-1, 1):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, case.get("axis", 0), 3] = sign * np.asarray(distances)
        parts.append(
            part_type(
                f"pad_{sign}",
                (trimesh.creation.box((0.02, 0.02, 0.02)),),
                poses,
                0.0,
                True,
                0.002,
            )
        )
    if case.get("body"):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = {
            "halo": [0.112] * 3,
            "first": [0.105] * 3,
            "ambiguous": [0.15, 0.105, 0.105],
        }[case["body"]]
        parts.append(
            part_type(
                "housing",
                (trimesh.creation.box((0.02, 0.02, 0.02)),),
                poses,
                0.0,
                False,
                0.002,
            )
        )
    options = {"revision": case["revision"], "target_contact_offset": 0.002}
    if case["revision"] == 9:
        options["gravity_in_object"] = np.asarray(
            case.get("gravity", [0, 0, -1]), dtype=np.float64
        )
    return tuple(parts), (trimesh.creation.box((0.2, 0.2, 0.2)),), options


def query_pose(case):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler(
        "y", case.get("angle", 0), degrees=True
    ).as_matrix()
    return pose


def exercise(query, case):
    pose = query_pose(case)
    inspected = query.inspect(pose)
    repeated = query.inspect(pose.copy())
    candidates = torch.as_tensor(pose, dtype=torch.float64).repeat(1, 2, 1, 1, 1)
    mask = query.mask(candidates, torch.eye(4, dtype=torch.float64)[None])
    if case["revision"] == 8:
        if hasattr(query, "record_deferred_proposal"):
            query.record_deferred_proposal()
        else:
            query.deferred_source_proposal_calls += 1
    return {
        "inspect": stable_report(inspected),
        "repeated": stable_report(repeated),
        "mask": mask.tolist(),
        "depth_offsets": list(query.depth_offsets()),
        "report": stable_report(query.report()),
    }


def stable_report(value):
    if isinstance(value, dict):
        return {
            {
                "clearance_lower_bound_m": "minimum_distance_m",
                "first_nonpad_contact_halo_clearance_lower_bound_m": "first_nonpad_contact_halo_minimum_distance_m",
                "mesh_clearance_lower_bound_m": "mesh_distance_m",
                "minimum_rest_clearance_lower_bound_m": "minimum_rest_clearance_m",
            }.get(key, key): stable_report(item)
            for key, item in value.items()
            if key
            not in {
                "query_seconds",
                "collision_backend",
                "collision_backend_version",
                "distance_semantics",
                "build_seconds",
                "query_scope",
                "object_pose_at_capture",
            }
        }
    if isinstance(value, (tuple, list)):
        return [stable_report(item) for item in value]
    return deepcopy(value)
