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

"""Provider-free source scene serialization into canonical simulation declarations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
import numpy as np

__all__ = ["source_scene_component_declaration"]


def source_scene_component_declaration(
    scene_id: str,
    *,
    background: Sequence[Mapping[str, Any]],
    rigid_objects: Sequence[Mapping[str, Any]],
    articulations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Serialize source poses, collider choices and articulation ownership.

    Args:
        scene_id: Explicit component identity chosen by the declaration caller.
        background: Ordered source background configurations.
        rigid_objects: Ordered dynamic/object configurations.
        articulations: Ordered source articulation configurations.

    Returns:
        Independent scene component with the source lighting profile. Intrinsic
        XYZ source rotations become explicit matrices; supplied matrices retain
        precedence. Legacy collider choices move to canonical collision fields.
        No source file, native object, robot or solver is opened or constructed.
    """
    if type(scene_id) is not str or not scene_id or scene_id.strip() != scene_id:
        raise ValueError("scene_id must be an exact nonempty identity.")

    def runtime_pose(config: Any) -> dict[str, Any]:
        from scipy.spatial.transform import Rotation

        result = deepcopy(dict(config))
        if result.get("init_local_pose") is None:
            # PreparedScene uses intrinsic XYZ; dictionary decoders infer
            # extrinsic xyz. Serialize the matrix to preserve the planner pose.
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler(
                "XYZ", result.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
            ).as_matrix()
            pose[:3, 3] = result.get("init_pos", [0.0, 0.0, 0.0])
            result["init_local_pose"] = pose.tolist()
        return result

    def runtime_rigid(config: Any) -> dict[str, Any]:
        result = runtime_pose(config)
        max_hulls = int(result.pop("max_convex_hull_num", 1))
        acd_method = str(result.pop("acd_method", "visacd"))
        shape = result.get("shape")
        if isinstance(shape, dict) and shape.get("shape_type") == "Mesh":
            shape.pop("max_convex_hull_num", None)
            shape.pop("acd_method", None)
            shape.setdefault(
                "collision",
                {
                    "approximation": (
                        "convex_hull" if max_hulls == 1 else "convex_decomposition"
                    ),
                    **(
                        {"max_hulls": max_hulls, "acd_method": acd_method}
                        if max_hulls > 1
                        else {}
                    ),
                },
            )
        return result

    articulations = [runtime_pose(value) for value in articulations]
    for articulation in articulations:
        for semantic_only_key in (
            "affordances",
            "attributes",
            "category",
            "description",
            "initial_state",
            "is_articulated",
            "name",
            "proxy_body_scale",
            "proxy_glb_fpath",
            "proxy_init_pos",
        ):
            articulation.pop(semantic_only_key, None)
        fixed_base = articulation.pop("fix_base", None)
        if fixed_base is not None:
            articulation.setdefault("root_props", {})["fixed_base"] = fixed_base
        if articulation.get("joint_drive_props") is not None:
            articulation.setdefault("asset_physics_mode", "overlay")
        suffix = Path(str(articulation.get("fpath", ""))).suffix.lower()
        if suffix in {".usd", ".usda", ".usdc"}:
            # pytorch-kinematics accepts URDF XML only.  These task skills use
            # DexSim's native link poses/geometry and do not need a PK chain.
            articulation["build_pk_chain"] = False
    return {
        "scene_id": scene_id,
        "simulation": {
            "light": {
                "direct": [
                    {
                        "uid": "main_light",
                        "color": [0.6, 0.6, 0.6],
                        "intensity": 30.0,
                        "init_pos": [0.5, 0.0, 3.0],
                    }
                ]
            },
            "background": [runtime_rigid(value) for value in background],
            "rigid_object": [runtime_rigid(value) for value in rigid_objects],
            "rigid_object_group": [],
            "articulation": articulations,
        },
    }
