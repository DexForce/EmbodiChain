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

"""Build grasp affordances and prepare their collision caches.

Grasp annotation and cache policy stay outside target dispatch so their heavy
toolkit dependencies do not leak into pure spec validation.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_parts import (
    _cfg_supported_kwargs,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.grasp_collision_cache import (
    GraspCollisionCachePreparationError as VhacdCachePreparationError,
    ensure_vhacd_grasp_collision_cache,
)
from embodichain.lab.sim.atomic_actions import AntipodalAffordance, ObjectSemantics
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
    GripperCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GRASP_ANNOTATOR_CACHE_DIR,
)
from embodichain.utils.logger import log_info

__all__ = [
    "_ActionAgentAntipodalAffordance",
    "_GraspRuntimeDefaults",
    "_build_object_semantics",
    "_prepare_grasp_collision_cache",
    "_stabilize_affordance_object",
    "_affordance_cache_path",
    "_max_decomposition_hulls",
    "_grasp_convex_decomposition_method",
    "_normalize_convex_decomposition_method",
]

_GRASP_DEFAULTS = defaults_section("grasp")
_GRASP_ALIGNMENT_CONFIG_KEY = "action_agent_max_approach_alignment_angle"


class _ActionAgentAntipodalAffordance(AntipodalAffordance):
    """Apply Action Agent grasp alignment without changing the shared affordance."""

    def get_valid_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self._generator is None:
            self._init_generator()
        alignment_angle = self.get_custom_config(_GRASP_ALIGNMENT_CONFIG_KEY)
        if alignment_angle is None:
            return super().get_valid_grasp_poses(obj_poses, approach_direction)

        previous_angle = self._generator.cfg.max_deviation_angle
        self._generator.cfg.max_deviation_angle = float(alignment_angle)
        try:
            return super().get_valid_grasp_poses(obj_poses, approach_direction)
        finally:
            self._generator.cfg.max_deviation_angle = previous_angle


@dataclass(frozen=True)
class _GraspRuntimeDefaults:
    antipodal_n_sample: int = int(_GRASP_DEFAULTS["antipodal_n_sample"])
    antipodal_max_angle: float = float(
        np.deg2rad(_GRASP_DEFAULTS["antipodal_max_angle_degrees"])
    )
    max_open_length: float = float(_GRASP_DEFAULTS["max_open_length"])
    min_open_length: float = float(_GRASP_DEFAULTS["min_open_length"])
    finger_length: float = float(_GRASP_DEFAULTS["finger_length"])
    point_sample_dense: float = float(_GRASP_DEFAULTS["point_sample_dense"])
    max_deviation_angle: float = float(
        np.deg2rad(_GRASP_DEFAULTS["max_deviation_angle_degrees"])
    )
    viser_port: int = int(_GRASP_DEFAULTS["viser_port"])


_GRASP_RUNTIME_DEFAULTS = _GraspRuntimeDefaults()


def _build_object_semantics(
    env,
    target: Mapping[str, Any],
    runtime_kwargs: dict[str, Any],
    *,
    max_approach_alignment_angle: float | None = None,
):
    obj_name = target.get("obj_name")
    if target.get("affordance", "antipodal") != "antipodal":
        raise ValueError("target_object only supports antipodal affordance.")
    target_obj = env.sim.get_rigid_object(obj_name)
    if target_obj is None:
        raise ValueError(f"No rigid object found for {obj_name}.")

    _stabilize_affordance_object(env, target_obj, runtime_kwargs)

    mesh_vertices = target_obj.get_vertices(env_ids=[0], scale=True)[0]
    mesh_triangles = target_obj.get_triangles(env_ids=[0])[0]
    mesh_vertices = torch.as_tensor(mesh_vertices, dtype=torch.float32)
    mesh_triangles = torch.as_tensor(mesh_triangles, dtype=torch.int64)
    if (
        mesh_vertices.numel() == 0
        or mesh_triangles.numel() == 0
        or mesh_vertices.shape[-1] != 3
        or mesh_triangles.shape[-1] != 3
    ):
        raise ValueError(f"Object {obj_name} has empty or invalid mesh geometry.")

    allow_annotation = bool(runtime_kwargs.get("allow_grasp_annotation", True))
    force_reannotate = bool(runtime_kwargs.get("force_grasp_reannotate", False))
    cache_path = _affordance_cache_path(mesh_vertices, mesh_triangles)
    if not os.path.exists(cache_path) and not allow_annotation:
        raise RuntimeError(
            "Grasp annotation cache is missing and annotation is disabled; "
            "set allow_grasp_annotation=True."
        )

    antipodal_sampler_cfg = AntipodalSamplerCfg(
        **_cfg_supported_kwargs(
            AntipodalSamplerCfg,
            {
                "n_sample": int(
                    runtime_kwargs.get(
                        "grasp_antipodal_n_sample",
                        _GRASP_RUNTIME_DEFAULTS.antipodal_n_sample,
                    )
                ),
                "max_angle": runtime_kwargs.get(
                    "grasp_antipodal_max_angle",
                    _GRASP_RUNTIME_DEFAULTS.antipodal_max_angle,
                ),
                "max_length": runtime_kwargs.get(
                    "max_open_length",
                    _GRASP_RUNTIME_DEFAULTS.max_open_length,
                ),
                "min_length": runtime_kwargs.get(
                    "min_open_length",
                    _GRASP_RUNTIME_DEFAULTS.min_open_length,
                ),
            },
        )
    )
    generator_cfg = GraspGeneratorCfg(
        **_cfg_supported_kwargs(
            GraspGeneratorCfg,
            {
                "viser_port": int(
                    runtime_kwargs.get(
                        "grasp_viser_port",
                        _GRASP_RUNTIME_DEFAULTS.viser_port,
                    )
                ),
                "antipodal_sampler_cfg": antipodal_sampler_cfg,
                "max_deviation_angle": runtime_kwargs.get(
                    "grasp_max_deviation_angle",
                    _GRASP_RUNTIME_DEFAULTS.max_deviation_angle,
                ),
                "n_deviated_approach_directions": 1,
            },
        )
    )
    max_decomposition_hulls = _max_decomposition_hulls(target_obj, runtime_kwargs)
    convex_decomposition_method = _grasp_convex_decomposition_method(runtime_kwargs)
    _prepare_grasp_collision_cache(
        obj_name=obj_name,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        max_decomposition_hulls=max_decomposition_hulls,
        convex_decomposition_method=convex_decomposition_method,
    )

    gripper_collision_cfg = GripperCollisionCfg(
        **_cfg_supported_kwargs(
            GripperCollisionCfg,
            {
                "max_open_length": runtime_kwargs.get(
                    "max_open_length",
                    _GRASP_RUNTIME_DEFAULTS.max_open_length,
                ),
                "finger_length": runtime_kwargs.get(
                    "grasp_finger_length",
                    _GRASP_RUNTIME_DEFAULTS.finger_length,
                ),
                "point_sample_dense": runtime_kwargs.get(
                    "grasp_point_sample_dense",
                    _GRASP_RUNTIME_DEFAULTS.point_sample_dense,
                ),
                "max_decomposition_hulls": max_decomposition_hulls,
            },
        )
    )
    affordance = _ActionAgentAntipodalAffordance(
        object_label=obj_name,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        generator_cfg=generator_cfg,
        gripper_collision_cfg=gripper_collision_cfg,
        force_reannotate=force_reannotate,
    )
    affordance.set_custom_config(
        _GRASP_ALIGNMENT_CONFIG_KEY,
        max_approach_alignment_angle,
    )
    grasp_pose_overrides = getattr(env, "agent_grasp_pose_overrides", {}) or {}
    if isinstance(grasp_pose_overrides, Mapping):
        grasp_pose_bias = grasp_pose_overrides.get(obj_name)
        if isinstance(grasp_pose_bias, Mapping):
            affordance.set_custom_config("grasp_pose_bias", dict(grasp_pose_bias))
    return ObjectSemantics(
        label=obj_name,
        geometry={
            "mesh_vertices": mesh_vertices,
            "mesh_triangles": mesh_triangles,
        },
        affordance=affordance,
        entity=target_obj,
    )


def _prepare_grasp_collision_cache(
    *,
    obj_name: str,
    mesh_vertices: torch.Tensor,
    mesh_triangles: torch.Tensor,
    max_decomposition_hulls: int,
    convex_decomposition_method: str,
    **_compat_kwargs: Any,
) -> None:
    """Prepare the only supported grasp collision cache backend.

    ``_compat_kwargs`` accepts historical private-call keywords without
    restoring their retired backend behavior. New callers must pass only the
    V-HACD inputs declared above.
    """
    if convex_decomposition_method != "vhacd":
        raise ValueError("convex_decomposition_method must be 'vhacd'")
    try:
        result = ensure_vhacd_grasp_collision_cache(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            max_decomposition_hulls=max_decomposition_hulls,
        )
    except VhacdCachePreparationError as exc:
        raise VhacdCachePreparationError(
            f"Failed to prepare V-HACD grasp collision cache for "
            f"target={obj_name}: {exc}"
        ) from exc
    if result.get("status") != "hit":
        log_info(
            "Prepared Main-compatible V-HACD grasp collision cache: "
            f"target={obj_name}, cache={result.get('grasp_cache_path')}.",
            color="green",
        )


def _stabilize_affordance_object(
    env,
    target_obj,
    runtime_kwargs: Mapping[str, Any],
) -> None:
    if not bool(runtime_kwargs.get("stabilize_affordance_object", True)):
        return

    update_steps = int(
        runtime_kwargs.get(
            "affordance_stabilization_steps",
            _GRASP_DEFAULTS["affordance_stabilization_steps"],
        )
    )
    if update_steps > 0 and hasattr(env.sim, "update"):
        env.sim.update(step=update_steps)
    if hasattr(target_obj, "clear_dynamics"):
        target_obj.clear_dynamics()


def _affordance_cache_path(mesh_vertices, mesh_triangles):
    vert_bytes = mesh_vertices.to("cpu").numpy().tobytes()
    face_bytes = mesh_triangles.to("cpu").numpy().tobytes()
    md5_hash = hashlib.md5(vert_bytes + face_bytes).hexdigest()
    return os.path.join(GRASP_ANNOTATOR_CACHE_DIR, f"antipodal_cache_{md5_hash}.npy")


def _max_decomposition_hulls(target_obj, runtime_kwargs: Mapping[str, Any]) -> int:
    if "grasp_max_decomposition_hulls" in runtime_kwargs:
        return int(runtime_kwargs["grasp_max_decomposition_hulls"])

    cfg = getattr(target_obj, "cfg", None)
    max_convex_hull_num = getattr(cfg, "max_convex_hull_num", MISSING)
    if max_convex_hull_num is MISSING or max_convex_hull_num is None:
        max_convex_hull_num = getattr(
            getattr(cfg, "shape", None),
            "max_convex_hull_num",
            1,
        )
    if max_convex_hull_num is MISSING or max_convex_hull_num is None:
        max_convex_hull_num = 1
    if int(max_convex_hull_num) > 1:
        return int(max_convex_hull_num)
    return 8


def _grasp_convex_decomposition_method(runtime_kwargs: Mapping[str, Any]) -> str:
    method = runtime_kwargs.get("grasp_convex_decomposition_method", "vhacd")
    return _normalize_convex_decomposition_method(method)


def _normalize_convex_decomposition_method(method: Any) -> str:
    method = str(method).lower()
    if method == "visacd":
        return "vhacd"
    if method == "vhacd":
        return method
    raise ValueError("convex_decomposition_method must be one of: 'vhacd', 'visacd'")
