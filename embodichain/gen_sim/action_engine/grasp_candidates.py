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

"""Grasp-candidate policies owned by the Action Engine runtime boundary."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

import torch

from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPolicy,
    GraspCandidateProvider,
    ParallelJawEefProfile,
)

__all__ = [
    "SupportCollisionFallbackProvider",
    "build_grasp_candidate_provider",
]


class SupportCollisionFallbackProvider:
    """Retry candidates when only the support-plane heuristic exhausts them.

    The relaxed pass still performs object/gripper collision filtering. Its
    candidates subsequently go through the Action Engine's live robot and
    scene motion planner, so this removes a conservative geometry heuristic
    rather than bypassing physical collision validation.
    """

    def __init__(
        self,
        strict_provider: Any,
        relaxed_provider: Any | None,
    ) -> None:
        self.strict_provider = strict_provider
        self.relaxed_provider = relaxed_provider
        self._diagnostics: dict[str, Any] | None = None

    @property
    def generator(self) -> SupportCollisionFallbackProvider:
        """Expose the generator surface expected by ``AntipodalAffordance``."""
        return self

    @property
    def eef_profile(self) -> ParallelJawEefProfile:
        return self.strict_provider.eef_profile

    @property
    def sampling_policy(self) -> AntipodalGraspPolicy:
        return self.strict_provider.sampling_policy

    @property
    def device(self) -> torch.device:
        return self.strict_provider.generator.device

    @property
    def diagnostics(self) -> dict[str, Any]:
        source = (
            self.strict_provider.diagnostics
            if self._diagnostics is None
            else self._diagnostics
        )
        return deepcopy(dict(source))

    @property
    def last_filter_diagnostics(self) -> dict[str, Any]:
        return self.diagnostics

    def get_valid_grasp_poses(self, **kwargs: Any) -> Any:
        strict_result = self.strict_provider.get_valid_grasp_poses(**kwargs)
        strict_diagnostics = self.strict_provider.diagnostics
        object_part = str(kwargs.get("object_part", "center"))
        should_relax = (
            not _single_succeeded(strict_result)
            and self.relaxed_provider is not None
            and _support_heuristic_exhausted(
                strict_diagnostics.get(object_part),
            )
        )
        if not should_relax:
            self._diagnostics = deepcopy(dict(strict_diagnostics))
            return strict_result

        relaxed_result = self.relaxed_provider.get_valid_grasp_poses(**kwargs)
        self._diagnostics = _fallback_diagnostics(
            strict_diagnostics,
            self.relaxed_provider.diagnostics,
            accepted=_single_succeeded(relaxed_result),
        )
        return relaxed_result

    def get_dual_arm_valid_grasp_poses(self, **kwargs: Any) -> Any:
        strict_result = self.strict_provider.get_dual_arm_valid_grasp_poses(**kwargs)
        strict_diagnostics = self.strict_provider.diagnostics
        failed_sides = _failed_dual_sides(strict_result)
        should_relax = (
            bool(failed_sides)
            and self.relaxed_provider is not None
            and all(
                _support_heuristic_exhausted(strict_diagnostics.get(side))
                for side in failed_sides
            )
        )
        if not should_relax:
            self._diagnostics = deepcopy(dict(strict_diagnostics))
            return strict_result

        relaxed_result = self.relaxed_provider.get_dual_arm_valid_grasp_poses(**kwargs)
        self._diagnostics = _fallback_diagnostics(
            strict_diagnostics,
            self.relaxed_provider.diagnostics,
            accepted=not _failed_dual_sides(relaxed_result),
        )
        return relaxed_result

    def get_grasp_poses(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate the legacy best-pose API to the strict generator."""
        return self.strict_provider.generator.get_grasp_poses(*args, **kwargs)


def build_grasp_candidate_provider(
    *,
    mesh_vertices: torch.Tensor,
    mesh_triangles: torch.Tensor,
    eef_profile: ParallelJawEefProfile,
    sampling_policy: AntipodalGraspPolicy,
    force_reannotate: bool = False,
) -> SupportCollisionFallbackProvider:
    """Build a strict provider with a diagnostic-gated relaxed fallback."""
    strict = GraspCandidateProvider(
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        eef_profile=eef_profile,
        sampling_policy=sampling_policy,
        force_reannotate=force_reannotate,
    )
    relaxed = None
    if sampling_policy.filter_support_collision:
        relaxed = GraspCandidateProvider(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            eef_profile=eef_profile,
            sampling_policy=replace(
                sampling_policy,
                filter_support_collision=False,
            ),
            force_reannotate=force_reannotate,
        )
    return SupportCollisionFallbackProvider(strict, relaxed)


def _support_heuristic_exhausted(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    collision = value.get("collision")
    if not isinstance(collision, Mapping):
        return False
    candidate_count = int(collision.get("candidate_count", 0))
    return (
        candidate_count > 0
        and collision.get("support_filter_enabled") is True
        and int(value.get("collision_free_pose_count", -1)) == 0
        and int(collision.get("combined_collision_count", -1)) == candidate_count
        and int(collision.get("support_collision_count", 0)) > 0
        and int(collision.get("object_collision_count", candidate_count))
        < candidate_count
    )


def _single_succeeded(result: Any) -> bool:
    return isinstance(result, tuple) and bool(result) and bool(result[0])


def _failed_dual_sides(result: Any) -> tuple[str, ...]:
    if not isinstance(result, Mapping):
        return ("left", "right")
    return tuple(
        side
        for side in ("left", "right")
        if not isinstance(result.get(side), Mapping)
        or not bool(result[side].get("is_success"))
    )


def _fallback_diagnostics(
    strict: Mapping[str, Any],
    relaxed: Mapping[str, Any],
    *,
    accepted: bool,
) -> dict[str, Any]:
    result = deepcopy(dict(strict))
    result["support_collision_fallback"] = {
        "attempted": True,
        "accepted": bool(accepted),
        "reason": "support_heuristic_exhausted",
        "relaxed": deepcopy(dict(relaxed)),
    }
    return result
