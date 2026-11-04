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

"""Contact-qualified paired grasp sampling with explicit data and query bounds."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING
import math
from typing import Any, Protocol, runtime_checkable

import torch

from embodichain.toolkits.graspkit import ParallelJawGraspPoseGenerator
from embodichain.utils import configclass, logger
from .pose_generator import (
    AntipodalGraspPoseGenerator,
    _drop_stale_mesh_versions,
    _mesh_geometry_key,
    _validate_mesh_geometry,
)
from .contact_fit import SampledHandGeometry, HandContactChecker, TriangleSurfaceQuery

__all__ = [
    "CoordinatedGraspCfg",
    "PairedGraspProvider",
    "CoordinatedGraspSampler",
    "CoordinatedGraspPoseGenerator",
    "grasp_clearance_costs",
]


@configclass
class CoordinatedGraspCfg:
    """Caller-owned contact thresholds, candidate budgets and clearance preference."""

    contact_tolerance: float = MISSING
    """Positive contact distance in metres."""
    minimum_normal_cosine: float = MISSING
    """Required outward-normal alignment in [0, 1]."""
    normal_probe_distance: float = MISSING
    """Positive signed-surface normal probe offset in metres."""
    max_candidates: int = MISSING
    """Maximum returned qualified candidates per hand."""
    candidate_chunk_size: int = MISSING
    """Maximum proposal rows processed together before checking the return budget."""
    cargo_clearance_reserve: float = MISSING
    """Positive world-box clearance preference in metres; not a collision margin."""

    def validate_values(self) -> None:
        """Validate required fields and semantic bounds after configuration assembly.

        Raises:
            TypeError: Required configuration values remain unresolved.
            ValueError: A threshold or candidate budget is invalid.
        """
        self.validate()
        for name in ("max_candidates", "candidate_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "contact_tolerance",
            "normal_probe_distance",
            "cargo_clearance_reserve",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        value = self.minimum_normal_cosine
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("minimum_normal_cosine must lie in [0, 1].")


def _owned_cfg(cfg: CoordinatedGraspCfg) -> CoordinatedGraspCfg:
    if not isinstance(cfg, CoordinatedGraspCfg):
        raise TypeError("cfg must be CoordinatedGraspCfg.")
    result = deepcopy(cfg)
    result.validate_values()
    return result


@runtime_checkable
class PairedGraspProvider(Protocol):
    """Numeric paired-query port; geometry capture and robot lifetimes stay outside."""

    def get_dual_arm_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
        cargo_bounds: torch.Tensor | None = None,
    ) -> list[dict[str, dict[str, object]] | None]:
        """Return per-row left/right candidates from explicit geometry and poses.

        Args:
            mesh_vertices: Object-local surface vertices.
            mesh_triangles: Matching triangle indices.
            obj_poses: Object-to-world transforms for the query rows.
            left_to_right_arm_direction: World-space pairing direction.
            approach_direction: World-space approach direction.
            middle_empty_ratio: Antipodal middle-region exclusion ratio.
            cargo_bounds: Optional world AABBs with shape ``(M, 2, 3)``.

        Returns:
            Existing left/right candidate records, or None for a failed row.
        """
        ...


def grasp_clearance_costs(
    geometry: SampledHandGeometry,
    poses: torch.Tensor,
    widths: torch.Tensor,
    costs: torch.Tensor,
    world_bounds: torch.Tensor | None,
    *,
    reserve: float,
) -> torch.Tensor:
    """Prefer clearance without removing or replacing qualified candidates.

    Args:
        geometry: Read-only closing surfaces in the candidate TCP frame.
        poses: Qualified TCP-to-world transforms, shape ``(K, 4, 4)``.
        widths: Qualified pad gaps, shape ``(K,)``.
        costs: Finite candidate costs, shape ``(K,)``.
        world_bounds: Explicit world-frame AABBs, shape ``(M, 2, 3)``, or None.
        reserve: Positive clearance preference distance in metres.

    Returns:
        Adjusted finite costs, or the original cost tensor when no bounds apply
        or preference geometry is unavailable. Inputs and candidate membership
        remain unchanged. This preference does not qualify collisions or cargo.

    Raises:
        ValueError: The configured preference reserve is not finite and positive.
    """
    if (
        isinstance(reserve, bool)
        or not isinstance(reserve, (int, float))
        or not math.isfinite(reserve)
        or reserve <= 0
    ):
        raise ValueError("reserve must be finite and positive.")
    if world_bounds is None or not len(costs):
        return costs
    try:
        if (
            not isinstance(world_bounds, torch.Tensor)
            or world_bounds.ndim != 3
            or world_bounds.shape[1:] != (2, 3)
        ):
            raise ValueError("World clearance bounds must have shape (M, 2, 3).")
        if not len(world_bounds):
            return costs
        if (
            not torch.isfinite(world_bounds).all()
            or not (world_bounds[:, 0] <= world_bounds[:, 1]).all()
        ):
            raise ValueError("World clearance bounds must be finite ordered boxes.")
        distance = torch.full_like(costs, torch.inf)
        opened = geometry.gaps[0].to(widths).expand_as(widths)
        for gaps in (opened, (opened + widths) / 2, widths):
            body, pads = geometry.at_width(gaps)
            points = torch.cat((body, pads.flatten(1, 2)), 1)
            world = points @ poses[:, :3, :3].transpose(-1, -2) + poses[:, None, :3, 3]
            for lower, upper in world_bounds:
                gaps = torch.maximum(lower.to(world) - world, world - upper.to(world))
                signed = torch.linalg.vector_norm(gaps.clamp_min(0), dim=-1)
                signed += gaps.amax(-1).clamp_max(0)
                distance = torch.minimum(distance, signed.amin(-1))
        near = distance < reserve
        if not near.any():
            return costs
        spread = (costs.max() - costs.min()).clamp_min(1)
        penalty = spread + 1 + (1 - distance / reserve).clamp_min(0) * spread
        ranked = costs + near.to(costs) * penalty
        return ranked if torch.isfinite(ranked).all() else costs
    except (ValueError, RuntimeError, TypeError, AttributeError) as error:
        logger.log_warning(
            f"Grasp clearance preference unavailable; retaining base costs: {error}"
        )
        return costs


class _HandSampler:
    """Compose explicit proposals with one hand's contact refinement."""

    def __init__(
        self,
        delegate: AntipodalGraspPoseGenerator,
        geometry: SampledHandGeometry,
        role: str,
        cfg: CoordinatedGraspCfg,
        surface_queries: dict[tuple[object, ...], TriangleSurfaceQuery],
    ) -> None:
        self._delegate = delegate
        self._geometry = geometry
        self._cfg = cfg
        self._role = role
        self._surface_queries = surface_queries
        self._validators: dict[tuple[object, ...], HandContactChecker] = {}

    def get_dual_arm_valid_grasp_poses(
        self, *, cargo_bounds: torch.Tensor | None = None, **kwargs: Any
    ) -> list[dict[str, dict[str, object]] | None]:
        vertices, triangles = kwargs["mesh_vertices"], kwargs["mesh_triangles"]
        key = _mesh_geometry_key(vertices, triangles)
        if key not in self._surface_queries:
            _validate_mesh_geometry(vertices, triangles)
            _drop_stale_mesh_versions(self._surface_queries, key)
            self._surface_queries[key] = TriangleSurfaceQuery(
                vertices,
                triangles,
                normal_probe_distance=self._cfg.normal_probe_distance,
            )
        if key not in self._validators:
            _drop_stale_mesh_versions(self._validators, key)
            self._validators[key] = HandContactChecker(
                self._geometry,
                self._surface_queries[key],
                contact_tolerance=self._cfg.contact_tolerance,
                minimum_normal_cosine=self._cfg.minimum_normal_cosine,
            )
        checker = self._validators[key]
        proposals = self._delegate.get_dual_arm_grasp_proposals(
            **kwargs, role=self._role
        )
        output = []
        for row, proposal in enumerate(proposals):
            if proposal is None:
                output.append(None)
                continue
            item = proposal[self._role]
            poses = item["grasp_poses"]
            empty = {
                "is_success": False,
                "grasp_poses": poses.new_empty((0, 4, 4)),
                "open_lengths": poses.new_empty(0),
                "total_cost": poses.new_empty(0),
            }
            result = {"left": dict(empty), "right": dict(empty)}
            if not item["is_success"]:
                output.append(result)
                continue
            widths, costs = item["open_lengths"], item["total_cost"]
            order = torch.argsort(costs)
            order = order[torch.isfinite(costs[order])]
            accepted_poses, accepted_widths, accepted_costs = [], [], []
            # Match the shared action's bounded reachability population, but
            # validate every returned candidate before it reaches that action.
            for start in range(0, len(order), self._cfg.candidate_chunk_size):
                ids = order[start : start + self._cfg.candidate_chunk_size]
                unresolved = torch.ones(len(ids), dtype=torch.bool, device=poses.device)
                _, pads = self._geometry.at_width(widths[ids])
                pad_length = (pads[..., 2].amax(-1) - pads[..., 2].amin(-1)).amin(-1)
                insertion = (
                    pads[..., 2].amax(-1).amin(-1) - 0.1 * pad_length
                ).clamp_min(0)
                # Prefer fingertip contact with a pad-length-relative reserve;
                # retain middle/deep alternatives for other object geometries.
                for fraction in (1.0, 0.5, 0.0):
                    subset = torch.where(unresolved)[0]
                    if not len(subset):
                        break
                    selected = ids[subset]
                    trial = poses[selected].clone()
                    trial[:, :3, 3] -= (
                        trial[:, :3, 2] * (fraction * insertion[subset])[:, None]
                    )
                    fitted, gaps, kept = checker.refine(
                        kwargs["obj_poses"][row], trial, widths[selected]
                    )
                    if kept.any():
                        accepted_poses.append(fitted[kept])
                        accepted_widths.append(gaps[kept])
                        accepted_costs.append(costs[selected[kept]])
                        unresolved[subset[kept]] = False
                if (
                    sum(len(value) for value in accepted_poses)
                    >= self._cfg.max_candidates
                ):
                    break
            if accepted_poses:
                combined_costs = torch.cat(accepted_costs)
                chosen = torch.argsort(combined_costs)[: self._cfg.max_candidates]
                result[self._role] = {
                    "is_success": True,
                    "grasp_poses": torch.cat(accepted_poses)[chosen],
                    "open_lengths": torch.cat(accepted_widths)[chosen],
                    "total_cost": combined_costs[chosen],
                }
                selected = result[self._role]
                selected["total_cost"] = grasp_clearance_costs(
                    self._geometry,
                    selected["grasp_poses"],
                    selected["open_lengths"],
                    selected["total_cost"],
                    cargo_bounds,
                    reserve=self._cfg.cargo_clearance_reserve,
                )
            output.append(result)
        return output


class CoordinatedGraspSampler:
    """Fit and qualify two hand populations from data-only geometry inputs.

    Each object mesh has one surface query shared by both hand refiners.
    Each hand retains its own proposal and refinement caches. The right proposal stream
    starts from the left stream's initial Torch RNG state, and the process RNG
    advances exactly as for the left query alone. No robot or scene is read.
    """

    def __init__(
        self,
        left: AntipodalGraspPoseGenerator,
        right: AntipodalGraspPoseGenerator,
        left_geometry: SampledHandGeometry,
        right_geometry: SampledHandGeometry,
        *,
        cfg: CoordinatedGraspCfg,
    ) -> None:
        """Bind explicit delegates, immutable samples and validated budgets.

        Args:
            left: Left-role antipodal generator with inferred ground filtering disabled.
            right: Right-role generator with the same explicit target-only policy.
            left_geometry: Left hand samples in its configured TCP frame.
            right_geometry: Right hand samples in its configured TCP frame.
            cfg: Explicit caller configuration, validated after assembly.
        """
        owned = _owned_cfg(cfg)
        if not all(
            isinstance(value, AntipodalGraspPoseGenerator) for value in (left, right)
        ):
            raise TypeError("Paired contact sampling requires antipodal generators.")
        if any(value.collision_cfg.filter_ground_collision for value in (left, right)):
            raise ValueError(
                "Paired contact sampling requires filter_ground_collision=False; "
                "world clearance belongs to an explicit caller check."
            )
        if not all(
            isinstance(value, SampledHandGeometry)
            for value in (left_geometry, right_geometry)
        ):
            raise TypeError("Paired contact sampling requires sampled hand geometry.")
        surfaces: dict[tuple[object, ...], TriangleSurfaceQuery] = {}
        self._left = _HandSampler(left, left_geometry, "left", owned, surfaces)
        self._right = _HandSampler(right, right_geometry, "right", owned, surfaces)

    def clear_cache(self) -> None:
        """Release target queries, hand refiners and both generators' mesh backends.

        Sampled hand geometry and configuration remain bound. Hosts can call
        this between unrelated asset cohorts without recapturing robot geometry.
        """
        self._left._surface_queries.clear()
        for hand in (self._left, self._right):
            hand._validators.clear()
            hand._delegate.clear_cache()

    def get_dual_arm_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
        cargo_bounds: torch.Tensor | None = None,
    ) -> list[dict[str, dict[str, object]] | None]:
        """Return contact-qualified pairs, preserving row order and failed rows.

        Args:
            mesh_vertices: Object-local vertices used by the antipodal service.
            mesh_triangles: Matching indexed triangle geometry.
            obj_poses: Object-to-world query transforms.
            left_to_right_arm_direction: Explicit world-space pairing direction.
            approach_direction: Explicit world-space approach direction.
            middle_empty_ratio: Existing antipodal middle-region exclusion ratio.
            cargo_bounds: Optional world AABBs shared by the query rows, shape
                ``(M, 2, 3)``. Callers own cohort selection and observation.

        Returns:
            The standard per-row left/right candidate mapping. Contact fitting,
            discrete sweeps and normal gates precede the returned population cap;
            clearance preference changes costs only after that selection.
        """
        query = dict(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            obj_poses=obj_poses,
            left_to_right_arm_direction=left_to_right_arm_direction,
            approach_direction=approach_direction,
            middle_empty_ratio=middle_empty_ratio,
            cargo_bounds=cargo_bounds,
        )
        device = mesh_vertices.device
        cuda = [device.index or 0] if device.type == "cuda" else []
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if cuda else None
        left = self._left.get_dual_arm_valid_grasp_poses(**query)
        with torch.random.fork_rng(devices=cuda):
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            right = self._right.get_dual_arm_valid_grasp_poses(**query)
        return [
            None if l is None or r is None else {"left": l["left"], "right": r["right"]}
            for l, r in zip(left, right, strict=True)
        ]


class CoordinatedGraspPoseGenerator(ParallelJawGraspPoseGenerator):
    """Delegate single-arm calls unchanged and route only paired queries.

    This facade reads no geometry or robot state. Its typed paired provider
    may own deferred capture in the caller's runtime layer; the concrete
    :class:`CoordinatedGraspSampler` consumes only explicit data and generators.
    """

    def __init__(
        self, delegate: ParallelJawGraspPoseGenerator, paired: PairedGraspProvider
    ) -> None:
        """Bind a single-arm delegate and a numeric paired-query port.

        Args:
            delegate: Existing standalone service, retained unchanged.
            paired: Paired numeric service. Construction never invokes it.
        """
        if not isinstance(delegate, ParallelJawGraspPoseGenerator):
            raise TypeError("delegate must be a parallel-jaw grasp generator.")
        if not isinstance(paired, PairedGraspProvider):
            raise TypeError("paired must implement PairedGraspProvider.")
        super().__init__(delegate.gripper_model)
        self._delegate = delegate
        self._paired = paired

    def get_valid_grasp_poses(
        self, **kwargs: Any
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Forward all caller keywords to the unchanged single-arm delegate.

        Args:
            **kwargs: Original standalone query arguments, forwarded unchanged.

        Returns:
            The delegate's pose and cost records without paired capture.
        """
        return self._delegate.get_valid_grasp_poses(**kwargs)

    def get_best_grasp_poses(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward all caller keywords without capturing paired geometry.

        Args:
            **kwargs: Original standalone query arguments, forwarded unchanged.

        Returns:
            The delegate's success mask, poses and opening widths.
        """
        return self._delegate.get_best_grasp_poses(**kwargs)

    def get_grasp_candidates(
        self, **kwargs: Any
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward candidate metadata to the unchanged single-arm delegate.

        Args:
            **kwargs: Original standalone query arguments, forwarded unchanged.

        Returns:
            The delegate's pose, width and cost records.
        """
        return self._delegate.get_grasp_candidates(**kwargs)

    def get_dual_arm_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
        cargo_bounds: torch.Tensor | None = None,
    ) -> list[dict[str, dict[str, object]] | None]:
        """Delegate one paired numeric query; no scene observation is performed.

        Args:
            mesh_vertices: Object-local surface vertices.
            mesh_triangles: Matching triangle indices.
            obj_poses: Object-to-world query transforms.
            left_to_right_arm_direction: World-space pairing direction.
            approach_direction: World-space approach direction.
            middle_empty_ratio: Existing middle-region exclusion ratio.
            cargo_bounds: Optional world AABBs, shape ``(M, 2, 3)``.

        Returns:
            The paired provider's standard candidate records.
        """
        return self._paired.get_dual_arm_valid_grasp_poses(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            obj_poses=obj_poses,
            left_to_right_arm_direction=left_to_right_arm_direction,
            approach_direction=approach_direction,
            middle_empty_ratio=middle_empty_ratio,
            cargo_bounds=cargo_bounds,
        )
