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

"""Owned, padded grasp candidates independent of simulation environments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch

__all__ = ["GraspCandidateBatch"]


def _se3_mask(poses: torch.Tensor) -> torch.Tensor:
    """Check transforms without passing non-finite values into linear algebra."""
    finite = torch.isfinite(poses).all(dim=(-2, -1))
    checked = torch.where(
        finite[..., None, None], poses, torch.eye(4, device=poses.device)
    ).to(torch.float64)
    rotation = checked[..., :3, :3]
    return (
        finite
        & torch.isclose(
            checked[..., 3, :], checked.new_tensor((0, 0, 0, 1)), atol=1e-6, rtol=0
        ).all(dim=-1)
        & torch.isclose(
            rotation.transpose(-2, -1) @ rotation,
            torch.eye(3, device=checked.device, dtype=checked.dtype),
            atol=1e-6,
            rtol=0,
        ).all(dim=(-2, -1))
        & torch.isclose(
            torch.linalg.det(rotation), checked.new_tensor(1), atol=1e-6, rtol=0
        )
    )


def _grasp_id(pose: torch.Tensor, width: float | None, namespace: str) -> str:
    """Hash canonical geometry, with sub-micrometre noise quantized away."""
    values = torch.round(pose.detach().to(device="cpu", dtype=torch.float64) * 1e6)
    payload = (namespace, values.to(torch.int64).reshape(-1).tolist(), width)
    return "grasp-" + hashlib.sha256(repr(payload).encode()).hexdigest()[:24]


@dataclass(frozen=True, slots=True, eq=False)
class GraspCandidateBatch:
    """Padded grasp candidates with explicit validity and reference frame.

    Rows are input objects, not simulator slots. Invalid individual candidates
    are safely masked and retain a rejection reason; malformed shared tensor
    shapes or metadata raise an error. Tensors are cloned on construction.
    Equal geometry may share an ID, allowing downstream geometric deduplication.

    Args:
        poses: Homogeneous grasp poses with shape ``(S, G, 4, 4)``.
        costs: Per-candidate ranking costs with shape ``(S, G)``.
        valid_mask: Boolean candidate eligibility with shape ``(S, G)``.
        opening_widths: Optional finger opening widths in metres, ``(S, G)``.
        grasp_ids: Stable IDs aligned with the padded rows; derived if omitted.
        frame: Named frame of ``poses``, such as ``local_arena`` or ``object``.
        rejection_reasons: Per-entry failure reason, or ``None`` for valid rows.
    """

    poses: torch.Tensor
    costs: torch.Tensor
    valid_mask: torch.Tensor
    opening_widths: torch.Tensor | None = None
    grasp_ids: tuple[tuple[str, ...], ...] = ()
    frame: str = "local_arena"
    rejection_reasons: tuple[tuple[str | None, ...], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.poses, torch.Tensor)
            or not self.poses.is_floating_point()
            or self.poses.ndim != 4
            or self.poses.shape[-2:] != (4, 4)
        ):
            raise ValueError("poses must be a floating tensor of shape (S,G,4,4).")
        shape = self.poses.shape[:2]
        for name, value in (("costs", self.costs), ("valid_mask", self.valid_mask)):
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(f"{name} must have shape (S,G).")
            if value.device != self.poses.device:
                raise ValueError(f"{name} must share the poses device.")
        if not self.costs.is_floating_point() or self.valid_mask.dtype != torch.bool:
            raise TypeError("costs must be floating and valid_mask must be boolean.")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("frame must be an explicit non-empty reference frame.")
        widths = self.opening_widths
        if widths is not None and (
            not isinstance(widths, torch.Tensor)
            or not widths.is_floating_point()
            or widths.shape != shape
            or widths.device != self.poses.device
        ):
            raise ValueError("opening_widths must be floating (S,G) on poses.device.")
        for name, rows in (
            ("grasp_ids", self.grasp_ids),
            ("rejection_reasons", self.rejection_reasons),
        ):
            if rows and (
                len(rows) != shape[0] or any(len(row) != shape[1] for row in rows)
            ):
                raise ValueError(f"{name} must align with padded (S,G) rows.")
        pose_ok = _se3_mask(self.poses)
        cost_ok = torch.isfinite(self.costs)
        width_ok = (
            torch.ones_like(self.valid_mask)
            if widths is None
            else torch.isfinite(widths) & (widths > 0)
        )
        valid = self.valid_mask & pose_ok & cost_ok & width_ok
        reasons: list[tuple[str | None, ...]] = []
        ids: list[tuple[str, ...]] = []
        for row in range(shape[0]):
            row_reasons: list[str | None] = []
            row_ids: list[str] = []
            for col in range(shape[1]):
                supplied_reason = (
                    self.rejection_reasons[row][col] if self.rejection_reasons else None
                )
                if supplied_reason is not None and (
                    not isinstance(supplied_reason, str) or not supplied_reason
                ):
                    raise ValueError(
                        "rejection reasons must be non-empty strings or None."
                    )
                if bool(valid[row, col]):
                    if supplied_reason is not None:
                        raise ValueError(
                            "A valid candidate cannot have a rejection reason."
                        )
                    reason = None
                elif not bool(self.valid_mask[row, col]):
                    reason = supplied_reason or "INELIGIBLE"
                elif not bool(pose_ok[row, col]):
                    reason = "INVALID_GRASP_POSE"
                elif not bool(cost_ok[row, col]):
                    reason = "NONFINITE_GRASP_COST"
                else:
                    reason = "INVALID_OPENING_WIDTH"
                row_reasons.append(reason)
                candidate_id = self.grasp_ids[row][col] if self.grasp_ids else ""
                if self.grasp_ids and (
                    not isinstance(candidate_id, str) or not candidate_id
                ):
                    raise ValueError("grasp_ids must contain non-empty strings.")
                if not candidate_id:
                    candidate_id = (
                        _grasp_id(
                            self.poses[row, col],
                            (
                                None
                                if widths is None
                                else round(float(widths[row, col]), 6)
                            ),
                            self.frame,
                        )
                        if bool(valid[row, col])
                        else f"invalid-{row}-{col}"
                    )
                row_ids.append(candidate_id)
            reasons.append(tuple(row_reasons))
            ids.append(tuple(row_ids))
        object.__setattr__(self, "valid_mask", valid.clone())
        object.__setattr__(
            self,
            "poses",
            torch.where(
                valid[..., None, None],
                self.poses,
                torch.eye(4, device=self.poses.device, dtype=self.poses.dtype),
            ).clone(),
        )
        object.__setattr__(
            self, "costs", torch.where(valid, self.costs, torch.inf).clone()
        )
        object.__setattr__(
            self,
            "opening_widths",
            None if widths is None else torch.where(valid, widths, 0).clone(),
        )
        object.__setattr__(self, "grasp_ids", tuple(ids))
        object.__setattr__(self, "rejection_reasons", tuple(reasons))

    @classmethod
    def from_ragged(
        cls,
        results: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        object_poses: torch.Tensor | None = None,
        opening_widths: Sequence[torch.Tensor] | None = None,
        frame: str = "local_arena",
        id_namespace: str = "",
    ) -> GraspCandidateBatch:
        """Pad legacy candidates while retaining invalid entries for diagnostics.

        Args:
            results: One ``(poses, costs)`` pair per object, with shapes
                ``(G_i,4,4)`` and ``(G_i,)``.
            object_poses: Optional ``(S,4,4)`` reference poses used to derive
                object-relative IDs that survive environment translation.
            opening_widths: Optional width vector for each ragged row.
            frame: Explicit reference frame for returned poses.
            id_namespace: Stable source/configuration namespace for IDs.

        Returns:
            Owned padded candidates; an empty input has shape ``(0,0,4,4)``.
        """
        if opening_widths is not None and len(opening_widths) != len(results):
            raise ValueError("opening_widths must contain one vector per object.")
        if results and (
            len(results[0]) != 2 or not isinstance(results[0][0], torch.Tensor)
        ):
            raise ValueError("results must contain (poses, costs) tensor pairs.")
        device = results[0][0].device if results else torch.device("cpu")
        dtype = results[0][0].dtype if results else torch.float32
        counts: list[int] = []
        for poses, costs in results:
            if (
                not isinstance(poses, torch.Tensor)
                or not poses.is_floating_point()
                or poses.ndim != 3
                or poses.shape[1:] != (4, 4)
            ):
                raise ValueError("ragged poses must have shape (G,4,4).")
            if (
                not isinstance(costs, torch.Tensor)
                or not costs.is_floating_point()
                or costs.shape != (poses.shape[0],)
            ):
                raise ValueError("ragged costs must have shape (G,).")
            if poses.device != device or costs.device != device:
                raise ValueError("All ragged candidate tensors must share a device.")
            counts.append(poses.shape[0])
        if object_poses is not None:
            if (
                not isinstance(object_poses, torch.Tensor)
                or not object_poses.is_floating_point()
                or object_poses.shape != (len(results), 4, 4)
                or not bool(_se3_mask(object_poses).all())
            ):
                raise ValueError("object_poses must contain one valid SE(3) per row.")
            object_poses = object_poses.to(device=device, dtype=dtype)
        count = max(counts, default=0)
        poses_out = torch.eye(4, dtype=dtype, device=device).repeat(
            len(results), count, 1, 1
        )
        costs_out = torch.full(
            (len(results), count), torch.inf, device=device, dtype=dtype
        )
        mask = torch.zeros((len(results), count), dtype=torch.bool, device=device)
        widths_out = None if opening_widths is None else torch.zeros_like(costs_out)
        for row, ((poses, costs), length) in enumerate(zip(results, counts)):
            poses_out[row, :length] = poses
            costs_out[row, :length] = costs
            mask[row, :length] = True
            if opening_widths is not None:
                widths = opening_widths[row]
                if (
                    not isinstance(widths, torch.Tensor)
                    or widths.shape != (length,)
                    or not widths.is_floating_point()
                    or widths.device != device
                ):
                    raise ValueError(
                        "Each opening-width vector must match its candidate row."
                    )
                assert widths_out is not None
                widths_out[row, :length] = widths
        padding_reasons = tuple(
            tuple(None if col < counts[row] else "PADDING" for col in range(count))
            for row in range(len(results))
        )
        result = cls(
            poses_out,
            costs_out,
            mask,
            widths_out,
            frame=frame,
            rejection_reasons=padding_reasons,
        )
        ids: list[tuple[str, ...]] = []
        for row in range(len(results)):
            canonical = result.poses[row]
            if object_poses is not None:
                canonical = torch.linalg.inv(object_poses[row]) @ canonical
            ids.append(
                tuple(
                    (
                        _grasp_id(
                            canonical[col],
                            (
                                None
                                if widths_out is None
                                else round(float(widths_out[row, col]), 6)
                            ),
                            id_namespace or frame,
                        )
                        if bool(result.valid_mask[row, col])
                        else result.grasp_ids[row][col]
                    )
                    for col in range(count)
                )
            )
        return cls(
            result.poses,
            result.costs,
            result.valid_mask,
            result.opening_widths,
            tuple(ids),
            frame,
            result.rejection_reasons,
        )

    def row(self, index: int) -> GraspCandidateBatch:
        """Return an owned single-row batch.

        Args:
            index: Zero-based source row.

        Returns:
            A candidate batch preserving its leading singleton dimension.
        """
        if not 0 <= index < self.poses.shape[0]:
            raise IndexError("candidate row index out of range")
        return GraspCandidateBatch(
            self.poses[index : index + 1],
            self.costs[index : index + 1],
            self.valid_mask[index : index + 1],
            (
                None
                if self.opening_widths is None
                else self.opening_widths[index : index + 1]
            ),
            (self.grasp_ids[index],),
            self.frame,
            (self.rejection_reasons[index],),
        )
