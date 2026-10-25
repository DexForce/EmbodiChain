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

"""Optional RigidFormer adapter with an explicit reference-geometry contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from embodichain.utils import configclass

__all__ = ["RIGIDFORMER_REVISION", "RigidFormerCfg", "RigidFormerModel"]

RIGIDFORMER_REVISION = "4ab4edac7271b49484748d6bf8086a497f13272d"


@configclass
class RigidFormerCfg:
    """Configure the optional upstream network and its distance-buffer budget.

    The defaults match the pinned upstream Box2D experiment. The byte budget
    bounds the dense nearest-neighbour distance tensor, not total GPU memory.
    Each distance counts at least four bytes because autocast may promote
    reduced-precision geometry to an FP32 ``cdist`` result.

    Args:
        dim: Object/anchor feature width.
        dim_head: Per-head attention width.
        heads: Attention heads per layer.
        num_register_tokens: Additional object-attention register tokens.
        object_self_attn_depth: Object interaction layers.
        anchor_cross_attn_depth: Anchor-to-object interaction layers.
        object_hidden_layers: Object-layer outputs consumed by anchor layers.
        num_anchors: Material points used to advance each rigid object.
        vertex_properties_dim: Number of object/point attribute channels.
        max_pairwise_bytes: Maximum dense nearest-neighbour distance bytes.
    """

    dim: int = 128
    dim_head: int = 32
    heads: int = 4
    num_register_tokens: int = 4
    object_self_attn_depth: int = 3
    anchor_cross_attn_depth: int = 3
    object_hidden_layers: tuple[int, ...] = (0, 1, 3)
    num_anchors: int = 4
    vertex_properties_dim: int = 3
    max_pairwise_bytes: int = 256 * 1024 * 1024


class RigidFormerModel(nn.Module):
    """Wrap the pinned RigidFormer architecture without importing simulation.

    Args:
        cfg: Upstream architecture and per-call pairwise-distance byte budget.

    Reference geometry and point identities must remain aligned with the
    current/previous point clouds. The adapter exposes forward prediction and
    parameter training; action gradients and a physics-solver interface are
    outside its contract. Finite padding with unequal point lengths is supported
    only when the padded point axis is at most 128; the pinned upstream
    hierarchical encoder does not preserve masking beyond that boundary.
    """

    action_conditioned: bool = False
    qualified_state_gradients: bool = False

    def __init__(self, cfg: RigidFormerCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else RigidFormerCfg()
        self.cfg.validate()
        if self.cfg.max_pairwise_bytes < 1:
            raise ValueError("max_pairwise_bytes must be positive")
        try:
            from rigidformer import Rigidformer
        except ImportError as exc:
            raise ImportError(
                "RigidFormer requires the optional rigidformer dependency; "
                "install embodichain[rigidformer] (rigidformer==0.2.0). "
                f"For frozen reference revision {RIGIDFORMER_REVISION}, use "
                "examples/learning/rigidformer/requirements-reference.txt"
            ) from exc
        fields = self.cfg.to_dict()
        fields.pop("max_pairwise_bytes")
        self.network = Rigidformer(**fields)

    def _check(self, values: Mapping[str, Any]) -> None:
        current = values["object_pos"]
        if current.ndim != 4 or current.shape[-1] != 3:
            raise ValueError("object_pos must have shape [batch, object, point, 3]")
        if min(current.shape[:3]) < 1:
            raise ValueError("Point batches must contain objects and points")
        for key in ("object_pos_prev", "object_first_frame_pos"):
            if values[key].shape != current.shape:
                raise ValueError(f"{key} must match the current point layout")
        target = values.get("object_pos_next")
        if target is not None and target.shape != current.shape:
            raise ValueError("object_pos_next must match the current point layout")
        delta_times = values["delta_times"]
        if delta_times.shape != current.shape[:1]:
            raise ValueError("delta_times must have shape [batch]")
        if not torch.isfinite(delta_times).all() or (delta_times <= 0).any():
            raise ValueError("delta_times must be finite and positive")
        batch, objects, points = current.shape[:3]
        object_lens = values.get("object_lens")
        point_lens = values.get("object_point_lens")
        for name, lengths, shape, maximum, minimum in (
            ("object_lens", object_lens, (batch,), objects, 1),
            (
                "object_point_lens",
                point_lens,
                (batch, objects),
                points,
                self.cfg.num_anchors,
            ),
        ):
            if lengths is not None and (
                lengths.shape != shape
                or lengths.dtype != torch.long
                or lengths.device != current.device
                or (lengths < minimum).any()
                or (lengths > maximum).any()
            ):
                raise ValueError(
                    f"{name} must be int64 {shape} counts in [{minimum}, {maximum}] "
                    "on the geometry device, including padded objects"
                )
        if point_lens is not None and points > 128 and (point_lens != points).any():
            raise ValueError(
                "Unequal object_point_lens requires a padded point axis <= 128; "
                "the pinned upstream PointNet FPS/KNN leaks finite padding above 128"
            )
        anchors = values.get("anchor_indices")
        if anchors is not None:
            valid_points = points if point_lens is None else point_lens[..., None]
            if (
                anchors.ndim != 3
                or anchors.shape[:2] != (batch, objects)
                or anchors.shape[-1] < 1
                or anchors.dtype != torch.long
                or anchors.device != current.device
                or (anchors < 0).any()
                or (anchors >= valid_points).any()
            ):
                raise ValueError(
                    "anchor_indices must index valid points in every object"
                )
        # cdist may run in FP32 under autocast, including for FP16/BF16 inputs.
        distance_bytes = (
            batch * (objects * points) ** 2 * max(current.element_size(), 4)
        )
        if distance_bytes > self.cfg.max_pairwise_bytes:
            raise ValueError(
                f"Pairwise distances require {distance_bytes} bytes, "
                f"exceeding {self.cfg.max_pairwise_bytes}"
            )

    def forward(
        self,
        *,
        object_pos: torch.Tensor,
        object_pos_prev: torch.Tensor,
        object_first_frame_pos: torch.Tensor,
        vertex_properties: torch.Tensor,
        delta_times: torch.Tensor,
        object_pos_next: torch.Tensor | None = None,
        anchor_indices: torch.Tensor | None = None,
        object_point_lens: torch.Tensor | None = None,
        object_lens: torch.Tensor | None = None,
    ) -> Any:
        """Predict next points, or return upstream loss with an explicit target.

        Args:
            object_pos: Current points with axes ``[B, O, P, 3]``.
            object_pos_prev: Previous points with identical point identities.
            object_first_frame_pos: Reference points for rigid projection.
            vertex_properties: Per-object or per-point attributes.
            delta_times: Positive integration intervals in seconds, ``[B]``.
            object_pos_next: Optional training target, never a model input.
            anchor_indices: Optional fixed per-object anchor indices.
            object_point_lens: Optional valid point counts, ``[B, O]``, each at
                least ``cfg.num_anchors`` including padded objects. Unequal
                counts require a padded point axis of at most 128.
            object_lens: Optional valid object counts, ``[B]``.

        Returns:
            Upstream prediction namedtuple, or scalar loss and loss breakdown.
        """
        values = dict(
            object_pos=object_pos,
            object_pos_prev=object_pos_prev,
            object_first_frame_pos=object_first_frame_pos,
            vertex_properties=vertex_properties,
            delta_times=delta_times,
            object_pos_next=object_pos_next,
            anchor_indices=anchor_indices,
            object_point_lens=object_point_lens,
            object_lens=object_lens,
        )
        self._check(values)
        return self.network(**values)

    def rollout(
        self,
        object_positions: Sequence[torch.Tensor],
        *,
        vertex_properties: torch.Tensor,
        delta_times: torch.Tensor,
        num_steps: int,
        anchor_indices: torch.Tensor | None = None,
        object_point_lens: torch.Tensor | None = None,
        object_lens: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Predict a fixed-step trajectory while preserving the first geometry.

        Args:
            object_positions: At least two aligned initial frames.
            vertex_properties: Static object/point attributes.
            delta_times: Fixed positive time interval per batch row.
            num_steps: Number of new frames, excluding the supplied history.
            anchor_indices: Optional initial anchors reused throughout rollout.
            object_point_lens: Optional valid point counts with the same masking
                limits as :meth:`forward`.
            object_lens: Optional valid object counts.

        Returns:
            Initial frames followed by the predicted frames.
        """
        if len(object_positions) < 2 or type(num_steps) is not int or num_steps < 1:
            raise ValueError("Rollout requires two frames and positive num_steps")
        self._check(
            dict(
                object_pos=object_positions[-1],
                object_pos_prev=object_positions[-2],
                object_first_frame_pos=object_positions[0],
                delta_times=delta_times,
                anchor_indices=anchor_indices,
                object_point_lens=object_point_lens,
                object_lens=object_lens,
            )
        )
        from rigidformer import RigidformerRolloutWrapper

        return RigidformerRolloutWrapper(self.network)(
            object_positions=list(object_positions),
            vertex_properties=vertex_properties,
            delta_times=delta_times,
            num_steps=num_steps,
            anchor_indices=anchor_indices,
            object_point_lens=object_point_lens,
            object_lens=object_lens,
        )
