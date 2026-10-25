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

"""RigidFormer objective and conversion of explicit geometry supervision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from embodichain.learning.sl import LossResult, SupervisedBatch

__all__ = ["RigidFormerObjective", "rigidformer_batch"]

_INPUTS = (
    "object_pos",
    "object_pos_prev",
    "object_first_frame_pos",
    "vertex_properties",
    "delta_times",
)
_OPTIONAL = ("anchor_indices", "object_point_lens", "object_lens")


def rigidformer_batch(
    values: Mapping[str, torch.Tensor], *, metadata: Mapping[str, Any] | None = None
) -> SupervisedBatch:
    """Separate next-frame targets from causal RigidFormer inputs.

    Args:
        values: Owned geometry tensors in the upstream keyword layout.
        metadata: Optional source identity and frame provenance.

    Returns:
        A supervised batch with target geometry outside the input mapping.
    """
    missing = set((*_INPUTS, "object_pos_next")) - values.keys()
    if missing:
        raise ValueError(f"Missing geometry fields: {sorted(missing)}")
    inputs = {key: values[key] for key in (*_INPUTS, *_OPTIONAL) if key in values}
    context = dict(metadata or {})
    context.update(
        {key: values[key] for key in ("episode_index", "frame_index") if key in values}
    )
    return SupervisedBatch(
        inputs=inputs,
        targets=values["object_pos_next"],
        metadata=context,
    )


class RigidFormerObjective:
    """Preserve upstream anchor position/acceleration losses and their weights."""

    def compute_loss(self, model: nn.Module, batch: SupervisedBatch) -> LossResult:
        """Compute loss without placing the next-frame target in model inputs.

        Args:
            model: RigidFormer-compatible model returning loss and breakdown.
            batch: Explicit geometry inputs and next-frame targets.

        Returns:
            Scalar loss and upstream position/acceleration diagnostic terms,
            weighted by the valid object count used by the upstream mean.
        """
        if batch.masks:
            raise ValueError(
                "RigidFormer masks use object_lens/object_point_lens inputs; "
                "arbitrary temporal loss masks are unsupported"
            )
        loss, terms = model(**batch.inputs, object_pos_next=batch.targets)
        object_lens = batch.inputs.get("object_lens")
        weight = (
            float(object_lens.sum())
            if object_lens is not None
            else float(batch.targets.shape[0] * batch.targets.shape[1])
        )
        return LossResult(
            loss=loss,
            metrics={"acceleration": terms.acceleration, "position": terms.position},
            weight=weight,
        )
