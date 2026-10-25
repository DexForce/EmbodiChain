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

"""Shared rollout metrics and teacher-content identity without simulator imports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch

__all__: list[str] = []


def _teacher_row_sha256(record: Mapping[str, Any], values: Sequence[np.ndarray]) -> str:
    """Hash ordered canonical state/control content and logical trajectory identity."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": "teacher-content-v1",
                "id": record["id"],
                "sampling_seed": record.get("metadata", {}).get("sampling_seed"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for name, value in zip(("q", "qd", "joint_f"), values):
        array = np.ascontiguousarray(value)
        digest.update(
            json.dumps(
                {"name": name, "shape": array.shape, "dtype": array.dtype.str},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


def _error_values(
    robot: str,
    q: torch.Tensor,
    qd: torch.Tensor,
    target_q: torch.Tensor,
    target_qd: torch.Tensor,
) -> torch.Tensor:
    # Accumulate errors in float64 so large but finite FP32 states remain
    # distinguishable from actual simulator NaN/Inf failures.
    q, qd, target_q, target_qd = (
        value.double() for value in (q, qd, target_q, target_qd)
    )
    start = 7 if robot == "ant" else 0
    dq = q[..., start:] - target_q[..., start:]
    if robot == "ant":
        # Ant scalar joints have finite limits, so a full-turn prediction error
        # is a real error. Only the Cartpole continuous pole coordinate wraps.
        position = (q[..., :3] - target_q[..., :3]).norm(dim=-1)
        dot = (
            (
                torch.nn.functional.normalize(q[..., 3:7], dim=-1)
                * torch.nn.functional.normalize(target_q[..., 3:7], dim=-1)
            )
            .sum(-1)
            .abs()
            .clamp(0, 1)
        )
        orientation = 2 * torch.acos(dot)
    else:
        dq = torch.stack(
            (
                dq[..., 0],
                torch.remainder(dq[..., 1] + torch.pi, 2 * torch.pi) - torch.pi,
            ),
            -1,
        )
        position = orientation = torch.zeros_like(q[..., 0])
    return torch.stack(
        (dq.abs().mean(-1), (qd - target_qd).abs().mean(-1), position, orientation), -1
    )


def _safe_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
