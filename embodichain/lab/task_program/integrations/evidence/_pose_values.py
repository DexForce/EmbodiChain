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

"""Frame/geometry consistency for offline observations."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any

__all__: list[str] = []


def region_poses_match(
    measurements: Mapping[str, Any],
    object_pose: torch.Tensor,
    reference_pose: torch.Tensor,
    num_envs: int,
) -> bool:
    import torch
    from embodichain.compute.geometry.rigid import (
        validate_rigid_pose_batch as _valid_pose,
    )

    try:
        observed = torch.tensor(measurements["object_pose"]).to(object_pose)
        reference = torch.tensor(measurements["reference_pose"]).to(reference_pose)
        _valid_pose(observed, num_envs)
        _valid_pose(reference, num_envs)
        return bool(
            torch.allclose(observed, object_pose, atol=0.0001, rtol=0)
            and torch.allclose(reference, reference_pose, atol=0.0001, rtol=0)
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
