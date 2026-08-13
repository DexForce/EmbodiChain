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

"""Load the policy weights stored in an EmbodiChain checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

__all__ = ["load_policy_state_dict"]


def load_policy_state_dict(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Load the ``policy`` state mapping from an EmbodiChain ``.pt`` file.

    Args:
        checkpoint: EmbodiChain training checkpoint.
        map_location: Device passed to :func:`torch.load`.

    Returns:
        Policy state mapping ready for ``module.load_state_dict()``.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        TypeError: If the checkpoint or policy payload is not a mapping.
        ValueError: If the checkpoint has no policy weights.
    """
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Policy checkpoint root must be a mapping")
    state = payload.get("policy")
    if not isinstance(state, Mapping):
        raise TypeError("Policy checkpoint field 'policy' must be a mapping")
    if not state:
        raise ValueError("Policy checkpoint field 'policy' is empty")
    return state
