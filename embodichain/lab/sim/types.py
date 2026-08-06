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

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from tensordict import TensorDict

__all__ = ["Array", "Device", "EnvObs", "EnvAction"]

Array = torch.Tensor | np.ndarray | Sequence
Device = str | torch.device

EnvObs = TensorDict[str, torch.Tensor | TensorDict[str, torch.Tensor]]

EnvAction = (
    torch.Tensor
    | np.ndarray
    | Sequence
    | Mapping[str, Array]
    | TensorDict[str, torch.Tensor]
)
