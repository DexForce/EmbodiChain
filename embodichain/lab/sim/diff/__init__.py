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
"""Differentiable Newton stepping for EmbodiChain.

Bridges DexSim's manager-owned differentiable trajectory transaction into
PyTorch autograd via a :class:`torch.autograd.Function`, and exposes a
:class:`tape_context` manager for advanced users who want to compose their
own Warp kernels.
"""

from __future__ import annotations

from .bridge import (
    NewtonStepFunc,
    differentiable_step,
    tape_context,
)

__all__ = [
    "NewtonStepFunc",
    "differentiable_step",
    "tape_context",
]
