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

"""Simulation-independent learned dynamics mathematics."""

from __future__ import annotations
from .codec import (
    NeRDStateCodec,
    quat_multiply,
    quat_rotate,
    quat_inverse,
    plane_contact_features,
    append_history,
    reset_history,
)

from .reference import reference_task_values

__all__ = [
    "reference_task_values",
    "NeRDStateCodec",
    "quat_multiply",
    "quat_rotate",
    "quat_inverse",
    "plane_contact_features",
    "append_history",
    "reset_history",
]
