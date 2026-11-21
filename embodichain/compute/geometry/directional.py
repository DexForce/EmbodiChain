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

"""World-axis operands for the existing directional spatial relations."""

from __future__ import annotations
from types import MappingProxyType

__all__ = ["DIRECTIONAL_RELATION_AXES"]

DIRECTIONAL_RELATION_AXES = MappingProxyType(
    {
        "left_of": ((1, -1),),
        "right_of": ((1, 1),),
        "front_of": ((0, -1),),
        "behind": ((0, 1),),
        "front_left_of": ((0, -1), (1, -1)),
        "front_right_of": ((0, -1), (1, 1)),
        "back_left_of": ((0, 1), (1, -1)),
        "back_right_of": ((0, 1), (1, 1)),
    }
)
