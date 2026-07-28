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

"""Shared object-classification vocabulary for generation and runtime heuristics.

These labels define reviewed classification behavior rather than tunable
numeric policy, so they intentionally remain in Python instead of defaults
YAML.
"""

from __future__ import annotations

__all__ = [
    "BOTTLE_LIKE_KEYWORDS",
    "CONTAINER_LIKE_KEYWORDS",
    "CUP_LIKE_KEYWORDS",
    "FLAT_CARRIER_KEYWORDS",
    "ROD_LIKE_KEYWORDS",
    "SHORT_BOTTLE_LIKE_KEYWORDS",
    "SHORT_CUP_LIKE_KEYWORDS",
    "UPRIGHTABLE_KEYWORDS",
]

BOTTLE_LIKE_KEYWORDS = (
    "bottle",
    "can",
    "jar",
    "tin",
    "soda",
    "cola",
    "罐头",
    "易拉罐",
    "瓶",
    "瓶子",
)
CUP_LIKE_KEYWORDS = (
    "cup",
    "mug",
    "paper cup",
    "water cup",
    "纸杯",
    "水杯",
    "杯子",
    "马克杯",
    "茶杯",
)
SHORT_BOTTLE_LIKE_KEYWORDS = frozenset({"can", "jar", "tin"})
SHORT_CUP_LIKE_KEYWORDS = frozenset({"cup", "mug"})
UPRIGHTABLE_KEYWORDS = (*BOTTLE_LIKE_KEYWORDS, *CUP_LIKE_KEYWORDS)

CONTAINER_LIKE_KEYWORDS = (
    "pot",
    "pan",
    "wok",
    "skillet",
    "saucepan",
    "tray",
    "plate",
    "bowl",
    "basket",
    "container",
    "dish",
    "basin",
    "cup",
    "mug",
    "锅",
    "平底锅",
    "炒锅",
    "托盘",
    "盘",
    "盘子",
    "碗",
    "篮",
    "篮子",
    "容器",
    "盆",
    "杯",
)
ROD_LIKE_KEYWORDS = (
    "umbrella",
    "rod",
    "bar",
    "stick",
    "tube",
    "cylinder",
    "cylindrical",
    "pole",
    "baton",
    "rectangular",
    "cuboid",
    "雨伞",
    "伞",
    "杆",
    "棒",
    "棍",
    "柱",
    "圆柱",
    "长方体",
    "矩形",
    "木条",
)
FLAT_CARRIER_KEYWORDS = ("plate", "dish", "platter", "盘", "盘子")
