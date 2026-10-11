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

"""Compatibility alias for :mod:`embodichain_robots.dexforce_w1.hand_specs`."""

from __future__ import annotations

from importlib import import_module
import sys

__all__ = [
    "W1HandSideSpec",
    "W1HandSpec",
    "get_default_w1_hand_version",
    "get_w1_hand_spec",
    "normalize_w1_hand_mappings",
]

sys.modules[__name__] = import_module("embodichain_robots.dexforce_w1.hand_specs")
