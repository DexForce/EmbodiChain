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

"""Compatibility facade for object semantics and relation prompt language.

Object classifiers are owned by ``domain.object_semantics``. Generation-only
English rendering is owned by ``generation.relation_language``. Keeping this
facade preserves historical imports while making new dependencies explicit.
"""

from __future__ import annotations

from embodichain.gen_sim.action_agent_pipeline.domain.object_semantics import (
    BOTTLE_LIKE_KEYWORDS,
    CONTAINER_LIKE_KEYWORDS,
    CUP_LIKE_KEYWORDS,
    FLAT_CARRIER_KEYWORDS,
    ROD_LIKE_KEYWORDS,
    SHORT_BOTTLE_LIKE_KEYWORDS,
    SHORT_CUP_LIKE_KEYWORDS,
    UPRIGHTABLE_KEYWORDS,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relation_language import (
    RELATIVE_RELATION_PHRASES,
    relative_relation_phrase,
)

__all__ = [
    "BOTTLE_LIKE_KEYWORDS",
    "CONTAINER_LIKE_KEYWORDS",
    "CUP_LIKE_KEYWORDS",
    "FLAT_CARRIER_KEYWORDS",
    "ROD_LIKE_KEYWORDS",
    "RELATIVE_RELATION_PHRASES",
    "SHORT_BOTTLE_LIKE_KEYWORDS",
    "SHORT_CUP_LIKE_KEYWORDS",
    "UPRIGHTABLE_KEYWORDS",
    "relative_relation_phrase",
]
